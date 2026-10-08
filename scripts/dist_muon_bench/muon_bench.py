# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DistMuon benchmark and numerics driver.

Modes (``--mb.mode``):
  optim  Build model and optimizer through the production path, then drive
         DistMuon alone with deterministic synthetic gradients. Isolates the
         optimizer from data loading and forward/backward.
  train  Run the unmodified training loop and record the optimizer phase.

Both modes write ``rank<r>.json`` with per-step optimizer wall time, peak and
resident allocator bytes, and SHA-256 digests of every Muon parameter and
momentum shard after the last step. Identical digests across two checkouts
mean the optimizer update is bitwise identical.

Run-level settings (seed, determinism, steps, TensorBoard, parallelism
degrees) are environment variables; see ``common.apply_harness_env``.

Options (``--mb.key=value``):
  out            output directory (required)
  steps          optimizer steps in optim mode (default 12)
  warmup         steps excluded from the timing summary (default 3)
  profile_step   1-based step captured with Kineto, 0 disables (default 0)
  profile_ranks  comma-separated ranks that write a trace (default 0)
  grad_scale     synthetic gradient scale (default 1e-3)
"""

from __future__ import annotations

import hashlib
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402  # pyrefly: ignore [missing-import]
    build_engine,
    dist_muon_optimizers,
    dump_json,
    harness_args,
    load_config,
    tensor_digest,
)


def _seed(*parts) -> int:
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:7], "little")


def _storage_offsets(param, rank: int) -> tuple:
    from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
        _dtensor_storage_region_for_participant,
    )

    try:
        return _dtensor_storage_region_for_participant(param, rank).offsets
    except RuntimeError:
        coordinate = param.device_mesh.get_coordinate()
        return ("coordinate", tuple(coordinate or ()))


class _SyntheticGradients:
    """Deterministic gradients keyed by parameter, step, and logical region."""

    def __init__(self, engine, scale: float) -> None:
        from torch.distributed.tensor import DTensor

        self._dtensor = DTensor
        self._scale = scale
        self._entries = []
        rank = dist.get_rank()
        for optimizer in engine.optim.optimizers:
            names = optimizer.param_groups[0].get("param_names")
            for group in optimizer.param_groups:
                for index, param in enumerate(group["params"]):
                    name = names[index] if names is not None else f"p{id(param)}"
                    offsets = (
                        _storage_offsets(param, rank)
                        if isinstance(param, DTensor)
                        else ()
                    )
                    self._entries.append((name, param, offsets))
        self._generator = torch.Generator(device=engine.device)

    def assign(self, step: int) -> None:
        for name, param, offsets in self._entries:
            is_dtensor = isinstance(param, self._dtensor)
            local = param.to_local() if is_dtensor else param
            self._generator.manual_seed(_seed(name, step, offsets))
            grad = torch.randn(
                local.shape,
                generator=self._generator,
                device=local.device,
                dtype=local.dtype,
            ).mul_(self._scale)
            if is_dtensor:
                grad = self._dtensor.from_local(
                    grad,
                    param.device_mesh,
                    param.placements,
                    shape=param.shape,
                    stride=param.stride(),
                    run_check=False,
                )
            param.grad = grad


class _StepRecorder:
    def __init__(self, device: torch.device) -> None:
        # pyrefly: ignore [read-only]
        self._device = device
        self.records: list[dict] = []

    def run(self, step: int, fn) -> None:
        arrival = time.perf_counter()
        torch.cuda.synchronize(self._device)
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)
        resident = torch.cuda.memory_allocated(self._device)
        start = time.perf_counter()
        fn()
        host = time.perf_counter() - start
        torch.cuda.synchronize(self._device)
        total = time.perf_counter() - start
        previous_end = self.records[-1]["end_time"] if self.records else None
        self.records.append(
            {
                "step": step,
                "host_ms": host * 1e3,
                "total_ms": total * 1e3,
                # Wall time since the previous optimizer step ended: in train
                # mode this is the forward/backward (plus data) of one step.
                "since_previous_ms": (
                    None if previous_end is None else (arrival - previous_end) * 1e3
                ),
                "end_time": time.perf_counter(),
                "resident_bytes": resident,
                "peak_bytes": torch.cuda.max_memory_allocated(self._device),
                "reserved_bytes": torch.cuda.memory_reserved(self._device),
            }
        )


def _digests(muons) -> dict:
    out = {}
    for muon in muons:
        for layout in muon._parameter_compute_layouts:
            param = layout.param
            state = muon.state.get(param, {})
            momentum = state.get("momentum_buffer")
            out[layout.fqn] = {
                "param": tensor_digest(param.to_local()),
                "momentum": (
                    tensor_digest(momentum.to_local()) if momentum is not None else None
                ),
            }
    return out


def _reserved_bytes(muons) -> int:
    total = 0
    for muon in muons:
        runtime = muon._redistribution_runtime
        slots = [runtime._local_slot]
        if runtime._context is not None:
            slots.extend(slot.buffers for slot in runtime._context.slots)
        for slot in slots:
            for reserved in slot.buffers.values():
                for name in (
                    "storage_exchange",
                    "compute_exchange",
                    "compute_scratch",
                    "storage_scratch",
                ):
                    tensor = getattr(reserved, name, None)
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
    return total


def _summary(records: list[dict], warmup: int) -> dict:
    timed = records[warmup:] or records
    totals = [record["total_ms"] for record in timed]
    hosts = [record["host_ms"] for record in timed]
    return {
        "num_timed_steps": len(timed),
        "total_ms_median": statistics.median(totals),
        "total_ms_mean": statistics.fmean(totals),
        "total_ms_min": min(totals),
        "total_ms_max": max(totals),
        "host_ms_median": statistics.median(hosts),
        "peak_bytes_max": max(record["peak_bytes"] for record in timed),
        "resident_bytes": timed[-1]["resident_bytes"],
        "transient_bytes_max": max(
            record["peak_bytes"] - record["resident_bytes"] for record in timed
        ),
    }


def _git_head() -> str:
    import subprocess

    import torchtitan

    repo = os.path.dirname(os.path.dirname(os.path.abspath(torchtitan.__file__)))
    marker = os.path.join(repo, ".commit")
    if os.path.exists(marker):
        return open(marker).read().strip()
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _write(out_dir, rank, mode, muons, recorder, warmup, extra=None) -> None:
    torch.cuda.synchronize()
    payload = {
        "rank": rank,
        "mode": mode,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "commit": _git_head(),
        "reserved_buffer_bytes": _reserved_bytes(muons),
        "summary": _summary(recorder.records, warmup) if recorder.records else {},
        "steps": recorder.records,
        "digests": _digests(muons),
    }
    payload.update(extra or {})
    dump_json(os.path.join(out_dir, f"rank{rank}.json"), payload)
    if rank == 0 and recorder.records:
        summary = payload["summary"]
        print(
            f"[muon_bench] rank0 optimizer median {summary['total_ms_median']:.2f} ms "
            f"(host {summary['host_ms_median']:.2f} ms), "
            f"transient {summary['transient_bytes_max'] / 2**20:.1f} MiB, "
            f"reserved buffers {payload['reserved_buffer_bytes'] / 2**20:.1f} MiB",
            flush=True,
        )


def _apply_runtime_overrides() -> None:
    """Experiment knobs that patch module constants before the plan is built."""
    slots = os.environ.get("MB_SLOTS")
    if slots:
        from torchtitan.distributed.flex_shard import _optimizer_reshard_runtime

        # pyrefly: ignore [bad-assignment]
        _optimizer_reshard_runtime._NUM_PIPELINE_SLOTS = int(slots)
    from torchtitan.distributed.flex_shard import dist_muon

    if os.environ.get("MB_NO_NS_GRAPHS"):
        if hasattr(dist_muon, "_NEWTON_SCHULZ_GRAPH_MAX_NUMEL"):
            dist_muon._NEWTON_SCHULZ_GRAPH_MAX_NUMEL = -1
    # Sweep knobs: largest Newton-Schulz input replayed from a CUDA graph and
    # the batch size above which a matrix batch is orthogonalized in pieces.
    # Both are plain integers (e.g. 16777216); they only exist on the
    # optimized branch and are ignored elsewhere.
    for env, name in (
        ("MB_NS_GRAPH_MAX_NUMEL", "_NEWTON_SCHULZ_GRAPH_MAX_NUMEL"),
        ("MB_NS_PIECE_NUMEL", "_NEWTON_SCHULZ_PIECE_NUMEL"),
    ):
        if os.environ.get(env) and hasattr(dist_muon, name):
            setattr(dist_muon, name, int(os.environ[env]))


def run_optim(options, argv) -> None:
    _apply_runtime_overrides()
    out_dir = options["out"]
    steps = int(options.get("steps", 12))
    warmup = int(options.get("warmup", 3))
    profile_step = int(options.get("profile_step", 0))
    profile_ranks = {int(r) for r in options.get("profile_ranks", "0").split(",")}

    config, engine = build_engine(argv)
    muons = dist_muon_optimizers(engine)
    rank = dist.get_rank() if dist.is_initialized() else 0
    gradients = _SyntheticGradients(engine, float(options.get("grad_scale", 1e-3)))
    recorder = _StepRecorder(engine.device)

    def optimizer_step() -> None:
        for muon in muons:
            muon.step()

    for step in range(1, steps + 1):
        gradients.assign(step)
        if step == profile_step and rank in profile_ranks:
            from torch.profiler import profile, ProfilerActivity

            torch.cuda.synchronize()
            dist.barrier()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                optimizer_step()
                torch.cuda.synchronize()
            os.makedirs(out_dir, exist_ok=True)
            prof.export_chrome_trace(os.path.join(out_dir, f"trace_rank{rank}.json.gz"))
        elif step == profile_step:
            # Keep every rank in the same collective sequence.
            torch.cuda.synchronize()
            dist.barrier()
            optimizer_step()
            torch.cuda.synchronize()
        else:
            recorder.run(step, optimizer_step)
        for optimizer in engine.optim.optimizers:
            optimizer.zero_grad(set_to_none=True)

    _write(out_dir, rank, "optim", muons, recorder, warmup)
    engine.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def run_train(options, argv) -> None:
    _apply_runtime_overrides()
    out_dir = options["out"]
    warmup = int(options.get("warmup", 3))
    config = load_config(argv)
    # Metrics (TensorBoard under MB_TB=1) land next to the harness output.
    config.dump_folder = os.path.join(out_dir, "dump")
    trainer = config.build()
    engine = trainer.engine
    muons = dist_muon_optimizers(engine)
    rank = dist.get_rank()
    recorder = _StepRecorder(engine.device)

    for muon in muons:
        original = muon.step

        def timed_step(closure=None, *, _original=original):
            recorder.run(len(recorder.records) + 1, lambda: _original(closure))

        muon.step = timed_step

    try:
        trainer.train()
    finally:
        _write(out_dir, rank, "train", muons, recorder, warmup)
        trainer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    options, argv = harness_args()
    if "out" not in options:
        raise SystemExit("--mb.out=<dir> is required")
    mode = options.get("mode", "optim")
    if mode == "optim":
        run_optim(options, argv)
    elif mode == "train":
        run_train(options, argv)
    else:
        raise SystemExit(f"unknown --mb.mode {mode!r}")


if __name__ == "__main__":
    main()
