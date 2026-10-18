# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run real K3 training in separate correctness and performance modes."""

import argparse
import hashlib
import math
import os
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from torchtitan.distributed.flex_shard import dist_muon, DistMuon
from torchtitan.distributed.flex_shard._optimizer_reshard_schedule import (
    _LocalBucketPlan,
)
from torchtitan.observability.logging import init_logger

from .common import (
    environment,
    experiment_hashes,
    file_digest,
    fingerprint,
    read_json,
    write_json,
)
from .workload import build_config


def state_digests(muon):
    def digest(tensor):
        data = tensor.to_local().detach().contiguous()
        return hashlib.sha256(
            data.view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()

    return {
        layout.fqn: {
            "parameter": digest(layout.param),
            "momentum": digest(muon.state[layout.param]["momentum_buffer"]),
        }
        for layout in muon._parameter_compute_layouts
    }


def placement_metadata(muon, rank):
    num_chunks = 0
    num_colocated_parameters = 0
    max_compute_bytes = 0
    for bucket in muon._bucket_plans:
        if isinstance(bucket, _LocalBucketPlan):
            continue
        for plan in bucket.redistribution_plans:
            partition = plan.compute_partition(rank)
            num_regions = sum(
                bool(region.numel) for region in partition.logical_regions
            )
            num_chunks += num_regions
            num_colocated_parameters += num_regions > 1
            max_compute_bytes = max(
                max_compute_bytes,
                math.prod(partition.tensor_shape)
                * torch.finfo(bucket.compute_dtype).bits
                // 8,
            )
    return {
        "num_chunks": num_chunks,
        "num_colocated_parameters": num_colocated_parameters,
        "max_parameter_compute_buffer_bytes": max_compute_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("main", "pr"), required=True)
    parser.add_argument("--phase", choices=("numerics", "performance"), required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    protocol = manifest["protocol"]
    source = args.source.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if commit != protocol["variants"][args.variant]:
        raise ValueError("Source checkout is not the pinned experiment commit")
    if not Path(dist_muon.__file__).resolve().is_relative_to(source):
        raise ValueError("Imported TorchTitan from the wrong source checkout")
    hashes = {path: file_digest(source / path) for path in protocol["source_files"]}
    if hashes != manifest["source_sha256"][args.variant]:
        raise ValueError("Source code, tokenizer, or dataset differs from the manifest")
    package_environment = environment()
    if fingerprint(package_environment) != manifest["environment_fingerprint"]:
        raise ValueError("Python packages changed after experiment preparation")
    harness_hashes = experiment_hashes()
    if harness_hashes != manifest["harness_sha256"]:
        raise ValueError("Experiment scripts changed after preparation")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    gpu = torch.cuda.get_device_name(local_rank)
    if "H20" not in gpu or torch.cuda.get_device_capability(local_rank) != (9, 0):
        raise ValueError(f"Expected H20/SM90, got {gpu}")

    init_logger()
    config = build_config(protocol, args.output)
    trainer = config.build()
    engine = trainer.engine
    rank = dist.get_rank()
    assert dist.get_world_size() == protocol["world_size"]
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    muons = [
        optimizer
        for optimizer in engine.optim.optimizers
        if isinstance(optimizer, DistMuon)
    ]
    assert len(muons) == 1
    muon = muons[0]
    is_performance = args.phase == "performance"
    records: list[dict[str, Any]] = []
    events = []
    current: dict[str, Any] | None = None
    previous_end = None
    timed_window_start = None
    num_forward_backward_calls = 0
    payload: dict[str, Any] = {
        "schema_version": protocol["schema_version"],
        "complete": False,
        "gradient_source": "trainer.forward_backward",
        "phase": args.phase,
        "variant": args.variant,
        "trial": args.trial,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "commit": commit,
        "protocol_fingerprint": fingerprint(protocol),
        "source_sha256": hashes,
        "harness_sha256": harness_hashes,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_fingerprint": fingerprint(package_environment),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "deterministic": True,
        "deterministic_warn_only": False,
        "per_step_state_digests": not is_performance,
        "num_muon_parameters": len(muon._parameter_compute_layouts),
        "placement": placement_metadata(muon, rank),
        "steps": records,
    }
    original_train_step = trainer.train_step
    original_forward_backward = engine.forward_backward
    original_optim_step = engine.optim_step
    original_muon_step = muon.step
    original_metrics_log = trainer.metrics_processor.log

    def forward_backward(*positional, **keywords):
        nonlocal num_forward_backward_calls
        num_forward_backward_calls += 1
        return original_forward_backward(*positional, **keywords)

    def optimizer_window(name, operation, positional, keywords):
        assert current is not None
        current[name + "_calls"] += 1
        if not is_performance:
            return operation(*positional, **keywords)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        host_start = time.perf_counter()
        result = operation(*positional, **keywords)
        current[name + "_host_ms"] = (time.perf_counter() - host_start) * 1000
        end.record()
        events.append((current, name, start, end))
        return result

    def muon_step(*positional, **keywords):
        return optimizer_window("muon", original_muon_step, positional, keywords)

    def optim_step(*positional, **keywords):
        return optimizer_window(
            "all_optimizers", original_optim_step, positional, keywords
        )

    def metrics_log(
        step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None
    ):
        assert current is not None
        if not all(math.isfinite(value) for value in (global_avg_loss, grad_norm)):
            raise RuntimeError(f"Nonfinite loss/grad_norm at step {step}")
        current["loss"] = global_avg_loss
        current["grad_norm"] = grad_norm
        current["loss_float64_hex"] = struct.pack("!d", global_avg_loss).hex()
        current["grad_norm_float64_hex"] = struct.pack("!d", grad_norm).hex()
        # The trainer resets peak statistics at the beginning of each log window.
        # Capture before any later metric reset or correctness-only state copies.
        current["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(engine.device)
        current["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(engine.device)
        return original_metrics_log(
            step, global_avg_loss, global_max_loss, grad_norm, extra_metrics
        )

    def train_step(data_iterator):
        nonlocal current, previous_end, timed_window_start
        step = engine.num_completed_steps + 1
        current = {"step": step, "muon_calls": 0, "all_optimizers_calls": 0}
        records.append(current)
        start = time.perf_counter()
        result = original_train_step(data_iterator)
        torch.cuda.synchronize(engine.device)
        end = time.perf_counter()
        if is_performance:
            current["train_step_wall_ms"] = (end - start) * 1000
            current["iteration_wall_ms"] = (
                None if previous_end is None else (end - previous_end) * 1000
            )
            previous_end = end
            if step == protocol["num_warmup_steps"]:
                timed_window_start = end
            payload["timed_window_wall_ms"] = (
                None
                if timed_window_start is None
                else (end - timed_window_start) * 1000
            )
        else:
            current["state_digests"] = state_digests(muon)
        if rank == 0 and (step % 10 == 0 or step == protocol["num_steps"]):
            print(
                f"H20_PROGRESS {args.phase}/{args.variant}/trial{args.trial} step={step}/{protocol['num_steps']}",
                flush=True,
            )
        return result

    engine.forward_backward = forward_backward
    engine.optim_step = optim_step
    muon.step = muon_step
    trainer.metrics_processor.log = metrics_log
    trainer.train_step = train_step
    try:
        trainer.train()
        assert (
            engine.num_completed_steps
            == num_forward_backward_calls
            == len(records)
            == protocol["num_steps"]
        )
        assert all(
            row["muon_calls"] == row["all_optimizers_calls"] == 1 for row in records
        )
        torch.cuda.synchronize(engine.device)
        for row, name, start, end in events:
            row[name + "_cuda_ms"] = start.elapsed_time(end)
        # This happens after the last training timing window has closed.
        payload["final_state_digests"] = state_digests(muon)
        payload["complete"] = True
    finally:
        payload["num_forward_backward_calls"] = num_forward_backward_calls
        payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            write_json(args.output / f"rank{rank}.json", payload)
        finally:
            trainer.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
