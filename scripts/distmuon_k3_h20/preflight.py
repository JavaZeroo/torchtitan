# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the actual selected H20 devices, NCCL, and fused KDA path."""

import argparse
import importlib
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from .common import environment, fingerprint, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    protocol = manifest["protocol"]
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != protocol["world_size"]:
        raise ValueError("This experiment requires one node with 8 selected H20 GPUs")
    torch.cuda.set_device(local_rank)
    name = torch.cuda.get_device_name(local_rank)
    capability = torch.cuda.get_device_capability(local_rank)
    if "H20" not in name or capability != (9, 0):
        raise ValueError(f"Expected H20/SM90, got rank {rank}: {name}, {capability}")
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True, warn_only=False)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    try:
        # Verify the dependency required by the official Hopper attention path.
        importlib.import_module("flash_attn_interface")
        kda = importlib.import_module("attn_gym.linear.kda")

        num_ranks = dist.get_world_size()
        source = torch.arange(num_ranks, device="cuda", dtype=torch.int64)
        source.add_(rank * num_ranks)
        destination = torch.empty_like(source)
        dist.all_to_all_single(destination, source)
        expected = torch.arange(num_ranks, device="cuda") * num_ranks + rank
        if not torch.equal(destination, expected):
            raise RuntimeError("NCCL all-to-all data check failed")

        shape = (1, protocol["seq_len"], 4, 128)
        q, k, v, raw_gate = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            for _ in range(4)
        ]
        a_log = torch.zeros(4, device="cuda", requires_grad=True)
        dt_bias = torch.zeros(4, 128, device="cuda", requires_grad=True)
        beta = torch.randn(1, shape[1], 4, device="cuda", requires_grad=True)
        gate = kda.bound_gate(raw_gate, a_log, dt_bias, lower_bound=-5.0, impl="fused")
        result, _ = kda.chunk_kda(
            kda.l2norm(q, autotune=False),
            kda.l2norm(k, autotune=False),
            v,
            gate,
            beta.sigmoid(),
            autotune=False,
            kernel_options=kda.KernelOptions(backend="fused"),
        )
        result.float().square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(result).all() or any(
            tensor.grad is None or not torch.isfinite(tensor.grad).all()
            for tensor in (q, k, v, raw_gate, a_log, dt_bias, beta)
        ):
            raise RuntimeError("Fused KDA forward/backward produced invalid values")
        write_json(
            args.output / f"rank{rank}.json",
            {
                "complete": True,
                "rank": rank,
                "world_size": num_ranks,
                "gpu": name,
                "capability": capability,
                "total_memory_bytes": torch.cuda.get_device_properties(
                    local_rank
                ).total_memory,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "environment_fingerprint": fingerprint(environment()),
                "nccl_all_to_all": "passed",
                "fused_kda_forward_backward": "passed",
                "flash_attention_3_import": "passed",
            },
        )
        print(f"H20_PREFLIGHT rank={rank} passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
