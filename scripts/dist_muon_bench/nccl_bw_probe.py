# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Measure collective bandwidth over the world and over sub-groups.

Launch with torchrun. For each group size (world, halves, quarters) and
message size, reports all-gather, reduce-scatter and all-to-all bus
bandwidth in GB/s, the numbers a replica-deduplicated DistMuon would pay
to exchange directions across the replicate axis.
"""

import os
import statistics

import torch
import torch.distributed as dist


def _time(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return statistics.median(times)


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    # Modulo lets several ranks share one device in local checks.
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]) % torch.cuda.device_count())
    device = torch.device("cuda")
    sizes_mb = (16, 64, 256, 1024)
    group_sizes = [g for g in (world, world // 2, world // 4) if g >= 2]
    rows: list[str] = []
    for group_size in group_sizes:
        groups = [
            dist.new_group(list(range(start, start + group_size)))
            for start in range(0, world, group_size)
        ]
        group = groups[rank // group_size]
        assert isinstance(group, dist.ProcessGroup)
        n = group_size
        for size_mb in sizes_mb:
            total = size_mb * 2**20 // 2  # BF16 elements of the gathered tensor
            shard = torch.randn(total // n, device=device, dtype=torch.bfloat16)
            full = torch.empty(total, device=device, dtype=torch.bfloat16)
            t_ag = _time(lambda: dist.all_gather_into_tensor(full, shard, group=group))
            t_rs = _time(lambda: dist.reduce_scatter_tensor(shard, full, group=group))
            t_a2a = _time(
                lambda: dist.all_to_all_single(full, full.clone(), group=group)
            )
            nbytes = total * 2
            # Bus bandwidth as in nccl-tests: data crossing the links per rank.
            bus = nbytes * (n - 1) / n
            rows.append(
                f"group {n:2d}  {size_mb:5d} MB  all_gather {bus / t_ag / 1e9:7.1f} GB/s  "
                f"reduce_scatter {bus / t_rs / 1e9:7.1f} GB/s  all_to_all {bus / t_a2a / 1e9:7.1f} GB/s"
            )
            del shard, full
    if rank == 0:
        print(torch.cuda.get_device_name(), torch.__version__, "world", world)
        print("\n".join(rows), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
