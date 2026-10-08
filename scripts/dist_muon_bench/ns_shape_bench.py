# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Time Newton-Schulz per matrix-view shape of one rank's static plan.

Usage: ns_shape_bench.py <plan.json> [--rank N]

For every (rows, cols) class the plan assigns to the rank, runs the DistMuon
kernel the way the optimizer issues it (one call per view, batch = the
view's matrix count) and once more with every view of the class merged into
a single batch. The gap between the two is what cross-parameter batching
could recover; the per-class share says where the GPU time goes.
"""

import json
import math
import statistics
import sys
from collections import defaultdict

import torch

from torchtitan.distributed.flex_shard.dist_muon import _zeropower_via_newtonschulz

COEF = (3.4445, -4.7750, 2.0315)
STEPS = 5
EPS = 1e-7
MAX_BATCH_BYTES = 8 * 2**30


def time_ms(fn, reps=8):
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
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def ns_flops(batch, rows, cols):
    short, long_ = sorted((rows, cols))
    return 2 * batch * STEPS * (2 * short * short * long_ + short**3)


def main():
    path = sys.argv[1]
    rank = int(sys.argv[sys.argv.index("--rank") + 1]) if "--rank" in sys.argv else 0
    plan = json.load(open(path))
    classes: dict[tuple[int, int], list[int]] = defaultdict(list)
    for optimizer in plan["optimizers"]:
        for bucket in optimizer["buckets"]:
            for item in bucket["items"]:
                shapes = item.get("view_shapes_by_rank", {}).get(str(rank))
                if not shapes:
                    continue
                for shape in shapes:
                    *batch, rows, cols = shape
                    classes[(rows, cols)].append(math.prod(batch) if batch else 1)
    device = torch.device("cuda")
    print(f"{torch.cuda.get_device_name()} {torch.__version__} plan={path} rank={rank}")
    print(
        f"{'rows x cols':>14s} {'views':>5s} {'mats':>5s} {'as issued ms':>13s} "
        f"{'TFLOPS':>7s} {'one batch ms':>13s} {'TFLOPS':>7s} {'gain ms':>8s}"
    )
    total_issued = 0.0
    total_batched = 0.0
    rows_out = []
    for (rows, cols), batches in sorted(classes.items(), key=lambda kv: -sum(kv[1])):
        flops = ns_flops(sum(batches), rows, cols)
        tensors = [
            torch.randn(b, rows, cols, device=device, dtype=torch.bfloat16)
            for b in batches
        ]

        def issued():
            for t in tensors:
                _zeropower_via_newtonschulz(
                    t.clone(), ns_coefficients=COEF, ns_steps=STEPS, eps=EPS
                )

        issued_ms = time_ms(issued)
        del tensors
        merged = sum(batches)
        if merged * rows * cols * 2 <= MAX_BATCH_BYTES:
            big = torch.randn(merged, rows, cols, device=device, dtype=torch.bfloat16)
            batched_ms = time_ms(
                lambda: _zeropower_via_newtonschulz(
                    big.clone(), ns_coefficients=COEF, ns_steps=STEPS, eps=EPS
                )
            )
            del big
        else:
            batched_ms = issued_ms
        total_issued += issued_ms
        total_batched += batched_ms
        rows_out.append(
            (rows, cols, len(batches), merged, issued_ms, flops, batched_ms)
        )
    for rows, cols, views, merged, issued_ms, flops, batched_ms in rows_out:
        print(
            f"{rows:>6d} x {cols:<6d} {views:>5d} {merged:>5d} {issued_ms:>13.2f} "
            f"{flops / issued_ms / 1e9:>7.1f} {batched_ms:>13.2f} "
            f"{flops / batched_ms / 1e9:>7.1f} {issued_ms - batched_ms:>8.2f}"
        )
    print(
        f"total as issued {total_issued:.1f} ms, fully batched per class "
        f"{total_batched:.1f} ms, gain {total_issued - total_batched:.1f} ms"
    )


if __name__ == "__main__":
    main()
