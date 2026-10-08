# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize a Kineto chrome trace of one optimizer step.

Reports GPU busy time per CUDA stream and per kernel family, the union of
busy intervals (how long the device had work at all), and host-side time in
the main CPU ops. Overlap = sum of per-stream busy time - union.
"""
import gzip
import json
import re
import sys
from collections import defaultdict


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


FAMILIES = (
    ("gemm", re.compile(r"gemm|cutlass|xmma|matmul|sgemm|bgemm|nvjet", re.I)),
    ("nccl", re.compile(r"ncclDevKernel|nccl", re.I)),
    ("copy/cast", re.compile(r"copy|Copy|memcpy|cast", re.I)),
    ("reduce/norm", re.compile(r"reduce|norm|Reduce", re.I)),
    (
        "elementwise",
        re.compile(r"elementwise|lerp|binary|unary|mul|add|div|fill|clamp", re.I),
    ),
)


def family(name):
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def union(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def main():
    path = sys.argv[1]
    events = load(path)
    kernels = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    if not kernels:
        print("no GPU kernel events in trace")
        return
    by_stream = defaultdict(list)
    by_family = defaultdict(float)
    count_family = defaultdict(int)
    by_name = defaultdict(lambda: [0, 0.0])
    for e in kernels:
        args = e.get("args", {})
        stream = args.get("stream", "?")
        start, dur = e["ts"], e.get("dur", 0)
        by_stream[stream].append((start, start + dur))
        fam = "memcpy/memset" if e["cat"] != "kernel" else family(e["name"])
        by_family[fam] += dur
        count_family[fam] += 1
        entry = by_name[e["name"][:90]]
        entry[0] += 1
        entry[1] += dur
    first = min(s for iv in by_stream.values() for s, _ in iv)
    last = max(t for iv in by_stream.values() for _, t in iv)
    all_intervals = [iv for ivs in by_stream.values() for iv in ivs]
    busy_union = union(all_intervals)
    busy_sum = sum(union(ivs) for ivs in by_stream.values())
    print(f"trace: {path}")
    print(
        f"GPU kernels: {len(kernels)}; span first->last kernel: {(last - first) / 1e3:.2f} ms"
    )
    print(
        f"device busy (union over streams): {busy_union / 1e3:.2f} ms  ({100 * busy_union / (last - first):.1f}% of span)"
    )
    print(
        f"sum of per-stream busy: {busy_sum / 1e3:.2f} ms  -> overlapped {(busy_sum - busy_union) / 1e3:.2f} ms"
    )
    print("per stream:")
    for stream, ivs in sorted(by_stream.items(), key=lambda kv: -union(kv[1])):
        print(
            f"  stream {stream!s:>6}: kernels {len(ivs):5d}  busy {union(ivs) / 1e3:9.2f} ms  "
            f"active window {(max(t for _, t in ivs) - min(s for s, _ in ivs)) / 1e3:9.2f} ms"
        )
    print("per kernel family (sum of durations):")
    for fam, dur in sorted(by_family.items(), key=lambda kv: -kv[1]):
        print(f"  {fam:<14s} n={count_family[fam]:5d}  {dur / 1e3:9.2f} ms")
    print("top kernels:")
    for name, (n, dur) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:14]:
        print(f"  {dur / 1e3:9.2f} ms  n={n:5d}  {name}")
    cpu = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    by_cpu = defaultdict(lambda: [0, 0.0])
    for e in cpu:
        entry = by_cpu[e["name"]]
        entry[0] += 1
        entry[1] += e.get("dur", 0)
    print("top host ops (inclusive):")
    for name, (n, dur) in sorted(by_cpu.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {dur / 1e3:9.2f} ms  n={n:5d}  {name}")
    # CUDA runtime calls: launches, graph launches, syncs and allocations.
    runtime = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")
    ]
    by_api = defaultdict(lambda: [0, 0.0])
    for e in runtime:
        entry = by_api[e["name"]]
        entry[0] += 1
        entry[1] += e.get("dur", 0)
    if cpu:
        cpu_span = max(e["ts"] + e.get("dur", 0) for e in cpu) - min(
            e["ts"] for e in cpu
        )
        print(f"host span first->last cpu op: {cpu_span / 1e3:.2f} ms")
    print("cuda runtime calls (inclusive host time):")
    for name, (n, dur) in sorted(by_api.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {dur / 1e3:9.2f} ms  n={n:6d}  {name}")
    # Idle gaps on the busiest stream: where the device waited for the host.
    busiest = max(by_stream.items(), key=lambda kv: union(kv[1]))[1]
    ordered = sorted(busiest)
    gaps = [
        ordered[i + 1][0] - ordered[i][1]
        for i in range(len(ordered) - 1)
        if ordered[i + 1][0] > ordered[i][1]
    ]
    big = [g for g in gaps if g >= 50]
    print(
        f"busiest-stream gaps: total {sum(gaps) / 1e3:.2f} ms over {len(gaps)} gaps; "
        f">=50us: {len(big)} gaps, {sum(big) / 1e3:.2f} ms; "
        f"largest {max(gaps) / 1e3 if gaps else 0:.2f} ms"
    )


if __name__ == "__main__":
    main()
