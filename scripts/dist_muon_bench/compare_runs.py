# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare two muon_bench output directories: digests, timing, memory."""
import glob
import json
import os
import statistics
import sys


def load(path):
    runs = {}
    for file in sorted(glob.glob(os.path.join(path, "rank*.json"))):
        data = json.load(open(file))
        runs[data["rank"]] = data
    return runs


def main():
    base_dir, test_dir = sys.argv[1], sys.argv[2]
    base, test = load(base_dir), load(test_dir)
    if set(base) != set(test):
        print(f"rank sets differ: {sorted(base)} vs {sorted(test)}")
        sys.exit(2)
    mismatched = []
    total = 0
    for rank in sorted(base):
        b, t = base[rank]["digests"], test[rank]["digests"]
        if set(b) != set(t):
            print(f"rank {rank}: parameter sets differ")
            sys.exit(2)
        for fqn in sorted(b):
            for kind in ("param", "momentum"):
                total += 1
                if b[fqn][kind] != t[fqn][kind]:
                    mismatched.append((rank, fqn, kind))
    print(
        f"commits: base {next(iter(base.values()))['commit'][:10]}  test {next(iter(test.values()))['commit'][:10]}"
    )
    print(f"digests compared: {total}, mismatched: {len(mismatched)}")
    for rank, fqn, kind in mismatched[:20]:
        print(f"  MISMATCH rank {rank} {kind:<8s} {fqn}")
    print(
        f"{'rank':>4s} {'base ms':>10s} {'test ms':>10s} {'speedup':>8s} "
        f"{'base host':>10s} {'test host':>10s} {'base transient MiB':>19s} "
        f"{'test transient MiB':>19s} {'base reserved MiB':>18s} "
        f"{'test reserved MiB':>18s}"
    )
    base_ms, test_ms = [], []
    for rank in sorted(base):
        bs, ts = base[rank]["summary"], test[rank]["summary"]
        base_ms.append(bs["total_ms_median"])
        test_ms.append(ts["total_ms_median"])
        print(
            f"{rank:>4d} {bs['total_ms_median']:>10.2f} {ts['total_ms_median']:>10.2f} "
            f"{bs['total_ms_median'] / ts['total_ms_median']:>8.3f} "
            f"{bs['host_ms_median']:>10.2f} {ts['host_ms_median']:>10.2f} "
            f"{bs['transient_bytes_max'] / 2**20:>19.1f} "
            f"{ts['transient_bytes_max'] / 2**20:>19.1f} "
            f"{base[rank]['reserved_buffer_bytes'] / 2**20:>18.1f} "
            f"{test[rank]['reserved_buffer_bytes'] / 2**20:>18.1f}"
        )
    print(
        f"slowest rank: base {max(base_ms):.2f} ms, test {max(test_ms):.2f} ms, speedup {max(base_ms) / max(test_ms):.3f}"
    )
    print(
        f"mean rank:    base {statistics.fmean(base_ms):.2f} ms, test {statistics.fmean(test_ms):.2f} ms"
    )
    sys.exit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
