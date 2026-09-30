# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Summarize a results directory: per-experiment, per-variant timing/memory tables,
digest equality between variants, TensorBoard loss comparison for train runs."""
import glob
import json
import os
import statistics
import subprocess
import sys


def load_ranks(path):
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "rank*.json"))):
        d = json.load(open(f))
        out[d["rank"]] = d
    return out


def fmt_ms(v):
    return f"{v:8.1f}"


def main():
    root = sys.argv[1]
    variants = sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    experiments = sorted(
        {
            e
            for v in variants
            for e in os.listdir(os.path.join(root, v))
            if os.path.isdir(os.path.join(root, v, e))
        }
    )
    for exp in experiments:
        print(f"\n### {exp}")
        rows = {}
        for v in variants:
            path = os.path.join(root, v, exp)
            if not os.path.isdir(path):
                continue
            ranks = load_ranks(path)
            if not ranks:
                print(f"  {v}: no rank files")
                continue
            rows[v] = ranks
            meds = [
                r["summary"]["total_ms_median"] for r in ranks.values() if r["summary"]
            ]
            hosts = [
                r["summary"]["host_ms_median"] for r in ranks.values() if r["summary"]
            ]
            trans = [
                r["summary"]["transient_bytes_max"] / 2**20
                for r in ranks.values()
                if r["summary"]
            ]
            peaks = [
                r["summary"]["peak_bytes_max"] / 2**20
                for r in ranks.values()
                if r["summary"]
            ]
            resv = [r["reserved_buffer_bytes"] / 2**20 for r in ranks.values()]
            gpu = next(iter(ranks.values()))["gpu"]
            commit = next(iter(ranks.values()))["commit"][:10]
            print(
                f"  {v:<8s} commit {commit} {gpu}: optimizer step median per rank (ms): "
                + " ".join(f"{m:7.1f}" for m in meds)
            )
            print(
                f"           slowest {max(meds):7.1f}  mean {statistics.fmean(meds):7.1f}  "
                f"host median {statistics.fmean(hosts):6.1f} ms  | reserved buffers max "
                f"{max(resv):7.1f} MiB  transient max {max(trans):7.1f} MiB  "
                f"peak allocated max {max(peaks):8.1f} MiB"
            )
        if len(rows) >= 2:
            names = list(rows)
            base = rows[names[0]]
            for other in names[1:]:
                test = rows[other]
                mism = 0
                total = 0
                for rank in base:
                    if rank not in test:
                        continue
                    for fqn, d in base[rank]["digests"].items():
                        for kind in ("param", "momentum"):
                            total += 1
                            if d[kind] != test[rank]["digests"].get(fqn, {}).get(kind):
                                mism += 1
                b_slow = max(
                    r["summary"]["total_ms_median"]
                    for r in base.values()
                    if r["summary"]
                )
                t_slow = max(
                    r["summary"]["total_ms_median"]
                    for r in test.values()
                    if r["summary"]
                )
                print(
                    f"  {names[0]} vs {other}: digests {total - mism}/{total} identical; "
                    f"slowest-rank speedup {b_slow / t_slow:.3f}x"
                )
        # TensorBoard losses for train runs
        tbs = {v: os.path.join(root, v, exp, "dump", "tb") for v in rows}
        tbs = {v: p for v, p in tbs.items() if os.path.isdir(p)}
        if len(tbs) >= 2:
            names = list(tbs)
            for other in names[1:]:
                r = subprocess.run(
                    [
                        sys.executable,
                        os.path.join(os.path.dirname(__file__), "tb_metrics.py"),
                        tbs[names[0]],
                        tbs[other],
                    ],
                    capture_output=True,
                    text=True,
                )
                tail = r.stdout.strip().splitlines()[-1:] or [r.stderr.strip()[-200:]]
                print(f"  loss/grad_norm {names[0]} vs {other}: {tail[0]}")


if __name__ == "__main__":
    main()
