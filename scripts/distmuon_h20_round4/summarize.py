# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print a short, paste-able summary of an h20-round4 session directory."""

import glob
import json
import os
import statistics
import sys

VARIANTS = ("main", "pr5468", "pr3")


def load_ranks(path):
    ranks = {}
    for file in glob.glob(os.path.join(path, "rank*.json")):
        data = json.load(open(file))
        ranks[data["rank"]] = data
    return ranks


def part_a(out):
    root = os.path.join(out, "k3debug-pr3")
    print("A. K3 debugmodel real training, main vs PR3")
    summary_path = os.path.join(root, "summary.json")
    if not os.path.exists(summary_path):
        status_path = os.path.join(root, "status.json")
        status = json.load(open(status_path)) if os.path.exists(status_path) else {}
        print(f"  status: {status.get('status', 'not run')} {status.get('error', '')}")
        return
    summary = json.load(open(summary_path))
    print(f"  status: {summary['status']}  numerics: {json.dumps(summary['numerics'])}")
    for metric, item in summary["summary"].items():
        main = [round(x, 2) for x in item["trial_values"]["main"]]
        pr = [round(x, 2) for x in item["trial_values"]["pr"]]
        paired = [round(x, 2) for x in item["paired_reductions_percent"]]
        print(f"  {metric}: main {main} | pr3 {pr} | paired % {paired}")


def part_b(out):
    print(
        "B. Real-shape slices, optimizer only, slowest rank (ms), memory max over ranks"
    )
    for workload in ("k3_slice", "k25_3"):
        rows = {
            v: {"ms": [], "peak": [], "transient": [], "reserved": []} for v in VARIANTS
        }
        digest_notes = []
        for round_ in (1, 2):
            runs = {
                v: load_ranks(os.path.join(out, "bench", f"{workload}_r{round_}", v))
                for v in VARIANTS
            }
            for v, ranks in runs.items():
                if len(ranks) != 8:
                    if ranks:
                        digest_notes.append(f"r{round_} {v}: only {len(ranks)} ranks")
                    continue
                rows[v]["ms"].append(
                    max(r["summary"]["total_ms_median"] for r in ranks.values())
                )
                rows[v]["peak"].append(
                    max(r["summary"]["peak_bytes_max"] for r in ranks.values())
                )
                rows[v]["transient"].append(
                    max(r["summary"]["transient_bytes_max"] for r in ranks.values())
                )
                rows[v]["reserved"].append(
                    max(r["reserved_buffer_bytes"] for r in ranks.values())
                )
            base = runs["main"]
            for v in ("pr5468", "pr3"):
                if len(base) != 8 or len(runs[v]) != 8:
                    continue
                diffs = sum(
                    base[rank]["digests"] != runs[v][rank]["digests"]
                    for rank in range(8)
                )
                digest_notes.append(
                    f"r{round_} {v} vs main: {'same bits' if not diffs else f'DIFFERENT on {diffs} ranks'}"
                )
        print(f"  {workload}:")
        for v in VARIANTS:
            r = rows[v]
            if not r["ms"]:
                print(f"    {v:7s} not run")
                continue
            print(
                f"    {v:7s} ms per round {[round(x, 1) for x in r['ms']]} "
                f"median {statistics.median(r['ms']):.1f} | peak {max(r['peak']) / 2**30:.2f} GiB "
                f"| transient {max(r['transient']) / 2**20:.0f} MiB "
                f"| reserved buffers {max(r['reserved']) / 2**20:.0f} MiB"
            )
        for note in digest_notes:
            print(f"    {note}")


def part_c(out):
    print("C. K3 debugmodel optimizer profile (timings per rank, trace of rank 0)")
    for v in ("main", "pr5468"):
        path = os.path.join(out, "bench", "k3debug_profile", v)
        ranks = load_ranks(path)
        if len(ranks) != 8:
            print(f"  {v}: {len(ranks)} rank results")
        else:
            totals = [
                round(ranks[r]["summary"]["total_ms_median"], 1) for r in range(8)
            ]
            hosts = [round(ranks[r]["summary"]["host_ms_median"], 1) for r in range(8)]
            print(f"  {v}: optimizer median per rank {totals} ms, host {hosts} ms")
        report = os.path.join(path, "trace_report.txt")
        if os.path.exists(report):
            lines = open(report).read().splitlines()
            keep = [
                line
                for line in lines
                if line.startswith(("GPU kernels", "device busy"))
                or line.startswith("  stream")
            ]
            families = (
                lines.index("per kernel family (sum of durations):")
                if ("per kernel family (sum of durations):" in lines)
                else None
            )
            if families is not None:
                keep += lines[families + 1 : families + 6]
            for line in keep:
                print(f"    {line.strip()}")


def part_d(out):
    print("D. flex_shard GPU unit tests")
    for v in VARIANTS:
        log = os.path.join(out, "logs", f"D-gputest-{v}.log")
        if not os.path.exists(log):
            print(f"  {v}: not run")
            continue
        lines = [line for line in open(log).read().splitlines() if line.strip()]
        failed = [
            line for line in lines if line.startswith(("FAILED", "ERROR", "E   "))
        ]
        print(f"  {v}: {lines[-1] if lines else 'empty log'}")
        for line in failed[:5]:
            print(f"    {line[:200]}")


def main():
    out = sys.argv[1]
    for part in (part_a, part_b, part_c, part_d):
        try:
            part(out)
        except Exception as error:  # keep summarizing the other parts
            print(f"  summary error: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
