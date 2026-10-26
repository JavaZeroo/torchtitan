# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Print a short, paste-able summary of an h20-round5 session directory."""

import glob
import json
import os
import re
import statistics
import sys

VARIANTS = ("main", "pr5468", "pr3", "both")
WARMUP = 3


def natural(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def load_ranks(path):
    ranks = {}
    for file in glob.glob(os.path.join(path, "rank*.json")):
        data = json.load(open(file))
        ranks[data["rank"]] = data
    return ranks


def last_error(log):
    if not os.path.exists(log):
        return "not run"
    lines = [
        line.strip()
        for line in open(log, errors="replace")
        if re.search(r"Error|error:|OutOfMemory", line)
        and "register_constant" not in line
    ]
    return lines[-1][:160] if lines else "no result"


def step_ms(rank):
    steps = [
        s["since_previous_ms"] + s["total_ms"]
        for s in rank["steps"][WARMUP:]
        if s["since_previous_ms"] is not None
    ]
    return statistics.median(steps) if steps else None


def row(out, part, name, v, main):
    ranks = load_ranks(os.path.join(out, part, name, v))
    if not ranks:
        return (
            f"    {v:7s} {last_error(os.path.join(out, 'logs', f'{part}-{name}-{v}.log'))}",
            None,
        )
    world = next(iter(ranks.values()))["world_size"]
    if len(ranks) != world:
        return f"    {v:7s} only {len(ranks)} of {world} rank results", None
    ms = max(r["summary"]["total_ms_median"] for r in ranks.values())
    text = f"    {v:7s} {ms:8.1f} ms"
    if main is not None and v != "main":
        text += f" ({(ms / main[0] - 1) * 100:+.1f}%)"
    text += (
        f" | host {max(r['summary']['host_ms_median'] for r in ranks.values()):.1f}"
        f" | peak {max(r['summary']['peak_bytes_max'] for r in ranks.values()) / 2**30:.2f} GiB"
        f" | transient {max(r['summary']['transient_bytes_max'] for r in ranks.values()) / 2**20:.0f} MiB"
        f" | reserved {max(r['reserved_buffer_bytes'] for r in ranks.values()) / 2**20:.0f} MiB"
    )
    if next(iter(ranks.values()))["mode"] == "train":
        steps = [step_ms(r) for r in ranks.values()]
        if None not in steps:
            text += (
                f" | step {max(steps):.1f} ms, optimizer {ms / max(steps) * 100:.0f}%"
            )
    if main is not None and v != "main":
        diffs = sum(
            ranks[r]["digests"] != main[1].get(r, {}).get("digests") for r in ranks
        )
        text += " | same bits" if not diffs else f" | DIFFERENT on {diffs} ranks"
    return text, (ms, ranks)


def trace_lines(path):
    report = os.path.join(path, "trace_report.txt")
    if not os.path.exists(report):
        return []
    lines = open(report).read().splitlines()
    keep = [line for line in lines if line.startswith(("GPU kernels", "device busy"))]
    marker = "per kernel family (sum of durations):"
    if marker in lines:
        keep += lines[lines.index(marker) + 1 : lines.index(marker) + 4]
    return keep


def main():
    out = sys.argv[1]
    for part in sorted(glob.glob(os.path.join(out, "part*")), key=natural):
        part = os.path.basename(part)
        for name in sorted(os.listdir(os.path.join(out, part)), key=natural):
            print(f"{part}/{name}")
            main_result = None
            for v in VARIANTS:
                if not os.path.isdir(os.path.join(out, part, name, v)):
                    continue
                try:
                    text, result = row(out, part, name, v, main_result)
                except Exception as error:  # keep summarizing the other runs
                    text, result = f"    {v:7s} summary error: {error}", None
                print(text)
                if v == "main":
                    main_result = result
                for line in trace_lines(os.path.join(out, part, name, v)):
                    print(f"        {line.strip()}")


if __name__ == "__main__":
    main()
