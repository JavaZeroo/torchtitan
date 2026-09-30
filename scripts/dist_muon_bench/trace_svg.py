# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render one optimizer step from a Kineto trace as an inline SVG timeline.

Usage: trace_svg.py <trace.json.gz> [<trace2.json.gz> ...] > figure.svg

Each trace becomes a group of lanes (one per CUDA stream); kernels are merged
into runs of the same family so the SVG stays small. Colors are CSS variables
so the figure follows the page theme.
"""
import gzip
import json
import re
import sys
from collections import defaultdict

FAMILIES = (
    ("gemm", re.compile(r"gemm|cutlass|xmma|matmul|nvjet", re.I)),
    ("nccl", re.compile(r"nccl", re.I)),
    ("copy", re.compile(r"copy|Copy|memcpy|cast|foreach|multi_tensor", re.I)),
    (
        "elementwise",
        re.compile(
            r"elementwise|lerp|binary|unary|mul|add|div|fill|clamp|reduce|norm", re.I
        ),
    ),
)
COLOR = {
    "gemm": "var(--compute)",
    "nccl": "var(--transfer)",
    "copy": "var(--copy)",
    "elementwise": "var(--elem)",
    "other": "var(--muted)",
}


def family(name, cat):
    if cat != "kernel":
        return "copy"
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def load(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    kernels = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    lanes = defaultdict(list)
    for e in kernels:
        stream = e.get("args", {}).get("stream", 0)
        lanes[stream].append(
            (e["ts"], e["ts"] + e.get("dur", 0), family(e["name"], e["cat"]))
        )
    return lanes


def binned(segments, t0, scale, px_width, bin_px=1.0):
    """Dominant kernel family per pixel bin, merged into runs (list of x0, x1, fam)."""
    num_bins = int(px_width / bin_px) + 1
    busy = [defaultdict(float) for _ in range(num_bins)]
    for start, end, fam in segments:
        x0 = (start - t0) * scale / bin_px
        x1 = (end - t0) * scale / bin_px
        b = int(x0)
        while b <= int(x1) and b < num_bins:
            lo, hi = max(x0, b), min(x1, b + 1)
            if hi > lo:
                busy[b][fam] += hi - lo
            b += 1
    runs = []
    for b, fams in enumerate(busy):
        if not fams:
            fam = None
        else:
            fam = max(fams, key=lambda name: fams[name])
            if sum(fams.values()) < 0.15:
                fam = None
        if fam is None:
            continue
        if runs and runs[-1][2] == fam and runs[-1][1] >= b * bin_px - 1e-6:
            runs[-1][1] = (b + 1) * bin_px
        else:
            runs.append([b * bin_px, (b + 1) * bin_px, fam])
    return runs


def lane_label(lanes, stream):
    fams = defaultdict(float)
    for s, e, f in lanes[stream]:
        fams[f] += e - s
    top = max(fams, key=lambda name: fams[name])
    return {"nccl": "NCCL stream", "gemm": "compute stream"}.get(top, "transfer stream")


def main():
    paths = sys.argv[1:]
    width = 900
    left = 120
    lane_h = 18
    group_gap = 26
    # Arguments are paths, or ``label=path`` to name a group.
    traces = [(p.split("=", 1)[0], load(p.split("=", 1)[-1])) for p in paths]
    span = max(
        max(e for lane in lanes.values() for _, e, _ in lane)
        - min(s for lane in lanes.values() for s, _, _ in lane)
        for _, lanes in traces
    )
    scale = (width - left - 20) / span
    y = 24
    out = []
    for path, lanes in traces:
        t0 = min(s for lane in lanes.values() for s, _, _ in lane)
        label = path if "/" not in path else "/".join(path.split("/")[-3:])
        out.append(
            f'<text x="{left}" y="{y}" font-size="12" font-weight="600" fill="currentColor">{label}</text>'
        )
        y += 8
        ordered = sorted(lanes, key=lambda st: -sum(e - s for s, e, _ in lanes[st]))
        for stream in ordered:
            y += lane_h + 4
            out.append(
                f'<text x="{left - 8}" y="{y - 4}" font-size="11" text-anchor="end" '
                f'fill="var(--muted)">{lane_label(lanes, stream)} #{stream}</text>'
            )
            out.append(
                f'<line x1="{left}" y1="{y}" x2="{width - 20}" y2="{y}" stroke="var(--line)" stroke-width="1"/>'
            )
            for x0, x1, fam in binned(
                list(lanes[stream]), t0, scale, width - left - 20
            ):
                out.append(
                    f'<rect x="{left + x0:.1f}" y="{y - lane_h}" width="{x1 - x0:.1f}" height="{lane_h - 2}" fill="{COLOR[fam]}"/>'
                )
        y += group_gap
    # axis
    ticks = 6
    axis_y = y
    out.append(
        f'<line x1="{left}" y1="{axis_y}" x2="{width - 20}" y2="{axis_y}" stroke="currentColor" stroke-width="1"/>'
    )
    for i in range(ticks + 1):
        t = span * i / ticks
        x = left + t * scale
        out.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 5}" stroke="currentColor"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{axis_y + 18}" font-size="11" text-anchor="middle" fill="currentColor">{t / 1e3:.0f} ms</text>'
        )
    height = axis_y + 28
    print(
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="GPU timeline per CUDA stream">'
    )
    print("\n".join(out))
    print("</svg>")


if __name__ == "__main__":
    main()
