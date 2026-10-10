# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static PNG figures for reports and issues.

  report_figures.py timeline --out step.png [--title T] label=trace.json.gz ...
      One panel per trace: the compute and transfer streams of one optimizer
      step as kernel-family runs (GEMM, NCCL, other) on a shared time axis.
  report_figures.py balance --out balance.png [--title T] label=plan.json ...
      Newton-Schulz TFLOPs per rank for each static plan, side by side.
  report_figures.py viewer --out view.png [--zoom a:b] label=trace.json.gz
      Trace-viewer style: CPU ops, CUDA runtime calls and every GPU kernel as
      its own slice with its name, like a Perfetto screenshot.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
# Categorical slots in fixed order: GEMM, NCCL, other kernels.
COLORS = {"gemm": "#2a78d6", "nccl": "#eb6834", "other": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GEMM = re.compile(r"gemm|cutlass|xmma|matmul|nvjet", re.I)
NCCL = re.compile(r"nccl", re.I)


def _family(name, cat):
    if cat != "kernel":
        return "other"
    if GEMM.search(name):
        return "gemm"
    if NCCL.search(name):
        return "nccl"
    return "other"


def _load_lanes(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    lanes = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            lanes[e.get("args", {}).get("stream", 0)].append(
                (e["ts"], e["ts"] + e.get("dur", 0), _family(e["name"], e["cat"]))
            )
    return lanes


def _runs(segments, t0, bin_us):
    """Dominant family per time bin, merged into runs of (start_ms, length_ms, family)."""
    busy = defaultdict(lambda: defaultdict(float))
    for start, end, fam in segments:
        b = int((start - t0) // bin_us)
        last = int((end - t0) // bin_us)
        while b <= last:
            lo = max(start, t0 + b * bin_us)
            hi = min(end, t0 + (b + 1) * bin_us)
            if hi > lo:
                busy[b][fam] += hi - lo
            b += 1
    runs = []
    for b in sorted(busy):
        fams = busy[b]
        if sum(fams.values()) < 0.15 * bin_us:
            continue
        fam = max(fams, key=lambda name: fams[name])
        start_ms = b * bin_us / 1e3
        if (
            runs
            and runs[-1][2] == fam
            and abs(runs[-1][0] + runs[-1][1] - start_ms) < 1e-9
        ):
            runs[-1][1] += bin_us / 1e3
        else:
            runs.append([start_ms, bin_us / 1e3, fam])
    return runs


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def timeline(args):
    traces = [
        (a.split("=", 1)[0], _load_lanes(a.split("=", 1)[-1])) for a in args.inputs
    ]
    span = 0.0
    panels = []
    for label, lanes in traces:
        ordered = sorted(lanes, key=lambda s: -sum(e - b for b, e, _ in lanes[s]))[:2]
        t0 = min(b for s in ordered for b, _, _ in lanes[s])
        t1 = max(e for s in ordered for _, e, _ in lanes[s])
        span = max(span, (t1 - t0) / 1e3)
        panels.append((label, ordered, t0, (t1 - t0) / 1e3, lanes))
    bin_us = span * 1e3 / 1400
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(9, 1.25 * len(panels) + 0.9),
        sharex=True,
        facecolor=SURFACE,
        constrained_layout=True,
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (label, ordered, t0, length, lanes) in zip(axes, panels):
        _style(ax)
        for row, stream in enumerate(ordered):
            runs = _runs(lanes[stream], t0, bin_us)
            ax.broken_barh(
                [(s, w) for s, w, _ in runs],
                (row * 1.0 + 0.15, 0.7),
                facecolors=[COLORS[f] for _, _, f in runs],
                linewidth=0,
            )
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(
            ["compute stream", "transfer stream"], fontsize=9, color=TEXT_2
        )
        ax.set_ylim(2.1, -0.1)
        ax.text(
            0, -0.2, label, fontsize=10, color=TEXT, fontweight="semibold", va="bottom"
        )
        ax.text(
            length,
            0.5,
            f" {length:.0f} ms",
            fontsize=10,
            color=TEXT,
            va="center",
            ha="left",
        )
    axes[-1].set_xlabel("time within one optimizer step (ms)", fontsize=9, color=TEXT_2)
    axes[-1].set_xlim(0, span * 1.12)
    handles = [
        Patch(color=COLORS[k], label=n)
        for k, n in (
            ("gemm", "Newton-Schulz GEMM"),
            ("nccl", "NCCL all-to-all"),
            ("other", "lerp, copies, casts"),
        )
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        frameon=False,
        fontsize=8,
        ncol=3,
        labelcolor=TEXT_2,
    )
    if args.title:
        fig.suptitle(args.title, fontsize=11, color=TEXT, x=0.01, ha="left")
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)


def balance(args):
    plans = []
    for a in args.inputs:
        label, path = a.split("=", 1)
        p = json.load(open(path))["optimizers"][0]["ns_flops_by_rank"]
        ranks = sorted(int(r) for r in p)
        plans.append((label, [p[str(r)] / 1e12 for r in ranks]))
    n = len(plans[0][1])
    fig, ax = plt.subplots(figsize=(7, 3), facecolor=SURFACE, constrained_layout=True)
    _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    width = 0.8 / len(plans)
    for i, (label, vals) in enumerate(plans):
        xs = [r + (i - (len(plans) - 1) / 2) * width for r in range(n)]
        ax.bar(
            xs,
            vals,
            width=width * 0.92,
            color=SERIES[i],
            linewidth=0,
            label=f"{label}  (max/mean {max(vals) / (sum(vals) / n):.2f})",
        )
    mean = sum(plans[0][1]) / n
    ax.axhline(mean, color=TEXT_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.45, mean, "mean", fontsize=8, color=TEXT_2, va="bottom", ha="left")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"rank {r}" for r in range(n)], fontsize=9)
    ax.set_ylabel("Newton-Schulz TFLOP per step", fontsize=9, color=TEXT_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2, loc="upper right")
    if args.title:
        ax.set_title(args.title, fontsize=11, color=TEXT, loc="left")
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)


def _slice_color(name):
    """Perfetto-like slice color: a hue hashed from the name, muted and light."""
    import colorsys
    import hashlib

    hue = int(hashlib.md5(name.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(hue, 0.42, 0.86)
    return (r, g, b)


def _depths(events):
    """Nesting depth of X events on one thread, by timestamp containment."""
    out = []
    stack = []
    for e in sorted(events, key=lambda e: (e["ts"], -e.get("dur", 0))):
        end = e["ts"] + e.get("dur", 0)
        while stack and stack[-1] <= e["ts"]:
            stack.pop()
        out.append((e, len(stack)))
        stack.append(end)
    return out


def viewer(args):
    """Trace-viewer style rendering of one step: every slice drawn, names inside."""
    label, path = args.inputs[0].split("=", 1)
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    events = [
        e
        for e in (data["traceEvents"] if isinstance(data, dict) else data)
        if e.get("ph") == "X"
    ]
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu = [e for e in events if e.get("cat") == "cpu_op"]
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    t0 = min(e["ts"] for e in gpu + cpu)
    t1 = max(e["ts"] + e.get("dur", 0) for e in gpu + cpu)
    if args.zoom:
        z0, z1 = (float(x) for x in args.zoom.split(":"))
        t0, t1 = t0 + z0 * 1e3, min(t1, t0 + z1 * 1e3)
    span_ms = (t1 - t0) / 1e3
    streams = sorted(
        {e["args"].get("stream", 0) for e in gpu},
        key=lambda s: -sum(
            e.get("dur", 0) for e in gpu if e["args"].get("stream", 0) == s
        ),
    )
    # Tracks: CPU ops (3 nesting rows), CUDA runtime (1 row), one per stream.
    tracks = [("CPU  main thread", 3), ("CUDA runtime", 1)] + [
        (f"GPU stream {s}", 1) for s in streams
    ]
    row_h = 0.42
    heights = [n * row_h for _, n in tracks]
    width_in = 14
    fig_h = sum(heights) + 0.9 + 0.25 * len(tracks)
    fig, ax = plt.subplots(figsize=(width_in, fig_h), facecolor="#ffffff")
    ax.set_facecolor("#f6f6f6")
    px_per_us = width_in * 0.9 * 200 / (t1 - t0)  # at dpi 200, after the left gutter

    def draw(evs, y, depth_fn, min_label_px=28):
        for e, depth in evs:
            start = max(e["ts"], t0)
            end = min(e["ts"] + e.get("dur", 0), t1)
            if end <= start:
                continue
            x0, w = (start - t0) / 1e3, (end - start) / 1e3
            yy = y + depth_fn(depth) * row_h
            ax.add_patch(
                plt.Rectangle(
                    (x0, yy + 0.03),
                    w,
                    row_h - 0.06,
                    facecolor=_slice_color(e["name"]),
                    edgecolor="#ffffff",
                    linewidth=0.4,
                )
            )
            if (end - start) * px_per_us > min_label_px:
                name = (
                    e["name"]
                    .replace("void at::native::", "")
                    .split("<")[0]
                    .split("(")[0]
                )
                chars = int((end - start) * px_per_us / 5.2)
                ax.text(
                    x0 + 0.03 * w / max(w, 1e-9) * 0,
                    yy + row_h / 2,
                    " " + name[:chars],
                    fontsize=5.2,
                    va="center",
                    ha="left",
                    color="#1a1a1a",
                    clip_on=True,
                )

    y = 0.0
    labels = []
    for (name, rows), h in zip(tracks, heights):
        ax.add_patch(
            plt.Rectangle(
                (0, y),
                span_ms,
                h,
                facecolor="#f6f6f6" if len(labels) % 2 else "#fbfbfb",
                edgecolor="#e2e2e2",
                linewidth=0.6,
            )
        )
        labels.append((name, y + h / 2))
        if name.startswith("CPU"):
            draw([(e, d) for e, d in _depths(cpu) if d < rows], y, lambda d: d)
        elif name.startswith("CUDA"):
            draw([(e, 0) for e in runtime], y, lambda d: 0, min_label_px=40)
        else:
            s = int(name.split()[-1])
            draw(
                [(e, 0) for e in gpu if e["args"].get("stream", 0) == s], y, lambda d: 0
            )
        y += h + 0.25
    ax.set_xlim(0, span_ms)
    ax.set_ylim(y, -0.6)
    ax.set_yticks([yy for _, yy in labels])
    ax.set_yticklabels([n for n, _ in labels], fontsize=8, color="#333333")
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", labelsize=8, colors="#333333", length=3)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([round(v, 1) for v in __import__("numpy").linspace(0, span_ms, 11)])
    ax.set_xticklabels(
        [
            f"{v:.0f} ms" if span_ms > 20 else f"{v:.1f} ms"
            for v in __import__("numpy").linspace(0, span_ms, 11)
        ]
    )
    for side in ("left", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["top"].set_color("#cccccc")
    ax.grid(True, axis="x", color="#e2e2e2", linewidth=0.5)
    ax.set_axisbelow(True)
    title = args.title or label
    fig.suptitle(
        f"{title}   ({label}, one optimizer step, {span_ms:.0f} ms shown)",
        fontsize=10,
        x=0.01,
        ha="left",
        color="#1a1a1a",
    )
    fig.subplots_adjust(left=0.1, right=0.995, top=0.9, bottom=0.02)
    fig.savefig(args.out, dpi=200, facecolor="#ffffff")
    print("wrote", args.out)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, fn in (("timeline", timeline), ("balance", balance), ("viewer", viewer)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--title", default="")
        if name == "viewer":
            p.add_argument("--zoom", default="", help="start:end in ms within the step")
        p.add_argument("inputs", nargs="+")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
