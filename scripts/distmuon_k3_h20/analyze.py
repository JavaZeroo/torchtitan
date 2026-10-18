# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate complete real-training records before reporting any speedup."""

import argparse
import csv
import math
import statistics
import struct
from pathlib import Path
from typing import Any

from .common import fingerprint, read_json, write_json


METRICS = {
    "muon_cuda_ms": ("DistMuon CUDA-event span", "ms"),
    "all_optimizers_cuda_ms": ("Optimizer phase, including AdamW", "ms"),
    "training_iteration_ms": ("Full training iteration, wall time", "ms"),
    "peak_allocated_bytes": ("Peak allocated GPU memory", "GiB"),
    "peak_reserved_bytes": ("Peak reserved GPU memory", "GiB"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_digests(digests, num_parameters):
    require(
        isinstance(digests, dict) and len(digests) == num_parameters > 0,
        "Incomplete Muon state digest set",
    )
    for name, values in digests.items():
        require(
            set(values) == {"parameter", "momentum"}, f"Incomplete states for {name}"
        )
        for value in values.values():
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value),
                f"Invalid SHA-256 for {name}",
            )


def load_run(root, manifest, phase, variant, trial):
    protocol = manifest["protocol"]
    rows = []
    for rank in range(protocol["world_size"]):
        path = root / "results" / phase / variant / f"trial{trial}" / f"rank{rank}.json"
        require(path.is_file(), f"Missing result: {path}")
        row = read_json(path)
        require(row["complete"] is True, f"Incomplete training: {path}")
        require(
            (row["phase"], row["variant"], row["trial"], row["rank"])
            == (phase, variant, trial, rank),
            f"Incorrect run identity: {path}",
        )
        require(
            row["world_size"] == protocol["world_size"]
            and row["commit"] == protocol["variants"][variant],
            f"Wrong world size or commit: {path}",
        )
        require(
            row["gradient_source"] == "trainer.forward_backward",
            "Not a real-training result",
        )
        require(
            row["protocol_fingerprint"] == fingerprint(protocol),
            "Protocol changed between runs",
        )
        require(
            row["source_sha256"] == manifest["source_sha256"][variant],
            "Source/dataset/tokenizer checksum mismatch",
        )
        require(
            row["harness_sha256"] == manifest["harness_sha256"],
            "Experiment code checksum mismatch",
        )
        require(
            row["environment_fingerprint"] == manifest["environment_fingerprint"],
            "Environment changed between runs",
        )
        require("H20" in row["gpu"], "Result was not recorded on H20")
        require(
            row["deterministic"] is True and row["deterministic_warn_only"] is False,
            "Deterministic execution was not enabled",
        )
        require(
            row["per_step_state_digests"] is (phase == "numerics"),
            "Incorrect digest/timing mode",
        )
        require(
            row["num_forward_backward_calls"] == protocol["num_steps"],
            "Missing real forward/backward steps",
        )
        require(
            [step["step"] for step in row["steps"]]
            == list(range(1, protocol["num_steps"] + 1)),
            "Missing, duplicate, or reordered training steps",
        )
        check_digests(row["final_state_digests"], row["num_muon_parameters"])
        for step in row["steps"]:
            require(
                step["muon_calls"] == step["all_optimizers_calls"] == 1,
                "Incomplete optimizer update",
            )
            for key in ("loss", "grad_norm"):
                require(math.isfinite(step[key]), f"Nonfinite {key}")
                require(
                    struct.pack("!d", step[key]).hex() == step[key + "_float64_hex"],
                    f"Inconsistent {key} bit pattern",
                )
            if phase == "numerics":
                check_digests(step["state_digests"], row["num_muon_parameters"])
                require(
                    set(step["state_digests"]) == set(row["final_state_digests"]),
                    "Muon parameter set changed during training",
                )
            else:
                require(
                    "state_digests" not in step,
                    "Per-step state copies contaminated a performance run",
                )
        if phase == "performance":
            measured = row["steps"][protocol["num_warmup_steps"] :]
            for step in measured:
                for key in (
                    "muon_cuda_ms",
                    "all_optimizers_cuda_ms",
                    "iteration_wall_ms",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                ):
                    require(
                        math.isfinite(step[key]) and step[key] > 0,
                        f"Invalid measurement: {key}",
                    )
            require(
                math.isclose(
                    sum(step["iteration_wall_ms"] for step in measured),
                    row["timed_window_wall_ms"],
                    rel_tol=1e-9,
                ),
                "Training-window time does not reconcile",
            )
        rows.append(row)
    return rows


def compare_numerics(reference, actual, *, every_step_states):
    num_scalars = 0
    num_states = 0
    for left, right in zip(reference, actual, strict=True):
        require(
            (left["gpu"], left["torch"], left["cuda"])
            == (right["gpu"], right["torch"], right["cuda"]),
            "Hardware/runtime changed between variants",
        )
        for a, b in zip(left["steps"], right["steps"], strict=True):
            for key in ("loss_float64_hex", "grad_norm_float64_hex"):
                require(
                    a[key] == b[key],
                    f"Numerical mismatch: rank {left['rank']}, step {a['step']}, {key}",
                )
                num_scalars += 1
            if every_step_states:
                require(
                    a["state_digests"] == b["state_digests"],
                    f"Muon state mismatch: rank {left['rank']}, step {a['step']}",
                )
                num_states += 2 * len(a["state_digests"])
        require(
            left["final_state_digests"] == right["final_state_digests"],
            f"Final Muon state mismatch: rank {left['rank']}",
        )
    return {
        "num_scalar_pairs": num_scalars,
        "num_per_step_state_pairs": num_states,
        "final_states_match": True,
    }


def validate_numerics(root, manifest):
    reference = load_run(root, manifest, "numerics", "main", 1)
    candidate = load_run(root, manifest, "numerics", "pr", 1)
    result = {
        **compare_numerics(reference, candidate, every_step_states=True),
        "status": "numerics_passed",
    }
    write_json(root / "numerics.json", result)
    return reference, result


def analyze(root, *, numerics_only=False) -> dict[str, Any]:
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    protocol = manifest["protocol"]
    reference, numerics = validate_numerics(root, manifest)
    if numerics_only:
        return numerics
    num_timed_steps = protocol["num_steps"] - protocol["num_warmup_steps"]
    trial_metrics = {}
    csv_rows = []
    performance_scalar_pairs = 0
    for trial in range(1, protocol["num_trials"] + 1):
        trial_metrics[trial] = {}
        for variant in protocol["variants"]:
            rows = load_run(root, manifest, "performance", variant, trial)
            checks = compare_numerics(reference, rows, every_step_states=False)
            performance_scalar_pairs += checks["num_scalar_pairs"]
            values = {}
            for key in ("muon_cuda_ms", "all_optimizers_cuda_ms"):
                samples = [
                    max(row["steps"][index][key] for row in rows)
                    for index in range(
                        protocol["num_warmup_steps"], protocol["num_steps"]
                    )
                ]
                values[key] = statistics.median(samples)
                for step, value in enumerate(samples, protocol["num_warmup_steps"] + 1):
                    csv_rows.append(
                        {
                            "trial": trial,
                            "variant": variant,
                            "step": step,
                            "metric": key,
                            "max_rank_value": value,
                        }
                    )
            values["training_iteration_ms"] = (
                max(row["timed_window_wall_ms"] for row in rows) / num_timed_steps
            )
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                values[key] = max(
                    step[key]
                    for row in rows
                    for step in row["steps"][protocol["num_warmup_steps"] :]
                )
            for index in range(protocol["num_warmup_steps"], protocol["num_steps"]):
                csv_rows.append(
                    {
                        "trial": trial,
                        "variant": variant,
                        "step": index + 1,
                        "metric": "iteration_wall_ms",
                        "max_rank_value": max(
                            row["steps"][index]["iteration_wall_ms"] for row in rows
                        ),
                    }
                )
            trial_metrics[trial][variant] = values
    summary = {}
    for metric, (label, unit) in METRICS.items():
        samples = {
            variant: [trial_metrics[trial][variant][metric] for trial in trial_metrics]
            for variant in protocol["variants"]
        }
        medians = {
            variant: statistics.median(values) for variant, values in samples.items()
        }
        summary[metric] = {
            "label": label,
            "display_unit": unit,
            "trial_values": samples,
            "median_of_trials": medians,
            "reduction_percent": 100 * (1 - medians["pr"] / medians["main"]),
            "paired_reductions_percent": [
                100 * (1 - pr / main)
                for main, pr in zip(samples["main"], samples["pr"], strict=True)
            ],
        }
    result = {
        "status": "validated_real_training",
        "protocol": protocol,
        "numerics": numerics,
        "performance_scalar_pairs_checked": performance_scalar_pairs,
        "summary": summary,
        "per_trial": trial_metrics,
        "scope": "Official K3 debugmodel only; not production-size convergence or throughput validation",
        "timing": (
            "Stage CUDA-event spans include waiting/launch gaps; training iteration "
            "uses max-rank wall-window time divided by measured steps and includes "
            "end-of-step device sync"
        ),
        "memory": "PyTorch allocator peaks over measured steps, excluding correctness-only state hashing and warmup",
    }
    write_json(root / "summary.json", result)
    with (root / "step_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("trial", "variant", "step", "metric", "max_rank_value")
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# K3 H20 real-training results",
        "",
        f"Main: `{protocol['variants']['main']}`",
        f"PR: `{protocol['variants']['pr']}`",
        "",
        "| Metric | main | PR | Reduction |",
        "|---|---:|---:|---:|",
    ]
    for item in summary.values():
        scale = 1024**3 if item["display_unit"] == "GiB" else 1
        main_value, pr_value = (
            item["median_of_trials"][variant] / scale for variant in ("main", "pr")
        )
        lines.append(
            f"| {item['label']} ({item['display_unit']}) | {main_value:.3f} | "
            f"{pr_value:.3f} | {item['reduction_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Per-step loss/grad_norm and Muon parameter/momentum digests match in "
            "the separate correctness runs. All performance runs also match the "
            "reference loss/grad_norm and final Muon states.",
            "",
            result["timing"] + ".",
            result["memory"] + ".",
            result["scope"] + ".",
            "See summary.json for all paired trials, including regressions and variability.",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    try:
        result = analyze(args.output, numerics_only=args.numerics_only)
    except (ValueError, KeyError, OSError, TypeError) as error:
        write_json(
            args.output / "validation_error.json",
            {"status": "invalid_or_incomplete", "error": str(error)},
        )
        raise SystemExit(f"No validated performance result: {error}") from error
    print(result["status"])
    if not args.numerics_only:
        print((args.output / "summary.md").read_text())


if __name__ == "__main__":
    main()
