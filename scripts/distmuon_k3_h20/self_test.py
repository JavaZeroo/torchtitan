# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Offline checks using synthetic result fixtures, never GPU measurements."""

import copy
import json
import math
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .analyze import analyze, load_run
from .common import fingerprint, load_protocol, read_json, SCRIPT_DIR, write_json
from .run import make_bundle, run_logged, run_schedule


def make_fixture(root):
    protocol = load_protocol()
    protocol.update(world_size=2, num_steps=12, num_warmup_steps=2)
    manifest = {
        "protocol": protocol,
        "source_sha256": {
            variant: {"fixture": "f" * 64} for variant in protocol["variants"]
        },
        "harness_sha256": {"synthetic_fixture": "e" * 64},
        "environment_fingerprint": "fixture-only",
    }
    write_json(root / "manifest.json", manifest)
    digests = {"fixture.weight": {"parameter": "1" * 64, "momentum": "2" * 64}}
    for phase, variant, trial in run_schedule(protocol):
        for rank in range(protocol["world_size"]):
            steps = []
            for step in range(1, protocol["num_steps"] + 1):
                row: dict[str, Any] = {
                    "step": step,
                    "muon_calls": 1,
                    "all_optimizers_calls": 1,
                    "loss": 1.0,
                    "grad_norm": 2.0,
                    "loss_float64_hex": struct.pack("!d", 1.0).hex(),
                    "grad_norm_float64_hex": struct.pack("!d", 2.0).hex(),
                }
                if phase == "numerics":
                    row["state_digests"] = copy.deepcopy(digests)
                else:
                    baseline = ([10, 30, 20] if variant == "main" else [9, 21, 16])[
                        trial - 1
                    ]
                    row.update(
                        muon_cuda_ms=baseline + rank + (step % 3) / 10,
                        all_optimizers_cuda_ms=baseline + rank + 4 + (step % 3) / 10,
                        iteration_wall_ms=(100 if variant == "main" else 95)
                        + rank
                        + step / 10,
                        peak_allocated_bytes=100 * 1024**2 + rank * 1024**2,
                        peak_reserved_bytes=120 * 1024**2 + rank * 1024**2,
                    )
                steps.append(row)
            payload = {
                "complete": True,
                "phase": phase,
                "variant": variant,
                "trial": trial,
                "rank": rank,
                "world_size": protocol["world_size"],
                "commit": protocol["variants"][variant],
                "gradient_source": "trainer.forward_backward",
                "protocol_fingerprint": fingerprint(protocol),
                "source_sha256": manifest["source_sha256"][variant],
                "harness_sha256": manifest["harness_sha256"],
                "environment_fingerprint": manifest["environment_fingerprint"],
                "gpu": "H20 SYNTHETIC SELF-TEST FIXTURE",
                "torch": "fixture",
                "cuda": "fixture",
                "deterministic": True,
                "deterministic_warn_only": False,
                "per_step_state_digests": phase == "numerics",
                "num_forward_backward_calls": protocol["num_steps"],
                "num_muon_parameters": 1,
                "final_state_digests": digests,
                "steps": steps,
            }
            if phase == "performance":
                payload["timed_window_wall_ms"] = sum(
                    row["iteration_wall_ms"] for row in steps[2:]
                )
            write_json(
                root
                / "results"
                / phase
                / variant
                / f"trial{trial}"
                / f"rank{rank}.json",
                payload,
            )
    return manifest


def expect_rejected(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("Invalid fixture was accepted")


def main():
    with tempfile.TemporaryDirectory(prefix="k3-h20-script-self-test-") as directory:
        root = Path(directory) / "results"
        manifest = make_fixture(root)
        result = analyze(root)
        assert result["status"] == "validated_real_training"
        assert math.isclose(
            result["summary"]["muon_cuda_ms"]["median_of_trials"]["main"], 21.1
        )
        assert math.isclose(
            result["summary"]["muon_cuda_ms"]["median_of_trials"]["pr"], 17.1
        )
        assert math.isclose(
            result["summary"]["training_iteration_ms"]["median_of_trials"]["main"],
            101.75,
        )

        target = root / "results/performance/pr/trial1/rank1.json"
        original = read_json(target)
        bad = copy.deepcopy(original)
        bad["steps"][3]["loss"] = 1.1
        bad["steps"][3]["loss_float64_hex"] = struct.pack("!d", 1.1).hex()
        write_json(target, bad)
        expect_rejected(lambda: analyze(root))
        bad = copy.deepcopy(original)
        bad["steps"].pop()
        write_json(target, bad)
        expect_rejected(lambda: load_run(root, manifest, "performance", "pr", 1))
        bad = copy.deepcopy(original)
        bad["commit"] = "f" * 40
        write_json(target, bad)
        expect_rejected(lambda: load_run(root, manifest, "performance", "pr", 1))
        write_json(target, original)
        hidden = target.with_suffix(".hidden")
        target.rename(hidden)
        expect_rejected(lambda: analyze(root))
        hidden.rename(target)
        analyze(root)

        for relative in (
            ".env",
            "source/credentials",
            "cache/credentials",
            "venv/credentials",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("DUMMY_SECRET_FOR_EXCLUSION_TEST\n")
        (root / "results/unsafe-link").symlink_to(root / ".env")
        archive = make_bundle(root)
        with tarfile.open(archive) as handle:
            names = handle.getnames()
        assert not any(
            "credentials" in name or "unsafe-link" in name or name.endswith(".env")
            for name in names
        )
        assert "k3-h20-results/summary.json" in names
        assert "k3-h20-results/experiment/train.py" in names

        plan = json.loads(
            subprocess.check_output(
                [sys.executable, "-m", "scripts.distmuon_k3_h20.run", "--dry-run"],
                cwd=SCRIPT_DIR.parents[1],
                text=True,
            )
        )
        assert plan["gpu_training_started"] is False and len(plan["schedule"]) == 8
        try:
            run_logged(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=root,
                env=os.environ.copy(),
                log_path=root / "timeout.log",
                timeout=0.1,
            )
        except TimeoutError:
            pass
        else:
            raise AssertionError("Subprocess timeout was not enforced")
    print(
        "SCRIPT_SELF_TEST_PASSED: synthetic fixtures only; no GPU training or H20 result"
    )


if __name__ == "__main__":
    main()
