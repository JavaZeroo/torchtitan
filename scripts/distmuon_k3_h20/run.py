# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare pinned checkouts, run manual H20 experiments, and bundle results."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .analyze import analyze, load_run, require
from .common import (
    experiment_hashes,
    file_digest,
    fingerprint,
    load_protocol,
    read_json,
    SCRIPT_DIR,
    write_json,
)


def output_text(command, *, cwd=None, env=None):
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def run_schedule(protocol):
    jobs = [("numerics", variant, 1) for variant in ("main", "pr")]
    for trial in range(1, protocol["num_trials"] + 1):
        order = ("pr", "main") if trial % 2 == 0 else ("main", "pr")
        jobs.extend(("performance", variant, trial) for variant in order)
    return jobs


def run_logged(command, *, cwd, env, log_path, timeout):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Run exceeded {timeout}s: {log_path}")
                try:
                    code = process.wait(timeout=min(20, remaining))
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"  running {elapsed:.0f}s; log: {log_path}", flush=True)
                    if elapsed >= timeout:
                        raise TimeoutError(
                            f"Run exceeded {timeout}s: {log_path}"
                        ) from None
            if code:
                raise RuntimeError(f"Command exited {code}; see {log_path}")
        except BaseException:
            # Only terminate this experiment subprocess group, never other jobs.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def make_bundle(root):
    archive = root.with_name(root.name + ".tar.gz")
    names = (
        "manifest.json",
        "status.json",
        "environment.json",
        "hardware.txt",
        "topology.txt",
        "numerics.json",
        "summary.json",
        "summary.md",
        "step_metrics.csv",
        "validation_error.json",
        "results",
        "preflight",
    )
    with tarfile.open(archive, "w:gz") as handle:
        for name in names:
            path = root / name
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            for item in paths:
                if item.is_file() and not item.is_symlink():
                    handle.add(
                        item,
                        arcname=str(Path("k3-h20-results") / item.relative_to(root)),
                        recursive=False,
                    )
        for path in sorted(SCRIPT_DIR.iterdir()):
            if path.is_file() and path.suffix in (".py", ".json", ".sh", ".md"):
                handle.add(
                    path,
                    arcname=f"k3-h20-results/experiment/{path.name}",
                    recursive=False,
                )
    checksum = file_digest(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{checksum}  {archive.name}\n"
    )
    print(f"RESULT_BUNDLE={archive}\nSHA256={checksum}", flush=True)
    return archive


def child_environment(root, python, source):
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTORCH_ALLOC_CONF="expandable_segments:True",
        TORCHINDUCTOR_CACHE_DIR=str(root / "cache/inductor"),
        ATTN_GYM_CUTE_CACHE_DIR=str(root / "cache/attn_gym"),
        HF_HOME=str(root / "cache/huggingface"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join((str(source), str(SCRIPT_DIR.parents[1]))),
    )
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    return env


def prepare_sources(repo, root, protocol):
    sources = {}
    hashes = {}
    for variant, commit in protocol["variants"].items():
        available = (
            subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=repo,
                capture_output=True,
            ).returncode
            == 0
        )
        if not available:
            subprocess.run(["git", "fetch", "origin", commit], cwd=repo, check=True)
        source = root / "source" / variant
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(source), commit],
                cwd=repo,
                check=True,
            )
        require(
            output_text(["git", "rev-parse", "HEAD"], cwd=source) == commit,
            f"Unexpected existing checkout: {source}",
        )
        require(
            not output_text(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source
            ),
            f"Tracked files were modified in {source}; do not overwrite them",
        )
        sources[variant] = source
        hashes[variant] = {
            path: file_digest(source / path) for path in protocol["source_files"]
        }
    for path in protocol["source_files"]:
        if path.startswith("tests/assets/"):
            require(
                hashes["main"][path] == hashes["pr"][path],
                f"Dataset/tokenizer differs between variants: {path}",
            )
    return sources, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="New result directory; default: outputs/k3-h20/<timestamp>",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python from the training environment; defaults to this interpreter",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Wall-time limit in seconds for each torchrun",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated completed runs; preserve failed attempts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the protocol without writing files or using GPUs",
    )
    parser.add_argument(
        "--bundle-only",
        action="store_true",
        help="Package an existing run without training",
    )
    args = parser.parse_args()
    protocol = load_protocol()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "schedule": run_schedule(protocol),
                    "gpu_training_started": False,
                },
                indent=2,
            )
        )
        return
    if sys.version_info < (3, 12):
        raise SystemExit("Use Python 3.12 or newer for the experiment driver")
    repo = Path(output_text(["git", "rev-parse", "--show-toplevel"], cwd=SCRIPT_DIR))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (
        (args.output or repo / "outputs" / "k3-h20" / timestamp).expanduser().absolute()
    )
    require(
        root != repo and root != Path(root.anchor),
        "Use a dedicated result directory, not the repository or filesystem root",
    )
    if args.bundle_only:
        require(
            (root / "manifest.json").is_file(),
            "No experiment manifest in this result directory",
        )
        make_bundle(root)
        return
    require(args.timeout > 0, "Timeout must be positive")
    if root.exists():
        require(
            args.resume and (root / "manifest.json").is_file(),
            "Result directory already exists; use --resume for this experiment or choose a new directory",
        )
    else:
        require(not args.resume, "Cannot resume a missing result directory")
        root.mkdir(parents=True)
    # Keep the venv executable path; resolving its symlink loses the venv.
    python = Path(shutil.which(args.python) or args.python).expanduser().absolute()
    require(python.is_file(), f"Python executable not found: {python}")
    status: dict[str, Any] = {"status": "preparing", "completed_runs": []}
    write_json(root / "status.json", status)
    try:
        sources, hashes = prepare_sources(repo, root, protocol)
        env = child_environment(root, python, sources["pr"])
        packages = json.loads(
            output_text(
                [
                    str(python),
                    "-c",
                    "import json; from scripts.distmuon_k3_h20.common import "
                    "environment; print(json.dumps(environment()))",
                ],
                cwd=sources["pr"],
                env=env,
            )
        )
        hardware = output_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        manifest = {
            "protocol": protocol,
            "experiment_commit": output_text(["git", "rev-parse", "HEAD"], cwd=repo),
            "source_sha256": hashes,
            "harness_sha256": experiment_hashes(),
            "environment_fingerprint": fingerprint(packages),
            "hardware_identity": hardware,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        manifest_path = root / "manifest.json"
        if args.resume:
            previous = read_json(manifest_path)
            for key in (
                "protocol",
                "source_sha256",
                "harness_sha256",
                "environment_fingerprint",
                "hardware_identity",
                "cuda_visible_devices",
            ):
                require(
                    previous[key] == manifest[key],
                    f"Cannot mix resumed runs: {key} changed",
                )
        write_json(manifest_path, manifest)
        write_json(root / "environment.json", packages)
        (root / "hardware.txt").write_text(hardware + "\n")
        (root / "topology.txt").write_text(
            output_text(["nvidia-smi", "topo", "-m"]) + "\n"
        )
        launcher = [
            str(python),
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={protocol['world_size']}",
            "--max_restarts=0",
            "--module",
        ]
        print(f"H20 preflight; results: {root}", flush=True)
        run_logged(
            launcher
            + [
                "scripts.distmuon_k3_h20.preflight",
                "--manifest",
                str(manifest_path),
                "--output",
                str(root / "preflight"),
            ],
            cwd=sources["pr"],
            env=env,
            log_path=root / "preflight/run.log",
            timeout=args.timeout,
        )
        for rank in range(protocol["world_size"]):
            check = read_json(root / "preflight" / f"rank{rank}.json")
            require(
                check["complete"] and check["rank"] == rank,
                "Incomplete hardware preflight",
            )
            require(
                check["environment_fingerprint"] == manifest["environment_fingerprint"],
                "Preflight environment differs",
            )
        for phase, variant, trial in run_schedule(protocol):
            output = root / "results" / phase / variant / f"trial{trial}"
            label = f"{phase}/{variant}/trial{trial}"
            if output.exists():
                try:
                    load_run(root, manifest, phase, variant, trial)
                except (ValueError, KeyError, OSError, TypeError):
                    failed = output.with_name(
                        output.name + f".failed-{timestamp}-{time.time_ns()}"
                    )
                    output.rename(failed)
                    print(f"Preserved incomplete attempt: {failed}", flush=True)
                else:
                    print(f"SKIP validated completed run: {label}", flush=True)
                    status["completed_runs"].append(label)
                    if phase == "numerics" and variant == "pr":
                        analyze(root, numerics_only=True)
                    continue
            require(
                not output_text(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=sources[variant],
                ),
                f"Source was edited during experiment: {variant}",
            )
            print(
                f"START {label}: {protocol['num_steps']} real training steps",
                flush=True,
            )
            command = launcher + [
                "scripts.distmuon_k3_h20.train",
                "--manifest",
                str(manifest_path),
                "--source",
                str(sources[variant]),
                "--output",
                str(output),
                "--variant",
                variant,
                "--phase",
                phase,
                "--trial",
                str(trial),
            ]
            run_logged(
                command,
                cwd=sources[variant],
                env=child_environment(root, python, sources[variant]),
                log_path=output / "run.log",
                timeout=args.timeout,
            )
            load_run(root, manifest, phase, variant, trial)
            status["completed_runs"].append(label)
            status["status"] = "running"
            write_json(root / "status.json", status)
            print(f"DONE {label}", flush=True)
            if phase == "numerics" and variant == "pr":
                analyze(root, numerics_only=True)
                print("NUMERICS_PASSED; starting performance runs", flush=True)
        analyze(root)
        status["status"] = "validated_complete"
        print((root / "summary.md").read_text(), flush=True)
    except BaseException as error:
        status["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(root / "status.json", status)
        make_bundle(root)


if __name__ == "__main__":
    main()
