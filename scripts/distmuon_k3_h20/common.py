# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dependency-free metadata helpers shared by the experiment commands."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def environment():
    """Record package versions, without private index URLs or credentials."""
    packages = sorted(
        (item.metadata["Name"], item.version)
        for item in importlib.metadata.distributions()
        if item.metadata["Name"]
    )
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "packages": packages,
    }


def experiment_hashes():
    return {
        path.name: file_digest(path)
        for path in sorted(SCRIPT_DIR.iterdir())
        if path.suffix in (".py", ".json", ".sh")
    }


def load_protocol():
    protocol = read_json(SCRIPT_DIR / "protocol.json")
    if not 10 <= protocol["num_steps"] or not (
        0 < protocol["num_warmup_steps"] < protocol["num_steps"]
    ):
        raise ValueError("Invalid training or warmup step counts")
    return protocol
