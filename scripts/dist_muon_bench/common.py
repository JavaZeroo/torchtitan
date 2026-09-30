# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the DistMuon analysis harness.

Everything here lives outside the repository on purpose: the harness drives
unmodified TorchTitan entry points and reads optimizer internals, so the same
scripts run against the baseline checkout and the optimized branch.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Any

import torch


def build_engine(argv: list[str]):
    """Build a TrainingEngine with model and optimizer, without a dataloader."""
    from typing import cast

    from torchtitan.config import ConfigManager
    from torchtitan.trainer import Trainer
    from torchtitan.training_engine import TrainingEngine

    config = cast(Trainer.Config, ConfigManager().parse_args(argv))
    model_config = config.model
    model_config.update_from_config(config=config)
    if config.override.imports:
        from torchtitan.config.override import apply_overrides

        apply_overrides(config.override, config)
    config.__post_init__()
    engine = TrainingEngine(
        config,
        model_config=model_config,
        max_num_documents=config.dataloader.max_num_documents,
        output_dir=config.dump_folder,
    )
    engine._initialize_model(
        compile_config=config.compile,
        hf_assets_path=config.hf_assets_path,
    )
    engine._initialize_optimizer()
    return config, engine


def dist_muon_optimizers(engine) -> list:
    from torchtitan.distributed.flex_shard import DistMuon

    return [opt for opt in engine.optimizers if isinstance(opt, DistMuon)]


def ns_flops(matrix_shape, ns_steps: int = 5) -> int:
    """FLOPs of one Newton-Schulz call on a [..., R, C] batch (2 per multiply-add)."""
    *batch, rows, cols = matrix_shape
    num = math.prod(batch) if batch else 1
    short, long_ = sorted((rows, cols))
    return 2 * num * ns_steps * (2 * short * short * long_ + short**3)


def tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous()
    if data.dtype == torch.bfloat16:
        data = data.view(torch.int16)
    return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()


def dump_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)


def harness_args() -> tuple[dict[str, str], list[str]]:
    """Split ``--mb.key=value`` harness options from TorchTitan's own arguments."""
    options: dict[str, str] = {}
    passthrough = []
    for arg in sys.argv[1:]:
        if arg.startswith("--mb."):
            key, _, value = arg[5:].partition("=")
            options[key] = value
        else:
            passthrough.append(arg)
    return options, passthrough
