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


def apply_harness_env(config):
    """Apply the harness environment knobs to a loaded Trainer config.

    The config loader only accepts a recipe name, so run-level settings are
    environment variables (read after the recipe is built):
      MB_SEED           debug seed (default 42; empty disables)
      MB_DETERMINISTIC  1 enables deterministic kernels (default 1)
      MB_STEPS          training steps for train mode
      MB_TIMEOUT        collective timeout in seconds (init and train)
      MB_TB             1 enables TensorBoard metrics every step under dump/tb
      MB_DP_SHARD, MB_DP_REPLICATE, MB_EP  parallelism degrees (rebuilt through
                        the recipe's __post_init__ so expert layouts realign)
    Returns the (possibly replaced) config.
    """
    from dataclasses import replace

    seed = os.environ.get("MB_SEED", "42")
    config.debug.seed = int(seed) if seed else None
    config.debug.deterministic = os.environ.get("MB_DETERMINISTIC", "1") == "1"
    if os.environ.get("MB_STEPS"):
        config.training.steps = int(os.environ["MB_STEPS"])
    if os.environ.get("MB_TB") == "1":
        config.metrics.enable_tensorboard = True
        config.metrics.log_freq = 1
        config.metrics.save_tb_folder = "tb"
    degrees = {}
    for env, field_name in (
        ("MB_DP_SHARD", "data_parallel_shard_degree"),
        ("MB_DP_REPLICATE", "data_parallel_replicate_degree"),
        ("MB_EP", "expert_parallel_degree"),
    ):
        if os.environ.get(env):
            degrees[field_name] = int(os.environ[env])
    if degrees:
        config = replace(config, parallelism=replace(config.parallelism, **degrees))
    return config


def load_config(argv: list[str]):
    """Load a Trainer config through TorchTitan's loader plus the harness env."""
    from typing import cast

    from torchtitan.config import ConfigLoader
    from torchtitan.trainer import Trainer

    config = cast(Trainer.Config, ConfigLoader().load(argv))
    return apply_harness_env(config)


def build_engine(argv: list[str]):
    """Build a TrainingEngine with model and optimizer, without a dataloader.

    Mirrors Trainer.__init__ up to the engine construction.
    """
    import copy

    from torchtitan.config import apply_overrides
    from torchtitan.config.validation import validate_model_training_config
    from torchtitan.training_engine import TrainingEngine

    config = load_config(argv)
    model_config = copy.deepcopy(config.model)
    model_config.set_sharding_(config.parallelism)
    config.model = model_config
    if config.override.imports:
        apply_overrides(config.override, config)
    model_config = config.model
    validate_model_training_config(
        model_config,
        parallelism=config.parallelism,
        training=config.training,
        debug=config.debug,
        activation_checkpoint=config.activation_checkpoint,
        max_num_documents=config.dataloader.max_num_documents,
    )
    engine = TrainingEngine(
        config,
        model_config=model_config,
        max_num_documents=config.dataloader.max_num_documents,
        output_dir=config.dump_folder,
    )
    engine._initialize_model(hf_assets_path=config.hf_assets_path)
    engine._initialize_optim()
    return config, engine


def dist_muon_optimizers(engine) -> list:
    from torchtitan.distributed.flex_shard import DistMuon

    return [opt for opt in engine.optim.optimizers if isinstance(opt, DistMuon)]


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
