# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model configurations used by tests."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3_1 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k3 import _debugmodel_trainer_config


def kimi_k3_1_debugmodel(
    seq_len: int | None = 256,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Use two debug top-k windows per microbatch to exercise sparse DSA."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        parallelism=parallelism,
    )
    return _debugmodel_trainer_config(model_config, parallelism=parallelism)
