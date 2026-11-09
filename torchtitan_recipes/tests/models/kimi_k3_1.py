# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model configurations used by tests."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.kimi_k3_1 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def kimi_k3_1_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    attn_backend: str = "varlen",
) -> Trainer.Config:
    config = kimi_k3_debugmodel(
        seq_len=seq_len,
        parallelism=parallelism,
        attn_backend=attn_backend,
    )
    config.model = build_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend=attn_backend,
        parallelism=config.parallelism,
    )
    return config
