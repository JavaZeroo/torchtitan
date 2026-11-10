# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model configurations used by tests."""

import dataclasses

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3_1 import build_model_config
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def kimi_k3_1_debugmodel(
    seq_len: int | None = 256,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Kimi K3 debug recipe with the K3.1 model; 256 tokens span two top-k windows."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    return dataclasses.replace(
        kimi_k3_debugmodel(seq_len, parallelism=parallelism),
        model=build_model_config(
            "debugmodel", seq_len=seq_len, parallelism=parallelism
        ),
    )
