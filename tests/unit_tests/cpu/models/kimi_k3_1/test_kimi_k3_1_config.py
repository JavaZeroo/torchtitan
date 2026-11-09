# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def test_kimi_k3_1_debug_recipe_builds_independent_model() -> None:
    from torchtitan.models.kimi_k3_1.dsa import KimiGatedDSAAttention
    from torchtitan.models.kimi_k3_1.model import KimiK31Model, KimiK31TransformerBlock
    from torchtitan_recipes.tests.models.kimi_k3_1 import kimi_k3_1_debugmodel

    config = kimi_k3_1_debugmodel()
    assert isinstance(config.model, KimiK31Model.Config)

    model = config.model.build()
    assert isinstance(model, KimiK31Model)
    assert all(
        isinstance(block, KimiK31TransformerBlock) for block in model.layers.values()
    )
    assert (
        sum(
            isinstance(block.attention, KimiGatedDSAAttention)
            for block in model.layers.values()
        )
        == 5
    )
