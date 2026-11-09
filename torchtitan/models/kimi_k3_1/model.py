# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.models.kimi_k3.model import KimiK3Model

__all__ = ["KimiK31Model"]


class KimiK31Model(KimiK3Model):
    """Kimi K3 with Gated DSA in place of the full-attention MLA layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3Model.Config):
        pass

    @classmethod
    def _register_optimizer_hooks(
        cls, optimizers, model_parts, parallelism_context
    ) -> None:
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook

        super()._register_optimizer_hooks(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)
