# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model flavors."""

from dataclasses import fields
from typing import Any

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.models.kimi_k3.flavors import (
    _debugmodel as _kimi_k3_debugmodel,
    _kimi_k3,
    _linear,
)
from torchtitan.models.kimi_k3.model import KimiK3Model

from .dsa import KimiDSAAttention, KimiDSAIndexer, KimiSparseAttention
from .model import KimiK31Model, KimiK31TransformerBlock

__all__ = ["build_model_config", "KimiK31Model", "MODEL_FLAVORS"]


def _fields_except(config: Any, name: str) -> dict[str, Any]:
    return {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.init and f.name != name
    }


def _dsa_config(
    mla: KimiMLAAttention.Config,
    *,
    index_topk: int,
    index_num_heads: int,
    index_head_dim: int,
    indexer_loss_coeff: float,
) -> KimiDSAAttention.Config:
    """Kimi K3 MLA projections and sharding, plus a replicated DSA indexer."""
    return KimiDSAAttention.Config(
        **_fields_except(mla, "inner_attention"),
        inner_attention=KimiSparseAttention.Config(
            wkv_b=mla.inner_attention.wkv_b,
            index_topk=index_topk,
            aux_loss=(
                None
                if indexer_loss_coeff == 0
                else SparseIndexerLoss.Config(
                    coeff=indexer_loss_coeff,
                    reduce_mesh="loss",
                    softmax_scale=mla.softmax_scale,
                    num_heads=mla.n_heads,
                    compress_ratio=1,
                )
            ),
            sharding_config=mla.inner_attention.sharding_config,
        ),
        indexer=KimiDSAIndexer.Config(
            index_num_heads=index_num_heads,
            index_head_dim=index_head_dim,
            wq_b=_linear(
                mla.wq_a.out_features,
                index_num_heads * index_head_dim,
                sharding_config=replicated_state_sharding_config("weight"),
            ),
            wk=_linear(
                mla.dim,
                index_head_dim,
                sharding_config=replicated_state_sharding_config("weight"),
            ),
            k_norm=LayerNorm.Config(
                normalized_shape=index_head_dim,
                eps=1e-5,
                param_init={"weight": nn.init.ones_, "bias": nn.init.zeros_},
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            weights_proj=_linear(
                mla.dim,
                index_num_heads,
                sharding_config=replicated_state_sharding_config("weight"),
            ),
        ),
    )


def _with_dsa(
    config: KimiK3Model.Config,
    *,
    index_topk: int,
    index_num_heads: int,
    index_head_dim: int,
    indexer_loss_coeff: float,
) -> KimiK31Model.Config:
    """Replace every full-attention MLA layer of a Kimi K3 config with Gated DSA."""
    layers = [
        (
            KimiK31TransformerBlock.Config(
                **_fields_except(layer, "attention"),
                attention=_dsa_config(
                    layer.attention,
                    index_topk=index_topk,
                    index_num_heads=index_num_heads,
                    index_head_dim=index_head_dim,
                    indexer_loss_coeff=indexer_loss_coeff,
                ),
            )
            if isinstance(layer.attention, KimiMLAAttention.Config)
            else layer
        )
        for layer in config.layers
    ]
    return KimiK31Model.Config(**_fields_except(config, "layers"), layers=layers)


def _debugmodel(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    indexer_loss_coeff: float,
) -> KimiK31Model.Config:
    return _with_dsa(
        _kimi_k3_debugmodel(attn_backend, seq_len=seq_len, parallelism=parallelism),
        index_topk=128,
        index_num_heads=4,
        index_head_dim=64,
        indexer_loss_coeff=indexer_loss_coeff,
    )


def _kimi_k3_1_pro(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    indexer_loss_coeff: float,
) -> KimiK31Model.Config:
    return _with_dsa(
        _kimi_k3(attn_backend, seq_len=seq_len, parallelism=parallelism),
        index_topk=2048,
        index_num_heads=32,
        index_head_dim=128,
        indexer_loss_coeff=indexer_loss_coeff,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "Kimi-K3.1-Pro": (_kimi_k3_1_pro, 1_048_576),
}


def build_model_config(
    flavor: str,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    indexer_loss_coeff: float = 0.01,
) -> KimiK31Model.Config:
    if indexer_loss_coeff < 0:
        raise ValueError("indexer_loss_coeff must be nonnegative.")
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        parallelism=parallelism,
        indexer_loss_coeff=indexer_loss_coeff,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
