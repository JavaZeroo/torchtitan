# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model flavors, reusing the K3 backbone configuration."""

from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3 import (
    build_model_config as build_kimi_k3_model_config,
    MODEL_FLAVORS as K3_MODEL_FLAVORS,
)
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.models.kimi_k3.dsa import (
    KimiDSAIndexer,
    KimiDSALatentAttention,
    KimiGatedDSAAttention,
)
from torchtitan.models.kimi_k3.model import KimiK3Model

__all__ = ["KimiK3Model", "MODEL_FLAVORS", "build_model_config"]

_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}


def _replicated_linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init=_LINEAR_INIT,
        sharding_config=replicated_state_sharding_config("weight"),
    )


def _convert_full_attention_to_dsa(
    config: KimiK3Model.Config,
    *,
    num_index_heads: int,
    index_head_dim: int,
    index_topk: int,
) -> KimiK3Model.Config:
    layers = []
    for layer in config.layers:
        attention = layer.attention
        if not isinstance(attention, KimiMLAAttention.Config):
            layers.append(layer)
            continue

        indexer = KimiDSAIndexer.Config(
            n_heads=num_index_heads,
            head_dim=index_head_dim,
            wq_b=_replicated_linear(
                attention.wq_a.out_features, num_index_heads * index_head_dim
            ),
            wk=_replicated_linear(attention.dim, index_head_dim),
            weights_proj=_replicated_linear(attention.dim, num_index_heads),
            k_norm=LayerNorm.Config(
                normalized_shape=index_head_dim,
                eps=1e-6,
                param_init=_NORM_INIT,
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
        )
        inner_attention = KimiDSALatentAttention.Config(
            n_heads=attention.n_heads,
            kv_lora_rank=attention.kv_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            index_topk=index_topk,
            wkv_b=attention.inner_attention.wkv_b,
            aux_loss=SparseIndexerLoss.Config(
                coeff=0.01,
                reduce_mesh="loss",
                softmax_scale=attention.softmax_scale,
                num_heads=attention.n_heads,
                compress_ratio=1,
            ),
        )
        attention = KimiGatedDSAAttention.Config(
            dim=attention.dim,
            n_heads=attention.n_heads,
            kv_lora_rank=attention.kv_lora_rank,
            qk_nope_head_dim=attention.qk_nope_head_dim,
            qk_rope_head_dim=attention.qk_rope_head_dim,
            v_head_dim=attention.v_head_dim,
            wq_a=attention.wq_a,
            q_norm=attention.q_norm,
            wq_b=attention.wq_b,
            wkv_a=attention.wkv_a,
            kv_norm=attention.kv_norm,
            gate=attention.gate,
            wo=attention.wo,
            inner_attention=inner_attention,
            indexer=indexer,
            sharding_config=attention.sharding_config,
        )
        layers.append(replace(layer, attention=attention))
    return replace(config, layers=layers)


def _build_config(
    *,
    base_flavor: str,
    num_index_heads: int,
    index_topk: int,
    attn_backend: str,
    parallelism: ParallelismConfig,
    seq_len: int,
) -> KimiK3Model.Config:
    base_max_context = K3_MODEL_FLAVORS[base_flavor][1]
    config = build_kimi_k3_model_config(
        base_flavor,
        attn_backend=attn_backend,
        parallelism=parallelism,
        seq_len=min(seq_len, base_max_context),
    )
    config = _convert_full_attention_to_dsa(
        config,
        num_index_heads=num_index_heads,
        index_head_dim=128,
        index_topk=index_topk,
    )
    # Neither the MLA-derived DSA branch nor its indexer uses RoPE.
    return replace(config, max_context_length=seq_len)


def _debugmodel(
    *, parallelism: ParallelismConfig, seq_len: int, attn_backend: str
) -> KimiK3Model.Config:
    return _build_config(
        base_flavor="debugmodel",
        num_index_heads=4,
        index_topk=4,
        attn_backend=attn_backend,
        parallelism=parallelism,
        seq_len=seq_len,
    )


def _kimi_k3_1(
    *, parallelism: ParallelismConfig, seq_len: int, attn_backend: str
) -> KimiK3Model.Config:
    return _build_config(
        base_flavor="Kimi-K3",
        num_index_heads=32,
        index_topk=2048,
        attn_backend=attn_backend,
        parallelism=parallelism,
        seq_len=seq_len,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "Kimi-K3.1-Pro": (_kimi_k3_1, 1048576),
}


def build_model_config(
    flavor: str,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> KimiK3Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        parallelism=parallelism, seq_len=context_len, attn_backend=attn_backend
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
