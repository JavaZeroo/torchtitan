# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model flavors."""

from dataclasses import replace
from functools import partial

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.kimi_k3.flavors import (
    _kimi_k3_config,
    _linear,
    _mla_config,
    _vision_encoder_config,
)
from torchtitan.models.kimi_k3.model import KimiK3Model

from .dsa import KimiDSAAttention, KimiDSAIndexer, KimiIndexerLoss, KimiSparseAttention
from .model import KimiK31Model

__all__ = ["build_model_config", "KimiK31Model", "MODEL_FLAVORS"]


def _dsa_config(
    *,
    dim: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    attn_backend: str,
    enable_sp: bool,
    index_topk: int,
    index_num_heads: int,
    index_head_dim: int,
    indexer_loss_coeff: float,
) -> KimiDSAAttention.Config:
    """Kimi K3 MLA projections and sharding, plus a replicated DSA indexer."""
    mla = _mla_config(
        dim=dim,
        num_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        attn_backend=attn_backend,
        enable_sp=enable_sp,
    )
    return KimiDSAAttention.Config(
        dim=dim,
        n_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        wq_a=mla.wq_a,
        q_norm=mla.q_norm,
        wq_b=mla.wq_b,
        wkv_a=mla.wkv_a,
        kv_norm=mla.kv_norm,
        gate=mla.gate,
        wo=mla.wo,
        inner_attention=replace(
            mla.inner_attention,
            inner_attention=KimiSparseAttention.Config(
                index_topk=index_topk,
                aux_loss=(
                    None
                    if indexer_loss_coeff == 0
                    else KimiIndexerLoss.Config(
                        coeff=indexer_loss_coeff,
                        num_heads=num_heads,
                        reduce_mesh="loss",
                    )
                ),
            ),
        ),
        indexer=KimiDSAIndexer.Config(
            index_num_heads=index_num_heads,
            index_head_dim=index_head_dim,
            wq_b=_linear(
                q_lora_rank,
                index_num_heads * index_head_dim,
                sharding_config=replicated_state_sharding_config("weight"),
            ),
            wk=_linear(
                dim,
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
                dim,
                index_num_heads,
                sharding_config=replicated_state_sharding_config("weight"),
            ),
        ),
        sharding_config=mla.sharding_config,
    )


def _debugmodel(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    indexer_loss_coeff: float,
) -> KimiK3Model.Config:
    dim = 256
    return _kimi_k3_config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        attn_res_block_size=4,
        num_heads=4,
        q_lora_rank=128,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=512,
        latent_dim=128,
        expert_hidden_dim=128,
        num_experts=8,
        top_k=2,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=256,
            qkv_dim=512,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend=attn_backend,
        parallelism=parallelism,
        full_attention_config_factory=partial(
            _dsa_config,
            index_topk=128,
            index_num_heads=4,
            index_head_dim=64,
            indexer_loss_coeff=indexer_loss_coeff,
        ),
        model_config_cls=KimiK31Model.Config,
    )


def _kimi_k3_1_pro(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
    indexer_loss_coeff: float,
) -> KimiK3Model.Config:
    dim = 7168
    return _kimi_k3_config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=163840,
        num_layers=93,
        full_attention_layers=set(range(3, 92, 4)) | {92},
        attn_res_block_size=12,
        num_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=33792,
        latent_dim=3584,
        expert_hidden_dim=3072,
        num_experts=896,
        top_k=16,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=1024,
            qkv_dim=1536,
            hidden_dim=4096,
            num_layers=27,
            num_heads=12,
            init_pos_emb_height=64,
            init_pos_emb_width=64,
        ),
        attn_backend=attn_backend,
        parallelism=parallelism,
        full_attention_config_factory=partial(
            _dsa_config,
            index_topk=2048,
            index_num_heads=32,
            index_head_dim=128,
            indexer_loss_coeff=indexer_loss_coeff,
        ),
        model_config_cls=KimiK31Model.Config,
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
) -> KimiK3Model.Config:
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
