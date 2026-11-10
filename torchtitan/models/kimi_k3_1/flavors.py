# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model flavors."""

import spmd_types as spmd

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import Embedding, Linear, RowParallelLinear
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    mla_attention_local_spmd_sharding_config,
    mla_kv_projection_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_input_colwise_sharding_config,
    replicated_state_sharding_config,
    residual_activation_placement,
    rowwise_sharding_config,
)
from torchtitan.models.common.nn_modules import LayerNorm, RMSNorm
from torchtitan.models.common.vision_encoder_sharding import (
    multimodal_token_embedding_sharding_config,
)
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3.flavors import (
    _EMBEDDING_INIT,
    _feed_forward_config,
    _kda_config,
    _latent_moe_config,
    _linear,
    _LINEAR_INIT,
    _norm,
    _output_linear_init,
    _RES_PROJ_INIT,
    _vision_encoder_config,
)
from torchtitan.models.kimi_k3.model import AttentionResidual, KimiK3TransformerBlock
from torchtitan.models.kimi_k3.sharding import (
    mla_attention_sharding_config,
    multimodal_first_layer_sharding_config,
    tp_unsharded_weight_sharding_config,
)
from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder

from .dsa import KimiDSAAttention, KimiDSAIndexer, KimiSparseAttention
from .model import KimiK31Model, KimiK31TransformerBlock

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
    index_topk: int,
    index_num_heads: int,
    index_head_dim: int,
    enable_sp: bool,
) -> KimiDSAAttention.Config:
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    return KimiDSAAttention.Config(
        dim=dim,
        n_heads=num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        wq_a=_linear(
            dim, q_lora_rank, sharding_config=replicated_state_sharding_config("weight")
        ),
        q_norm=_norm(
            q_lora_rank, sharding_config=replicated_state_sharding_config("weight")
        ),
        wq_b=_linear(
            q_lora_rank,
            num_heads * q_head_dim,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wkv_a=_linear(
            dim,
            kv_lora_rank + qk_rope_head_dim,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        kv_norm=_norm(
            kv_lora_rank, sharding_config=replicated_state_sharding_config("weight")
        ),
        gate=_linear(
            dim,
            num_heads * v_head_dim,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wo=RowParallelLinear.Config(
            in_features=num_heads * v_head_dim,
            out_features=dim,
            param_init=_LINEAR_INIT,
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        inner_attention=KimiSparseAttention.Config(
            wkv_b=_linear(
                kv_lora_rank,
                num_heads * (qk_nope_head_dim + v_head_dim),
                sharding_config=mla_kv_projection_sharding_config(cp=spmd.S(0)),
            ),
            index_topk=index_topk,
            aux_loss=SparseIndexerLoss.Config(
                coeff=0.01,
                reduce_mesh="loss",
                softmax_scale=q_head_dim**-0.5,
                num_heads=num_heads,
                compress_ratio=1,
            ),
            sharding_config=mla_attention_local_spmd_sharding_config(),
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
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            weights_proj=_linear(
                dim,
                index_num_heads,
                sharding_config=replicated_state_sharding_config("weight"),
            ),
        ),
        sharding_config=mla_attention_sharding_config(enable_sp=enable_sp),
    )


def _kimi_k3_1_config(
    *,
    max_context_length: int,
    dim: int,
    vocab_size: int,
    num_layers: int,
    full_attention_layers: set[int],
    attn_res_block_size: int,
    num_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_topk: int,
    index_num_heads: int,
    index_head_dim: int,
    kda_head_dim: int,
    conv_kernel_size: int,
    dense_hidden_dim: int,
    latent_dim: int,
    expert_hidden_dim: int,
    num_experts: int,
    top_k: int,
    num_shared_experts: int,
    vision_encoder: KimiK3VisionEncoder.Config,
    parallelism: ParallelismConfig,
) -> KimiK31Model.Config:
    enable_sp = parallelism.enable_sequence_parallel
    enable_ep = parallelism.expert_parallel_degree > 1
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)

    def residual_norm() -> RMSNorm.Config:
        return _norm(
            dim,
            sharding_config=tp_unsharded_weight_sharding_config(
                token_sharded=enable_sp
            ),
        )

    def residual_proj() -> Linear.Config:
        return _linear(
            dim,
            1,
            param_init=_RES_PROJ_INIT,
            sharding_config=tp_unsharded_weight_sharding_config(
                token_sharded=enable_sp
            ),
        )

    layers = []
    for layer_idx in range(num_layers):
        is_full_attention = layer_idx in full_attention_layers
        block = KimiK31TransformerBlock if is_full_attention else KimiK3TransformerBlock
        layers.append(
            block.Config(
                layer_id=layer_idx,
                attn_res_block_size=attn_res_block_size,
                attention=(
                    _dsa_config(
                        dim=dim,
                        num_heads=num_heads,
                        q_lora_rank=q_lora_rank,
                        kv_lora_rank=kv_lora_rank,
                        qk_nope_head_dim=qk_nope_head_dim,
                        qk_rope_head_dim=qk_rope_head_dim,
                        v_head_dim=v_head_dim,
                        index_topk=index_topk,
                        index_num_heads=index_num_heads,
                        index_head_dim=index_head_dim,
                        enable_sp=enable_sp,
                    )
                    if is_full_attention
                    else _kda_config(
                        dim=dim,
                        num_heads=num_heads,
                        head_dim=kda_head_dim,
                        conv_kernel_size=conv_kernel_size,
                        enable_sp=enable_sp,
                    )
                ),
                feed_forward=(
                    _feed_forward_config(
                        dim=dim, hidden_dim=dense_hidden_dim, enable_sp=enable_sp
                    )
                    if layer_idx == 0
                    else _latent_moe_config(
                        dim=dim,
                        latent_dim=latent_dim,
                        expert_hidden_dim=expert_hidden_dim,
                        num_experts=num_experts,
                        top_k=top_k,
                        num_shared_experts=num_shared_experts,
                        enable_sp=enable_sp,
                        enable_ep=enable_ep,
                    )
                ),
                attention_norm=_norm(dim, sharding_config=norm_sharding),
                ffn_norm=_norm(dim, sharding_config=norm_sharding),
                attention_res_norm=None if layer_idx == 0 else residual_norm(),
                attention_res_proj=None if layer_idx == 0 else residual_proj(),
                ffn_res_norm=residual_norm(),
                ffn_res_proj=residual_proj(),
                attention_residual=AttentionResidual.Config(),
                sharding_config=(
                    multimodal_first_layer_sharding_config(enable_sp=enable_sp)
                    if layer_idx == 0
                    else None
                ),
            )
        )

    return KimiK31Model.Config(
        max_context_length=max_context_length,
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_sharding_config(),
        ),
        layers=layers,
        norm=_norm(
            dim, sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
        ),
        lm_head=_linear(
            dim,
            vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        output_res_norm=residual_norm(),
        output_res_proj=residual_proj(),
        vision_encoder=vision_encoder,
    )


def _debugmodel(*, seq_len: int, parallelism: ParallelismConfig) -> KimiK31Model.Config:
    dim = 256
    return _kimi_k3_1_config(
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
        index_topk=128,
        index_num_heads=4,
        index_head_dim=64,
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
        parallelism=parallelism,
    )


def _kimi_k3_1_pro(
    *, seq_len: int, parallelism: ParallelismConfig
) -> KimiK31Model.Config:
    dim = 7168
    return _kimi_k3_1_config(
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
        index_topk=2048,
        index_num_heads=32,
        index_head_dim=128,
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
        parallelism=parallelism,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 16384),
    "Kimi-K3.1-Pro": (_kimi_k3_1_pro, 1_048_576),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> KimiK31Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    return get_config(seq_len=context_len, parallelism=parallelism)
