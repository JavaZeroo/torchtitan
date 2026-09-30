# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark configs: production Kimi-family shapes cut down to a few layers.

DistMuon work is per layer, so a slice with the real matrix shapes reproduces
the optimizer's per-layer compute and communication while fitting on rented
GPUs. The vocabulary is shrunk because embeddings and the LM head belong to
AdamW and only cost memory here.

Environment knobs (read when the config function runs):
  MB_LAYERS       total transformer layers, first one dense (default per flavor)
  MB_EXPERTS      routed experts per MoE layer (default per flavor)
  MB_VOCAB        vocabulary size (default 4096)
  MB_SEQ_LEN      context length (default 512)
"""

from __future__ import annotations

import os

import torchtitan.models.kimi_k2_7 as kimi_models
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import LRSchedulersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm, Sigmoid
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k2_7 import config_registry as kimi, KimiK25Model
from torchtitan.models.kimi_k2_7.sharding import set_kimi_k2_5_sharding_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _model(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    q_lora_rank: int,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    seq_len: int,
    vocab_size: int,
) -> KimiK25Model.Config:
    layers = kimi_models._build_kimi_layers(
        enable_sp=True,
        n_layers=n_layers,
        n_dense_layers=1,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=router_top_k,
        router_score_func=Sigmoid.Config(),
        router_route_scale=2.446,
        router_route_norm=True,
        attn_backend="flex",
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        rope=ComplexRoPE.Config(
            dim=64,
            max_context_length=seq_len,
            theta=50000.0,
        ),
    )
    return KimiK25Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=kimi_models._EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=kimi_models._NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=kimi_models._output_linear_init(dim),
        ),
        layers=layers,
        vision_encoder=None,
    )


def _trainer_config(model_config: KimiK25Model.Config, *, ep: int) -> Trainer.Config:
    parallelism = ParallelismConfig(expert_parallel_degree=ep)
    return kimi._KimiTrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optimizer=kimi._dist_muon_optimizer(
            model_config,
            muon_lr=3e-4,
            adamw_lr=3e-4,
            parallelism=parallelism,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=model_config.max_context_length,
            max_context_length=model_config.max_context_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=SelectiveAC.Config(),
    )


def moonlight_slice() -> Trainer.Config:
    """Moonlight 16B-A3B shapes: dim 2048, 64 experts of 1408x2048, no q-LoRA."""
    model_config = _model(
        dim=2048,
        n_layers=_env_int("MB_LAYERS", 5),
        n_heads=16,
        q_lora_rank=0,
        dense_hidden_dim=11264,
        moe_hidden_dim=1408,
        num_experts=_env_int("MB_EXPERTS", 64),
        num_shared_experts=2,
        router_top_k=6,
        seq_len=_env_int("MB_SEQ_LEN", 512),
        vocab_size=_env_int("MB_VOCAB", 4096),
    )
    return _trainer_config(model_config, ep=8)


def kimi_k2_5_slice() -> Trainer.Config:
    """Kimi K2.5 shapes: dim 7168, experts of 2048x7168, q-LoRA attention."""
    model_config = _model(
        dim=7168,
        n_layers=_env_int("MB_LAYERS", 3),
        n_heads=64,
        q_lora_rank=1536,
        dense_hidden_dim=18432,
        moe_hidden_dim=2048,
        num_experts=_env_int("MB_EXPERTS", 16),
        num_shared_experts=1,
        router_top_k=8,
        seq_len=_env_int("MB_SEQ_LEN", 512),
        vocab_size=_env_int("MB_VOCAB", 4096),
    )
    return _trainer_config(model_config, ep=8)


def debugmodel() -> Trainer.Config:
    """The CI recipe (Kimi K2.5 debugmodel, FSDP 8 x EP 8) for loss validation."""
    from torchtitan_recipes.tests.models import kimi_k2_5_debugmodel_muon_fsdp8_ep8

    return kimi_k2_5_debugmodel_muon_fsdp8_ep8()


__all__ = ["moonlight_slice", "kimi_k2_5_slice", "debugmodel"]
_ = set_kimi_k2_5_sharding_config
