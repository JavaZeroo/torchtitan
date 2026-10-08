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
  MB_BUCKET_LAYERS  MoE layers per DistMuon bucket (default: the recipe's 2)
  MB_TOKENS_PER_MB  tokens per microbatch per dp rank for train mode (default: seq_len)
Run-level knobs (seed, steps, TensorBoard, parallelism degrees) are applied
afterwards by ``common.apply_harness_env``.
"""

from __future__ import annotations

import os

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import DistMuon, LRSchedulersContainer, Optim
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.flex_shard import BucketConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm, Sigmoid
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k2_7 import flavors as kimi_models, KimiK25Model
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models import kimi_k2_7 as kimi


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


def _regroup_buckets(optimizer_config, layers_per_bucket: int):
    """Rebuild the recipe's per-layer buckets with a different layer grouping.

    The recipe keeps the dense layer 0 alone and pairs the MoE layers; each
    pair also gets a routed-experts bucket. The FQN patterns carry the layer
    index, so they can be regrouped without touching the compute layouts.
    """
    import re
    from dataclasses import replace

    muon = next(
        optimizer
        for optimizer in optimizer_config.optimizers
        if isinstance(optimizer, DistMuon.Config)
    )
    by_layer: dict[tuple[int, bool], list[str]] = {}
    for bucket in muon.bucket_configs:
        for fqn in bucket.patterns:
            match = re.match(r"layers\.(\d+)\.", fqn)
            assert match is not None, fqn
            layer = int(match.group(1))
            routed = ".routed_experts." in fqn
            by_layer.setdefault((layer, routed), []).append(fqn)
    num_layers = max(layer for layer, _ in by_layer) + 1
    groups = [(0,)] + [
        tuple(range(first, min(first + layers_per_bucket, num_layers)))
        for first in range(1, num_layers, layers_per_bucket)
    ]
    buckets = []
    for layer_ids in groups:
        name = "layers." + "-".join(map(str, layer_ids))
        non_routed = [
            f for layer in layer_ids for f in by_layer.get((layer, False), [])
        ]
        routed = [f for layer in layer_ids for f in by_layer.get((layer, True), [])]
        if non_routed:
            buckets.append(BucketConfig(name=name, patterns=tuple(non_routed)))
        if routed:
            buckets.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=tuple(routed))
            )
    optimizers = [
        replace(o, bucket_configs=tuple(buckets)) if o is muon else o
        for o in optimizer_config.optimizers
    ]
    return replace(optimizer_config, optimizers=optimizers)


def _trainer_config(model_config: KimiK25Model.Config, *, ep: int) -> Trainer.Config:
    parallelism = ParallelismConfig(expert_parallel_degree=ep)
    optimizer = kimi._dist_muon_optimizer(
        model_config,
        muon_lr=3e-4,
        adamw_lr=3e-4,
        parallelism=parallelism,
    )
    if os.environ.get("MB_BUCKET_LAYERS"):
        optimizer = _regroup_buckets(optimizer, int(os.environ["MB_BUCKET_LAYERS"]))
    tokens_per_microbatch = _env_int(
        "MB_TOKENS_PER_MB", model_config.max_context_length
    )
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
        optim=Optim.Config(
            optimizer=optimizer,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=tokens_per_microbatch,
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
    from torchtitan_recipes.tests.suites.models import (
        kimi_k2_5_debugmodel_muon_fsdp8_ep8,
    )

    return kimi_k2_5_debugmodel_muon_fsdp8_ep8()


def kimi_k3_debug() -> Trainer.Config:
    """The Kimi K3 debugmodel (per-head Muon, KDA) at FSDP 8 x EP 8."""
    from dataclasses import replace

    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    config = kimi_k3_debugmodel(seq_len=512)
    return replace(
        config,
        parallelism=replace(
            config.parallelism,
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
        ),
    )


def moonlight_16b() -> Trainer.Config:
    """The Moonlight 16B-A3B production recipe on the repository's c4_test data.

    Same model, vocabulary and optimizer as the production recipe; only the
    dataset is local, so training steps need no Hugging Face streaming. The
    tokenizer still has to be downloaded to ./assets/hf/Moonlight-16B-A3B.
    """
    from dataclasses import replace

    from torchtitan_recipes.tests.models.kimi_k2_7 import moonlight_16b_a3b

    config = moonlight_16b_a3b(seq_len=_env_int("MB_SEQ_LEN", 4096))
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = _env_int(
        "MB_TOKENS_PER_MB", config.model.max_context_length
    )
    config.metrics = MetricsProcessor.Config(log_freq=1)
    return replace(config)


__all__ = [
    "moonlight_slice",
    "kimi_k2_5_slice",
    "debugmodel",
    "kimi_k3_debug",
    "moonlight_16b",
]
