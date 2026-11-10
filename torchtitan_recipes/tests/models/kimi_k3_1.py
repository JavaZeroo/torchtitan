# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 model configurations used by tests."""

from typing import cast

import torch
from torch.distributed.tensor import Shard

from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    DistMuon,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import PrecisionConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
    Owned,
)
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS
from torchtitan.models.kimi_k3_1 import build_model_config, KimiK31Model
from torchtitan.models.kimi_k3_1.dsa import KimiDSAAttention
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.models.kimi_k2_7 import _per_expert_compute_layout
from torchtitan_recipes.tests.models.kimi_k3 import (
    _kimi_k3_multimodal_dataloader,
    _KimiK3TrainerConfig,
)


def kimi_k3_1_debugmodel(
    seq_len: int | None = 256,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=parallelism
    )
    return _KimiK3TrainerConfig(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_config),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"]),
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model_config,
                muon_lr=8e-4,
                adamw_lr=8e-4,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        precision=PrecisionConfig(dtype="bfloat16"),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=1 * model_config.max_context_length,
            steps=10,
            disable_accelerator_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def _dist_muon_optimizer(
    model_config: KimiK31Model.Config,
    *,
    muon_lr: float,
    adamw_lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    attention = cast(KimiDSAAttention.Config, model_config.first_full_attention)
    delta_attention = next(
        layer.attention
        for layer in model_config.layers
        if isinstance(layer.attention, KDA.Config)
    )
    owned = ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.DP_SHARD.value: Owned(),
            f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": Owned(),
        },
        compute_dtype=torch.bfloat16,
    )

    def blocks_of(*num_rows: int) -> ComputeLayout:
        return ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: BlockShard(dim=0, block_sizes=num_rows),
                f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": BlockShard(
                    dim=0, block_sizes=num_rows
                ),
            },
            compute_dtype=torch.bfloat16,
        )

    attention_shardings = {
        "wq_a": owned,
        "wq_b": blocks_of(attention.qk_nope_head_dim, attention.qk_rope_head_dim),
        "wkv_a": blocks_of(attention.kv_lora_rank, attention.qk_rope_head_dim),
        "inner_attention.wkv_b": blocks_of(
            attention.qk_nope_head_dim, attention.v_head_dim
        ),
        "gate": blocks_of(attention.v_head_dim),
        "wo": owned,
        "indexer.wq_b": blocks_of(attention.indexer.index_head_dim),
        "indexer.wk": owned,
        "indexer.weights_proj": owned,
    }
    per_kda_head = blocks_of(delta_attention.head_dim)
    delta_attention_shardings = {
        "q_proj": per_kda_head,
        "k_proj": per_kda_head,
        "v_proj": per_kda_head,
        "forget_a": owned,
        "forget_b": per_kda_head,
        "output_gate": per_kda_head,
        "output_proj": owned,
    }
    per_expert = _per_expert_compute_layout(parallelism)
    feed_forward_shardings = {
        "w13": ComputeLayout(
            shardings_by_mesh_axis={
                MeshAxisName.DP_SHARD.value: Shard(0),
                f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}": Shard(0),
            },
            compute_dtype=torch.bfloat16,
        ),
        "w2": owned,
    }
    expert_projections = ("w13.weight", "w2.weight")

    def compute_shardings_for_layer(layer_id: int) -> dict[str, ComputeLayout]:
        layer = model_config.layers[layer_id]
        prefix = f"layers.{layer_id}"
        shardings = {}
        if isinstance(layer.attention, KimiDSAAttention.Config):
            shardings.update(
                {
                    f"{prefix}.attention.{projection}.weight": compute_sharding
                    for projection, compute_sharding in attention_shardings.items()
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.attention.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (
                        delta_attention_shardings.items()
                    )
                }
            )
        if isinstance(layer.feed_forward, DenseFeedForward.Config):
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (feed_forward_shardings.items())
                }
            )
        else:
            shardings.update(
                {
                    f"{prefix}.feed_forward.routed_experts.{projection}": per_expert
                    for projection in expert_projections
                }
            )
            shardings[f"{prefix}.feed_forward.router.gate.weight"] = owned
            shardings.update(
                {
                    f"{prefix}.feed_forward.shared_experts.{projection}.weight": compute_sharding
                    for projection, compute_sharding in (feed_forward_shardings.items())
                }
            )
            shardings.update(
                {
                    f"{prefix}.feed_forward.{projection}.weight": owned
                    for projection in ("routed_down", "routed_up")
                }
            )
        return shardings

    num_layers = len(model_config.layers)
    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    layer_fqns = []
    for layer_id in range(num_layers):
        layer_shardings = compute_shardings_for_layer(layer_id)
        compute_sharding_by_fqn.update(layer_shardings)
        layer_fqns.append(tuple(layer_shardings))
    bucket_layer_ids = [(0,)] + [
        tuple(range(first_layer_id, min(first_layer_id + 2, num_layers)))
        for first_layer_id in range(1, num_layers, 2)
    ]
    bucket_configs = []
    for layer_ids in bucket_layer_ids:
        name = "layers." + "-".join(map(str, layer_ids))
        fqns = [fqn for layer_id in layer_ids for fqn in layer_fqns[layer_id]]
        routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is per_expert
        )
        non_routed_fqns = tuple(
            fqn for fqn in fqns if compute_sharding_by_fqn[fqn] is not per_expert
        )
        bucket_configs.append(BucketConfig(name=name, patterns=non_routed_fqns))
        if routed_fqns:
            bucket_configs.append(
                BucketConfig(name=f"{name}.routed-experts", patterns=routed_fqns)
            )
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_shardings)}|"
        rf"{'|'.join(delta_attention_shardings)})\.weight|"
        r"routed_experts\.w(?:13|2)\.weight|"
        r"feed_forward\.w(?:13|2)\.weight|"
        r"feed_forward\.router\.gate\.weight|"
        r"feed_forward\.shared_experts\.w(?:13|2)\.weight|"
        r"feed_forward\.routed_(?:down|up)\.weight"
        r")$"
    )
    return OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern=muon_pattern,
                bucket_configs=tuple(bucket_configs),
                compute_sharding_by_fqn=compute_sharding_by_fqn,
                lr=muon_lr,
                weight_decay=0.1,
                adjust_lr_fn="match_rms_adamw",
                ns_device="accelerator",
            ),
            AdamW.Config(
                pattern=r".*",
                foreach=True,
                fused=False,
                lr=adamw_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
        ],
    )
