# Experiment-only recipes for the 8x H20 validation of Kimi K3.1 (#6007).
# Not for upstream. Run through exp_h20/run.sh, which sets the EXP_* env vars.
#
# Groups (each group shares one seed checkpoint, so every run starts from the
# same weights and sees the same data with data_parallel_shard_degree=2):
#   p_*  debug model (16 heads), DistMuon: FSDP / EP / PP
#   t_*  debug model with 32 heads, AdamW: TP needs >= 16 heads per rank for
#        the Attention Gym pair kernel, and DistMuon has no TP yet
#   n_*  configs that are expected to fail with a clear error

import os
from functools import partial

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
from torchtitan.models.kimi_k3.flavors import kimi_k3_config, vision_encoder_config
from torchtitan.models.kimi_k3_1.flavors import _dsa_config
from torchtitan.models.kimi_k3_1.flavors import build_model_config as kimi_k3_1_build
from torchtitan.models.kimi_k3_1.model import KimiK31Model
from torchtitan.observability.metrics import TensorBoardLogger
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests import _set_spmd_typechecking
from torchtitan_recipes.tests.models.kimi_k3_1 import kimi_k3_1_debugmodel


def _adamw(config: Trainer.Config) -> None:
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
    )


def _debugmodel_32_heads(parallelism: ParallelismConfig) -> KimiK31Model.Config:
    # Same as kimi_k3_1/flavors.py:_debugmodel with seq_len 256, except num_heads.
    dim = 256
    return kimi_k3_config(
        config_cls=KimiK31Model.Config,
        max_context_length=256,
        dim=dim,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        full_attention=partial(
            _dsa_config,
            index_topk=128,
            num_index_heads=4,
            index_head_dim=64,
            indexer_loss_coeff=0.01,
        ),
        attn_res_block_size=4,
        num_heads=32,
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
        vision_encoder=vision_encoder_config(
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


def _pp_split(config: Trainer.Config) -> None:
    parallelism = config.parallelism
    split = _generate_llm_fqn_per_model_part(
        2 * parallelism.pipeline_parallel_degree,
        len(config.model.layers),
        parallelism.pipeline_parallel_first_stage_less_layers,
        parallelism.pipeline_parallel_last_stage_less_layers,
    )
    split[0][:0] = KimiK31Model.pipeline_first_stage_module_fqns
    split[-1].extend(KimiK31Model.pipeline_last_stage_module_fqns)
    parallelism.pipeline_parallel_module_fqns_per_model_part = split


def _finish(config: Trainer.Config) -> Trainer.Config:
    config.training.steps = int(os.environ.get("EXP_STEPS", "20"))
    config.debug.seed = 42
    # The (latent, rope) pair backward has no deterministic mode yet
    # (meta-pytorch/attention-gym#795), so these runs are nondeterministic.
    config.debug.deterministic = False
    if not any(isinstance(c, TensorBoardLogger.Config) for c in config.metrics.loggers):
        config.metrics.loggers.append(TensorBoardLogger.Config())
    config.metrics.log_freq = 1
    config.metrics.save_tb_folder = os.environ.get("EXP_TB", "tb")
    mode = os.environ.get("EXP_MODE") or None
    if mode is None:
        return config
    if config.checkpointer is None:
        config.checkpointer = CheckpointManager.Config()
    if mode == "seed":
        p = config.parallelism
        p.data_parallel_replicate_degree = 1
        p.data_parallel_shard_degree = 1
        p.context_parallel_degree = 1
        p.tensor_parallel_degree = 1
        p.pipeline_parallel_degree = 1
        p.expert_parallel_degree = 1
        config.checkpointer.last_save_model_only = True
        config.create_seed_checkpoint = True
    elif mode == "load":
        config.checkpointer.load_only = True
    else:
        raise ValueError(f"Unknown EXP_MODE {mode}")
    return config


def _p(**kwargs) -> Trainer.Config:
    pp = kwargs.get("pipeline_parallel_degree", 1)
    if pp > 1:
        kwargs.setdefault("pipeline_parallel_schedule", "Interleaved1F1B")
        kwargs.setdefault("num_pp_microbatches", 4)
    config = kimi_k3_1_debugmodel(parallelism=ParallelismConfig(**kwargs))
    if pp > 1:
        _set_spmd_typechecking(config, typechecking=False)
        _pp_split(config)
    else:
        _set_spmd_typechecking(config, typechecking=True)
    return config


def _t(**kwargs) -> Trainer.Config:
    # The K3.1 debug recipe builds DistMuon, which rejects TP, so build it
    # without parallelism and swap in the TP model, parallelism and AdamW.
    pp = kwargs.get("pipeline_parallel_degree", 1)
    if pp > 1:
        kwargs.setdefault("pipeline_parallel_schedule", "Interleaved1F1B")
        kwargs.setdefault("num_pp_microbatches", 4)
    config = kimi_k3_1_debugmodel()
    config.parallelism = ParallelismConfig(**kwargs)
    config.model = _debugmodel_32_heads(config.parallelism)
    _adamw(config)
    if pp > 1:
        _set_spmd_typechecking(config, typechecking=False)
        _pp_split(config)
    else:
        _set_spmd_typechecking(config, typechecking=True)
    return config


# --- group p: 16 heads, DistMuon ---
def p_seed() -> Trainer.Config:
    config = _p()
    _adamw(config)
    return _finish(config)


def p_fsdp2() -> Trainer.Config:
    return _finish(_p(data_parallel_shard_degree=2))


def p_fsdp2_ep2() -> Trainer.Config:
    return _finish(_p(data_parallel_shard_degree=2, expert_parallel_degree=2))


def p_fsdp2_pp2() -> Trainer.Config:
    return _finish(_p(data_parallel_shard_degree=2, pipeline_parallel_degree=2))


def p_fsdp2_ep2_pp2() -> Trainer.Config:
    return _finish(
        _p(
            data_parallel_shard_degree=2,
            expert_parallel_degree=2,
            pipeline_parallel_degree=2,
        )
    )


# --- group t: 32 heads, AdamW ---
def t_seed() -> Trainer.Config:
    return _finish(_t())


def t_fsdp2() -> Trainer.Config:
    return _finish(_t(data_parallel_shard_degree=2))


def t_fsdp2_ep2() -> Trainer.Config:
    return _finish(_t(data_parallel_shard_degree=2, expert_parallel_degree=2))


def t_fsdp2_tp2_ep2() -> Trainer.Config:
    return _finish(
        _t(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            enable_sequence_parallel=True,
            expert_parallel_degree=2,
        )
    )


def t_fsdp2_tp2_ep2_pp2() -> Trainer.Config:
    return _finish(
        _t(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
            enable_sequence_parallel=True,
            expert_parallel_degree=2,
            pipeline_parallel_degree=2,
        )
    )


# --- group n: expected to fail ---
def n_tp2_16_heads() -> Trainer.Config:
    # 16 heads / TP 2 = 8 heads per rank, below the pair kernel's minimum of 16.
    config = kimi_k3_1_debugmodel()
    config.parallelism = ParallelismConfig(
        tensor_parallel_degree=2,
        enable_sequence_parallel=True,
        expert_parallel_degree=2,
    )
    config.model = kimi_k3_1_build("debugmodel", seq_len=256, parallelism=config.parallelism)
    _adamw(config)
    _set_spmd_typechecking(config, typechecking=True)
    return _finish(config)
