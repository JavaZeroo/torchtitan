# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Official K3 debug recipe, real CC12M data, and deterministic execution."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import KimiK3Model
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel


def build_config(protocol, output):
    config = kimi_k3_debugmodel(
        seq_len=protocol["seq_len"],
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=protocol["world_size"],
            expert_parallel_degree=protocol["world_size"],
        ),
    )
    config.training.steps = protocol["num_steps"]
    config.debug.seed = protocol["seed"]
    config.debug.deterministic = True
    config.debug.deterministic_warn_only = False
    config.comm.init_timeout_seconds = 300
    config.comm.train_timeout_seconds = 300
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = False
    config.dump_folder = str(output / "dump")
    assert not config.profiler.enable_profiling
    assert not config.profiler.enable_memory_snapshot
    assert config.checkpointer is None and config.validator is None
    assert isinstance(config.model, KimiK3Model.Config)
    assert len(config.model.layers) == 17 and config.model.vision_encoder is not None
    return config
