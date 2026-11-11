# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torchtitan.training_engine import TrainingEngine


def test_optimizer_and_checkpoint_capture_bound_parameter():
    class Engine(TrainingEngine):
        def __init__(self):
            self.device_memory_monitor = SimpleNamespace(get_peak_stats=lambda: None)
            self.state_dict_adapter = None

        def _initialize_model(self, **kwargs):
            self.model_parts = [torch.nn.Linear(2, 2, bias=False)]
            self.original = self.model_parts[0].weight

        def _initialize_model_resources(self):
            self.resident = torch.nn.Parameter(self.original.detach().clone())
            self.model_parts[0].weight = self.resident

        def _initialize_optim(self):
            self.optim = torch.optim.SGD(self.model_parts[0].parameters(), lr=0.1)

        def _initialize_checkpointer(self, **kwargs):
            self.saved_parameters = list(self.model_parts[0].parameters())

        def _initialize_forward_backward(self):
            self.ready = True

    engine = Engine()
    engine.initialize(hf_assets_path="")
    assert engine.ready
    assert engine.optim.param_groups[0]["params"][0] is engine.resident
    assert engine.saved_parameters[0] is engine.resident
    original = engine.original.detach().clone()
    engine.model_parts[0](torch.ones(1, 2)).sum().backward()
    engine.optim.step()
    assert torch.equal(engine.original, original)
    assert not torch.equal(engine.resident, original)
