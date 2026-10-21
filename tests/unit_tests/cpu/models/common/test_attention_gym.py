# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the real reference path without replacing kernels with mocks."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("attn_gym")

from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.config.validation import validate_model_training_config
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    LinearAttentionMetadata,
)
from torchtitan.models.common.attention_gym import (
    AttentionGymConfig,
    causal_conv1d,
    l2norm,
)
from torchtitan.models.common.gdn_reference import paged_gdn_reference
from torchtitan.models.kimi_k3.flavors import _kda_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.models.qwen3_5.flavors import _qwen35_deltanet_config


def _config(kind, *, impl="fused"):
    if kind == "gdn":
        config = _qwen35_deltanet_config(
            dim=16,
            n_key_heads=1,
            n_value_heads=2,
            key_head_dim=8,
            value_head_dim=8,
            layer_id=0,
            enable_sp=False,
        )
    else:
        config = _kda_config(
            dim=16, num_heads=1, head_dim=128, conv_kernel_size=4, enable_sp=False
        )
    for _, kernel, _, _ in config.traverse(AttentionGymConfig):
        kernel.impl = impl
    return config


@pytest.mark.parametrize("kind", ["gdn", "kda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_documents_gradients_and_checkpoints(kind, dtype):
    torch.manual_seed(42)
    config = _config(kind)
    fused = config.build().to(dtype)
    fused.init_states()
    model = _config(kind, impl="reference").build().to(dtype)
    model.load_state_dict(fused.state_dict(), strict=True)
    assert model.state_dict().keys() == fused.state_dict().keys()

    lengths = (3, 5, 2)
    x = torch.randn(sum(lengths), 16, dtype=dtype, requires_grad=True)
    positions = torch.tensor([i for length in lengths for i in range(length)])
    metadata = LinearAttentionMetadata(
        varlen=create_varlen_metadata_for_document(positions),
        num_conv_history_tokens=config.conv_kernel_size - 1,
    )
    packed = model(x, metadata)
    independent = torch.cat([model(document) for document in x.split(lengths)])
    tolerance = 3e-3 if dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(packed, independent, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(packed)
    inputs = (x, *model.parameters())
    packed_grads = torch.autograd.grad(packed, inputs, grad)
    independent_grads = torch.autograd.grad(independent, inputs, grad)
    torch.testing.assert_close(
        packed_grads, independent_grads, atol=tolerance, rtol=tolerance
    )
    assert all(torch.isfinite(gradient).all() for gradient in packed_grads)


def test_invalid_selection():
    with pytest.raises(ValueError, match="unknown impl"):
        AttentionGymConfig(impl="invalid")


@pytest.mark.parametrize("impl", ["fused", "reference"])
@pytest.mark.parametrize(
    "batch_invariant,regions,disable_cuda_graphs,graphs_supported,error",
    [
        (False, None, True, True, None),
        (True, None, True, True, "debug.batch_invariant=False"),
        (False, ["partial_rope", "gated_rmsnorm"], True, True, None),
        (False, None, False, True, "training.disable_cuda_graphs=True"),
        (False, None, False, False, None),
    ],
)
def test_training_config_compatibility(
    monkeypatch,
    impl,
    batch_invariant,
    regions,
    disable_cuda_graphs,
    graphs_supported,
    error,
):
    monkeypatch.setattr(
        "torchtitan.distributed.cuda_graph.cuda_graphs_supported",
        lambda: graphs_supported,
    )
    expectation = (
        pytest.raises(ValueError, match=error)
        if impl == "reference" and error
        else nullcontext()
    )
    config = build_model_config(
        "debugmodel", seq_len=16, attn_backend="flex", parallelism=ParallelismConfig()
    )
    config.local_compile_regions = regions or []
    for _, kernel, _, _ in config.traverse(AttentionGymConfig):
        kernel.impl = impl
    with expectation:
        validate_model_training_config(
            config,
            parallelism=ParallelismConfig(),
            training=TrainingConfig(
                disable_cuda_graphs=disable_cuda_graphs, max_context_length=16
            ),
            debug=DebugConfig(batch_invariant=batch_invariant),
            activation_remat=None,
            max_num_documents=None,
        )


def test_reference_recipe_validates_before_build(monkeypatch):
    from torchtitan_recipes.tests.models.qwen3_5 import qwen35_debugmodel

    monkeypatch.setattr(
        "torchtitan.distributed.cuda_graph.cuda_graphs_supported", lambda: True
    )
    config = qwen35_debugmodel(attn_backend="flex")
    for _, kernel, _, _ in config.model.traverse(AttentionGymConfig):
        kernel.impl = "reference"
    with pytest.raises(ValueError, match="layers.0.*training.disable_cuda_graphs=True"):
        config.__post_init__()

    config.training.disable_cuda_graphs = True
    config.__post_init__()


def test_reference_rejects_context_parallelism():
    parallelism = ParallelismConfig(context_parallel_degree=2)
    config = build_model_config(
        "debugmodel", seq_len=16, attn_backend="flex", parallelism=parallelism
    )
    for _, kernel, _, _ in config.traverse(AttentionGymConfig):
        kernel.impl = "reference"
    with pytest.raises(ValueError, match="context_parallel_degree=1"):
        validate_model_training_config(
            config,
            parallelism=parallelism,
            training=TrainingConfig(disable_cuda_graphs=True, max_context_length=16),
            debug=DebugConfig(),
            activation_remat=None,
            max_num_documents=None,
        )


def test_reference_convolution_padding_and_empty_documents():
    torch.manual_seed(42)
    x = torch.randn(1, 8, 4, requires_grad=True)
    weight = torch.randn(4, 3, requires_grad=True)
    offsets = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    actual = causal_conv1d(x, weight, cu_seqlens=offsets, impl="reference")
    expected = torch.cat(
        [
            F.silu(
                F.conv1d(
                    F.pad(x[:, start:end].transpose(1, 2), (2, 0)),
                    weight[:, None],
                    groups=4,
                )
            ).transpose(1, 2)
            for start, end in [(0, 2), (2, 5)]
        ]
        + [torch.zeros_like(x[:, 5:])],
        dim=1,
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert not x.grad[:, 5:].any()
    assert torch.isfinite(weight.grad).all()


def test_reference_l2norm_uses_additive_epsilon_and_preserves_dtype():
    x = torch.tensor([[0.0, 1e-4, -1e-4]], dtype=torch.bfloat16, requires_grad=True)
    actual = l2norm(x, impl="reference")
    expected = (
        x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
    ).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_paged_reference_prefill_decode_padding_and_fresh_slots():
    torch.manual_seed(42)
    num_key_heads, num_value_heads, key_dim, value_dim = 1, 2, 4, 3
    channels = 2 * num_key_heads * key_dim + num_value_heads * value_dim
    x = torch.randn(5, channels)
    a, b = torch.randn(5, num_value_heads), torch.randn(5, num_value_heads)
    weight = torch.randn(channels, 3)
    A_log, dt_bias = torch.zeros(num_value_heads), torch.zeros(num_value_heads)
    conv = torch.randn(4, 2, channels)
    state = torch.randn(4, num_value_heads, value_dim, key_dim)
    original_conv, original_state = conv.clone(), state.clone()

    def run(x, a, b, conv, state, offsets, slots, resumed):
        return paged_gdn_reference(
            x,
            a,
            b,
            weight,
            A_log,
            dt_bias,
            conv,
            state,
            torch.tensor(offsets, dtype=torch.int32),
            torch.tensor(slots, dtype=torch.int32),
            torch.tensor(resumed),
            num_key_heads=num_key_heads,
            key_head_dim=key_dim,
        )

    whole_conv, whole_state = conv.clone(), state.clone()
    whole = run(x, a, b, whole_conv, whole_state, [0, 5], [1], [False])
    first = run(x[:3], a[:3], b[:3], conv, state, [0, 3], [1], [False])
    last = []
    for index in (3, 4):
        last.append(
            run(
                x[index : index + 1],
                a[index : index + 1],
                b[index : index + 1],
                conv,
                state,
                [0, 1],
                [1],
                [True],
            )
        )
    torch.testing.assert_close(torch.cat([first, *last]), whole)
    torch.testing.assert_close(conv, whole_conv)
    torch.testing.assert_close(state, whole_state)
    torch.testing.assert_close(conv[[0, 2, 3]], original_conv[[0, 2, 3]])
    torch.testing.assert_close(state[[0, 2, 3]], original_state[[0, 2, 3]])

    before_conv, before_state = conv.clone(), state.clone()
    padding = run(
        x,
        a,
        b,
        conv,
        state,
        [0, 1, 2, 2, 2],
        [0, -1, 2, 3],
        [False, False, False, True],
    )
    assert not padding.any()
    assert not conv[2].any() and not state[2].any()
    torch.testing.assert_close(conv[[0, 1, 3]], before_conv[[0, 1, 3]])
    torch.testing.assert_close(state[[0, 1, 3]], before_state[[0, 1, 3]])

    # A reused slot must forget both previous convolution and recurrent state.
    restarted = run(x, a, b, conv, state, [0, 5], [1], [False])
    torch.testing.assert_close(restarted, whole)
