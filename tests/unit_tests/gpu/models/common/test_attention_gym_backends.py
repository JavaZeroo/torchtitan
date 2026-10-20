# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare selectable Attention Gym implementations on supported CUDA devices."""

import pytest
import torch
from attn_gym.linear import (
    causal_conv1d_decode,
    gate_transform,
    paged_causal_conv1d,
    paged_chunk_gdn,
    recurrent_gdn_decode,
)
from torchtitan.models.common.attention.gdn import GDNKernel
from torchtitan.models.common.attention.kda import KDAKernel

from torchtitan.models.common.attention_gym import causal_conv1d, l2norm
from torchtitan.models.common.gdn_reference import paged_gdn_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


@pytest.mark.parametrize("kind", ["gdn", "kda"])
def test_fused_matches_reference_outputs_and_gradients(kind):
    if kind == "kda" and torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Fused KDA test requires SM90 or newer")
    torch.manual_seed(42)
    offsets = torch.tensor([0, 17, 67], device="cuda", dtype=torch.int32)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    q, k, v = (randn(67, 1, 128) for _ in range(3))
    if kind == "gdn":
        inputs = (q, k, v, -torch.rand(67, 1, device="cuda"), randn(67, 1).sigmoid())
        kernel_cls = GDNKernel
    else:
        inputs = (
            q[None],
            k[None],
            v[None],
            randn(1, 67, 1, 128),
            randn(1, 67, 1),
            torch.zeros(1, device="cuda"),
            torch.zeros(1, 128, device="cuda"),
        )
        kernel_cls = KDAKernel
    inputs = tuple(tensor.requires_grad_() for tensor in inputs)
    fused = kernel_cls.Config().build()(*inputs, cu_seqlens=offsets)
    reference = kernel_cls.Config(impl="reference").build()(*inputs, cu_seqlens=offsets)
    torch.testing.assert_close(fused, reference, atol=2e-3, rtol=2e-2)
    gradient = torch.randn_like(fused)
    actual = torch.autograd.grad(fused, inputs, gradient)
    expected = torch.autograd.grad(reference, inputs, gradient)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_fused_convolution_matches_reference_outputs_and_gradients():
    torch.manual_seed(42)
    x = torch.randn(1, 67, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(128, 4, device="cuda", dtype=x.dtype, requires_grad=True)
    offsets = torch.tensor([0, 17, 67], device="cuda", dtype=torch.int32)
    actual = causal_conv1d(x, weight, cu_seqlens=offsets)
    expected = causal_conv1d(x, weight, cu_seqlens=offsets, impl="reference")
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    gradient = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, (x, weight), gradient),
        torch.autograd.grad(expected, (x, weight), gradient),
        atol=3e-2,
        rtol=2e-2,
    )


@torch.no_grad()
def test_paged_reference_matches_fused_outputs_and_state():
    torch.manual_seed(42)
    dtype, device = torch.bfloat16, "cuda"
    num_tokens, channels = 20, 512
    x = torch.randn(num_tokens, channels, dtype=dtype, device=device)
    a, b = (torch.randn(num_tokens, 2, device=device) for _ in range(2))
    weight = torch.randn(channels, 4, dtype=dtype, device=device) * 0.1
    A_log, dt_bias = torch.zeros(2, device=device), torch.zeros(2, device=device)
    conv = torch.randn(4, 3, channels, dtype=dtype, device=device)
    state = torch.randn(4, 2, 128, 128, device=device)
    ref_conv, ref_state = conv.clone(), state.clone()
    offsets = torch.tensor([0, 7, 17, 20], dtype=torch.int32, device=device)
    slots = torch.tensor([1, 2, 0], dtype=torch.int32, device=device)
    resumed = torch.tensor([True, False, False], device=device)
    expected = paged_gdn_reference(
        x,
        a,
        b,
        weight,
        A_log,
        dt_bias,
        ref_conv,
        ref_state,
        offsets,
        slots,
        resumed,
        num_key_heads=1,
        key_head_dim=128,
    )
    convolved = paged_causal_conv1d(
        x[None],
        weight,
        conv,
        slots,
        cu_seqlens=offsets,
        has_initial_state=resumed,
        activation="silu",
    )
    q, k, v = (
        tensor.unflatten(-1, (-1, 128))
        for tensor in convolved.split((128, 128, 256), dim=-1)
    )
    actual = paged_chunk_gdn(
        l2norm(q),
        l2norm(k),
        v,
        gate_transform(a[None], A_log, dt_bias, kind="softplus", impl="reference"),
        b[None].sigmoid(),
        state,
        slots,
        cu_seqlens=offsets,
        has_initial_state=resumed,
    )[0]
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, ref_state, atol=1e-2, rtol=2e-2)

    # Continue both caches through the fused single-token decode ABI.
    decode_offsets = torch.arange(4, dtype=torch.int32, device=device)
    resumed.fill_(True)
    for _ in range(2):
        token = torch.randn(3, channels, dtype=dtype, device=device)
        raw_a, raw_b = (torch.randn(3, 2, dtype=dtype, device=device) for _ in range(2))
        expected = paged_gdn_reference(
            token,
            raw_a,
            raw_b,
            weight,
            A_log,
            dt_bias,
            ref_conv,
            ref_state,
            decode_offsets,
            slots,
            resumed,
            num_key_heads=1,
            key_head_dim=128,
        )
        convolved = causal_conv1d_decode(
            token,
            weight,
            conv,
            state_indices=slots,
            has_initial_state=resumed,
            activation="silu",
        )
        actual = recurrent_gdn_decode(
            convolved,
            raw_a[None],
            raw_b[None],
            A_log,
            dt_bias,
            state,
            slots,
            has_initial_state=resumed,
            scale=128**-0.5,
        )[0]
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=2e-2)
        torch.testing.assert_close(conv, ref_conv, atol=0, rtol=0)
        torch.testing.assert_close(state, ref_state, atol=1e-2, rtol=2e-2)
