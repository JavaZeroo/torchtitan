# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager reference execution for the GDN paged-cache contract."""

# Shape suffixes:
# T = packed tokens, C = projection channels, W = convolution width,
# H = attention heads, K = key head dimension, V = value head dimension,
# S = cache slots.

from itertools import pairwise

import torch

from torchtitan.models.common.attention_gym import causal_conv1d_reference, l2norm

from torchtitan.models.common.attn_gym import gate_transform, recurrent_gdn


@torch.compiler.disable
@torch.no_grad()
def paged_gdn_reference(
    mixed_qkv_TC: torch.Tensor,
    a_TH: torch.Tensor,
    b_TH: torch.Tensor,
    weight_CW: torch.Tensor,
    A_log_H: torch.Tensor,
    dt_bias_H: torch.Tensor,
    conv_cache_SWC: torch.Tensor,
    state_cache_SHVK: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    *,
    num_key_heads: int,
    key_head_dim: int,
) -> torch.Tensor:
    """Advance paged convolution and recurrent state using upstream references.

    Inputs obey the paged metadata contract: positive slots are unique and valid;
    non-positive slots are padding. Empty fresh sequences clear their state.
    Both prefill and single-token decode use this same eager implementation.
    """
    num_tokens = mixed_qkv_TC.shape[0]
    num_value_heads, value_head_dim = state_cache_SHVK.shape[1:3]
    key_dim = num_key_heads * key_head_dim
    output_THV = mixed_qkv_TC.new_zeros(num_tokens, num_value_heads, value_head_dim)
    offsets = cu_seqlens.tolist()
    slots = state_indices.tolist()
    resumed = has_initial_state.tolist()

    for (begin, end), slot, has_state in zip(
        pairwise(offsets), slots, resumed, strict=True
    ):
        if slot <= 0:
            continue
        if not has_state:
            conv_cache_SWC[slot].zero_()
            state_cache_SHVK[slot].zero_()
        if begin == end:
            continue

        input_1TC = mixed_qkv_TC[None, begin:end]
        history_1WC = conv_cache_SWC[slot : slot + 1]
        convolved_1TC = causal_conv1d_reference(
            input_1TC, weight_CW, initial_state_BWC=history_1WC
        )
        num_history_tokens = conv_cache_SWC.shape[1]
        conv_cache_SWC[slot].copy_(
            torch.cat((history_1WC, input_1TC), dim=1)[0, -num_history_tokens:]
        )
        query_1TC, key_1TC, value_1TC = convolved_1TC.split(
            (key_dim, key_dim, num_value_heads * value_head_dim), dim=-1
        )
        query_1THK = query_1TC.unflatten(-1, (num_key_heads, key_head_dim))
        key_1THK = key_1TC.unflatten(-1, (num_key_heads, key_head_dim))
        value_1THV = value_1TC.unflatten(-1, (num_value_heads, value_head_dim))
        decay_1TH = gate_transform(
            a_TH[None, begin:end],
            A_log_H.float(),
            dt_bias_H.float(),
            kind="softplus",
            impl="reference",
        )
        output_1THV, final_state_1HVK = recurrent_gdn(
            l2norm(query_1THK, impl="reference"),
            l2norm(key_1THK, impl="reference"),
            value_1THV,
            decay_1TH,
            b_TH[None, begin:end].sigmoid(),
            state_cache_SHVK[slot : slot + 1],
            scale=key_head_dim**-0.5,
            output_final_state=True,
            impl="reference",
        )
        assert final_state_1HVK is not None
        state_cache_SHVK[slot].copy_(final_state_1HVK[0])
        output_THV[begin:end] = output_1THV[0]
    return output_THV
