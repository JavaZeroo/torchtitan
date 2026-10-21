# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation selection for Attention Gym and its preprocessing operations."""

# Shape suffixes: B = batch, T = tokens, C = channels, W = convolution width.

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
import torch.nn.functional as F

from torchtitan.models.base.module import Module

AttentionGymImpl = Literal["fused", "reference"]


@dataclass(kw_only=True, slots=True)
class AttentionGymConfig(Module.Config):
    """Select the implementation of an Attention Gym component."""

    impl: AttentionGymImpl = "fused"

    def __post_init__(self) -> None:
        # Model configs must remain usable without the optional kernel package.
        if self.impl not in ("fused", "reference"):
            raise ValueError(
                f"unknown impl {self.impl!r}; expected 'fused' or 'reference'"
            )


def l2norm(
    x: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    impl: AttentionGymImpl = "fused",
    autotune: bool = True,
) -> torch.Tensor:
    """Use the upstream normalization with the selected implementation."""
    if impl == "reference":
        from torchtitan.models.common.attn_gym import l2norm_fwd_ref

        return l2norm_fwd_ref(x)

    from torchtitan.models.common.attn_gym import l2norm as fused_l2norm

    return fused_l2norm(x, cu_seqlens=cu_seqlens, autotune=autotune)


def causal_conv1d_reference(
    x_BTC: torch.Tensor,
    weight_CW: torch.Tensor,
    *,
    initial_state_BWC: torch.Tensor | None = None,
) -> torch.Tensor:
    """Depthwise causal convolution with SiLU and optional trailing input history."""
    history = weight_CW.shape[-1] - 1
    input_BCT = x_BTC.transpose(1, 2)
    if initial_state_BWC is None:
        input_BCT = F.pad(input_BCT, (history, 0))
    else:
        input_BCT = torch.cat((initial_state_BWC.transpose(1, 2), input_BCT), dim=-1)
    # Match the fused operator's FP32 accumulation and activation before casting.
    with torch.autocast(device_type=x_BTC.device.type, enabled=False):
        output_BCT = F.conv1d(
            input_BCT.float(), weight_CW.float().unsqueeze(1), groups=x_BTC.shape[-1]
        )
        return F.silu(output_BCT).transpose(1, 2).to(x_BTC.dtype)


def causal_conv1d(
    x_BTC: torch.Tensor,
    weight_CW: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    impl: AttentionGymImpl = "fused",
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SiLU short convolution, resetting history at document boundaries."""
    if impl == "fused":
        from torchtitan.models.common.attn_gym import (
            causal_conv1d as fused_causal_conv1d,
        )

        output_BTC = fused_causal_conv1d(
            x_BTC,
            weight_CW,
            activation="silu",
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
        )
        assert isinstance(output_BTC, torch.Tensor)
        return output_BTC

    if cu_seqlens is None:
        return causal_conv1d_reference(
            x_BTC, weight_CW, initial_state_BWC=initial_state
        )
    if x_BTC.shape[0] != 1:
        raise ValueError("Packed convolution requires batch size 1.")
    offsets = cu_seqlens.tolist()
    if (
        offsets[0] != 0
        or any(begin > end for begin, end in pairwise(offsets))
        or offsets[-1] > x_BTC.shape[1]
    ):
        raise ValueError(
            "cu_seqlens must start at zero, be nondecreasing, and end within "
            "the token capacity."
        )
    output_BTC = torch.zeros_like(x_BTC)
    for sequence, (begin, end) in enumerate(pairwise(offsets)):
        if begin != end:
            output_BTC[:, begin:end] = causal_conv1d_reference(
                x_BTC[:, begin:end],
                weight_CW,
                initial_state_BWC=None
                if initial_state is None
                else initial_state[sequence : sequence + 1],
            )
    return output_BTC
