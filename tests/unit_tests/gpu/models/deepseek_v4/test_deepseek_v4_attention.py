# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks the DeepSeek-V4 attention cores are wired to ``gather_attn``
correctly (layouts, window, compressed positions, sink, scale) against dense
masked attention, on CPU (reference impl) and CUDA (fused impl). Kernel
numerics are tested in Attention Gym.
"""

import pytest
import torch
from torchtitan.models.common.rope import ComplexRoPE

from torchtitan.models.deepseek_v4.attention import (
    CompressedSparseAttention,
    DeepSeekV4Attention,
    DSV4InnerAttention,
    HeavilyCompressedAttention,
    SlidingWindowAttention,
)
from torchtitan.models.deepseek_v4.compressor import Indexer
from torchtitan.models.deepseek_v4.flavors import _make_v4_attn_config

SEQLEN, N_HEADS, HEAD_DIM, WINDOW, SCALE, TOPK = 32, 4, 8, 6, 0.1234, 5


def dense_attention(q, swa_k, cmp_k, cmp_allowed, attn_sink):
    """Softmax over [swa_k; cmp_k; sink] with a causal window over swa_k and
    ``cmp_allowed`` [T, S] over cmp_k."""
    t = torch.arange(SEQLEN).unsqueeze(1)
    window = (torch.arange(SEQLEN) <= t) & (t - torch.arange(SEQLEN) < WINDOW)
    sink = torch.ones(SEQLEN, 1, dtype=torch.bool)
    allowed = torch.cat([window, cmp_allowed, sink], dim=-1)

    kv = torch.cat([swa_k, cmp_k, swa_k.new_zeros(1, HEAD_DIM)])
    scores_HTS = torch.einsum("thd,sd->hts", q, kv) * SCALE
    scores_HTS[:, :, -1] = attn_sink.unsqueeze(-1)
    scores_HTS = scores_HTS.masked_fill(~allowed, float("-inf"))
    return torch.einsum("hts,sd->thd", scores_HTS.softmax(dim=-1), kv)


def causal_cmp_allowed(ratio):
    """Compressed block s summarizes tokens [s * ratio, (s + 1) * ratio)."""
    t = torch.arange(SEQLEN).unsqueeze(1)
    block_end = (torch.arange(SEQLEN // ratio) + 1) * ratio - 1
    return block_end <= t


# HCA runs at ratio 8 instead of 128 so SEQLEN 32 has compressed blocks; the
# all-causal-blocks pattern is the same.
@pytest.mark.parametrize(
    "cls, ratio",
    [
        (SlidingWindowAttention, 1),
        (HeavilyCompressedAttention, 8),
        (CompressedSparseAttention, 4),
    ],
    ids=["swa", "hca", "csa"],
)
@pytest.mark.parametrize(
    "device, dtype, tol",
    [
        pytest.param("cpu", torch.float64, {}, id="cpu"),
        pytest.param(
            "cuda",
            torch.float32,
            {"atol": 1e-3, "rtol": 1e-2},
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_matches_dense_masked_attention(cls, ratio, device, dtype, tol):
    torch.manual_seed(0)
    module = cls(
        cls.Config(
            window_size=WINDOW,
            compress_ratio=ratio,
            softmax_scale=SCALE,
            index_topk=TOPK,
        )
    )
    n_cmp = 0 if ratio == 1 else SEQLEN // ratio
    # fp64 CPU copies feed the oracle; the module runs on `device` in `dtype`.
    ref_inputs = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in (
            (SEQLEN, N_HEADS, HEAD_DIM),
            (SEQLEN, HEAD_DIM),
            (n_cmp, HEAD_DIM),
            (N_HEADS,),
        )
    ]
    if ratio == 1:
        ref_inputs[2].requires_grad_(False)
    inputs = [
        t.detach().to(device, dtype).requires_grad_(t.requires_grad) for t in ref_inputs
    ]
    q, swa_k, cmp_k, attn_sink = inputs

    match module:
        case SlidingWindowAttention():
            out = module(q, swa_k, attn_sink)
            cmp_allowed = torch.zeros(SEQLEN, 0, dtype=torch.bool)
        case HeavilyCompressedAttention():
            out = module(q, swa_k, cmp_k, attn_sink)
            cmp_allowed = causal_cmp_allowed(ratio)
        case CompressedSparseAttention():
            idx_q, idx_k, idx_w = (
                torch.randn(*shape)
                for shape in ((SEQLEN, 3, 6), (n_cmp, 6), (SEQLEN, 3))
            )
            idx = [t.to(device) for t in (idx_q, idx_k, idx_w)]
            out = module(q, swa_k, cmp_k, *idx, attn_sink)
            cmp_topk, _ = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                max_seqlen=SEQLEN,
                ratio=ratio,
                topk=TOPK,
                return_scores=False,
            )
            cmp_topk = cmp_topk.long()
            cmp_allowed = torch.zeros(SEQLEN, n_cmp, dtype=torch.bool)
            rows = torch.arange(SEQLEN).unsqueeze(1).expand_as(cmp_topk)
            cmp_allowed[rows[cmp_topk >= 0], cmp_topk[cmp_topk >= 0]] = True
            cmp_allowed &= causal_cmp_allowed(ratio)

    ref = dense_attention(*ref_inputs[:3], cmp_allowed, ref_inputs[3])
    diff_inputs = [t for t in inputs if t.requires_grad]
    diff_ref_inputs = [t for t in ref_inputs if t.requires_grad]
    grads = torch.autograd.grad(out.sum(), diff_inputs)
    ref_grads = torch.autograd.grad(ref.sum(), diff_ref_inputs)

    torch.testing.assert_close(out.cpu().double(), ref, **tol)
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        torch.testing.assert_close(grad.cpu().double(), ref_grad, **tol)


def test_query_rescale_can_be_overridden_without_replacing_projections():
    """V4 removes query magnitude; a variant can preserve it with one override."""

    class UnnormalizedQueryAttention(DeepSeekV4Attention):
        def _normalize_query(self, q):
            return q

    cfg = _make_v4_attn_config(
        layer_id=0,
        dim=16,
        n_heads=4,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=4,
        n_groups=2,
        compress_ratio=1,
        window_size=6,
        norm_eps=0.0,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        n_layers=1,
        rope=ComplexRoPE.Config(dim=4, max_context_length=16),
        enable_sp=False,
    )
    torch.manual_seed(0)
    base = cfg.build()
    base.init_states()
    variant = UnnormalizedQueryAttention(cfg)
    variant.load_state_dict(base.state_dict(), strict=True)
    x = torch.randn(8, 16)
    positions = torch.arange(8)
    for module, scale in ((base, 1), (variant, 2)):
        qr, q, kv = module._project_qkv(x, positions)
        with torch.no_grad():
            module.wq_b.weight.mul_(2)
        scaled_qr, scaled_q, scaled_kv = module._project_qkv(x, positions)
        torch.testing.assert_close(scaled_qr, qr, rtol=0, atol=0)
        torch.testing.assert_close(scaled_kv, kv, rtol=0, atol=0)
        torch.testing.assert_close(scaled_q, q * scale)


def test_ratio_one_compressed_pool_preserves_packed_document_boundaries():
    """A non-empty ratio-1 pool can reuse the core without mixing documents."""

    class RatioOnePoolAttention(DSV4InnerAttention):
        def _compressed_cu_seqlens(self, cu_seqlens):
            return cu_seqlens

    module = RatioOnePoolAttention(
        DSV4InnerAttention.Config(
            window_size=2, compress_ratio=1, softmax_scale=0.5, index_topk=1
        )
    )
    torch.manual_seed(0)
    inputs = [
        torch.randn(*shape, dtype=torch.float64, requires_grad=True)
        for shape in ((8, 2, 8), (8, 8), (8, 8), (2,))
    ]
    q, swa_k, cmp_k, sink = inputs
    # Select the first compressed entry of each query's document.
    topk = torch.zeros(8, 1, dtype=torch.int32)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    packed = module._gather_attn(q, swa_k, cmp_k, topk, sink, cu)
    separate = torch.cat(
        [
            module._gather_attn(
                q[start:end],
                swa_k[start:end],
                cmp_k[start:end],
                topk[start:end],
                sink,
                None,
            )
            for start, end in ((0, 3), (3, 8))
        ]
    )
    torch.testing.assert_close(packed, separate)
    packed_grads = torch.autograd.grad(packed.square().sum(), inputs)
    separate_grads = torch.autograd.grad(separate.square().sum(), inputs)
    for actual, expected in zip(packed_grads, separate_grads, strict=True):
        torch.testing.assert_close(actual, expected)
