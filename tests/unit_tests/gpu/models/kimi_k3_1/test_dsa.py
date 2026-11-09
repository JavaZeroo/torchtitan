# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.kimi_k3_1 import build_model_config
from torchtitan.models.kimi_k3_1.dsa import KimiGatedDSAAttention


def _build_dsa_layer(*, training: bool = False) -> KimiGatedDSAAttention:
    config = build_model_config(
        "debugmodel", seq_len=16, parallelism=ParallelismConfig()
    )
    config = replace(config, vision_encoder=None)
    layer_config = config.layers[3].attention
    assert isinstance(layer_config, KimiGatedDSAAttention.Config)
    layer = layer_config.build()
    layer.init_states()
    layer = layer.to(device="cuda", dtype=torch.bfloat16)
    return layer.train() if training else layer.eval()


def _reference_dsa_forward(
    layer: KimiGatedDSAAttention,
    x_TD: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Materialize per-head MLA K/V and evaluate each sparse attention row."""
    inner = layer.inner_attention
    num_tokens = x_TD.shape[0]
    q_a = layer.wq_a(x_TD)
    q_normed = layer.q_norm(q_a)
    q_THK = F.linear(q_normed, layer.wq_b.weight).view(
        num_tokens, layer.n_heads, layer.q_head_dim
    )

    compressed_kv = layer.wkv_a(x_TD)
    kv_latent, k_shared = compressed_kv.split(
        [layer.kv_lora_rank, layer.qk_rope_head_dim], dim=-1
    )
    kv_latent = layer.kv_norm(kv_latent)
    kv_THP = F.linear(kv_latent, inner.wkv_b.weight).view(
        num_tokens,
        layer.n_heads,
        inner.packed_kv_head_dim
        if hasattr(inner, "packed_kv_head_dim")
        else inner.qk_nope_head_dim + inner.v_head_dim,
    )
    k_nope_THN, v_THV = kv_THP.split([inner.qk_nope_head_dim, inner.v_head_dim], dim=-1)
    k_THK = torch.cat(
        (k_nope_THN, k_shared.unsqueeze(1).expand(-1, layer.n_heads, -1)), dim=-1
    )

    indexer = layer.indexer
    idx_q_THD = F.linear(q_normed, indexer.wq_b.weight).view(
        num_tokens, indexer.n_heads, indexer.head_dim
    )
    idx_k_TD = F.layer_norm(
        F.linear(x_TD, indexer.wk.weight),
        (indexer.head_dim,),
        indexer.k_norm.weight,
        indexer.k_norm.bias,
        indexer.k_norm.eps,
    )
    idx_w_TH = F.linear(x_TD.float(), indexer.weights_proj.weight.float()) * (
        indexer.n_heads**-0.5 * indexer.head_dim**-0.5
    )
    if indexer.apply_hadamard:
        matrix = torch.ones((1, 1), dtype=torch.bfloat16, device=x_TD.device)
        while matrix.shape[0] < indexer.head_dim:
            matrix = torch.cat(
                (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)),
                dim=0,
            )
        idx_q_THD = F.linear(idx_q_THD, matrix) * (indexer.head_dim**-0.5)
        idx_k_TD = F.linear(idx_k_TD, matrix) * (indexer.head_dim**-0.5)

    metadata = create_varlen_metadata_for_document(positions)
    starts = metadata.cu_seq_q.cpu().tolist()
    out_THV = torch.zeros_like(v_THV)
    for token_idx in range(num_tokens):
        doc_idx = next(
            idx for idx in range(len(starts) - 1) if starts[idx + 1] > token_idx
        )
        candidates = list(range(starts[doc_idx], token_idx + 1))
        scores = []
        for key_idx in candidates:
            logits = idx_q_THD[token_idx].float() @ idx_k_TD[key_idx].float()
            score = (logits.relu() * idx_w_TH[token_idx].float()).sum()
            scores.append(score)
        scores = torch.stack(scores)
        selected = torch.topk(
            scores, min(inner.index_topk, len(candidates)), sorted=True
        ).indices
        key_indices = torch.tensor(candidates, device=x_TD.device)[selected]
        for head_idx in range(layer.n_heads):
            logits = (
                k_THK[key_indices, head_idx].float()
                @ q_THK[token_idx, head_idx].float()
            ) * inner.softmax_scale
            probs = torch.softmax(logits, dim=-1)
            out_THV[token_idx, head_idx] = probs @ v_THV[key_indices, head_idx].float()

    gate = F.linear(x_TD, layer.gate.weight)
    return F.linear(out_THV.flatten(-2) * torch.sigmoid(gate), layer.wo.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dsa_forward_matches_materialized_mla_reference() -> None:
    torch.manual_seed(42)
    layer = _build_dsa_layer()
    x_TD = torch.randn(12, layer.wq_a.in_features, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2],
        dtype=torch.int32,
        device="cuda",
    )
    with torch.no_grad():
        actual = layer(
            x_TD,
            attention_metadata=create_varlen_metadata_for_document(positions),
            positions=positions,
        )
        expected = _reference_dsa_forward(layer, x_TD, positions)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sparse_indexer_loss_updates_indexer() -> None:
    torch.manual_seed(7)
    layer = _build_dsa_layer(training=True)
    x_TD = torch.randn(12, layer.wq_a.in_features, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(12, dtype=torch.int32, device="cuda")
    output = layer(
        x_TD,
        positions=positions,
        aux_loss_denominator=torch.tensor(12.0, device="cuda"),
    )
    output.float().square().mean().backward()
    for name, parameter in layer.indexer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_model_backward_with_activation_remat() -> None:
    torch.manual_seed(7)
    config = build_model_config(
        "debugmodel", seq_len=16, parallelism=ParallelismConfig()
    )
    config = replace(config, vision_encoder=None)
    model = config.build()
    model.init_states()
    model = model.to(device="cuda", dtype=torch.bfloat16).train()
    DefaultActivationRematConfig().apply(model)

    tokens = torch.randint(0, config.vocab_size, (12,), device="cuda")
    positions = torch.arange(12, dtype=torch.int32, device="cuda")
    output = model(
        tokens,
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
        aux_loss_denominators=torch.tensor([12.0], device="cuda"),
    )
    output.float().square().mean().backward()
    for block in model.layers.values():
        if not isinstance(block.attention, KimiGatedDSAAttention):
            continue
        for name, parameter in block.attention.indexer.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name
