# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from unittest.mock import patch

import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import materialize_mla_kv
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3.flavors import _kimi_k3_config, _vision_encoder_config
from torchtitan.models.kimi_k3_1.dsa import KimiSparseAttention
from torchtitan.models.kimi_k3_1.flavors import _with_dsa


def _reference(
    q_THK,
    k_THK,
    v_THV,
    idx_q_TIE,
    idx_k_TE,
    idx_w_TI,
    *,
    topk,
    scale,
    lengths,
    padding_mask
):
    """Per-token sparse attention outputs and the summed indexer KL."""
    outputs_THV, losses = [], []
    start = 0
    for length in lengths:
        for local_t in range(length):
            token = start + local_t
            candidates = torch.arange(start, token + 1)
            dots_IC = torch.einsum("ie,ce->ic", idx_q_TIE[token], idx_k_TE[candidates])
            scores_C = (dots_IC.relu() * idx_w_TI[token, :, None]).sum(dim=0)
            scores_S, slots_S = scores_C.topk(min(topk, local_t + 1))
            selected = candidates[slots_S]
            logits_HS = (
                torch.einsum("hk,shk->hs", q_THK[token], k_THK[selected]) * scale
            )
            probs_HS = logits_HS.softmax(dim=-1)
            outputs_THV.append(torch.einsum("hs,shv->hv", probs_HS, v_THV[selected]))
            if not padding_mask[token]:
                teacher_S = probs_HS.detach().sum(dim=0)
                teacher_S = teacher_S / teacher_S.sum()
                student_log_S = scores_S.log_softmax(dim=-1)
                losses.append((teacher_S * (teacher_S.log() - student_log_S)).sum())
        start += length
    return torch.stack(outputs_THV), torch.stack(losses).sum()


def test_sparse_attention_and_indexer_loss_match_oracle() -> None:
    torch.manual_seed(19)
    lengths = [3, 5, 2]
    padding_mask = torch.tensor([False] * 8 + [True, True])
    q_THK, kv_c_normed_TL, k_shared_TR, idx_q_TIE, idx_k_TE, idx_w_TI = (
        torch.randn(*shape, dtype=torch.float64)
        for shape in ((10, 2, 6), (10, 14), (10, 2), (10, 3, 8), (10, 8), (10, 3))
    )
    idx_k_TE.abs_()
    idx_q_TIE[:, 0].abs_()
    main = (q_THK, kv_c_normed_TL, k_shared_TR)
    indexer = (idx_q_TIE, idx_k_TE, idx_w_TI)
    for tensor in main + indexer:
        tensor.requires_grad_()
    with patch.dict(AuxLoss._group_counts, clear=True), patch.dict(
        AuxLoss.group_acc, clear=True
    ):
        config = KimiSparseAttention.Config(
            wkv_b=Linear.Config(in_features=14, out_features=2 * (4 + 5)),
            index_topk=3,
            aux_loss=SparseIndexerLoss.Config(
                coeff=0.01,
                reduce_mesh="loss",
                softmax_scale=0.173,
                num_heads=2,
                compress_ratio=1,
            ),
        )
        attention = config.build().train()
        attention.wkv_b.double()
        with torch.no_grad():
            attention.wkv_b.weight.normal_()
        positions = torch.cat([torch.arange(length) for length in lengths]).int()
        actual = attention(
            *main,
            idx_q_TIE=idx_q_TIE,
            idx_k_TE=idx_k_TE,
            idx_w_TI=idx_w_TI,
            attention_metadata=config.build_attention_metadata(
                positions, padding_mask=padding_mask, max_context_length=10
            ),
            scale=0.173,
            aux_loss_denominator=torch.tensor(8.0),
            padding_mask=padding_mask,
        )
        kv_THP = F.linear(kv_c_normed_TL, attention.wkv_b.weight).unflatten(-1, (2, -1))
        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        expected, expected_kl = _reference(
            q_THK,
            k_THK,
            v_THV,
            *indexer,
            topk=3,
            scale=0.173,
            lengths=lengths,
            padding_mask=padding_mask,
        )

        grad = torch.randn_like(actual)
        main += (attention.wkv_b.weight,)
        actual_grads = torch.autograd.grad(actual, main + indexer, grad)
        expected_grads = torch.autograd.grad(
            expected, main, grad, retain_graph=True
        ) + torch.autograd.grad(expected_kl * (0.01 / 8.0), indexer)

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)
        assert attention.aux_loss is not None
        torch.testing.assert_close(
            attention.aux_loss.instance_acc, (expected_kl.detach() / 8.0).float()
        )


def test_full_causal_dsa_model_matches_dense_k3_mla() -> None:
    torch.manual_seed(31)
    dense_config = _kimi_k3_config(
        max_context_length=8,
        dim=8,
        vocab_size=16,
        num_layers=1,
        full_attention_layers={0},
        attn_res_block_size=1,
        num_heads=2,
        q_lora_rank=4,
        kv_lora_rank=5,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=3,
        kda_head_dim=4,
        conv_kernel_size=2,
        dense_hidden_dim=12,
        latent_dim=4,
        expert_hidden_dim=4,
        num_experts=2,
        top_k=1,
        num_shared_experts=1,
        vision_encoder=_vision_encoder_config(
            text_dim=8,
            dim=8,
            qkv_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            init_pos_emb_height=2,
            init_pos_emb_width=2,
        ),
        attn_backend="flex",
        parallelism=ParallelismConfig(),
    )
    dsa_config = _with_dsa(
        dense_config,
        index_topk=16,
        index_num_heads=3,
        index_head_dim=4,
        indexer_loss_coeff=0.01,
    )
    with torch.device("meta"):
        dense_model = dense_config.build()
        dsa_model = dsa_config.build()
    for model in (dense_model, dsa_model):
        model.to(dtype=torch.float32)
        model.to_empty(device="cpu")
        model.init_states(buffer_device=torch.device("cpu"))
        model.eval()
    load_result = dsa_model.load_state_dict(dense_model.state_dict(), strict=False)
    assert all(".indexer." in key for key in load_result.missing_keys)
    assert not load_result.unexpected_keys

    tokens = torch.arange(8) % dense_config.vocab_size
    positions = torch.arange(8, dtype=torch.int32)
    with torch.no_grad():
        dense = dense_model(
            tokens,
            positions=positions,
            attention_metadata=dense_model._get_attention_metadata(positions),
        )
        dsa = dsa_model(
            tokens,
            positions=positions,
            attention_metadata=dsa_model._get_attention_metadata(positions),
        )
    torch.testing.assert_close(dsa, dense, atol=1e-6, rtol=1e-6)
