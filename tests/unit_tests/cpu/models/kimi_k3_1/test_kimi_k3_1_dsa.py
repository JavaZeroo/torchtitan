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


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def _reference_sparse_attention(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    idx_q_TIE: torch.Tensor,
    idx_k_TE: torch.Tensor,
    idx_w_TI: torch.Tensor,
    *,
    topk: int,
    scale: float,
    lengths: list[int],
) -> torch.Tensor:
    outputs_THV = []
    start = 0
    for length in lengths:
        for local_t in range(length):
            token = start + local_t
            candidates = torch.arange(start, token + 1, device=q_THK.device)
            dots_IC = torch.einsum("ie,ce->ic", idx_q_TIE[token], idx_k_TE[candidates])
            scores_C = (dots_IC.relu() * idx_w_TI[token, :, None]).sum(dim=0)
            selected = candidates[scores_C.topk(min(topk, local_t + 1)).indices]
            logits_HS = (
                torch.einsum("hk,shk->hs", q_THK[token], k_THK[selected]) * scale
            )
            probs_HS = logits_HS.softmax(dim=-1)
            outputs_THV.append(torch.einsum("hs,shv->hv", probs_HS, v_THV[selected]))
        start += length
    return torch.stack(outputs_THV)


def _random_sparse_inputs(
    lengths: list[int], *, topk: int
) -> tuple[dict[str, torch.Tensor], KimiSparseAttention.Config]:
    num_tokens = sum(lengths)
    num_heads, nope_dim, shared_dim, value_dim, latent_dim = 2, 4, 2, 5, 14
    index_heads, index_dim = 3, 8

    def tensor(*shape: int, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            *shape,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    inputs = {
        "q_THK": tensor(
            num_tokens, num_heads, nope_dim + shared_dim, requires_grad=True
        ),
        "kv_c_normed_TL": tensor(num_tokens, latent_dim, requires_grad=True),
        "k_shared_TR": tensor(num_tokens, shared_dim, requires_grad=True),
        "idx_q_TIE": tensor(num_tokens, index_heads, index_dim),
        "idx_k_TE": tensor(num_tokens, index_dim).abs(),
        "idx_w_TI": tensor(num_tokens, index_heads),
    }
    inputs["idx_q_TIE"][:, 0].abs_()
    config = KimiSparseAttention.Config(
        wkv_b=_linear(latent_dim, num_heads * (nope_dim + value_dim)),
        index_topk=topk,
    )
    return inputs, config


def _build_sparse_attention(config: KimiSparseAttention.Config) -> KimiSparseAttention:
    attention = config.build()
    attention.wkv_b.double()
    with torch.no_grad():
        attention.wkv_b.weight.normal_()
    return attention


def _expanded_inputs(
    inputs: dict[str, torch.Tensor], wkv_b_weight: torch.Tensor
) -> dict[str, torch.Tensor]:
    q_THK = inputs["q_THK"]
    kv_THP = F.linear(inputs["kv_c_normed_TL"], wkv_b_weight).unflatten(
        -1, (q_THK.shape[1], -1)
    )
    k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, inputs["k_shared_TR"])
    expanded = {
        name: value
        for name, value in inputs.items()
        if name not in ("kv_c_normed_TL", "k_shared_TR")
    }
    return {**expanded, "k_THK": k_THK, "v_THV": v_THV}


_MAIN_INPUTS = ("q_THK", "kv_c_normed_TL", "k_shared_TR")
_INDEXER_INPUTS = ("idx_q_TIE", "idx_k_TE", "idx_w_TI")


def _reference_indexer_kl(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    idx_q_TIE: torch.Tensor,
    idx_k_TE: torch.Tensor,
    idx_w_TI: torch.Tensor,
    *,
    topk: int,
    scale: float,
    lengths: list[int],
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    del v_THV
    losses = []
    start = 0
    for length in lengths:
        for local_t in range(length):
            token = start + local_t
            if padding_mask is not None and padding_mask[token]:
                continue
            candidates = torch.arange(start, token + 1, device=q_THK.device)
            dots_IC = torch.einsum("ie,ce->ic", idx_q_TIE[token], idx_k_TE[candidates])
            scores_C = (dots_IC.relu() * idx_w_TI[token, :, None]).sum(dim=0)
            topk_result = scores_C.topk(min(topk, local_t + 1))
            selected = candidates[topk_result.indices]
            logits_HS = (
                torch.einsum("hk,shk->hs", q_THK[token], k_THK[selected]) * scale
            )
            teacher_S = logits_HS.softmax(dim=-1).sum(dim=0).detach()
            teacher_S = teacher_S / q_THK.shape[1]
            teacher_S = teacher_S / teacher_S.sum()
            student_log_S = topk_result.values.log_softmax(dim=-1)
            losses.append((teacher_S * (teacher_S.log() - student_log_S)).sum())
        start += length
    return torch.stack(losses).sum()


def test_sparse_attention_and_indexer_loss_match_oracle() -> None:
    torch.manual_seed(19)
    lengths = [3, 5, 2]
    padding_mask = torch.tensor([False] * 8 + [True, True])
    with patch.dict(AuxLoss._group_counts, clear=True), patch.dict(
        AuxLoss.group_acc, clear=True
    ):
        inputs, config = _random_sparse_inputs(lengths, topk=3)
        for name in _INDEXER_INPUTS:
            inputs[name].requires_grad_()
        config.aux_loss = SparseIndexerLoss.Config(
            coeff=0.01,
            reduce_mesh="loss",
            softmax_scale=0.173,
            num_heads=2,
            compress_ratio=1,
        )
        attention = _build_sparse_attention(config).train()
        positions = torch.cat([torch.arange(length) for length in lengths]).int()
        metadata = config.build_attention_metadata(
            positions, padding_mask=padding_mask, max_context_length=sum(lengths)
        )
        denominator = torch.tensor(8.0)
        actual = attention(
            **inputs,
            attention_metadata=metadata,
            scale=0.173,
            aux_loss_denominator=denominator,
            padding_mask=padding_mask,
        )
        expanded = _expanded_inputs(inputs, attention.wkv_b.weight)
        expected = _reference_sparse_attention(
            **expanded, topk=3, scale=0.173, lengths=lengths
        )
        expected_kl = _reference_indexer_kl(
            **expanded,
            topk=3,
            scale=0.173,
            lengths=lengths,
            padding_mask=padding_mask,
        )

        grad = torch.randn_like(actual)
        main_inputs = (
            *(inputs[name] for name in _MAIN_INPUTS),
            attention.wkv_b.weight,
        )
        indexer_inputs = tuple(inputs[name] for name in _INDEXER_INPUTS)
        actual_grads = torch.autograd.grad(actual, main_inputs + indexer_inputs, grad)
        expected_grads = torch.autograd.grad(
            expected, main_inputs, grad, retain_graph=True
        ) + torch.autograd.grad(expected_kl * (0.01 / denominator), indexer_inputs)

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)
        assert attention.aux_loss is not None
        torch.testing.assert_close(
            attention.aux_loss.instance_acc,
            (expected_kl.detach() / denominator).float(),
        )


def _one_layer_k3_config():
    return _kimi_k3_config(
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


def test_full_causal_dsa_model_matches_dense_k3_mla() -> None:
    torch.manual_seed(31)
    dense_config = _one_layer_k3_config()
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
