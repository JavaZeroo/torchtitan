# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models import kimi_k3, kimi_k3_1
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.models.kimi_k3.flavors import _kimi_k3_config, _mla_config
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3_1.dsa import (
    KimiDSAAttention,
    KimiDSAIndexer,
    KimiIndexerLoss,
    KimiSparseAttention,
)
from torchtitan.models.kimi_k3_1.flavors import _dsa_config
from torchtitan.models.kimi_k3_1.model import KimiK31Model


def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(in_features=in_features, out_features=out_features)


def test_dsa_flavor_replaces_exactly_the_k3_full_attention_layers() -> None:
    parallelism = ParallelismConfig()
    dense_config = kimi_k3.build_model_config("debugmodel", parallelism=parallelism)
    dsa_config = kimi_k3_1.build_model_config("debugmodel", parallelism=parallelism)
    assert isinstance(dsa_config, KimiK31Model.Config)

    dense_layers = {
        layer_id
        for layer_id, layer in enumerate(dense_config.layers)
        if isinstance(layer.attention, KimiMLAAttention.Config)
    }
    dsa_layers = {
        layer_id
        for layer_id, layer in enumerate(dsa_config.layers)
        if isinstance(layer.attention, KimiDSAAttention.Config)
    }
    assert dsa_layers == dense_layers
    assert all(
        isinstance(layer.attention, KimiDSAAttention.Config)
        == (layer_id in dense_layers)
        for layer_id, layer in enumerate(dsa_config.layers)
    )

    released_config = kimi_k3_1.build_model_config(
        "Kimi-K3.1-Pro", parallelism=parallelism
    )
    released_dsa = [
        layer.attention
        for layer in released_config.layers
        if isinstance(layer.attention, KimiDSAAttention.Config)
    ]
    assert len(released_dsa) == 24
    assert released_config.max_context_length == 1_048_576
    for attention in released_dsa:
        assert attention.indexer.index_num_heads == 32
        assert attention.indexer.index_head_dim == 128
        assert attention.inner_attention.inner_attention.index_topk == 2048


def test_indexer_projection_matches_reference_math() -> None:
    torch.manual_seed(42)
    num_tokens, dim, q_lora_rank = 5, 6, 4
    index_num_heads, index_head_dim = 3, 8
    config = KimiDSAIndexer.Config(
        index_num_heads=index_num_heads,
        index_head_dim=index_head_dim,
        wq_b=_linear(q_lora_rank, index_num_heads * index_head_dim),
        wk=_linear(dim, index_head_dim),
        k_norm=LayerNorm.Config(
            normalized_shape=index_head_dim,
            eps=1e-5,
        ),
        weights_proj=_linear(dim, index_num_heads),
    )
    indexer = config.build()
    q_TC = torch.randn(num_tokens, q_lora_rank)
    x_TD = torch.randn(num_tokens, dim)

    idx_q_TIE, idx_k_TE, idx_w_TI = indexer(q_TC, x_TD)

    expected_q_TIE = F.linear(q_TC, indexer.wq_b.weight).view(
        num_tokens, index_num_heads, index_head_dim
    )
    expected_k_TE = F.layer_norm(
        F.linear(x_TD, indexer.wk.weight).float(),
        (index_head_dim,),
        indexer.k_norm.weight.float(),
        indexer.k_norm.bias.float(),
        indexer.k_norm.eps,
    )
    scale = index_head_dim**-0.5 * index_num_heads**-0.5
    expected_w_TI = F.linear(x_TD, indexer.weights_proj.weight) * scale
    torch.testing.assert_close(idx_q_TIE, expected_q_TIE)
    torch.testing.assert_close(idx_k_TE, expected_k_TE)
    torch.testing.assert_close(idx_w_TI, expected_w_TI)
    assert idx_k_TE.dtype == x_TD.dtype


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
    num_heads, head_dim, value_dim = 2, 6, 5
    index_heads, index_dim = 3, 8

    def tensor(*shape: int, requires_grad: bool = False) -> torch.Tensor:
        return torch.randn(
            *shape,
            dtype=torch.float64,
            requires_grad=requires_grad,
        )

    inputs = {
        "q_THK": tensor(num_tokens, num_heads, head_dim, requires_grad=True),
        "k_THK": tensor(num_tokens, num_heads, head_dim, requires_grad=True),
        "v_THV": tensor(num_tokens, num_heads, value_dim, requires_grad=True),
        "idx_q_TIE": tensor(num_tokens, index_heads, index_dim),
        "idx_k_TE": tensor(num_tokens, index_dim),
        "idx_w_TI": tensor(num_tokens, index_heads),
    }
    return inputs, KimiSparseAttention.Config(index_topk=topk)


def test_sparse_attention_output_and_gradients_match_independent_oracle() -> None:
    torch.manual_seed(7)
    for topk in (3, 16):
        inputs, config = _random_sparse_inputs([8], topk=topk)
        attention = config.build().train()
        actual = attention(**inputs, scale=0.173)
        expected = _reference_sparse_attention(
            **inputs, topk=topk, scale=0.173, lengths=[8]
        )
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, (inputs["q_THK"], inputs["k_THK"], inputs["v_THV"]), grad
        )
        expected_grads = torch.autograd.grad(
            expected, (inputs["q_THK"], inputs["k_THK"], inputs["v_THV"]), grad
        )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_grads, expected_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)


def test_packed_documents_match_isolated_forward_and_backward() -> None:
    torch.manual_seed(11)
    lengths = [3, 5, 2]
    inputs, config = _random_sparse_inputs(lengths, topk=3)
    attention = config.build().train()
    positions = torch.cat([torch.arange(length) for length in lengths]).int()
    padding_mask = torch.tensor([False, False, False] + [False] * 5 + [True, True])
    metadata = config.build_attention_metadata(
        positions, padding_mask=padding_mask, max_context_length=sum(lengths)
    )

    packed = attention(
        **inputs,
        attention_metadata=metadata,
        padding_mask=padding_mask,
        scale=0.173,
    )
    grad = torch.randn_like(packed)
    packed_grads = torch.autograd.grad(
        packed, (inputs["q_THK"], inputs["k_THK"], inputs["v_THV"]), grad
    )

    isolated = []
    isolated_grads = [
        torch.zeros_like(inputs[name]) for name in ("q_THK", "k_THK", "v_THV")
    ]
    start = 0
    for length in lengths:
        doc_inputs = {
            name: value[start : start + length].detach().clone().requires_grad_()
            for name, value in inputs.items()
        }
        doc = attention(**doc_inputs, scale=0.173)
        isolated.append(doc)
        doc_grads = torch.autograd.grad(
            doc,
            (doc_inputs["q_THK"], doc_inputs["k_THK"], doc_inputs["v_THV"]),
            grad[start : start + length],
        )
        for accumulated, doc_grad in zip(isolated_grads, doc_grads, strict=True):
            accumulated[start : start + length] = doc_grad
        start += length

    torch.testing.assert_close(packed, torch.cat(isolated))
    for packed_grad, isolated_grad in zip(packed_grads, isolated_grads, strict=True):
        torch.testing.assert_close(packed_grad, isolated_grad)


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


def test_indexer_loss_matches_independent_kl_and_preserves_main_gradients() -> None:
    torch.manual_seed(19)
    with patch.dict(AuxLoss._group_counts, clear=True), patch.dict(
        AuxLoss.group_acc, clear=True
    ):
        inputs, config = _random_sparse_inputs([8], topk=3)
        for name in ("idx_q_TIE", "idx_k_TE", "idx_w_TI"):
            inputs[name].requires_grad_()
        config.aux_loss = KimiIndexerLoss.Config(
            coeff=0.01, num_heads=2, reduce_mesh="loss"
        )
        attention = config.build().train()
        denominator = torch.tensor(8.0)
        actual = attention(**inputs, scale=0.173, aux_loss_denominator=denominator)
        expected = _reference_sparse_attention(
            **inputs, topk=3, scale=0.173, lengths=[8]
        )
        expected_kl = _reference_indexer_kl(**inputs, topk=3, scale=0.173, lengths=[8])

        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            actual, tuple(inputs.values()), grad, allow_unused=True
        )
        expected_main_grads = torch.autograd.grad(
            expected,
            (inputs["q_THK"], inputs["k_THK"], inputs["v_THV"]),
            grad,
        )
        expected_idx_grads = torch.autograd.grad(
            expected_kl * (0.01 / denominator),
            (inputs["idx_q_TIE"], inputs["idx_k_TE"], inputs["idx_w_TI"]),
        )

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(
            actual_grads[:3], expected_main_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(
            actual_grads[3:], expected_idx_grads, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(
            attention.aux_loss.instance_acc,
            (expected_kl.detach() / denominator).float(),
        )


def test_indexer_loss_ignores_padding_and_eval_and_resets_at_step() -> None:
    torch.manual_seed(23)
    with patch.dict(AuxLoss._group_counts, clear=True), patch.dict(
        AuxLoss.group_acc, clear=True
    ):
        inputs, config = _random_sparse_inputs([3, 2], topk=2)
        for name in ("idx_q_TIE", "idx_k_TE", "idx_w_TI"):
            inputs[name].requires_grad_()
        config.aux_loss = KimiIndexerLoss.Config(
            coeff=0.01, num_heads=2, reduce_mesh="loss"
        )
        attention = config.build().train()
        padding_mask = torch.tensor([False, False, False, True, True])
        positions = torch.cat([torch.arange(3), torch.arange(2)]).int()
        metadata = config.build_attention_metadata(
            positions, padding_mask=padding_mask, max_context_length=5
        )
        denominator = torch.tensor(3.0)
        actual = attention(
            **inputs,
            attention_metadata=metadata,
            padding_mask=padding_mask,
            scale=0.173,
            aux_loss_denominator=denominator,
        )
        expected_kl = _reference_indexer_kl(
            **inputs,
            topk=2,
            scale=0.173,
            lengths=[3, 2],
            padding_mask=padding_mask,
        )
        torch.testing.assert_close(
            attention.aux_loss.instance_acc,
            (expected_kl.detach() / denominator).float(),
        )
        actual.sum().backward()
        idx_grads = [
            inputs[name].grad for name in ("idx_q_TIE", "idx_k_TE", "idx_w_TI")
        ]
        for grad in idx_grads:
            assert grad is not None
            torch.testing.assert_close(grad[3:], torch.zeros_like(grad[3:]))

        attention.eval()
        metric_before_eval = attention.aux_loss.instance_acc.clone()
        eval_output = attention(
            **inputs,
            attention_metadata=metadata,
            padding_mask=padding_mask,
            scale=0.173,
            aux_loss_denominator=denominator,
        )
        torch.testing.assert_close(attention.aux_loss.instance_acc, metric_before_eval)
        eval_idx_grads = torch.autograd.grad(
            eval_output.sum(),
            (inputs["idx_q_TIE"], inputs["idx_k_TE"], inputs["idx_w_TI"]),
            allow_unused=True,
        )
        assert all(grad is None for grad in eval_idx_grads)

        callbacks = []
        optimizers = SimpleNamespace(register_step_pre_hook=callbacks.append)
        KimiK31Model._register_optimizer_hooks(optimizers, [attention], None)
        assert len(callbacks) == 1
        callbacks[0]()
        torch.testing.assert_close(
            attention.aux_loss.instance_acc,
            torch.zeros_like(attention.aux_loss.instance_acc),
        )
        torch.testing.assert_close(
            AuxLoss.group_acc[("loss", "kimi_indexer_loss")], metric_before_eval
        )


def test_dsa_attention_uses_detached_normalized_query_and_trains_indexer() -> None:
    torch.manual_seed(29)
    config = _dsa_config(
        dim=8,
        num_heads=2,
        q_lora_rank=4,
        kv_lora_rank=5,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=3,
        attn_backend="varlen",
        enable_sp=False,
        index_topk=2,
        index_num_heads=3,
        index_head_dim=4,
        indexer_loss_coeff=0.01,
    )
    assert isinstance(config, KimiDSAAttention.Config)
    attention = config.build().train()
    attention.init_states()
    captured = {}

    def capture_indexer_inputs(_module, args, _output) -> None:
        captured["q"] = args[0]
        captured["x"] = args[1]

    attention.indexer.register_forward_hook(capture_indexer_inputs)
    x_TD = torch.randn(5, 8, requires_grad=True)
    output = attention(x_TD, aux_loss_denominator=torch.tensor(5.0))
    with torch.no_grad():
        expected_q = attention.q_norm(attention.wq_a(x_TD))
    torch.testing.assert_close(captured["q"], expected_q)
    torch.testing.assert_close(captured["x"], x_TD)
    assert not captured["q"].requires_grad
    assert not captured["x"].requires_grad

    output.sum().backward()
    assert x_TD.grad is not None and torch.isfinite(x_TD.grad).all()
    for parameter in attention.indexer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0


def test_debug_recipe_selects_the_dsa_flavor() -> None:
    from torchtitan_recipes.tests.models.kimi_k3_1 import kimi_k3_1_debugmodel

    config = kimi_k3_1_debugmodel(seq_len=16)
    dsa_layers = [
        layer.attention
        for layer in config.model.layers
        if isinstance(layer.attention, KimiDSAAttention.Config)
    ]
    assert len(dsa_layers) == 5
    for attention in dsa_layers:
        assert attention.inner_attention.inner_attention.aux_loss is not None


def _one_layer_dsa_model_config(*, dsa: bool):
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
        vision_encoder=None,
        attn_backend="flex",
        parallelism=ParallelismConfig(),
        full_attention_config_factory=(
            partial(
                _dsa_config,
                index_topk=16,
                index_num_heads=3,
                index_head_dim=4,
                indexer_loss_coeff=0.01,
            )
            if dsa
            else _mla_config
        ),
        model_config_cls=KimiK31Model.Config if dsa else KimiK3Model.Config,
    )


def test_full_causal_dsa_model_matches_dense_k3_mla() -> None:
    torch.manual_seed(31)
    dense_config = _one_layer_dsa_model_config(dsa=False)
    dsa_config = _one_layer_dsa_model_config(dsa=True)
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


def test_indexer_key_norm_computes_fp32_with_bf16_model_default() -> None:
    config = KimiDSAIndexer.Config(
        index_num_heads=3,
        index_head_dim=8,
        wq_b=_linear(4, 24),
        wk=_linear(6, 8),
        k_norm=LayerNorm.Config(
            normalized_shape=8,
        ),
        weights_proj=_linear(6, 3),
    )
    from torchtitan.tools.utils import set_default_dtype

    with torch.device("meta"), set_default_dtype(torch.bfloat16):
        indexer = config.build()
    indexer.to_empty(device="cpu")
    indexer.init_states(buffer_device=torch.device("cpu"))

    assert indexer.k_norm.weight.dtype == torch.bfloat16
    assert indexer.k_norm.bias.dtype == torch.bfloat16
    q_TC = torch.randn(5, 4, dtype=torch.bfloat16)
    x_TD = torch.randn(5, 6, dtype=torch.bfloat16)
    _, idx_k_TE, _ = indexer(q_TC, x_TD)
    assert idx_k_TE.dtype == torch.bfloat16
    assert torch.isfinite(idx_k_TE).all()


def test_dsa_flops_use_sparse_window_and_indexer_geometry() -> None:
    common = {
        "dim": 8,
        "num_heads": 2,
        "q_lora_rank": 4,
        "kv_lora_rank": 5,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 2,
        "v_head_dim": 3,
        "attn_backend": "varlen",
        "enable_sp": False,
    }
    dense = _mla_config(**common)
    dsa = _dsa_config(
        **common,
        index_topk=4,
        index_num_heads=3,
        index_head_dim=8,
        indexer_loss_coeff=0.01,
    )
    seq_len = 32
    expected = 6 * 2 * (6 + 3) * 4 + 6 * 3 * 8 * seq_len
    assert dsa.flops_per_token(seq_len) == expected
    assert dense.flops_per_token(seq_len) == 6 * 2 * (6 + 3) * seq_len


def test_dsa_model_backward_with_activation_remat() -> None:
    torch.manual_seed(41)
    config = _one_layer_dsa_model_config(dsa=True)
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))
    model.train()
    DefaultActivationRematConfig().apply(model)

    tokens = torch.arange(8) % config.vocab_size
    positions = torch.arange(8, dtype=torch.int32)
    output = model(
        tokens,
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
        aux_loss_denominators=torch.tensor([8.0]),
    )
    output.sum().backward()
    attention = model.layers["0"].attention
    assert isinstance(attention, KimiDSAAttention)
    for parameter in attention.indexer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_zero_indexer_loss_coefficient_disables_kl() -> None:
    config = kimi_k3_1.build_model_config(
        "debugmodel",
        seq_len=8,
        parallelism=ParallelismConfig(),
        indexer_loss_coeff=0.0,
    )
    attention_config = next(
        layer.attention
        for layer in config.layers
        if isinstance(layer.attention, KimiDSAAttention.Config)
    )
    attention = attention_config.build().train()
    attention.init_states()
    assert attention.inner_attention.inner_attention.aux_loss is None

    x_TD = torch.randn(8, config.dim, requires_grad=True)
    output = attention(x_TD)
    output.sum().backward()
    for parameter in attention.indexer.parameters():
        assert parameter.grad is None
