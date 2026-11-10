# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss

from tests.unit_tests.cpu.models.kimi_k3_1.test_kimi_k3_1_dsa import (
    _build_sparse_attention,
    _expanded_inputs,
    _INDEXER_INPUTS,
    _MAIN_INPUTS,
    _random_sparse_inputs,
    _reference_indexer_kl,
    _reference_sparse_attention,
)


class TestKimiK31DSA(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
    def test_sparse_attention_bf16_matches_float64_oracle(self):
        torch.manual_seed(37)
        with patch.dict(AuxLoss._group_counts, clear=True), patch.dict(
            AuxLoss.group_acc, clear=True
        ):
            reference_inputs, config = _random_sparse_inputs([8], topk=3)
            for name in _INDEXER_INPUTS:
                reference_inputs[name].requires_grad_()
            reference_weight = _build_sparse_attention(config).wkv_b.weight
            expanded = _expanded_inputs(reference_inputs, reference_weight)
            expected = _reference_sparse_attention(
                **expanded, topk=3, scale=0.173, lengths=[8]
            )
            expected_kl = _reference_indexer_kl(
                **expanded, topk=3, scale=0.173, lengths=[8]
            )
            output_grad = torch.randn_like(expected)
            expected_main_grads = torch.autograd.grad(
                expected,
                (*(reference_inputs[name] for name in _MAIN_INPUTS), reference_weight),
                output_grad,
                retain_graph=True,
            )
            reference_denominator = torch.tensor(8.0)
            expected_idx_grads = torch.autograd.grad(
                expected_kl * (0.01 / reference_denominator),
                tuple(reference_inputs[name] for name in _INDEXER_INPUTS),
            )

            config.aux_loss = SparseIndexerLoss.Config(
                coeff=0.01,
                reduce_mesh="loss",
                softmax_scale=0.173,
                num_heads=2,
                compress_ratio=1,
            )
            attention = config.build().to(device="cuda").train()
            attention.wkv_b.to(dtype=torch.bfloat16)
            with torch.no_grad():
                attention.wkv_b.weight.copy_(reference_weight)
            inputs = {
                name: value.detach()
                .to(device="cuda", dtype=torch.bfloat16)
                .requires_grad_()
                for name, value in reference_inputs.items()
            }
            actual = attention(
                **inputs,
                scale=0.173,
                aux_loss_denominator=torch.tensor(8.0, device="cuda"),
            )
            actual_grads = torch.autograd.grad(
                actual,
                (
                    *(inputs[name] for name in _MAIN_INPUTS),
                    attention.wkv_b.weight,
                    *(inputs[name] for name in _INDEXER_INPUTS),
                ),
                output_grad.to(device="cuda", dtype=torch.bfloat16),
            )

            def check(name, actual_value, expected_value, *, atol, rtol=0.03):
                actual_value = actual_value.detach().double().cpu()
                difference = (actual_value - expected_value).abs()
                max_abs = difference.max().item()
                max_rel = (
                    (difference / expected_value.abs().clamp_min(1e-6)).max().item()
                )
                print(
                    f"K3.1 DSA BF16 {name}: max_abs={max_abs:.6e} "
                    f"max_rel={max_rel:.6e}"
                )
                torch.testing.assert_close(
                    actual_value, expected_value, atol=atol, rtol=rtol
                )

            check("output", actual, expected, atol=0.03)
            for name, actual_grad, expected_grad in zip(
                ("q", "kv_c", "k_shared", "wkv_b"),
                actual_grads[:4],
                expected_main_grads,
                strict=True,
            ):
                check(f"{name}_grad", actual_grad, expected_grad, atol=0.03)
            for name, actual_grad, expected_grad in zip(
                ("idx_q", "idx_k", "idx_w"),
                actual_grads[4:],
                expected_idx_grads,
                strict=True,
            ):
                assert expected_grad.abs().max() > 5e-4
                check(f"{name}_grad", actual_grad, expected_grad, atol=1e-4, rtol=0.05)
            check(
                "indexer_loss",
                attention.aux_loss.instance_acc,
                (expected_kl.detach() / reference_denominator).double(),
                atol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
