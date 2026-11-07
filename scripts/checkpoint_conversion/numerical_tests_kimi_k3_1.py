#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 DSA forward parity against pinned external references.

The K3.1 HF repository publishes weights and config, but no model forward code.
The sparse indexer and attention checks therefore use SGLang at revision
``0afb736d0aab96e3ab80b4c54a376213a1db67a6``. Full-model logits use the
official K3 HF implementation with full-selection DSA, which is mathematically
equivalent to dense MLA. Separate Python environments avoid mixing the two
runtimes' PyTorch installations.

Run with CUDA available in all environments:

    python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3_1 \
        --reference-python /path/to/sglang/venv/bin/python \
        --hf-reference-python /path/to/hf/venv/bin/python

The SGLang environment needs the pinned source and its dependencies. The HF
environment needs transformers==4.56.2 and fla-core==0.5.2. These tests use
randomly initialized debug weights, not the released checkpoint.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

_REFERENCE_REVISION = "0afb736d0aab96e3ab80b4c54a376213a1db67a6"


def _build_reference_layer():
    from sglang.srt.models.kimi_k3 import (  # pyrefly: ignore [missing-import]
        KimiK3MLAAttention,
    )

    config = SimpleNamespace(
        architectures=["KimiK31ForConditionalGeneration"],
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=128,
        kv_lora_rank=64,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-5,
        mla_use_output_gate=True,
        mla_use_nope=True,
        index_topk=4,
        index_n_heads=4,
        index_head_dim=128,
        index_rope_dim=0,
        index_k_norm_type="layernorm",
        index_topk_freq=1,
        index_skip_topk_offset=1,
    )
    return KimiK3MLAAttention(config, layer_idx=3).to(
        device="cuda", dtype=torch.bfloat16
    )


def _dump_torchtitan(path: Path) -> None:
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.kimi_k3_1 import build_model_config

    config = build_model_config(
        "debugmodel", seq_len=16, parallelism=ParallelismConfig()
    )
    layer = config.layers[3].attention.build()
    layer.init_states()
    layer = layer.to(device="cuda", dtype=torch.bfloat16).eval()
    torch.manual_seed(42)
    x = torch.randn(12, 256, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(12, device="cuda")
    q_a = layer.wq_a(x)
    q_normed = layer.q_norm(q_a)
    idx_q, idx_k, idx_w = layer.indexer(x, q_normed)
    q_main = layer.wq_b(q_normed).view(12, 4, 192)
    compressed = layer.wkv_a(x)
    kv_latent, k_shared = compressed.split([64, 64], dim=-1)
    kv_latent = layer.kv_norm(kv_latent)
    q_nope, q_rope = q_main.split([128, 64], dim=-1)
    from torchtitan.models.kimi_k3.dsa import _select_topk

    wkv = layer.inner_attention.wkv_b.weight.view(4, 256, 64)
    w_uk, w_uv = wkv.split([128, 128], dim=1)
    q_latent = torch.einsum("txn,xnl->txl", q_nope, w_uk)
    inner_out = layer.inner_attention(
        q_main,
        kv_latent,
        k_shared,
        idx_q_THD=idx_q,
        idx_k_SD=idx_k,
        idx_w_TH=idx_w,
        attention_metadata=None,
    )
    topk, _ = _select_topk(
        idx_q,
        idx_k,
        idx_w,
        top_k=4,
        attention_metadata=None,
        padding_mask_T=None,
        return_scores=False,
    )
    tensors = {
        "x": x,
        "positions": positions,
        "q_normed": q_normed,
        "idx_q": idx_q,
        "idx_k": idx_k,
        "idx_w": idx_w,
        "q_latent_nope": q_latent,
        "q_rope": q_rope,
        "kv_latent": kv_latent,
        "k_shared": k_shared,
        "w_uv": w_uv,
        "inner_out": inner_out,
        "topk": topk,
    }
    for name, parameter in layer.named_parameters():
        tensors[f"param.{name}"] = parameter.detach()
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(path),
    )


def _run_sglang_reference(input_path: Path, output_path: Path) -> None:
    from sglang.srt.server_args import ServerArgs  # pyrefly: ignore [missing-import]
    from sglang.test.parallel_groups import publish  # pyrefly: ignore [missing-import]

    publish(ServerArgs(model_path="dummy"), role="worker")
    from sglang.srt.model_executor.forward_mode import (  # pyrefly: ignore [missing-import]
        ForwardMode,
    )
    from sglang.test.kits.attention_unittest.attention_methods.dsa_attention import (  # pyrefly: ignore [missing-import]
        DSAAttentionCase,
        expected_dsa_sparse_fixture_output,
    )

    tensors = {name: value.cuda() for name, value in load_file(str(input_path)).items()}
    reference = _build_reference_layer()
    with torch.no_grad():
        reference.fused_qkv_a_proj_with_mqa.weight.copy_(
            torch.cat(
                (
                    tensors["param.wq_a.weight"],
                    tensors["param.wkv_a.weight"],
                ),
                dim=0,
            )
        )
        reference.q_a_layernorm.weight.copy_(tensors["param.q_norm.weight"])
        reference.q_b_proj.weight.copy_(tensors["param.wq_b.weight"])
        reference.kv_a_layernorm.weight.copy_(tensors["param.kv_norm.weight"])
        reference.kv_b_proj.weight.copy_(tensors["param.inner_attention.wkv_b.weight"])
        reference.g_proj.weight.copy_(tensors["param.gate.weight"])
        reference.o_proj.weight.copy_(tensors["param.wo.weight"])
        reference.indexer.wq_b.weight.copy_(tensors["param.indexer.wq_b.weight"])
        reference.indexer.wk.weight.copy_(tensors["param.indexer.wk.weight"])
        reference.indexer.weights_proj.weight.copy_(
            tensors["param.indexer.weights_proj.weight"]
        )
        reference.indexer.k_norm.weight.copy_(tensors["param.indexer.k_norm.weight"])
        reference.indexer.k_norm.bias.copy_(tensors["param.indexer.k_norm.bias"])

        idx_q, idx_k, _ = reference.indexer._get_q_k_bf16(
            tensors["q_normed"],
            tensors["x"],
            tensors["positions"],
            False,
            SimpleNamespace(forward_mode=None),
        )
        idx_w = reference.indexer._project_head_gates(tensors["x"]) * (
            reference.indexer.n_heads**-0.5 * reference.indexer.head_dim**-0.5
        )

    class SparseReferenceModule:
        qk_nope_head_dim = 64
        qk_rope_head_dim = 64
        attn = SimpleNamespace(scaling=192**-0.5)

        def project_q(self, unused):
            return tensors["q_latent_nope"], tensors["q_rope"]

        def project_k(self, unused):
            return tensors["kv_latent"], tensors["k_shared"]

        def o_proj(self, x):
            latent = x.view(12, 4, 64).float()
            return torch.einsum(
                "thl,hvl->thv", latent, tensors["w_uv"].float()
            ).flatten(-2)

    case = DSAAttentionCase(
        name="kimi_k31_numerics",
        backend="dsa",
        forward_mode=ForwardMode.EXTEND,
        num_heads=4,
        num_kv_heads=1,
        page_size=64,
        prefix_lens=(0,),
        extend_lens=(12,),
    )
    fixture = SimpleNamespace(
        case=case,
        prefix_hidden=[torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)],
        input_hidden=torch.empty(12, 256, device="cuda", dtype=torch.bfloat16),
        topk_rows=tensors["topk"].tolist(),
        actual_module=SparseReferenceModule(),
    )
    inner_out = expected_dsa_sparse_fixture_output(fixture)
    save_file(
        {
            "idx_q": idx_q.contiguous(),
            "idx_k": idx_k.contiguous(),
            "idx_w": idx_w.contiguous(),
            "inner_out_flat": inner_out.contiguous(),
        },
        str(output_path),
    )


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float().view_as(actual)
    difference = (actual - expected).abs()
    print(
        f"{name}: max_abs={difference.max().item():.8g}, "
        f"mean_abs={difference.mean().item():.8g}, "
        f"reference_max={expected.abs().max().item():.8g}"
    )
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


_HF_REPO_ID = "moonshotai/Kimi-K3"
_HF_REFERENCE_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


@torch.no_grad()
def _dump_full_selection_logits(workdir: Path) -> None:
    import shutil
    from dataclasses import replace

    from huggingface_hub import snapshot_download

    from scripts.checkpoint_conversion.numerical_tests_kimi_k3 import reduce_hf_config
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.kimi_k3.dsa import KimiGatedDSAAttention
    from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
    from torchtitan.models.kimi_k3_1 import build_model_config
    from transformers import AutoConfig

    seq_len = 12
    config = build_model_config(
        "debugmodel", seq_len=seq_len, parallelism=ParallelismConfig()
    )
    layers = []
    for layer in config.layers:
        attention = layer.attention
        if isinstance(attention, KimiGatedDSAAttention.Config):
            attention = replace(
                attention,
                inner_attention=replace(attention.inner_attention, index_topk=seq_len),
            )
        layers.append(replace(layer, attention=attention))
    config = replace(config, layers=layers)

    torch.manual_seed(42)
    with torch.device("meta"):
        model = config.build()
    model.to_empty(device="cpu")
    model.to(dtype=torch.bfloat16)
    model.init_states(buffer_device=torch.device("cpu"))
    model = model.eval()

    shared_state = {
        key: value
        for key, value in model.state_dict().items()
        if ".indexer." not in key
    }
    hf_state = KimiK3StateDictAdapter(config, hf_assets_path=None).to_hf(shared_state)
    save_file(
        {key: value.contiguous() for key, value in hf_state.items()},
        str(workdir / "hf_state.safetensors"),
    )

    source = Path(
        snapshot_download(
            repo_id=_HF_REPO_ID,
            revision=_HF_REFERENCE_REVISION,
            allow_patterns=["*.py", "config.json"],
        )
    )
    for path in source.iterdir():
        if path.suffix == ".py":
            shutil.copy(path, workdir / path.name)
    hf_config = AutoConfig.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    reduce_hf_config(hf_config, config, str(source))
    # The released special-token IDs exceed the reduced vocabulary. Text-only
    # random inputs do not use them, so use valid IDs for HF model construction.
    for subconfig in (hf_config, hf_config.text_config):
        subconfig.pad_token_id = 0
        subconfig.bos_token_id = 0
        subconfig.eos_token_id = 1
    hf_config.save_pretrained(workdir)

    model.vision_encoder = None
    model = model.to(device="cuda")
    tokens = torch.randint(0, config.vocab_size, (seq_len,), device="cuda")
    positions = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    logits = model(
        tokens,
        positions=positions,
        attention_metadata=model._get_attention_metadata(positions),
    )
    save_file(
        {"tokens": tokens, "logits": logits.contiguous()},
        str(workdir / "torchtitan_logits.safetensors"),
    )


@torch.no_grad()
def _run_hf_reference(input_path: Path, output_path: Path) -> None:
    import transformers.utils

    # The released model imports FlashAttention while being selected for eager.
    transformers.utils.is_flash_attn_2_available = lambda: False
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        input_path, trust_remote_code=True, local_files_only=True
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # The K3 decoder constructor forces FlashAttention 2; restore eager for the
    # same reference path used by the K3 numerical test.
    model.language_model.config._attn_implementation = "eager"
    model.language_model._use_flash_attention_2 = False
    model.to(dtype=torch.bfloat16)
    model.load_state_dict(
        load_file(str(input_path / "hf_state.safetensors")), strict=True
    )
    model = model.to(device="cuda").eval()
    tokens = load_file(str(input_path / "torchtitan_logits.safetensors"))["tokens"]
    logits = model(input_ids=tokens.cuda().unsqueeze(0), use_cache=False).logits[0]
    save_file({"logits": logits.cpu().contiguous()}, str(output_path))


def _compare_logits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    _compare("logits", actual, expected)
    actual_last = actual[-1].float()
    expected_last = expected[-1].float()
    cosine = torch.nn.functional.cosine_similarity(
        actual_last, expected_last, dim=0
    ).item()
    top5_overlap = len(
        set(actual_last.topk(5).indices.tolist())
        & set(expected_last.topk(5).indices.tolist())
    )
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(actual_last, dim=-1),
        torch.log_softmax(expected_last, dim=-1),
        log_target=True,
        reduction="sum",
    ).item()
    print(
        f"last_logits: max_abs={(actual_last - expected_last).abs().max().item():.8g}, "
        f"cosine={cosine:.8g}, "
        f"top1_match={actual_last.argmax().item() == expected_last.argmax().item()}, "
        f"top5_overlap={top5_overlap}/5, kl_hf_to_tt={kl:.8g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--hf-reference-python", type=Path)
    parser.add_argument("--sglang-reference", action="store_true")
    parser.add_argument("--hf-reference", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sglang_reference or args.hf_reference:
        if args.input is None or args.output is None:
            parser.error("Reference subprocesses require --input and --output")
        if args.sglang_reference:
            _run_sglang_reference(args.input, args.output)
        else:
            _run_hf_reference(args.input, args.output)
        return
    if args.reference_python is None and args.hf_reference_python is None:
        parser.error("At least one reference Python environment is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Kimi K3.1 numerical test.")

    with tempfile.TemporaryDirectory(prefix="kimi_k3_1_numerics_") as directory:
        workdir = Path(directory)
        if args.reference_python is not None:
            tt_path = workdir / "torchtitan.safetensors"
            reference_path = workdir / "sglang.safetensors"
            _dump_torchtitan(tt_path)
            subprocess.run(
                [
                    str(args.reference_python),
                    __file__,
                    "--sglang-reference",
                    "--input",
                    str(tt_path),
                    "--output",
                    str(reference_path),
                ],
                check=True,
            )
            actual = load_file(str(tt_path))
            expected = load_file(str(reference_path))
            for name in ("idx_q", "idx_k", "idx_w"):
                _compare(name, actual[name], expected[name])
            _compare(
                "inner_out",
                actual["inner_out"],
                expected["inner_out_flat"].view_as(actual["inner_out"]),
            )
            print(f"PASS: SGLang reference revision={_REFERENCE_REVISION}")

        if args.hf_reference_python is not None:
            hf_workdir = workdir / "hf"
            hf_workdir.mkdir()
            _dump_full_selection_logits(hf_workdir)
            reference_path = hf_workdir / "hf_logits.safetensors"
            subprocess.run(
                [
                    str(args.hf_reference_python),
                    __file__,
                    "--hf-reference",
                    "--input",
                    str(hf_workdir),
                    "--output",
                    str(reference_path),
                ],
                check=True,
            )
            actual = load_file(str(hf_workdir / "torchtitan_logits.safetensors"))
            expected = load_file(str(reference_path))
            _compare_logits(actual["logits"], expected["logits"])
            print(
                "PASS: HF K3 full-selection DSA/dense equivalence "
                f"revision={_HF_REFERENCE_REVISION}"
            )


if __name__ == "__main__":
    main()
