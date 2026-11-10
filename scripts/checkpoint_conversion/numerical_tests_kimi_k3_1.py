#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Forward parity of one released Kimi K3.1 Pro DSA layer.

No Hugging Face model code is released for Kimi K3.1, so the reference follows
vLLM's Kimi K3.1 DSA (``vllm/models/kimi_k3/nvidia/dsa.py``) in plain PyTorch,
without its FP8 inference quantization. The layer is loaded from a local
checkpoint through ``KimiK31StateDictAdapter``; only the shard that holds the
layer is read. In bf16 the indexer uses attn-gym's fused kernel, which needs
SM90 or SM100; ``--fp32`` runs on any GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m \
        scripts.checkpoint_conversion.numerical_tests_kimi_k3_1 \
        --hf-path /path/to/Kimi-K3.1-Pro
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4.compressor import Indexer
from torchtitan.models.kimi_k3_1 import build_model_config
from torchtitan.models.kimi_k3_1.state_dict_adapter import KimiK31StateDictAdapter


def _load_layer(hf_path: str, layer: int) -> dict[str, torch.Tensor]:
    with open(os.path.join(hf_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"language_model.model.layers.{layer}.self_attn."
    tensors = {}
    for shard in sorted({v for k, v in weight_map.items() if k.startswith(prefix)}):
        with safe_open(os.path.join(hf_path, shard), "pt") as f:
            for key in f.keys():
                if key.startswith(prefix):
                    tensors[key] = f.get_tensor(key)
    return tensors


def _reference(x, w, text_config):
    """Return the fp32 attention output and the ``[T, T]`` top-k selection."""
    num_tokens = x.shape[0]
    num_heads = text_config["num_attention_heads"]
    nope_dim = text_config["qk_nope_head_dim"]
    rope_dim = text_config["qk_rope_head_dim"]
    v_dim = text_config["v_head_dim"]
    latent_dim = text_config["kv_lora_rank"]
    eps = text_config["rms_norm_eps"]
    dsa = text_config["dsa_config"]
    topk, index_dim = dsa["index_topk"], dsa["index_head_dim"]
    index_heads = dsa["index_num_heads"]

    def rms_norm(t, weight):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * weight

    q_c = rms_norm(x @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"])
    q = (q_c @ w["q_b_proj.weight"].T).view(num_tokens, num_heads, -1)
    kv_c, k_pe = (x @ w["kv_a_proj_with_mqa.weight"].T).split(
        [latent_dim, rope_dim], -1
    )
    kv_c = rms_norm(kv_c, w["kv_a_layernorm.weight"])
    k_nope, value = (
        (kv_c @ w["kv_b_proj.weight"].T)
        .view(num_tokens, num_heads, nope_dim + v_dim)
        .split([nope_dim, v_dim], -1)
    )
    k = torch.cat([k_nope, k_pe[:, None].expand(-1, num_heads, -1)], -1)

    index_q = (q_c @ w["indexer.wq_b.weight"].T).view(num_tokens, index_heads, -1)
    index_k = F.layer_norm(
        x @ w["indexer.wk.weight"].T,
        (index_dim,),
        w["indexer.k_norm.weight"],
        w["indexer.k_norm.bias"],
        eps,
    )
    index_w = (x @ w["indexer.weights_proj.weight"].T) * (
        index_dim**-0.5 * index_heads**-0.5
    )
    causal = torch.ones(num_tokens, num_tokens, dtype=torch.bool, device=x.device)
    causal = causal.tril()
    selected = torch.zeros_like(causal)
    out = torch.empty(num_tokens, num_heads, v_dim, device=x.device)
    for start in range(0, num_tokens, 256):
        stop = start + 256
        scores = torch.einsum("qhd,kd->qhk", index_q[start:stop], index_k)
        scores = (scores.relu() * index_w[start:stop, :, None]).sum(1)
        scores = scores.masked_fill(~causal[start:stop], -torch.inf)
        indices = scores.topk(min(topk, num_tokens), dim=-1).indices
        selected[start:stop].scatter_(1, indices, True)
        selected[start:stop] &= causal[start:stop]
        logits = torch.einsum("qhd,khd->qhk", q[start:stop], k)
        logits = logits * (nope_dim + rope_dim) ** -0.5
        logits = logits.masked_fill(~selected[start:stop, None], -torch.inf)
        out[start:stop] = torch.einsum("qhk,khv->qhv", logits.softmax(-1), value)
    gate = torch.sigmoid(x @ w["g_proj.weight"].T)
    return (out.flatten(1) * gate) @ w["o_proj.weight"].T, selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=2304)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    hf_state = _load_layer(args.hf_path, args.layer)
    with open(os.path.join(args.hf_path, "config.json")) as f:
        text_config = json.load(f)["text_config"]
    config = build_model_config(
        "Kimi-K3.1-Pro",
        parallelism=ParallelismConfig(),
        seq_len=args.seq_len,
        indexer_loss_coeff=0.0,
    )
    attention_config = config.layers[args.layer].attention
    adapter = KimiK31StateDictAdapter(config, hf_assets_path=None)
    prefix = f"layers.{args.layer}.attention."
    state = {
        k.removeprefix(prefix): v.to("cuda", dtype)
        for k, v in adapter.from_hf(hf_state).items()
    }
    with torch.device("meta"):
        attention = attention_config.build()
    attention = attention.to_empty(device="cuda").to(dtype)
    attention.load_state_dict(state, strict=True)
    attention.eval()

    torch.manual_seed(0)
    x = torch.randn(args.seq_len, config.dim, device="cuda").to(dtype)
    positions = torch.arange(args.seq_len, device="cuda", dtype=torch.int32)
    metadata = attention_config.inner_attention.build_attention_metadata(positions)
    topk = attention_config.inner_attention.index_topk
    with torch.no_grad():
        out = attention(x, metadata).float()
        index_q, index_k, index_w = attention.indexer(
            attention.q_norm(attention.wq_a(x)), x
        )
        indices, _ = Indexer.select(
            index_q,
            index_k,
            index_w,
            max_seqlen=metadata.max_k,
            ratio=1,
            topk=topk,
            cu_seqlens=metadata.cu_seq_q,
            return_scores=False,
        )
        weights = {
            k.removeprefix(
                f"language_model.model.layers.{args.layer}.self_attn."
            ): v.to("cuda", torch.float32)
            for k, v in hf_state.items()
        }
        ref_out, ref_selected = _reference(x.float(), weights, text_config)

    rows = torch.arange(topk, args.seq_len, device="cuda")
    selected = torch.zeros_like(ref_selected)
    valid = indices >= 0
    token = torch.arange(args.seq_len, device="cuda")[:, None].expand_as(indices)
    selected[token[valid], indices[valid].long()] = True
    overlap = (selected[rows] & ref_selected[rows]).sum(-1).float() / topk
    error = (out - ref_out).norm() / ref_out.norm()
    print(f"{dtype}, {args.seq_len} tokens, layer {args.layer}")
    print(
        f"top-k overlap over {len(rows)} rows: mean {overlap.mean():.5f}, min {overlap.min():.5f}"
    )
    print(
        f"output: max abs diff {(out - ref_out).abs().max():.3e}, relative L2 {error:.3e}"
    )


if __name__ == "__main__":
    main()
