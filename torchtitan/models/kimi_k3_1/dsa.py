# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 DeepSeek Sparse Attention components.

Forward semantics follow vLLM at b65f94500866d7277a0863baff4369888819091a
(``vllm/models/kimi_k3/nvidia/dsa.py``), with HF config revision
d9e99b0792a5e93b78ea9bea62a68167e36ddd9e. The released configuration uses
index_topk_freq=1 and index_skip_topk_offset=1, so every layer selected by
``dsa_layers`` computes its own sparse indices.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    Attention,
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.deepseek_v4.compressor import (
    compressed_gather_indices,
    Indexer,
    SparseIndexerLoss,
)
from torchtitan.models.kimi_k3.attention import KimiMLAAttention

__all__ = [
    "KimiDSAAttention",
    "KimiDSAIndexer",
    "KimiSparseAttention",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# S = selected token slots
# D = model dimension (dim)
# C = low-rank query channels (q_lora_rank)
# H = main attention head count (n_heads)
# K = main query/key head dimension (N + R)
# N = per-head non-positional key dimension (qk_nope_head_dim)
# R = head-shared key dimension (qk_rope_head_dim)
# V = main value head dimension
# L = KV latent dimension (kv_lora_rank)
# P = packed per-head KV up-projection channels (N + V)
# G = shared latent key channels (L + R)
# I = indexer head count (index_num_heads)
# E = indexer head dimension (index_head_dim)
# J = flattened projection channels


class KimiDSAIndexer(Module):
    """Project normalized low-rank queries and raw inputs into DSA scores."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        index_num_heads: int
        index_head_dim: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config

        def __post_init__(self) -> None:
            if self.index_num_heads <= 0 or self.index_head_dim <= 0:
                raise ValueError("Indexer head count and dimension must be positive.")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.index_num_heads = config.index_num_heads
        self.index_head_dim = config.index_head_dim
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()

    def forward(
        self, q_TC: torch.Tensor, x_TD: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_q_TJ = self.wq_b(q_TC)
        remat.recompute_needs_tensor(idx_q_TJ)
        idx_q_TIE = idx_q_TJ.view(
            q_TC.shape[0], self.index_num_heads, self.index_head_dim
        )
        idx_k_TE = self.wk(x_TD)
        remat.recompute_needs_tensor(idx_k_TE)
        # The released parameters are BF16; vLLM evaluates this norm in FP32.
        idx_k_TE = F.layer_norm(
            idx_k_TE.float(),
            (self.index_head_dim,),
            self.k_norm.weight.float(),
            self.k_norm.bias.float(),
            self.k_norm.eps,
        ).to(x_TD.dtype)
        idx_w_TI = self.weights_proj(x_TD)
        remat.recompute_needs_tensor(idx_w_TI)
        idx_w_TI = idx_w_TI * (
            self.index_head_dim**-0.5 * self.index_num_heads**-0.5
        )
        return idx_q_TIE, idx_k_TE, idx_w_TI


class KimiSparseAttention(InnerAttention):
    """Eager absorbed MLA over indexer-selected tokens.

    Each query head absorbs its key up-projection, so all heads attend one
    shared latent ``[kv_c_normed, k_shared]`` that is also the value. The
    first ``kv_lora_rank`` output channels are up-projected per head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        wkv_b: Linear.Config
        index_topk: int
        aux_loss: SparseIndexerLoss.Config | None = None

        def __post_init__(self) -> None:
            if self.index_topk <= 0:
                raise ValueError("index_topk must be positive.")
            if self.aux_loss is not None and self.aux_loss.compress_ratio != 1:
                raise ValueError("Kimi DSA selects uncompressed tokens.")

        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> VarlenAttentionMetadata:
            return create_varlen_metadata_for_document(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.wkv_b = config.wkv_b.build()
        if type(self.wkv_b) is not Linear or self.wkv_b.bias is not None:
            raise ValueError(
                "KimiSparseAttention reads wkv_b.weight directly, so wkv_b must be "
                f"a bias-free Linear, but got {type(self.wkv_b).__qualname__} "
                f"(bias={self.wkv_b.bias is not None})."
            )
        self.index_topk = config.index_topk
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        *,
        idx_q_TIE: torch.Tensor,
        idx_k_TE: torch.Tensor,
        idx_w_TI: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None = None,
        scale: float,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        aux_loss = self.aux_loss if self.training else None
        cu_seqlens = None if attention_metadata is None else attention_metadata.cu_seq_q
        topk_indices_TS, topk_scores_TS = Indexer.select(
            idx_q_TIE,
            idx_k_TE,
            idx_w_TI,
            max_seqlen=(
                q_THK.shape[0]
                if attention_metadata is None
                else attention_metadata.max_k
            ),
            ratio=1,
            topk=self.index_topk,
            cu_seqlens=cu_seqlens,
            return_scores=aux_loss is not None,
        )

        latent_dim = kv_c_normed_TL.shape[-1]
        nope_dim = q_THK.shape[-1] - k_shared_TR.shape[-1]
        wkv_b_HPL = self.wkv_b.weight.view(q_THK.shape[1], -1, latent_dim)
        w_uk_HNL, w_uv_HVL = wkv_b_HPL.split(
            [nope_dim, wkv_b_HPL.shape[1] - nope_dim], dim=1
        )
        q_nope_THN, q_shared_THR = q_THK.split(
            [nope_dim, k_shared_TR.shape[-1]], dim=-1
        )
        q_latent_THL = torch.einsum(
            "thn,hnl->thl",  # codespell:ignore thn
            q_nope_THN,
            w_uk_HNL,
        )
        q_THG = torch.cat((q_latent_THL, q_shared_THR), dim=-1)
        kv_TG = torch.cat((kv_c_normed_TL, k_shared_TR), dim=-1)
        # TODO: replace with the Attention Gym kernel.
        kv_TSG = kv_TG[compressed_gather_indices(topk_indices_TS, cu_seqlens, 1)]
        logits_THS = (
            torch.einsum(
                "thg,tsg->ths",  # codespell:ignore ths
                q_THG,
                kv_TSG,
            ).to(torch.promote_types(q_THG.dtype, torch.float32))
            * scale
        )
        logits_THS = logits_THS.masked_fill(
            (topk_indices_TS < 0).unsqueeze(1), -torch.inf
        )
        probs_THS = logits_THS.softmax(dim=-1).to(q_THG.dtype)
        out_THL = torch.einsum(
            "ths,tsl->thl",  # codespell:ignore ths
            probs_THS,
            kv_TSG[..., :latent_dim],
        )
        out_THV = torch.einsum("thl,hvl->thv", out_THL, w_uv_HVL)
        if aux_loss is not None:
            assert topk_scores_TS is not None
            if aux_loss_denominator is None:
                raise RuntimeError("SparseIndexerLoss requires aux_loss_denominator.")
            out_THV = aux_loss(
                q_THG.detach(),
                kv_TG.detach(),
                topk_indices_TS,
                topk_scores_TS,
                logits_THS.detach().logsumexp(dim=-1),
                cu_seqlens,
                carrier=out_THV,
                denominator=aux_loss_denominator,
                padding_mask_T=padding_mask,
            )
        return out_THV


class KimiDSAAttention(KimiMLAAttention):
    """Kimi K3 gated MLA with a DeepSeek sparse-attention indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiMLAAttention.Config):
        inner_attention: KimiSparseAttention.Config  # pyrefly: ignore[bad-override]
        indexer: KimiDSAIndexer.Config

        def __post_init__(self) -> None:
            Attention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, KimiSparseAttention.Config):
                raise ValueError(
                    "KimiDSAAttention requires a KimiSparseAttention.Config, "
                    f"but got {type(self.inner_attention).__qualname__}."
                )

        def flops_per_token(self, seq_len: int) -> int:
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            attention_flops = quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                seq_len=min(seq_len, self.inner_attention.index_topk),
            )
            indexer_flops = (
                6 * self.indexer.index_num_heads * self.indexer.index_head_dim * seq_len
            )
            return attention_flops + indexer_flops

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.indexer = config.indexer.build()

    def forward(  # pyrefly: ignore[bad-override]
        self,
        x_TD: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
        *,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        x_TD = maybe_gather_tp_input(self, x_TD)
        q_TC = self.wq_a(x_TD)
        remat.recompute_needs_tensor(q_TC)
        q_normed_TC = self.q_norm(q_TC)
        q_TJ = self.wq_b(q_normed_TC)
        remat.recompute_needs_tensor(q_TJ)
        q_THK = local_head_split(q_TJ, self.q_head_dim, cp_shard_dim=0)

        compressed_kv_TC = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_latent_TC, k_shared_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_c_normed_TL = self.kv_norm(kv_latent_TC)
        idx_q_TIE, idx_k_TE, idx_w_TI = self.indexer(
            q_normed_TC.detach(), x_TD.detach()
        )
        out_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR,
            attention_metadata=attention_metadata,
            scale=self.scale,
            idx_q_TIE=idx_q_TIE,
            idx_k_TE=idx_k_TE,
            idx_w_TI=idx_w_TI,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask=padding_mask,
        )
        gate_TD = self.gate(x_TD)
        remat.recompute_needs_tensor(out_THV, gate_TD)
        out_TD = out_THV.flatten(-2)
        out_TD = out_TD * torch.sigmoid(gate_TD)
        return self.wo(out_TD)
