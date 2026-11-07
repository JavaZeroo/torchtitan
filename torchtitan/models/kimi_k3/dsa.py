# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3.1 Gated DeepSeek Sparse Attention."""

from dataclasses import dataclass
from functools import cache

import torch
import torch.nn.functional as F
import torch_remat as remat
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.base.module import Module
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.attention import (
    Attention,
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.linear import maybe_gather_tp_input
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss

from .attention import KimiMLAAttention

__all__ = ["KimiDSAIndexer", "KimiDSALatentAttention", "KimiGatedDSAAttention"]

# Shape suffixes:
# T = query tokens, S = candidate tokens, D = model/indexer head dimension,
# H = attention/indexer heads, L = MLA KV latent dimension,
# N = per-head non-positional key dimension, R = shared key dimension,
# V = per-head value dimension, K = selected tokens.


@cache
def _hadamard_matrix(
    dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if dim & (dim - 1):
        raise ValueError("Hadamard dimension must be a power of two.")
    matrix = torch.ones((1, 1), dtype=dtype, device=device)
    while matrix.shape[0] < dim:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return matrix


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    dim = x.size(-1)
    matrix = _hadamard_matrix(dim, x.dtype, x.device)
    if isinstance(x, DTensor):
        matrix = DTensor.from_local(
            matrix,
            x.device_mesh,
            [Replicate()] * x.device_mesh.ndim,
            run_check=False,
        )
    return F.linear(x, matrix) * (dim**-0.5)


class KimiDSAIndexer(Module):
    """NoPE lightning indexer used by Kimi K3.1 DSA layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        head_dim: int
        wq_b: Linear.Config
        wk: Linear.Config
        k_norm: LayerNorm.Config
        weights_proj: Linear.Config
        apply_hadamard: bool = True

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.apply_hadamard = config.apply_hadamard
        self.wq_b = config.wq_b.build()
        self.wk = config.wk.build()
        self.k_norm = config.k_norm.build()
        self.weights_proj = config.weights_proj.build()

    def forward(
        self, x_TD: torch.Tensor, q_latent_TC: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = x_TD.shape[0]
        idx_q_THD = self.wq_b(q_latent_TC).view(num_tokens, self.n_heads, self.head_dim)
        idx_k_TD = self.k_norm(self.wk(x_TD))
        # The reference CUDA path produces FP32 head gates directly from BF16
        # operands rather than rounding the projection output to BF16 first.
        idx_w_TH = F.linear(x_TD.float(), self.weights_proj.weight.float()) * (
            self.n_heads**-0.5 * self.head_dim**-0.5
        )
        if self.apply_hadamard:
            idx_q_THD = _hadamard(idx_q_THD)
            idx_k_TD = _hadamard(idx_k_TD)
        remat.recompute_needs_tensor(idx_q_THD, idx_k_TD)
        return idx_q_THD, idx_k_TD, idx_w_TH


def _document_ids(offsets: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return torch.searchsorted(
        offsets[1:],
        torch.arange(num_tokens, device=offsets.device, dtype=torch.int32),
        right=True,
        out_int32=True,
    )


def _select_topk(
    idx_q_THD: torch.Tensor,
    idx_k_SD: torch.Tensor,
    idx_w_TH: torch.Tensor,
    *,
    top_k: int,
    attention_metadata: VarlenAttentionMetadata | None,
    padding_mask_T: torch.Tensor | None,
    return_scores: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    num_queries, num_candidates = idx_q_THD.shape[0], idx_k_SD.shape[0]
    selected_width = min(top_k, num_candidates)
    if selected_width == 0:
        empty_indices = torch.empty(
            num_queries, 0, dtype=torch.int32, device=idx_q_THD.device
        )
        return empty_indices, None

    with torch.no_grad():
        scores_TS = (
            torch.einsum("txd,sd->txs", idx_q_THD.float(), idx_k_SD.float())
            .relu()
            .mul(idx_w_TH.float().unsqueeze(-1))
            .sum(dim=1)
        )
        query_positions = torch.arange(num_queries, device=scores_TS.device)
        key_positions = torch.arange(num_candidates, device=scores_TS.device)
        valid_TS = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        if attention_metadata is not None:
            query_docs = _document_ids(attention_metadata.cu_seq_q, num_queries)
            key_docs = _document_ids(attention_metadata.cu_seq_k, num_candidates)
            valid_TS = valid_TS & (query_docs.unsqueeze(1) == key_docs.unsqueeze(0))
        if padding_mask_T is not None:
            valid_TS = valid_TS & ~padding_mask_T.unsqueeze(1)
            valid_TS = valid_TS & ~padding_mask_T.unsqueeze(0)
        scores_TS = scores_TS.masked_fill(~valid_TS, -torch.inf)
        selected_scores_TS, topk_indices_TK = torch.topk(
            scores_TS, selected_width, dim=-1, sorted=True
        )
        topk_indices_TK = topk_indices_TK.to(torch.int32)
        topk_indices_TK = topk_indices_TK.masked_fill(
            ~torch.isfinite(selected_scores_TS), -1
        )

    if not return_scores:
        return topk_indices_TK, None

    gather_indices_TK = topk_indices_TK.clamp_min(0).long()
    selected_k_TKD = idx_k_SD[gather_indices_TK]
    selected_logits_THK = torch.einsum(
        "thd,tkd->thk", idx_q_THD.float(), selected_k_TKD.float()
    )
    topk_scores_TK = (
        selected_logits_THK.relu()
        .mul(idx_w_TH.float().unsqueeze(-1))
        .sum(dim=1)
        .masked_fill(topk_indices_TK < 0, -torch.inf)
    )
    return topk_indices_TK, topk_scores_TK


class KimiDSALatentAttention(InnerAttention):
    """Eager sparse attention over shared MLA latent K/V entries."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        n_heads: int
        kv_lora_rank: int
        qk_nope_head_dim: int
        qk_rope_head_dim: int
        v_head_dim: int
        index_topk: int
        wkv_b: Linear.Config
        aux_loss: SparseIndexerLoss.Config | None = None

        @property
        def is_full_attention(self) -> bool:
            return True

        @property
        def softmax_scale(self) -> float:
            return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

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
        self.n_heads = config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = config.softmax_scale
        self.wkv_b = config.wkv_b.build()
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        *,
        idx_q_THD: torch.Tensor,
        idx_k_SD: torch.Tensor,
        idx_w_TH: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata,
        aux_loss_denominator: torch.Tensor | None = None,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_dtype = q_THK.dtype
        num_tokens, num_local_heads = q_THK.shape[:2]
        q_nope_THN, q_rope_THR = torch.split(
            q_THK, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        wkv_HPL = self.wkv_b.weight.view(
            num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_uk_HNL, w_uv_HVL = wkv_HPL.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=1
        )
        q_latent_THL = torch.einsum("txn,xnl->txl", q_nope_THN, w_uk_HNL)
        q_latent_THK = torch.cat((q_latent_THL, q_rope_THR), dim=-1).float()
        latent_k_TC = torch.cat((kv_c_normed_TL, k_shared_TR), dim=-1).float()

        topk_indices_TK, topk_scores_TK = _select_topk(
            idx_q_THD,
            idx_k_SD,
            idx_w_TH,
            top_k=self.index_topk,
            attention_metadata=attention_metadata,
            padding_mask_T=padding_mask_T,
            return_scores=self.training and self.aux_loss is not None,
        )
        if topk_indices_TK.shape[1] == 0:
            return q_THK.new_zeros(num_tokens, num_local_heads, self.v_head_dim)

        gather_indices_TK = topk_indices_TK.clamp_min(0).long()
        selected_k_TKC = latent_k_TC[gather_indices_TK]
        logits_THK = (
            torch.einsum("thc,tkc->thk", q_latent_THK, selected_k_TKC)
            * self.softmax_scale
        )
        valid_TK = topk_indices_TK >= 0
        logits_THK = logits_THK.masked_fill(~valid_TK.unsqueeze(1), -torch.inf)
        lse_TH = torch.logsumexp(logits_THK, dim=-1)
        safe_lse_TH = torch.where(torch.isfinite(lse_TH), lse_TH, 0.0)
        probs_THK = torch.exp(logits_THK - safe_lse_TH.unsqueeze(-1))
        selected_v_TKL = kv_c_normed_TL.float()[gather_indices_TK]
        out_latent_THL = torch.einsum("thk,tkl->thl", probs_THK, selected_v_TKL)
        out_THV = torch.einsum("thl,hvl->thv", out_latent_THL, w_uv_HVL.float()).to(
            input_dtype
        )

        if self.training and self.aux_loss is not None:
            if aux_loss_denominator is None:
                raise RuntimeError("SparseIndexerLoss requires aux_loss_denominator.")
            out_THV = self.aux_loss(
                q_latent_THK.detach(),
                latent_k_TC.detach(),
                topk_indices_TK,
                topk_scores_TK,
                lse_TH.detach(),
                None,
                carrier=out_THV,
                denominator=aux_loss_denominator,
                padding_mask_T=padding_mask_T,
            )
        return out_THV


class KimiGatedDSAAttention(KimiMLAAttention):
    """Kimi K3 MLA projections and output gate with DSA token selection."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiMLAAttention.Config):
        indexer: KimiDSAIndexer.Config
        inner_attention: KimiDSALatentAttention.Config  # pyrefly: ignore[bad-override]

        def __post_init__(self) -> None:
            Attention.Config.__post_init__(self)
            if not isinstance(self.inner_attention, KimiDSALatentAttention.Config):
                raise ValueError("Kimi DSA requires KimiDSALatentAttention.Config.")

        def flops_per_token(self, seq_len: int) -> int:
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            inner = self.inner_attention
            attention_flops = quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=inner.kv_lora_rank + inner.qk_rope_head_dim,
                v_head_dim=inner.kv_lora_rank,
                seq_len=min(seq_len, inner.index_topk),
            )
            indexer_flops = 6 * self.indexer.n_heads * self.indexer.head_dim * seq_len
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
        q_THK = local_head_split(
            self.wq_b(q_normed_TC), self.q_head_dim, cp_shard_dim=0
        )
        compressed_kv_TC = self.wkv_a(x_TD)
        remat.recompute_needs_tensor(compressed_kv_TC)
        kv_latent_TC, k_shared_TR = torch.split(
            compressed_kv_TC,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv_c_normed_TL = self.kv_norm(kv_latent_TC)
        idx_q_THD, idx_k_SD, idx_w_TH = self.indexer(
            x_TD.detach(), q_normed_TC.detach()
        )
        out_THV = self.inner_attention(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR,
            idx_q_THD=idx_q_THD,
            idx_k_SD=idx_k_SD,
            idx_w_TH=idx_w_TH,
            attention_metadata=attention_metadata,
            aux_loss_denominator=aux_loss_denominator,
            padding_mask_T=padding_mask,
        )
        gate_TD = self.gate(x_TD)
        remat.recompute_needs_tensor(out_THV, gate_TD)
        out_TD = out_THV.flatten(-2) * torch.sigmoid(gate_TD)
        return self.wo(out_TD)
