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
from torch.distributed._functional_collectives import all_reduce, wait_tensor

from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_mesh_size
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.attn_gym import lightning_indexer
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.tools.utils import has_cuda_capability

__all__ = [
    "KimiDSAAttention",
    "KimiDSAIndexer",
    "KimiIndexerLoss",
    "KimiSparseAttention",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# S = selected token slots
# D = model dimension (dim)
# C = low-rank query channels (q_lora_rank)
# H = main attention head count (n_heads)
# K = main query/key head dimension
# V = main value head dimension
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


class KimiIndexerLoss(AuxLoss):
    """DeepSeek-style KL from sparse main attention to the indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        num_heads: int

        def __post_init__(self) -> None:
            if self.num_heads <= 0:
                raise ValueError("num_heads must be positive.")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.num_heads = config.num_heads

    def forward(
        self,
        topk_scores_TS: torch.Tensor,
        attention_probs_THS: torch.Tensor,
        topk_indices_TS: torch.Tensor,
        *,
        carrier: torch.Tensor,
        denominator: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        work_dtype = (
            torch.float32
            if attention_probs_THS.dtype in (torch.float16, torch.bfloat16)
            else attention_probs_THS.dtype
        )
        with torch.no_grad():
            teacher_TS = attention_probs_THS.detach().to(work_dtype).sum(dim=1)
            tp_group = spmd_mesh_group("tp")
            if tp_group is not None:
                teacher_TS = wait_tensor(all_reduce(teacher_TS, "sum", tp_group))
            teacher_TS = teacher_TS / self.num_heads
            teacher_TS = teacher_TS / teacher_TS.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(work_dtype).tiny
            )

        slot_valid_TS = torch.isfinite(topk_scores_TS) & (topk_indices_TS >= 0)
        if padding_mask is not None:
            slot_valid_TS = slot_valid_TS & ~padding_mask.unsqueeze(1)
        student_input_TS = topk_scores_TS.to(work_dtype).masked_fill(
            ~slot_valid_TS, -torch.inf
        )
        student_input_TS = student_input_TS.masked_fill(
            ~slot_valid_TS.any(dim=-1, keepdim=True), 0.0
        )
        student_log_TS = F.log_softmax(student_input_TS, dim=-1)
        student_log_TS = student_log_TS.masked_fill(~slot_valid_TS, 0.0)
        kl_TS = torch.special.xlogy(teacher_TS, teacher_TS) - (
            teacher_TS * student_log_TS
        )
        raw_sum = kl_TS.masked_fill(~slot_valid_TS, 0.0).sum()
        tp_size = spmd_mesh_size("tp")
        raw_sum = raw_sum.detach() + (raw_sum - raw_sum.detach()) / tp_size
        return self.inject(raw_sum, carrier=carrier, denominator=denominator)


class KimiSparseAttention(InnerAttention):
    """Eager causal attention over indexer-selected token positions."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        index_topk: int
        aux_loss: KimiIndexerLoss.Config | None = None

        def __post_init__(self) -> None:
            if self.index_topk <= 0:
                raise ValueError("index_topk must be positive.")

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
        self.index_topk = config.index_topk
        self.aux_loss = config.aux_loss.build() if config.aux_loss is not None else None

    def _select(
        self,
        idx_q_TIE: torch.Tensor,
        idx_k_TE: torch.Tensor,
        idx_w_TI: torch.Tensor,
        attention_metadata: VarlenAttentionMetadata | None,
        *,
        return_scores: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_tokens = idx_q_TIE.shape[0]
        max_seqlen = (
            num_tokens if attention_metadata is None else attention_metadata.max_k
        )
        cu_seqlens = None if attention_metadata is None else attention_metadata.cu_seq_q
        fused = (
            idx_q_TIE.is_cuda
            and idx_q_TIE.dtype in (torch.float16, torch.bfloat16)
            and has_cuda_capability(9, 0)
        )
        with torch.no_grad():
            local_indices_TS = lightning_indexer(
                idx_q_TIE.unsqueeze(0),
                idx_k_TE.unsqueeze(0),
                idx_w_TI.unsqueeze(0),
                min(self.index_topk, max_seqlen),
                causal=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                impl="fused" if fused else "reference",
            ).squeeze(0)
        if attention_metadata is not None:
            assert cu_seqlens is not None
            document_starts_T = torch.repeat_interleave(
                cu_seqlens[:-1],
                torch.diff(cu_seqlens),
                output_size=num_tokens,
            )
            topk_indices_TS = torch.where(
                local_indices_TS >= 0,
                local_indices_TS + document_starts_T.unsqueeze(1),
                -1,
            )
        else:
            topk_indices_TS = local_indices_TS
        if not return_scores:
            return topk_indices_TS, None

        gather_indices_TS = topk_indices_TS.clamp_min(0).long()
        selected_idx_k_TSE = idx_k_TE[gather_indices_TS]
        logits_TIS = torch.einsum("tie,tse->tis", idx_q_TIE, selected_idx_k_TSE)
        topk_scores_TS = (logits_TIS.relu() * idx_w_TI.unsqueeze(-1)).sum(dim=1)
        topk_scores_TS = topk_scores_TS.masked_fill(topk_indices_TS < 0, -torch.inf)
        return topk_indices_TS, topk_scores_TS

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
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
        topk_indices_TS, topk_scores_TS = self._select(
            idx_q_TIE,
            idx_k_TE,
            idx_w_TI,
            attention_metadata,
            return_scores=aux_loss is not None,
        )
        gather_indices_TS = topk_indices_TS.clamp_min(0).long()
        selected_k_TSHK = k_THK[gather_indices_TS]
        selected_v_TSHV = v_THV[gather_indices_TS]
        logits_THS = (
            torch.einsum(
                "thk,tshk->ths",  # codespell:ignore ths
                q_THK,
                selected_k_TSHK,
            )
            * scale
        )
        logits_THS = logits_THS.masked_fill(
            (topk_indices_TS < 0).unsqueeze(1), -torch.inf
        )
        probs_THS = logits_THS.softmax(dim=-1)
        out_THV = torch.einsum(
            "ths,tshv->thv",  # codespell:ignore ths
            probs_THS,
            selected_v_TSHV,
        )
        if aux_loss is not None:
            assert topk_scores_TS is not None
            if aux_loss_denominator is None:
                raise RuntimeError("KimiIndexerLoss requires aux_loss_denominator.")
            out_THV = aux_loss(
                topk_scores_TS,
                probs_THS,
                topk_indices_TS,
                carrier=out_THV,
                denominator=aux_loss_denominator,
                padding_mask=padding_mask,
            )
        return out_THV


class KimiDSAAttention(KimiMLAAttention):
    """Kimi K3 gated MLA with a DeepSeek sparse-attention indexer."""

    @dataclass(kw_only=True, slots=True)
    class Config(KimiMLAAttention.Config):
        indexer: KimiDSAIndexer.Config

        def flops_per_token(self, seq_len: int) -> int:
            from torchtitan.models.common.flops import (
                quadratic_attention_flops_per_token,
            )

            sparse_attention = self.inner_attention.inner_attention
            assert isinstance(sparse_attention, KimiSparseAttention.Config)
            attention_flops = quadratic_attention_flops_per_token(
                num_heads=self.n_heads,
                qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                seq_len=min(seq_len, sparse_attention.index_topk),
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
        out_THV = self.inner_attention(
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
