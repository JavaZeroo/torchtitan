# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce, wait_tensor

from torchtitan.distributed.spmd_types import spmd_mesh_group, spmd_mesh_size
from torchtitan.models.common.attn_gym import lightning_indexer
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.tools.utils import has_cuda_capability

# Shape suffixes: T = tokens, H = heads, D = head width, K = selected slots,
# S = compressed pool size.


def compressed_cu_seqlens(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Offsets of each document's compressed entries: ``len_d // ratio`` per document.

    A document's trailing partial group forms no entry, matching Attention Gym's
    packed ``cu_seqlens_k`` convention.
    """
    counts = torch.diff(cu_seqlens) // ratio
    return F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))


def compressed_gather_indices(topk_indices_TK, cu_seqlens, ratio):
    """Translate document-local selections into safe compressed-pool indices."""
    gather_TK = topk_indices_TK.clamp_min(0).long()
    if cu_seqlens is not None:
        doc_T = torch.searchsorted(
            cu_seqlens[1:],
            torch.arange(topk_indices_TK.size(0), device=topk_indices_TK.device),
            right=True,
            out_int32=True,
        )
        gather_TK = gather_TK + compressed_cu_seqlens(cu_seqlens, ratio)[doc_T, None]
    # Empty documents can start at the end of the pool; invalid slots read zero.
    return gather_TK.masked_fill(topk_indices_TK < 0, 0)


def select_topk(
    idx_q,
    idx_k,
    idx_w,
    *,
    max_seqlen: int,
    ratio: int,
    topk: int,
    cu_seqlens: torch.Tensor | None = None,
    return_scores: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select top-k compressed positions per folded query token.

    Uses Attention Gym's ``lightning_indexer`` for discrete selection, then
    recomputes the selected logits inside autograd for the distillation loss.
    With packed ``cu_seqlens``, returned indices are local to the query's
    document in the packed compressed-KV pool.

    Returns:
        ``(indices, scores)``. ``indices`` is int32
        ``[T, min(topk, max_seqlen // ratio)]`` with ``-1`` in unusable
        slots. ``scores`` holds live indexer logits for those selected
        entries and ``-inf`` in unusable slots, or None when return_scores
        is False (evaluation and disabled auxiliary loss).
    """
    # The fused kernels need fp16/bf16 on NVIDIA SM90 or newer and do not
    # fall back; CPU, fp32, ROCm, and older GPUs use the reference
    # implementation.
    fused = (
        idx_q.is_cuda
        and idx_q.dtype in (torch.float16, torch.bfloat16)
        and has_cuda_capability(9, 0)
    )
    k = min(topk, max_seqlen // ratio)
    cu_seqlens_k = (
        None if cu_seqlens is None else compressed_cu_seqlens(cu_seqlens, ratio)
    )
    with torch.no_grad():
        topk_indices = lightning_indexer(
            idx_q.unsqueeze(0),
            idx_k.unsqueeze(0),
            idx_w.unsqueeze(0),
            k,
            causal=True,
            compress_ratio=ratio,
            cu_seqlens=cu_seqlens,
            cu_seqlens_k=cu_seqlens_k,
            impl="fused" if fused else "reference",
        ).squeeze(0)

    if not return_scores:
        return topk_indices, None
    valid_TK = topk_indices >= 0
    gather_indices_TK = compressed_gather_indices(topk_indices, cu_seqlens, ratio)

    selected_TKD = idx_k[gather_indices_TK]
    logits_THK = torch.einsum("thd,tkd->thk", idx_q, selected_TKD)
    topk_scores = (logits_THK.relu() * idx_w.unsqueeze(-1)).sum(dim=1)
    topk_scores = topk_scores.masked_fill(~valid_TK, -torch.inf)
    return topk_indices, topk_scores


class SparseIndexerLoss(AuxLoss):
    """Distill selected compressed attention using the main kernel's LSE.

    The head-aggregated, L1-normalized target follows DeepSeek-V3.2 Sec. 2.1
    (https://arxiv.org/html/2512.02556v1#S2.SS1.SSS1), inherited by V4's CSA.
    Optional mass weighting multiplies each row's KL by its mean compressed
    attention mass: rows dominated by the window or sink contribute less.
    This weighting is an implementation choice, not a claim about the paper's
    objective. The default is unweighted normalized KL; set mass_weighted=True
    to enable mass weighting explicitly.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(AuxLoss.Config):
        """Aux-loss fields plus sparse-attention geometry."""

        softmax_scale: float
        num_heads: int
        compress_ratio: int = 4
        chunk_size: int = 128
        mass_weighted: bool = False

        def __post_init__(self):
            if self.num_heads <= 0 or self.chunk_size <= 0 or self.compress_ratio <= 0:
                raise ValueError(
                    "num_heads, chunk_size and compress_ratio must be positive."
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.softmax_scale = config.softmax_scale
        self.num_heads = config.num_heads
        self.compress_ratio = config.compress_ratio
        self.chunk_size = config.chunk_size
        self.mass_weighted = config.mass_weighted

    @torch.no_grad()
    def _teacher(
        self,
        q_THD: torch.Tensor,
        cmp_k_SD: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        attn_lse_TH: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tp_size = spmd_mesh_size("tp")
        if q_THD.size(1) * tp_size != self.num_heads:
            raise ValueError("Teacher head count does not match the TP-sharded query.")
        gather_TK = compressed_gather_indices(
            topk_indices_TK, cu_seqlens, self.compress_ratio
        )
        targets = []
        # Bound gathered keys and fp32 logits by chunk_size, not context length.
        with torch.autocast(device_type=q_THD.device.type, enabled=False):
            for start in range(0, q_THD.size(0), self.chunk_size):
                stop = start + self.chunk_size
                valid_TK = topk_indices_TK[start:stop] >= 0
                if padding_mask_T is not None:
                    valid_TK = valid_TK & ~padding_mask_T[start:stop, None]
                keys_TKD = cmp_k_SD[gather_TK[start:stop]].float()
                logits_THK = (
                    torch.einsum("thd,tkd->thk", q_THD[start:stop].float(), keys_TKD)
                    * self.softmax_scale
                )
                # The main kernel includes window, compressed slots and sink.
                logits_THK = logits_THK - attn_lse_TH[start:stop, :, None].float()
                logits_THK = logits_THK.masked_fill(~valid_TK[:, None], -torch.inf)
                targets.append(logits_THK.exp().sum(dim=1) / self.num_heads)
        p_TK = torch.cat(targets)
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            p_TK = wait_tensor(all_reduce(p_TK, "sum", tp_group))
        return p_TK

    def _logged_kl(
        self,
        p_TK: torch.Tensor,
        t_TK: torch.Tensor,
        logits_TK: torch.Tensor,
        slot_valid_TK: torch.Tensor,
    ) -> torch.Tensor:
        log_student_TK = F.log_softmax(
            logits_TK.masked_fill(~slot_valid_TK, -torch.inf).masked_fill(
                ~slot_valid_TK.any(dim=-1, keepdim=True), 0.0
            ),
            dim=-1,
        )
        weight_TK = p_TK if self.mass_weighted else t_TK
        log_student_TK = log_student_TK.masked_fill(~slot_valid_TK, 0.0)
        weighted_TK = torch.special.xlogy(weight_TK, t_TK) - weight_TK * log_student_TK
        return weighted_TK.masked_fill(~slot_valid_TK, 0.0)

    def forward(
        self,
        q_THD: torch.Tensor,
        cmp_k_SD: torch.Tensor,
        topk_indices_TK: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        attn_lse_TH: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        *,
        carrier: torch.Tensor,
        denominator: torch.Tensor,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss and inject its gradient on ``carrier``."""
        if topk_scores_TK.numel() == 0:
            return carrier

        p_TK = self._teacher(
            q_THD,
            cmp_k_SD,
            topk_indices_TK,
            attn_lse_TH,
            cu_seqlens,
            padding_mask_T,
        )
        tiny = torch.finfo(torch.float32).tiny
        t_TK = p_TK / p_TK.sum(dim=-1, keepdim=True).clamp_min(tiny)

        slot_valid_TK = torch.isfinite(topk_scores_TK)
        if padding_mask_T is not None:
            slot_valid_TK = slot_valid_TK & ~padding_mask_T[:, None]
        raw_sum = self._logged_kl(
            p_TK, t_TK, topk_scores_TK.float(), slot_valid_TK
        ).sum()
        # TP-replicated students sum their gradients. Keep the globally correct
        # metric, but contribute only 1/TP of its gradient on each rank.
        tp_size = spmd_mesh_size("tp")
        raw_sum = raw_sum.detach() + (raw_sum - raw_sum.detach()) / tp_size
        return self.inject(raw_sum, carrier=carrier, denominator=denominator)
