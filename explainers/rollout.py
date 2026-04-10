"""
rollout.py  —  E2  AttentionRolloutExplainer
============================================
Recursive attention rollout across ALL transformer blocks.

Reference
---------
Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers."
https://arxiv.org/abs/2005.00928

Algorithm
---------
For a model with L blocks, let A_l ∈ (H, N+1, N+1) be the attention matrix
at block l (averaged over heads after dropout).

  R[0] = I                  (identity matrix, (N+1, N+1))
  for l = 0, 1, …, L-1:
      Ã_l = 0.5·Ã_avg(A_l) + 0.5·I   (residual approximation)
      Ã_l /= Ã_l.sum(dim=-1, keepdim=True)   (re-normalise rows)
      R[l+1] = Ã_l @ R[l]
  attribution = R[L][0, 1:]     (CLS row, patch tokens only)

discard_ratio
    Before averaging heads, the ``discard_ratio`` fraction of heads
    with the lowest minimum attention are zeroed out.  This reduces
    noise from heads that attend diffusely.  Default 0.9.

Architecture notes
------------------
Requires a global CLS token.  Raises UnsupportedArchitectureError for
Swin-B.  Use GradCAMExplainer instead.
"""

from __future__ import annotations

import torch

from .base import (
    BaseExplainer,
    UnsupportedArchitectureError,
    _capture_attn_weights,
    _get_timm_blocks,
    _has_cls_token,
    _to_patch_grid,
)


class AttentionRolloutExplainer(BaseExplainer):
    """
    E2 — Attention Rollout (Abnar & Zuidema, 2020).

    Parameters
    ----------
    model         : fine-tuned timm ViT / DeiT / BEiT / DINO model.
    patch_size    : patch size in pixels (default 16).
    img_size      : input image spatial size (default 224).
    discard_ratio : fraction of lowest-attention heads discarded before
                    head-averaging (default 0.9).

    Returns from explain()
    ----------------------
    (P, P) float32 tensor, P = img_size // patch_size.
    Values in [0, 1] after rollout accumulation.
    """

    def __init__(
        self,
        model:        "torch.nn.Module",
        patch_size:   int   = 16,
        img_size:     int   = 224,
        discard_ratio: float = 0.9,
    ) -> None:
        super().__init__(model, patch_size)
        self.img_size      = img_size
        self.discard_ratio = discard_ratio

    def explain(
        self,
        x:            torch.Tensor,
        target_class: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute attention rollout for one image.

        Parameters
        ----------
        x            : (3, H, W) float32.
        target_class : ignored (rollout is not class-specific).

        Returns
        -------
        (P, P) float32 tensor.
        """
        if not _has_cls_token(self.model):
            raise UnsupportedArchitectureError(
                f"{type(self.model).__name__} has no CLS token. "
                "AttentionRolloutExplainer requires a global CLS token. "
                "Use GradCAMExplainer for Swin-B."
            )

        blocks   = _get_timm_blocks(self.model)
        L        = len(blocks)
        all_attn = _capture_attn_weights(
            self.model, x.unsqueeze(0), block_indices=list(range(L))
        )  # list of (1, H, N+1, N+1)

        N_plus_1 = all_attn[0].shape[-1]
        device   = x.device
        rollout  = torch.eye(N_plus_1, device=device)   # (N+1, N+1)

        for A in all_attn:
            if A is None:
                continue
            A = A[0].float()   # (H, N+1, N+1)

            # Head discard: zero out the discard_ratio% lowest-attending heads
            if self.discard_ratio > 0 and A.shape[0] > 1:
                flat_mins = A.min(dim=-1).values.min(dim=-1).values  # (H,)
                threshold = flat_mins.quantile(self.discard_ratio)
                A = A * (flat_mins >= threshold).float().view(-1, 1, 1)

            # Average over surviving heads
            A_avg = A.mean(dim=0)   # (N+1, N+1)

            # Residual connection approximation + re-normalisation
            eye   = torch.eye(N_plus_1, device=device)
            A_res = 0.5 * A_avg + 0.5 * eye
            row_sum = A_res.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            A_res   = A_res / row_sum

            rollout = A_res @ rollout

        # CLS row → patch tokens
        cls_row = rollout[0, 1:]   # (N,)
        P       = self.img_size // self.patch_size
        return _to_patch_grid(cls_row, P)
