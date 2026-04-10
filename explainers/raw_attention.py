"""
raw_attention.py  —  E1  RawAttentionExplainer
===============================================
CLS-to-patch attention from the LAST transformer block, averaged over heads.

Reference
---------
Dosovitskiy et al. (2020), "An Image is Worth 16x16 Words."
Head-averaging is the standard baseline for ViT attention visualisation.

Algorithm
---------
1. Hook the last block's ``attn.attn_drop`` module.
2. Forward pass: capture A ∈ (B, H, N+1, N+1) where N = num_patches.
3. Average over heads → A_avg ∈ (N+1, N+1).
4. Extract CLS→patch row: A_cls = A_avg[0, 1:] ∈ (N,).
5. Reshape to (P, P) where P = img_size // patch_size.

Architecture notes
------------------
Requires a global CLS token.  Raises UnsupportedArchitectureError for
Swin-B (no CLS token; window attention).  Use GradCAMExplainer instead.
"""

from __future__ import annotations

import torch

from .base import (
    BaseExplainer,
    UnsupportedArchitectureError,
    _capture_attn_weights,
    _has_cls_token,
    _to_patch_grid,
)


class RawAttentionExplainer(BaseExplainer):
    """
    E1 — Raw CLS-to-patch attention, last layer, head-averaged.

    Parameters
    ----------
    model      : fine-tuned timm ViT / DeiT / BEiT / DINO model.
    patch_size : patch size in pixels (default 16).
    img_size   : input image spatial size (default 224).

    Returns from explain()
    ----------------------
    (P, P) float32 tensor, P = img_size // patch_size.
    Values ∈ [0, 1] (raw attention probabilities, head-averaged).
    """

    def __init__(
        self,
        model:      "torch.nn.Module",
        patch_size: int = 16,
        img_size:   int = 224,
    ) -> None:
        super().__init__(model, patch_size)
        self.img_size = img_size

    def explain(
        self,
        x:            torch.Tensor,
        target_class: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute raw attention attribution for one image.

        Parameters
        ----------
        x            : (3, H, W) float32.
        target_class : ignored (attention is not class-specific).

        Returns
        -------
        (P, P) float32 tensor.
        """
        if not _has_cls_token(self.model):
            raise UnsupportedArchitectureError(
                f"{type(self.model).__name__} has no CLS token. "
                "RawAttentionExplainer requires a global CLS token. "
                "Use GradCAMExplainer for Swin-B."
            )

        # Capture attention from the LAST block (index -1)
        attn_list = _capture_attn_weights(
            self.model, x.unsqueeze(0), block_indices=[-1]
        )
        A = attn_list[0]  # (1, H, N+1, N+1)

        if A is None:
            raise RuntimeError(
                "Attention weights not captured.  Ensure model has 'attn_drop' "
                "and is a timm-style ViT/DeiT/BEiT/DINO model."
            )

        # Average over heads, take the CLS→patch row
        A_avg  = A[0].mean(dim=0)   # (N+1, N+1)
        A_cls  = A_avg[0, 1:]       # (N,) — CLS attends to each patch

        P = self.img_size // self.patch_size
        return _to_patch_grid(A_cls.float(), P)
