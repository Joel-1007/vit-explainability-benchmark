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

from typing import Optional

import torch

from .base import (
    BaseExplainer,
    UnsupportedArchitectureError,
    _capture_attn_weights,
    _has_cls_token,
    _to_patch_grid,
    _num_prefix_tokens,
    _get_timm_blocks,
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

    # Ablation knobs (defaults reproduce E1 exactly):
    #   token  : "cls"   → attention from the CLS token to each patch   (E1)
    #            "patch" → mean attention each patch receives from all
    #                      patch-token queries                          (A1)
    #   layers : number of final blocks whose head-averaged attention is
    #            averaged; 1 = last layer (E1), None = all layers       (A2)
    token:  str = "cls"
    layers: Optional[int] = 1

    def __init__(
        self,
        model:      "torch.nn.Module",
        patch_size: int = 16,
        img_size:   int = 224,
    ) -> None:
        super().__init__(model, patch_size)
        self.img_size = img_size
        if self.token not in ("cls", "patch"):
            raise ValueError(f"token must be 'cls' or 'patch', got {self.token!r}")

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

        n_blocks = len(_get_timm_blocks(self.model))
        k = n_blocks if self.layers is None else min(self.layers, n_blocks)
        attn_list = _capture_attn_weights(
            self.model, x.unsqueeze(0), block_indices=list(range(-k, 0))
        )
        if any(A is None for A in attn_list):
            raise RuntimeError(
                "Attention weights not captured.  Ensure model has 'attn_drop' "
                "and is a timm-style ViT/DeiT/BEiT/DINO model."
            )

        # Average over heads, then over the selected layers
        A_avg = torch.stack([A[0].mean(dim=0) for A in attn_list]).mean(dim=0)  # (T, T)
        n_pre = _num_prefix_tokens(self.model, A_avg.shape[-1])
        if self.token == "cls":
            att = A_avg[0, n_pre:]                    # CLS → each patch
        else:
            att = A_avg[n_pre:, n_pre:].mean(dim=0)   # patches → each patch

        P = self.img_size // self.patch_size
        return _to_patch_grid(att.float(), P)


# ---------------------------------------------------------------------------
# Ablation variants (Phase 4, A1 token resolution / A2 layer depth).
# Subclasses so Phase3Runner can build them as ``cls(model, patch_size=...)``.
# ---------------------------------------------------------------------------

class RawPatchAttentionExplainer(RawAttentionExplainer):
    """A1 — mean patch-token attention received by each patch, last layer."""
    token, layers = "patch", 1


class RawAttentionLast3Explainer(RawAttentionExplainer):
    """A2 — CLS attention averaged over the last 3 layers."""
    token, layers = "cls", 3


class RawAttentionAllLayersExplainer(RawAttentionExplainer):
    """A2 — CLS attention averaged over all layers."""
    token, layers = "cls", None
