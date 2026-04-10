"""
chefer_lrp.py  —  E4  CheferLRPExplainer
=========================================
Attention-aware LRP for Vision Transformers (pure-PyTorch, no external dep).

Reference
---------
Chefer H., Gur S., Wolf L. (2021), "Transformer Interpretability Beyond
Attention Visualization."  CVPR 2021.  arXiv:2012.09838.

Implementation note
-------------------
The official repo (hila-chefer/Transformer-Explainability) is NOT
pip-installable and requires deep modifications to timm model internals.
This self-contained implementation replicates the core LRP rule for ViTs:

  For each block l, the relevance map R is updated by:
      cam_l = ReLU( A_l ⊙ ∂logit / ∂A_l ).mean(heads)
      cam_l += I                    (residual skip-connection)
      cam_l /= cam_l.sum(-1, keepdim=True)
      R      = cam_l @ R            (rollout accumulation)

  attribution = R[CLS, patch_tokens]

This matches Chefer et al. Eq. 5-7 for the ViT case where LRP of the
MLP blocks reduces to identity propagation (standard ε-LRP).

Gradient handling
-----------------
Requires a backward pass → called inside ``torch.enable_grad()``.
Attention tensors are registered with ``.retain_grad()`` during forward
to allow gradient extraction after ``.backward()``.

Architecture notes
------------------
Requires a global CLS token.  Raises UnsupportedArchitectureError for Swin-B.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .base import (
    BaseExplainer,
    UnsupportedArchitectureError,
    _get_timm_blocks,
    _has_cls_token,
    _to_patch_grid,
)


class CheferLRPExplainer(BaseExplainer):
    """
    E4 — Attention-aware LRP (Chefer et al., CVPR 2021).

    Pure-PyTorch self-contained implementation.
    Does not require the external Transformer-Explainability repo.

    Parameters
    ----------
    model      : fine-tuned timm ViT / DeiT / BEiT / DINO model.
    patch_size : patch size in pixels (default 16).
    img_size   : input image spatial resolution (default 224).

    Returns from explain()
    ----------------------
    (P, P) float32 tensor, P = img_size // patch_size.
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
        Compute LRP-based attribution for one image.

        Parameters
        ----------
        x            : (3, H, W) float32.
        target_class : class index whose logit relevance is propagated.

        Returns
        -------
        (P, P) float32 tensor.
        """
        if not _has_cls_token(self.model):
            raise UnsupportedArchitectureError(
                f"{type(self.model).__name__} has no CLS token. "
                "CheferLRPExplainer requires a global CLS token. "
                "Use GradCAMExplainer for Swin-B."
            )

        blocks   = _get_timm_blocks(self.model)
        L        = len(blocks)

        # Step 1: forward pass — register attention tensors + hooks
        attn_tensors: List[torch.Tensor] = []
        grad_tensors: List[torch.Tensor] = []
        fwd_states: dict = {}
        handles = []

        for idx, blk in enumerate(blocks):
            attn_mod = blk.attn
            # Disable fused SDPA
            if getattr(attn_mod, "fused_attn", False):
                fwd_states[idx] = True
                attn_mod.fused_attn = False

            def _make_hook(store: list, grad_store: list):
                def _hook(module, inp, out):
                    # inp[0] = attention weights (B, H, N+1, N+1)
                    attn = inp[0]
                    attn.retain_grad()
                    store.append(attn)
                    # Register gradient hook on this tensor
                    def _grad_hook(g, gs=grad_store):
                        gs.append(g.detach())
                    attn.register_hook(_grad_hook)
                return _hook

            if hasattr(attn_mod, "attn_drop"):
                h = attn_mod.attn_drop.register_forward_hook(
                    _make_hook(attn_tensors, grad_tensors)
                )
            else:
                # Fallback: hook whole attn module
                def _make_fallback(store, gs):
                    def _hook(module, inp, out):
                        if hasattr(module, "_attn_weights"):
                            w = module._attn_weights
                            w.retain_grad()
                            store.append(w)
                            w.register_hook(lambda g, s=gs: s.append(g.detach()))
                    return _hook
                h = attn_mod.register_forward_hook(
                    _make_fallback(attn_tensors, grad_tensors)
                )
            handles.append(h)

        try:
            with torch.enable_grad():
                x_in = x.unsqueeze(0)
                self.model.zero_grad()
                out  = self.model(x_in)
                out[0, target_class].backward()
        finally:
            for h in handles:
                h.remove()
            # Restore fused attention
            for idx, _ in fwd_states.items():
                blocks[idx].attn.fused_attn = True
            self.model.zero_grad()

        # Step 2: LRP rollout — from last block backwards
        device   = x.device
        N_plus_1 = attn_tensors[0].shape[-1] if attn_tensors else 2
        rollout  = torch.eye(N_plus_1, device=device)

        for attn_t, grad_t in zip(attn_tensors, grad_tensors):
            # attn_t: (B, H, N+1, N+1) — attention weights
            # grad_t: (B, H, N+1, N+1) — ∂logit/∂attn_t
            A = attn_t[0].float()    # (H, N+1, N+1)
            G = grad_t[0].float()    # (H, N+1, N+1)

            # Chefer LRP rule: relevance through attention
            cam = (A * G).clamp(min=0).mean(dim=0)   # (N+1, N+1)
            # Residual skip-connection approximation
            eye = torch.eye(N_plus_1, device=device)
            cam = cam + eye
            # Row normalisation
            cam = cam / (cam.sum(dim=-1, keepdim=True).clamp(min=1e-8))
            rollout = cam @ rollout

        # CLS row → patch tokens
        cls_row = rollout[0, 1:]   # (N,)
        P       = self.img_size // self.patch_size
        return _to_patch_grid(cls_row, P)
