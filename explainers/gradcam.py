"""
gradcam.py  —  E3  GradCAMExplainer
=====================================
GradCAM adapted for Vision Transformers.

Reference
---------
Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization."  Adapted for ViT patch embeddings.

Algorithm
---------
Let F ∈ (B, N, D) be the output activations of the LAST transformer block
(patch tokens only, CLS excluded) and let g ∈ (B, N, D) be the gradient of
the target-class logit w.r.t. those activations.

Global-average-pool gradients over patch dimension:
    α_d = mean_n(g[b=0, n, d])     d = 1..D

Weighted combination then ReLU:
    CAM_n = ReLU( Σ_d α_d · F[0, n, d] )   n = 1..N

Reshape (N,) → (P, P) where P = img_size // patch_size.

Gradient handling
-----------------
GradCAM requires gradients.  The model is called inside ``torch.enable_grad()``
with its own ``zero_grad()`` cycle.  The outer ``BenchmarkRunner`` must NOT wrap
GradCAM calls in ``torch.no_grad()``.

Architecture notes
------------------
GradCAM works on ANY model with a block-structured backbone, including Swin-B.
It is the required fallback for models without a CLS token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseExplainer, _get_timm_blocks, _to_patch_grid


class GradCAMExplainer(BaseExplainer):
    """
    E3 — GradCAM on ViT / Swin patch embeddings.

    Parameters
    ----------
    model      : fine-tuned timm model (ViT, DeiT, BEiT, DINO, Swin-B …).
    patch_size : ViT patch size in pixels (default 16).
    img_size   : input image spatial resolution (default 224).

    Returns from explain()
    ----------------------
    (P, P) float32 tensor, P = img_size // patch_size.
    Values ≥ 0 (ReLU applied).
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
        Compute GradCAM attribution for one image.

        Parameters
        ----------
        x            : (3, H, W) float32.
        target_class : target class index for gradient computation.

        Returns
        -------
        (P, P) float32 tensor with ReLU-gated attribution.
        """
        blocks = _get_timm_blocks(self.model)
        target_block = blocks[-1]

        activations = [None]
        gradients   = [None]

        fwd_h = target_block.register_forward_hook(
            lambda m, inp, out: activations.__setitem__(0, out)
        )
        bwd_h = target_block.register_full_backward_hook(
            lambda m, g_in, g_out: gradients.__setitem__(0, g_out[0])
        )

        try:
            with torch.enable_grad():
                x_in = x.unsqueeze(0).detach()
                self.model.zero_grad()
                out  = self.model(x_in)
                out[0, target_class].backward()
        finally:
            fwd_h.remove()
            bwd_h.remove()
            self.model.zero_grad()

        act  = activations[0]  # (1, N+1, D)  or (1, N, D) for Swin-B
        grad = gradients[0]    # same shape

        if act is None or grad is None:
            raise RuntimeError(
                "GradCAM: activations or gradients not captured from "
                f"last block ({type(target_block).__name__}).  "
                "Ensure the model is not in 'no_grad' mode at call time."
            )

        act  = act.float()
        grad = grad.float()

        # Drop CLS token from ViT-style models (first token)
        if act.shape[1] > (self.img_size // self.patch_size) ** 2:
            act  = act[:, 1:, :]    # (1, N, D)
            grad = grad[:, 1:, :]

        act  = act[0]   # (N, D)
        grad = grad[0]  # (N, D)

        # GradCAM: global-average-pool gradients over patches → importance per feature
        alpha = grad.mean(dim=0)             # (D,)
        cam   = torch.relu((act * alpha).sum(dim=-1))  # (N,)

        P = self.img_size // self.patch_size

        # Handle Swin-B: spatial tokens may be flattened from H/32 × W/32 feature map
        if cam.numel() != P * P:
            # Interpolate to target grid
            cam = F.interpolate(
                cam.view(1, 1, -1),
                size=P * P,
                mode="linear",
                align_corners=False,
            ).squeeze()

        return _to_patch_grid(cam, P)
