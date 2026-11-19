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

from .base import BaseExplainer, _get_timm_blocks, _num_prefix_tokens


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

        act  = act[0].float()   # (T, D) for ViT-style, (h, w, D) for Swin
        grad = grad[0].float()

        if act.dim() == 3:
            # Swin (timm, NHWC): last stage is an (h, w, D) feature map.
            h, w = act.shape[0], act.shape[1]
            act  = act.reshape(h * w, -1)
            grad = grad.reshape(h * w, -1)
        else:
            # ViT-style: strip prefix tokens (CLS; DeiT also has a
            # distillation token) so only the h x w patch tokens remain.
            n_prefix = _num_prefix_tokens(self.model, act.shape[0])
            act, grad = act[n_prefix:], grad[n_prefix:]
            h = w = int(round(act.shape[0] ** 0.5))
            if h * w != act.shape[0]:
                raise RuntimeError(
                    f"GradCAM: {act.shape[0]} patch tokens do not form a square grid."
                )

        # GradCAM: global-average-pool gradients over patches → importance per feature
        alpha = grad.mean(dim=0)                       # (D,)
        cam   = torch.relu((act * alpha).sum(dim=-1))  # (h*w,)
        cam   = cam.view(h, w)

        P = self.img_size // self.patch_size
        if (h, w) != (P, P):
            # Resample the native token grid (e.g. Swin's 7x7) to the
            # benchmark's P x P attribution grid, preserving 2-D layout.
            cam = F.interpolate(
                cam[None, None], size=(P, P), mode="bilinear", align_corners=False,
            )[0, 0]

        return cam

