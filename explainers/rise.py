"""
rise.py  —  E5  RISEExplainer
==============================
RISE: Randomized Input Sampling for Explanation — vectorised implementation.

Reference
---------
Petsiuk V., Das A., Saenko K. (2018), "RISE: Randomized Input Sampling for
Explanation of Black-box Models."  BMVC 2018.  arXiv:1806.07421.

Algorithm
---------
Pre-generate M random binary masks m_k ∈ {0,1}^{H×W} at small patch-level
resolution, then upsample smoothly to image resolution.

  saliency = Σ_k f_c(x ⊙ m_k) · m_k  /  (M · p)

where f_c is the softmax probability of target class c and p = mask_prob.

Vectorisation strategy
----------------------
All M masks are pre-generated once at ``__init__`` time (in float16 to
reduce memory) and stored as a (M, 1, H, W) buffer.  At inference time,
masks are processed in chunks of ``chunk_size=100``, vectorised across
the batch dimension — giving ~40 GPU forward passes for M=4000, vs. 4000
serial passes without batching.

This is consistent with the guide's explicit vectorisation requirement for
perturbation-based methods.

Gradient handling
-----------------
RISE is a black-box method: ``torch.no_grad()`` is used throughout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseExplainer


class RISEExplainer(BaseExplainer):
    """
    E5 — RISE (Petsiuk et al., BMVC 2018).

    Parameters
    ----------
    model      : fine-tuned nn.Module (treated as black box).
    patch_size : ViT patch size in pixels (default 16).
    img_size   : input image spatial resolution (default 224).
    n_masks    : number of random masks (default 4000 per guide).
    chunk_size : masks processed per GPU forward pass (default 100).
    mask_prob  : probability that each patch/pixel is kept (default 0.5).
    seed       : RNG seed for reproducibility (default 42).

    Production note
    ---------------
    For M=4000 on ImageNet (5,000 val images), RISE makes
    4,000×5,000 = 20 million additional forward passes.  Run on a GPU
    subsample (~500 images) and report in a table footnote.
    """

    def __init__(
        self,
        model:      "torch.nn.Module",
        patch_size: int   = 16,
        img_size:   int   = 224,
        n_masks:    int   = 4000,
        chunk_size: int   = 100,
        mask_prob:  float = 0.5,
        seed:       int   = 42,
    ) -> None:
        super().__init__(model, patch_size)
        self.img_size   = img_size
        self.n_masks    = n_masks
        self.chunk_size = chunk_size
        self.mask_prob  = mask_prob
        self.seed       = seed
        self._masks     = self._pregenerate_masks()

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    def _pregenerate_masks(self) -> torch.Tensor:
        """
        Pre-generate all M masks once.

        Strategy:
          1. Create (M, 1, P, P) binary base at PATCH resolution (small).
          2. Upsample bilinearly to (M, 1, img_size, img_size).
          3. Clamp to [0, 1]; store as float16 to save VRAM.

        Returns
        -------
        (M, 1, img_size, img_size) float16 CPU tensor.
        """
        P    = self.img_size // self.patch_size   # patches per side
        gen  = torch.Generator()
        gen.manual_seed(self.seed)

        # Binary base masks at patch resolution
        base = (
            torch.rand(self.n_masks, 1, P, P, generator=gen) < self.mask_prob
        ).float()

        # Upsample smoothly to image resolution
        masks = F.interpolate(
            base, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        ).clamp(0.0, 1.0)

        return masks.half()   # float16 to halve memory (4000 × 224² ≈ 400 MB fp32)

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(
        self,
        x:            torch.Tensor,
        target_class: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute RISE attribution for one image.

        Parameters
        ----------
        x            : (3, H, W) float32 in [0, 1].
        target_class : target class index.

        Returns
        -------
        (P, P) float32 tensor, values ≥ 0.
        """
        device   = x.device
        H, W     = x.shape[-2:]
        masks    = self._masks.to(device).float()        # (M, 1, H, W)
        saliency = torch.zeros(H, W, device=device)

        with torch.no_grad():
            for start in range(0, self.n_masks, self.chunk_size):
                chunk   = masks[start : start + self.chunk_size]   # (C, 1, H, W)
                masked  = x.unsqueeze(0) * chunk                   # (C, 3, H, W)
                logits  = self.model(masked)                        # (C, num_classes)
                probs   = torch.softmax(logits, dim=-1)[:, target_class]  # (C,)
                # Weighted sum of masks by probability
                saliency += (probs.view(-1, 1, 1) * chunk.squeeze(1)).sum(dim=0)

        saliency = saliency / (self.n_masks * self.mask_prob + 1e-8)
        saliency = saliency.clamp(min=0.0)

        P = self.img_size // self.patch_size

        # If H != img_size, interpolate (handles test images at smaller resolution)
        if H != self.img_size or W != self.img_size:
            saliency = F.interpolate(
                saliency.unsqueeze(0).unsqueeze(0),
                size=(P, P),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        else:
            saliency = F.adaptive_avg_pool2d(
                saliency.unsqueeze(0).unsqueeze(0), (P, P)
            ).squeeze()

        return saliency.float()

    def explain_batch(
        self,
        xs:             torch.Tensor,
        target_classes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Batch explanation.  Masks are shared across images (amortised).
        Calls ``explain`` per image; masks stay on device across calls.
        """
        return torch.stack([
            self.explain(xs[i], int(target_classes[i].item()), **kwargs)
            for i in range(len(xs))
        ])
