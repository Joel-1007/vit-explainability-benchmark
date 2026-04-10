"""
lime.py  —  E6  LIMEExplainer
==============================
LIME adapted for Vision Transformers using ViT patch grid as superpixels.

Reference
---------
Ribeiro M.T., Singh S., Guestrin C. (2016), "Why Should I Trust You?
Explaining the Predictions of Any Classifier."  KDD 2016.

ViT adaptation
--------------
Instead of SLIC superpixels (which require scikit-image), this implementation
uses the ViT PATCH GRID as superpixels.  This is the natural segmentation for
ViTs and avoids scikit-image dependency:
  • For ViT-B/16 (224×224): 14×14 patches = 196 superpixels.
  • For DINO-ViT-B/8  (224×224): 28×28 patches = 784 superpixels.

Algorithm
---------
1. Partition the image into P×P non-overlapping patches (superpixels).
2. Sample N binary perturbation vectors z ∈ {0,1}^S (S = P²).
   z[0] = [1,…,1] (original image always included).
3. For each z: reconstruct image by replacing absent patches with mean colour.
4. Batch all N reconstructed images through the model → class probability.
5. Fit weighted ridge regression:
      w = argmin_w Σ_i π(z_i) · (f(z_i) - w^T z_i)²  +  λ‖w‖²
   where π(z) = exp(-d(z, 1)² / 2σ²) and d = cosine distance to original.
6. Reshape w ∈ R^S to (P, P).

Gradient handling
-----------------
Black-box method (no gradients required).  Uses ``torch.no_grad()``.

Note
----
n_samples=500 is the default (fast).  For production, n_samples ≥ 1000
is recommended for stable regression coefficients.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import BaseExplainer, _to_patch_grid


class LIMEExplainer(BaseExplainer):
    """
    E6 — LIME with ViT patch-grid superpixels (Ribeiro et al., KDD 2016).

    Parameters
    ----------
    model      : fine-tuned nn.Module (treated as black box).
    patch_size : ViT patch size in pixels (default 16).
    img_size   : input image spatial resolution (default 224).
    n_samples  : number of binary perturbation samples (default 500).
    sigma      : cosine-distance kernel bandwidth (default 0.25).
    seed       : RNG seed for reproducible sampling (default 42).
    ridge_alpha: L2 regularisation for weighted ridge regression (default 1e-4).
    """

    def __init__(
        self,
        model:       "torch.nn.Module",
        patch_size:  int   = 16,
        img_size:    int   = 224,
        n_samples:   int   = 500,
        sigma:       float = 0.25,
        seed:        int   = 42,
        ridge_alpha: float = 1e-4,
    ) -> None:
        super().__init__(model, patch_size)
        self.img_size    = img_size
        self.n_samples   = n_samples
        self.sigma       = sigma
        self.seed        = seed
        self.ridge_alpha = ridge_alpha

    # ------------------------------------------------------------------
    # Internal: image reconstruction from binary superpixel mask
    # ------------------------------------------------------------------

    def _reconstruct(
        self,
        x:       torch.Tensor,
        z:       torch.Tensor,
        mean_c:  torch.Tensor,
        P:       int,
        ps:      int,
    ) -> torch.Tensor:
        """
        Reconstruct image from binary superpixel mask z ∈ {0,1}^S.

        Absent patches (z[i]=0) are filled with the per-channel mean colour.

        Parameters
        ----------
        x      : (3, H, W) original image.
        z      : (S,) binary mask, S = P*P.
        mean_c : (3,) per-channel image mean.
        P      : patches per side.
        ps     : patch size in pixels.

        Returns
        -------
        (3, H, W) reconstructed image.
        """
        img = x.clone()
        for i in range(P * P):
            if z[i].item() < 0.5:   # patch is absent
                row = (i // P) * ps
                col = (i %  P) * ps
                img[:, row:row + ps, col:col + ps] = mean_c.view(3, 1, 1)
        return img

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
        Compute LIME attribution for one image.

        Parameters
        ----------
        x            : (3, H, W) float32 in [0, 1].
        target_class : target class index.

        Returns
        -------
        (P, P) float32 tensor with LIME coefficients (can be negative).
        """
        device  = x.device
        P       = self.img_size // self.patch_size
        S       = P * P
        ps      = self.patch_size
        mean_c  = x.mean(dim=(-2, -1))   # (3,) per-channel mean

        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        # Sample binary perturbations: (n_samples, S) ∈ {0,1}
        Z = (torch.rand(self.n_samples, S, generator=gen) > 0.5).float()
        Z[0] = 1.0   # first sample = original image  → max similarity

        # Collect model predictions for all perturbations
        y_vals = torch.zeros(self.n_samples, device=device)
        with torch.no_grad():
            for i in range(self.n_samples):
                recon  = self._reconstruct(x, Z[i], mean_c, P, ps)
                logit  = self.model(recon.unsqueeze(0))
                y_vals[i] = torch.softmax(logit, dim=-1)[0, target_class]

        # LIME kernel: cosine similarity to original (all-1 vector)
        Z_d   = Z.to(device)
        norms = Z_d.norm(dim=-1).clamp(min=1e-8)   # (N,)
        cosim = Z_d.sum(dim=-1) / (norms * math.sqrt(S))  # (N,)
        weights = torch.exp(-(1.0 - cosim) ** 2 / (2 * self.sigma ** 2))  # (N,)

        # Weighted ridge regression: (Z^T W Z + αI)^{-1} Z^T W y
        W_diag = torch.diag(weights)
        ZtWZ   = Z_d.T @ W_diag @ Z_d + self.ridge_alpha * torch.eye(S, device=device)
        ZtWy   = Z_d.T @ (weights * y_vals)

        try:
            coefs = torch.linalg.solve(ZtWZ, ZtWy)   # (S,)
        except RuntimeError:
            # Fallback to lstsq if matrix is singular
            coefs = torch.linalg.lstsq(ZtWZ, ZtWy.unsqueeze(-1)).solution.squeeze()

        return _to_patch_grid(coefs, P)
