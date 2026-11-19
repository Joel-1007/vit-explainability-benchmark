"""
integrated_gradients.py  —  E8  IntegratedGradientsExplainer
=============================================================
Integrated Gradients (Sundararajan et al., 2017), aggregated to patches.

    IG_i(x) = (x_i − x'_i) · ∫₀¹ ∂f_y(x' + α(x − x')) / ∂x_i  dα

approximated with an ``n_steps``-node **Gauss–Legendre** quadrature on α ∈ (0, 1)
(default 400 nodes; Captum's default method), with baseline x' = 0 in [0, 1] pixel space (the **black** image; the model applies
its own normalisation, so the baseline is the same for every model) and f_y
the target-class logit. Pixel attributions are summed over channels and
over the pixels of each patch:

    φ_p = Σ_{i ∈ patch p} IG_i(x)

Quadrature (verified on real ViT-B/16 weights, 2026-11-19). Near the black
baseline the path gradient of a pre-trained ViT is extremely spiky (directional
derivative +2.2 at α = 0, −23.4 at α = 0.02, smooth beyond α ≈ 0.3; autograd
matches finite differences). A uniform trapezoidal grid misses this feature and
does not converge monotonically (completeness gaps 28% / 146% / 47% at 50 / 100 /
200 steps); Gauss–Legendre reaches 0.5–2.2% at 200 nodes on ViT-B/16, and DeiT-B
needs 400 (9.4% → 0.0% / 0.2%), hence the 400-node default. ``method="trapezoid"``
keeps the uniform rule for comparison.

Axiomatic properties (exact for the integral; the quadrature converges to
it, and ``last_completeness_gap`` measures the remaining error):

  • Completeness (A2): Σ_p φ_p = f_y(x) − f_y(x')  — summing over patches
    preserves it.
  • Dummy (A1): a patch the model does not depend on gets φ_p = 0.
  • Linearity (A4): IG of α·g + β·h is α·IG(g) + β·IG(h).
  • Symmetry-preserving (A3) for inputs symmetric in the function and equal
    in x and x'.

Model-agnostic: works on Swin-B as well (only gradients are needed).
Attributions are signed; the benchmark's normalisation maps them to [0, 1].
"""

from __future__ import annotations

import torch

from .base import BaseExplainer


class IntegratedGradientsExplainer(BaseExplainer):
    """
    E8 — Integrated Gradients, patch-aggregated.

    Parameters
    ----------
    model      : fine-tuned classifier.
    patch_size : attribution-grid cell size in pixels (default 16).
    img_size   : input resolution (default 224).
    n_steps    : quadrature nodes along the path (default 400).
    chunk_size : path points per forward/backward pass (default 25).
    method     : "gausslegendre" (default) or "trapezoid".
    """

    def __init__(
        self,
        model:      "torch.nn.Module",
        patch_size: int = 16,
        img_size:   int = 224,
        n_steps:    int = 400,
        chunk_size: int = 25,
        method:     str = "gausslegendre",
    ) -> None:
        super().__init__(model, patch_size)
        if method not in ("gausslegendre", "trapezoid"):
            raise ValueError(f"method must be 'gausslegendre' or 'trapezoid', got {method!r}")
        self.img_size   = img_size
        self.n_steps    = n_steps
        self.chunk_size = chunk_size
        self.method     = method
        self.last_completeness_gap: float | None = None

    def pixel_attributions(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """(C, H, W) Integrated Gradients for one image."""
        x = x.detach()
        baseline = torch.zeros_like(x)
        alphas, w = self.quadrature(self.n_steps, self.method, x.device)

        grad_sum = torch.zeros_like(x)
        with torch.enable_grad():
            for start in range(0, len(alphas), self.chunk_size):
                a = alphas[start:start + self.chunk_size].view(-1, 1, 1, 1)
                path = (baseline + a * (x - baseline)).requires_grad_(True)
                logits = self.model(path)[:, target_class]
                grads, = torch.autograd.grad(logits.sum(), path)
                grad_sum += (grads * w[start:start + self.chunk_size].view(-1, 1, 1, 1)).sum(0)
            with torch.no_grad():
                f = self.model(torch.stack([x, baseline]))[:, target_class]
        attr = (x - baseline) * grad_sum
        self.last_completeness_gap = float(attr.sum() - (f[0] - f[1]))
        return attr.detach()

    @staticmethod
    def quadrature(n: int, method: str, device=None):
        """Nodes α ∈ [0, 1] and weights (sum to 1) for ∫₀¹ g(α) dα."""
        if method == "trapezoid":
            a = torch.linspace(0.0, 1.0, n + 1)
            w = torch.full_like(a, 1.0 / n)
            w[0] = w[-1] = 0.5 / n
        else:
            import numpy as np
            nodes, weights = np.polynomial.legendre.leggauss(n)
            a = torch.tensor((nodes + 1) / 2, dtype=torch.float32)
            w = torch.tensor(weights / 2, dtype=torch.float32)
        return a.to(device), w.to(device)

    def explain(self, x: torch.Tensor, target_class: int, **kwargs) -> torch.Tensor:
        attr = self.pixel_attributions(x, target_class).sum(dim=0)   # (H, W)
        P, ps = self.img_size // self.patch_size, self.patch_size
        H, W = attr.shape
        if (H, W) != (P * ps, P * ps):
            raise ValueError(f"Input {H}x{W} does not match a {P}x{P} grid of {ps}px patches")
        return attr.view(P, ps, P, ps).sum(dim=(1, 3)).float()        # (P, P)
