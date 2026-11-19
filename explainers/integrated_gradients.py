"""
integrated_gradients.py  —  E8  IntegratedGradientsExplainer
=============================================================
Integrated Gradients (Sundararajan et al., 2017), aggregated to patches.

    IG_i(x) = (x_i − x'_i) · ∫₀¹ ∂f_y(x' + α(x − x')) / ∂x_i  dα

approximated with an ``n_steps``-point Riemann sum (trapezoidal rule), with
baseline x' = 0 in normalised input space (the dataset-mean image) and f_y
the target-class logit. Pixel attributions are summed over channels and
over the pixels of each patch:

    φ_p = Σ_{i ∈ patch p} IG_i(x)

Axiomatic properties (exact for the integral; the Riemann sum converges to
it, and ``completeness_gap`` measures the remaining error):

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
    n_steps    : Riemann-sum points along the path (default 50).
    chunk_size : path points per forward/backward pass (default 25).
    """

    def __init__(
        self,
        model:      "torch.nn.Module",
        patch_size: int = 16,
        img_size:   int = 224,
        n_steps:    int = 50,
        chunk_size: int = 25,
    ) -> None:
        super().__init__(model, patch_size)
        self.img_size   = img_size
        self.n_steps    = n_steps
        self.chunk_size = chunk_size
        self.last_completeness_gap: float | None = None

    def pixel_attributions(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """(C, H, W) Integrated Gradients for one image."""
        x = x.detach()
        baseline = torch.zeros_like(x)
        alphas = torch.linspace(0.0, 1.0, self.n_steps + 1, device=x.device)
        # trapezoidal weights
        w = torch.full_like(alphas, 1.0 / self.n_steps)
        w[0] = w[-1] = 0.5 / self.n_steps

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

    def explain(self, x: torch.Tensor, target_class: int, **kwargs) -> torch.Tensor:
        attr = self.pixel_attributions(x, target_class).sum(dim=0)   # (H, W)
        P, ps = self.img_size // self.patch_size, self.patch_size
        H, W = attr.shape
        if (H, W) != (P * ps, P * ps):
            raise ValueError(f"Input {H}x{W} does not match a {P}x{P} grid of {ps}px patches")
        return attr.view(P, ps, P, ps).sum(dim=(1, 3)).float()        # (P, P)
