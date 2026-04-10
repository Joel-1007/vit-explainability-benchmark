"""
complexity.py  —  Task 2.4 Complexity Metrics
==============================================
Implements three complexity metrics C1–C3 that measure whether an
attribution map is sparse, concentrated, and spatially compact.

C1  gini_coefficient      — Gini concentration index of attribution mass
C2  sparsity              — Fraction of total mass in top-k% pixels
C3  effective_resolution  — Bounding-box area of thresholded attribution

Design principles
-----------------
• Stateless class — identical style to LocalizationMetrics & RobustnessMetrics.
• Pure PyTorch — no scipy, no scikit-image.
• All methods accept raw (H, W) attribution tensors (any device).
• Outputs are Python float in documented range.
• Code style: from __future__ import annotations, NumPy-style docstrings,
  dash-separated section comments.

Usage
-----
    from metrics.complexity import ComplexityMetrics

    cm = ComplexityMetrics(k_fraction=0.05, threshold=0.5)

    g  = cm.gini_coefficient(att_map)          # float in [0, 1]
    s  = cm.sparsity(att_map)                  # float in [0, 1]
    er = cm.effective_resolution(att_map)      # float in (0, 1]

    scores = cm.compute_all(att_map)
    # → {'gini': 0.72, 'sparsity': 0.41, 'effective_resolution': 0.18}
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8   # numerical stability floor


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """
    Squeeze tensor to shape (H, W).

    Accepts (H, W), (1, H, W), (1, 1, H, W), (B=1, 1, H, W).
    Matches the helper in localization.py / robustness.py for consistency.
    """
    t = t.float()
    while t.dim() > 2:
        if t.shape[0] == 1:
            t = t.squeeze(0)
        else:
            raise ValueError(
                f"Cannot reduce tensor of shape {tuple(t.shape)} to 2D — "
                "batch dimension > 1.  Pass one sample at a time."
            )
    return t


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ComplexityMetrics:
    """
    Stateless collection of complexity metrics C1–C3.

    Complexity metrics quantify whether an attribution map is **sparse,
    concentrated, and spatially compact** — properties that measure
    user-facing interpretability.  A map that uniformly attributes all
    pixels is technically valid but practically uninterpretable.

    Parameters
    ----------
    k_fraction : float
        Fraction of total pixels used for C2 (Sparsity).  Default 0.05
        means the top 5% of pixels.  Must be in (0, 1].
    threshold : float
        Binarisation threshold (on min-max normalised attribution) used
        for C3 (EffRes).  Default 0.5.  Must be in (0, 1).
    """

    def __init__(
        self,
        k_fraction: float = 0.05,
        threshold:  float = 0.50,
    ) -> None:
        if not (0.0 < k_fraction <= 1.0):
            raise ValueError(
                f"k_fraction must be in (0, 1], got {k_fraction}."
            )
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold}."
            )
        self.k_fraction = k_fraction
        self.threshold  = threshold

    # ------------------------------------------------------------------
    # C1 — Gini Coefficient
    # ------------------------------------------------------------------

    def gini_coefficient(self, att_map: torch.Tensor) -> float:
        """
        C1 — Gini Coefficient.

        Formal definition (Task 2.4 §1.1)
        -----------------------------------
        Let ẽ be the attribution flattened and shifted to non-negative
        values: ẽ = e - min(e).  Sort ascending: ẽ₁ ≤ ẽ₂ ≤ … ≤ ẽₙ.

            Gini(e) = 1 - (2/n) · Σᵢ (n - i + 0.5) · ẽᵢ / Σⱼ ẽⱼ

        This is the Lorenz-curve formulation equivalent to the double-sum
        definition, numerically stable for large maps.

        Interpretation: Gini = 0 means all pixels have equal attribution
        (perfectly uniform — uninterpretable).  Gini near 1 means nearly
        all attribution mass is concentrated on a single pixel.

        Parameters
        ----------
        att_map : (H, W) tensor — raw attribution (any sign, any scale).

        Returns
        -------
        float in [0, 1].  Higher = more concentrated/sparse attribution.
        """
        e = _to_2d(att_map).float().flatten()

        # Shift to non-negative (Gini is defined for non-negative values)
        e = e - e.min()

        total = e.sum()
        if total < _EPS:
            # Constant-zero map is perfectly uniform → Gini = 0
            return 0.0

        n    = e.numel()
        e, _ = e.sort()                             # ascending

        # Lorenz-curve Gini (vectorised)
        idx     = torch.arange(1, n + 1, dtype=torch.float32, device=e.device)
        # (n - i + 0.5) for i = 1..n  →  (n - idx + 0.5)
        weights = n - idx + 0.5
        gini    = 1.0 - 2.0 / n * float((weights * e).sum().item()) / float(total.item())

        # Clamp to [0, 1] — floating-point can produce tiny negatives
        return float(max(0.0, min(1.0, gini)))

    # ------------------------------------------------------------------
    # C2 — Sparsity (top-k mass fraction)
    # ------------------------------------------------------------------

    def sparsity(
        self,
        att_map:    torch.Tensor,
        k_fraction: float | None = None,
    ) -> float:
        """
        C2 — Sparsity (top-k mass fraction).

        Formal definition (Task 2.4 §1.2)
        -----------------------------------
        Let k = ⌈k_fraction · HW⌉ (minimum 1).  Let ẽ be the attribution
        shifted to non-negative values.

            Sparsity_k(e) = Σ_{p ∈ top-k} ẽₚ / Σₚ ẽₚ

        Interpretation: Sparsity = 1 means all non-negative attribution
        mass is in k pixels.  Sparsity = k_fraction for a uniform map.

        Parameters
        ----------
        att_map    : (H, W) tensor — raw attribution.
        k_fraction : overrides constructor default if provided.

        Returns
        -------
        float in [0, 1].  Higher = more concentrated attribution.
        """
        kf = k_fraction if k_fraction is not None else self.k_fraction
        e  = _to_2d(att_map).float().flatten()

        # Shift to non-negative
        e = e - e.min()

        total = e.sum()
        if total < _EPS:
            return 0.0

        k         = max(1, math.ceil(kf * e.numel()))
        topk_vals = torch.topk(e, k, largest=True).values
        return float(min(1.0, (topk_vals.sum() / total).item()))

    # ------------------------------------------------------------------
    # C3 — Effective Resolution
    # ------------------------------------------------------------------

    def effective_resolution(
        self,
        att_map:   torch.Tensor,
        threshold: float | None = None,
    ) -> float:
        """
        C3 — Effective Resolution.

        Formal definition (Task 2.4 §1.3)
        -----------------------------------
        Let ẽ = MinMax(e) ∈ [0, 1].  Binarise at threshold τ:

            M̂_τ = {p : ẽₚ ≥ τ}

        The effective resolution is the bounding-box area of M̂_τ as a
        fraction of total image area:

            EffRes(e) = |BBox(M̂_τ)| / HW

        where |BBox(·)| is the area of the smallest axis-aligned
        bounding box containing all foreground pixels.

        Interpretation: EffRes near 0 means attribution is spatially
        compact (good).  EffRes = 1 means the bounding box covers the
        entire image (poor compactness).

        Edge cases:
          • No pixel above τ → return 1.0 (worst-case — attribution fully
            diffuse) rather than 0 (which would be misleadingly "good").
          • Single pixel above τ → BBox = 1 pixel → EffRes ≈ 0.

        Parameters
        ----------
        att_map   : (H, W) tensor — raw attribution.
        threshold : overrides constructor default if provided.

        Returns
        -------
        float in (0, 1].  Lower = more spatially compact.
        """
        tau = threshold if threshold is not None else self.threshold
        e   = _to_2d(att_map).float()
        H, W = e.shape
        total = H * W

        # Min-max normalise
        e_min, e_max = e.min(), e.max()
        if (e_max - e_min) < _EPS:
            # Constant map → trivially full bounding box (diffuse)
            return 1.0

        e_norm = (e - e_min) / (e_max - e_min + _EPS)
        mask   = e_norm >= tau

        if not mask.any():
            return 1.0   # nothing above threshold → worst case

        # Axis-aligned bounding box
        rows = mask.any(dim=1).nonzero(as_tuple=True)[0]
        cols = mask.any(dim=0).nonzero(as_tuple=True)[0]

        r_min, r_max = int(rows.min().item()), int(rows.max().item())
        c_min, c_max = int(cols.min().item()), int(cols.max().item())

        bbox_area = (r_max - r_min + 1) * (c_max - c_min + 1)
        return float(min(1.0, bbox_area / total))

    # ------------------------------------------------------------------
    # Convenience: compute C1, C2, C3 in one call
    # ------------------------------------------------------------------

    def compute_all(self, att_map: torch.Tensor) -> Dict[str, float]:
        """
        Compute C1, C2, and C3 in a single call for one attribution map.

        Parameters
        ----------
        att_map : (H, W) tensor — raw attribution.

        Returns
        -------
        dict with keys:
            'gini'                 — C1, float in [0, 1]
            'sparsity'             — C2, float in [0, 1]
            'effective_resolution' — C3, float in (0, 1]
        """
        return {
            "gini":                 self.gini_coefficient(att_map),
            "sparsity":             self.sparsity(att_map),
            "effective_resolution": self.effective_resolution(att_map),
        }
