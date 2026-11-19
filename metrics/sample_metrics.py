"""
sample_metrics.py — per-sample metric computation for Phase3Runner.

Computes, for every image of a (dataset, model, explainer) combination:

    localization  L1 iou@τ / miou, L2 pointing_game, L3 egt   (needs gt_mask)
    complexity    C1 gini, C2 entropy_norm, C3 emr_50/90/95, k_star_90
    fidelity      F1 sufficiency@k, F2 comprehensiveness@k, F3 log_odds_drop@k
    robustness    R1 max_sensitivity, R2 model_randomisation, R3 label_randomisation
                  (first ``robustness_subset`` images only — it re-runs the
                  explainer 2 + n_samples times per image)

plus ``pred`` and ``correct``. L4 (CalibGap) is a dataset-level statistic,
computed from ``egt`` and ``correct`` in :mod:`metrics.aggregate`.

All metrics receive the *normalised* attribution map (Phase 3 Task 3.2), so
values are comparable across explainers whose raw scales differ.
Fidelity targets the explained class (the ground-truth label).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .complexity import ComplexityMetrics
from .fidelity import FidelityMetrics
from .localization import LocalizationMetrics
from .normalize import normalize_attribution
from .robustness import (
    RobustnessMetrics,
    randomise_classifier_labels,
    randomise_model_weights,
)

FAMILIES = ("localization", "complexity", "fidelity", "robustness")

_COMPLEXITY_KEYS = ("gini", "entropy_norm", "emr_50", "emr_90", "emr_95", "k_star_90")

# +1 = higher is better, -1 = lower is better. Matched by exact name, then prefix.
METRIC_DIRECTIONS: Dict[str, int] = {
    "miou": +1, "iou@": +1, "pointing_game": +1, "egt": +1, "calibration_gap": +1,
    "gini": +1, "entropy_norm": -1, "emr_": -1, "k_star_90": -1,
    "sufficiency@": -1,          # f(x) - f(x_keep): small = kept patches suffice
    "comprehensiveness@": +1,    # f(x) - f(x_drop): large = dropped patches mattered
    "log_odds_drop@": +1,
    "max_sensitivity": -1, "model_randomisation": +1, "label_randomisation": +1,
}


def metric_direction(name: str) -> Optional[int]:
    """+1 / -1 for a known metric column, None otherwise."""
    if name in METRIC_DIRECTIONS:
        return METRIC_DIRECTIONS[name]
    for key, d in METRIC_DIRECTIONS.items():
        if key.endswith(("@", "_")) and name.startswith(key):
            return d
    return None


class SampleMetricComputer:
    """
    Configuration for per-sample metrics. ``bind(explainer)`` returns a
    :class:`BoundMetrics` for one combination (it holds the randomised model
    copies needed by R2/R3, so create one per combination).
    """

    def __init__(
        self,
        families:            Iterable[str] = FAMILIES,
        norm_mode:           str   = "minmax",
        thresholds:          Sequence[float] = (0.25, 0.50, 0.75),
        fidelity_k:          Sequence[float] = (0.1, 0.2, 0.5),
        fidelity_mask_mode:  str   = "mean",
        robustness_subset:   Optional[int] = 100,
        robustness_samples:  int   = 20,
        robustness_epsilon:  float = 0.05,
        seed:                int   = 42,
    ) -> None:
        self.families = tuple(families)
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"Unknown metric families {sorted(unknown)}; choose from {FAMILIES}")
        self.norm_mode          = norm_mode
        self.thresholds         = tuple(thresholds)
        self.fidelity_k         = tuple(fidelity_k)
        self.fidelity_mask_mode = fidelity_mask_mode
        self.robustness_subset  = robustness_subset
        self.robustness_samples = robustness_samples
        self.robustness_epsilon = robustness_epsilon
        self.seed               = seed

    def config(self) -> Dict[str, Any]:
        """Settings stored in every checkpoint for traceability."""
        return {k: v for k, v in vars(self).items()}

    def bind(self, explainer: Any) -> "BoundMetrics":
        return BoundMetrics(self, explainer)


class BoundMetrics:
    """Per-combination metric state (metric objects + randomised models)."""

    def __init__(self, cfg: SampleMetricComputer, explainer: Any) -> None:
        self.cfg       = cfg
        self.explainer = explainer
        self.model     = explainer.model
        self.loc  = LocalizationMetrics(thresholds=cfg.thresholds, seed=cfg.seed)
        self.comp = ComplexityMetrics()
        self.fid  = FidelityMetrics(mask_mode=cfg.fidelity_mask_mode, k_fractions=cfg.fidelity_k)
        self.rob  = RobustnessMetrics(
            epsilon=cfg.robustness_epsilon, n_samples=cfg.robustness_samples, seed=cfg.seed,
        )
        self._rand_explainer = None
        self._shuf_explainer = None
        self._shuf_error: Optional[str] = None
        self.n_robust = 0

    # -- helpers ------------------------------------------------------------

    def _norm(self, att: torch.Tensor) -> torch.Tensor:
        return normalize_attribution(att.detach().float().cpu(), mode=self.cfg.norm_mode)

    def _explain_one(self, explainer: Any, x1: torch.Tensor, y: int) -> torch.Tensor:
        """Normalised (Hp, Wp) map for a single (1, C, H, W) image."""
        target = torch.tensor([y], device=x1.device)
        return self._norm(explainer.explain_batch(x1, target))[0]

    def _robust_explainers(self):
        """Explainers on weight-randomised (R2) and label-permuted (R3) copies."""
        if self._rand_explainer is None:
            cls, ps = type(self.explainer), self.explainer.patch_size
            device = next(self.model.parameters()).device
            rand = randomise_model_weights(self.model, seed=self.cfg.seed).to(device).eval()
            self._rand_explainer = cls(rand, patch_size=ps)
            try:
                shuf = randomise_classifier_labels(self.model, seed=self.cfg.seed).to(device).eval()
                self._shuf_explainer = cls(shuf, patch_size=ps)
            except AttributeError as e:
                self._shuf_error = str(e)
        return self._rand_explainer, self._shuf_explainer

    # -- main entry ---------------------------------------------------------

    def compute_batch(
        self,
        x:        torch.Tensor,            # (B, C, H, W) on model device
        y:        torch.Tensor,            # (B,) on model device
        gt_mask:  Optional[torch.Tensor],  # (B, H, W) or None
        att_norm: torch.Tensor,            # (B, Hp, Wp) normalised, cpu
    ) -> List[Dict[str, float]]:
        cfg, B = self.cfg, x.shape[0]
        with torch.no_grad():
            preds = self.model(x).argmax(dim=1)
        rows: List[Dict[str, float]] = [
            {"pred": int(preds[i]), "correct": int(preds[i] == y[i])} for i in range(B)
        ]

        if "localization" in cfg.families and gt_mask is not None:
            for i in range(B):
                rows[i].update(self.loc.compute_all(att_norm[i], gt_mask[i].cpu()))

        if "complexity" in cfg.families:
            for i in range(B):
                c = self.comp.compute(att_norm[i]).to_dict()
                rows[i].update({k: float(c[k]) for k in _COMPLEXITY_KEYS})

        if "fidelity" in cfg.families:
            fid = self.fid.compute_all(self.model, x, y, att_norm.to(x.device))
            for k, v in fid.items():
                for i in range(B):
                    rows[i][k] = float(v[i])

        if "robustness" in cfg.families:
            for i in range(B):
                if cfg.robustness_subset is not None and self.n_robust >= cfg.robustness_subset:
                    break
                rows[i].update(self._robustness_one(x[i:i + 1], int(y[i]), att_norm[i]))
                self.n_robust += 1

        return rows

    def _robustness_one(self, x1: torch.Tensor, y: int, att: torch.Tensor) -> Dict[str, float]:
        def explain_fn(_model, img4d):
            return self._explain_one(self.explainer, img4d.to(x1.device), y)

        out = {"max_sensitivity": self.rob.max_sensitivity(explain_fn, self.model, x1, att)}

        rand_exp, shuf_exp = self._robust_explainers()
        att_rand = self._explain_one(rand_exp, x1, y)
        # A N(0,1)-randomised ViT can overflow; record NaN rather than a fake score.
        out["model_randomisation"] = (
            self.rob.model_randomisation(att, att_rand)
            if torch.isfinite(att_rand).all() else math.nan
        )
        if shuf_exp is not None:
            att_shuf = self._explain_one(shuf_exp, x1, y)
            out["label_randomisation"] = (
                self.rob.label_randomisation(att, att_shuf)
                if torch.isfinite(att_shuf).all() else math.nan
            )
        else:
            out["label_randomisation"] = math.nan
        return out
