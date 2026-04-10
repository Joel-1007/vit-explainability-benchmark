"""
axiom_verifier.py
=================
Task 2.5 — Axiomatic analysis for the ViT Explainability Benchmark.

This module provides:

1. Toy models for controlled axiom tests:
   - ``LinearPatchModel``   — output = Σ w_i · x_i (exact analytic attribution)
   - ``XORInteractionModel``— multiplicative patch interaction

2. ``AxiomVerifier`` — empirically tests axiomatic sensitivity of all benchmark
   metrics against the four Shapley-value axioms:

       A1  Dummy (D)             — irrelevant patches receive zero attribution
       A2  Completeness (CE)     — attributions sum to model output difference
       A3  Symmetry (S)          — equal-contribution patches receive equal attr.
       A4  Linearity (L)         — attribution is linear in model combination

3. Standalone verification functions:
   - ``verify_completeness()``          — checks A2 for one sample
   - ``run_completeness_verification()``— runs A2 checks across a dataset
   - ``generate_completeness_error_table()`` — paper-ready Table T3
   - ``verify_dummy_axiom()``           — approximate A1 check via masking
   - ``verify_rollout_dummy_violation()``— empirical Theorem T3 verifier
   - ``generate_axiom_satisfaction_heatmap()``— Figure F1 for the paper

References
----------
- Shapley (1953), "A Value for N-Person Games"
- Lundberg & Lee (2017), "A Unified Approach to Interpreting Model
  Predictions" (SHAP)
- Adebayo et al. (2018), "Sanity Checks for Saliency Maps"
- Chefer, Gur & Wolf (2021), "Transformer Interpretability Beyond Attention"
"""

from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Toy models for controlled axiom tests
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class LinearPatchModel(nn.Module):
        """
        Toy model: output = Σ_i w_i · x_i   for patch activations x_i.

        The analytic attribution equals the weight vector ``w``, making this
        model ideal for constructing controlled A1 / A2 / A3 / A4 test cases.

        Parameters
        ----------
        weights : np.ndarray
            Shape ``(N,)``.  Zero weights define dummy patches (A1 tests).
        """

        def __init__(self, weights: np.ndarray) -> None:
            super().__init__()
            self.w = torch.tensor(weights, dtype=torch.float32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """``x``: ``(batch, N)`` → scalar predictions ``(batch,)``."""
            return (x * self.w).sum(dim=-1)


    class XORInteractionModel(nn.Module):
        """
        Toy model: output = 1.0 if both p1 and p2 are present, else 0.5 / 0.0.

        Used in completeness counterexample (supports Theorem T1):
        multiplicative interactions make it impossible for additive attributions
        to be both complete and correctly ranking.

        Parameters
        ----------
        p1, p2 : int
            Patch indices that must both be present for maximum output.
        """

        def __init__(self, p1: int = 0, p2: int = 1) -> None:
            super().__init__()
            self.p1 = p1
            self.p2 = p2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            both = (x[:, self.p1] > 0.5) & (x[:, self.p2] > 0.5)
            one  = (x[:, self.p1] > 0.5) ^ (x[:, self.p2] > 0.5)
            result = torch.zeros(x.shape[0])
            result[both] = 1.0
            result[one]  = 0.5
            return result

else:
    # Provide stub classes when torch is not installed
    class LinearPatchModel:    # type: ignore[no-redef]
        """Stub (torch not installed)."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for LinearPatchModel")

    class XORInteractionModel: # type: ignore[no-redef]
        """Stub (torch not installed)."""
        def __init__(self, *args, **kwargs):
            raise ImportError("torch is required for XORInteractionModel")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AxiomTestResult:
    """Result of a single (metric, axiom) empirical test."""
    metric_name: str
    axiom_name: str         # 'A1' | 'A2' | 'A3' | 'A4'
    axiom_label: str        # 'Dummy' | 'Completeness' | 'Symmetry' | 'Linearity'
    satisfies: bool         # True if the metric rewards axiom-compliant explanation
    test_description: str
    value_satisfying: float # metric value when axiom is satisfied
    value_violating: float  # metric value when axiom is violated
    delta: float            # (satisfying − violating), signed for higher-is-better
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "metric_name":      self.metric_name,
            "axiom_name":       self.axiom_name,
            "axiom_label":      self.axiom_label,
            "satisfies":        self.satisfies,
            "test_description": self.test_description,
            "value_satisfying": self.value_satisfying,
            "value_violating":  self.value_violating,
            "delta":            self.delta,
            "note":             self.note,
        }


# ---------------------------------------------------------------------------
# AxiomVerifier
# ---------------------------------------------------------------------------

class AxiomVerifier:
    """
    Empirically verifies axiomatic sensitivity of benchmark metrics on
    synthetic test cases built from :class:`LinearPatchModel` and
    :class:`XORInteractionModel`.

    The four tests ``test_a1``–``test_a4`` may be called individually for any
    metric callable.  ``run_all()`` iterates over *all* registered metrics;
    metrics from the F / L / R families require a ``MetricSuite`` instance
    (available in Phase 3).  C1–C3 metrics work standalone.

    Usage
    -----
    >>> from metrics.complexity import gini_coefficient, attribution_entropy
    >>> from metrics.complexity import effective_mass_ratio
    >>> verifier = AxiomVerifier(n_patches=16)
    >>> # Test Gini against A3 (Theorem T6: anti-alignment expected)
    >>> result = verifier.test_a3(
    ...     lambda e, x, m: gini_coefficient(e),
    ...     "C1-Gini", higher_is_better=True
    ... )
    >>> print(result.satisfies, result.delta)   # False (anti-aligned), negative

    Parameters
    ----------
    metric_suite : optional
        A ``MetricSuite`` instance for F / L / R metric wrappers.
        If ``None``, only C1–C3 standalone tests are available.
    n_patches : int
        Number of patches in synthetic test inputs.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        metric_suite=None,
        n_patches: int = 16,
        seed: int = 42,
    ) -> None:
        self.ms  = metric_suite
        self.n   = n_patches
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # A1 (Dummy) test
    # ------------------------------------------------------------------

    def test_a1(
        self,
        metric_fn: Callable,
        metric_name: str,
        higher_is_better: bool,
    ) -> AxiomTestResult:
        """
        Setup: linear model with one dummy patch (weight=0).

        - Good attribution (A1 satisfied): assigns 0 to the dummy patch.
        - Bad attribution (A1 violated): assigns 0.5 to the dummy patch.

        A metric *rewards* A1 compliance if it scores the good attribution
        higher (for higher-is-better metrics) or lower (for lower-is-better).
        """
        N = self.n
        weights = self.rng.uniform(0.1, 1.0, N)
        dummy_patch = N // 2
        weights[dummy_patch] = 0.0

        if _TORCH_AVAILABLE:
            import torch
            model = LinearPatchModel(weights)
            x = torch.ones(1, N)
        else:
            model = None
            x = None

        e_good = weights.copy(); e_good[dummy_patch] = 0.0
        e_bad  = weights.copy(); e_bad[dummy_patch]  = 0.5

        score_good = metric_fn(e_good, x, model)
        score_bad  = metric_fn(e_bad,  x, model)
        delta = (
            (score_good - score_bad)
            if higher_is_better
            else (score_bad - score_good)
        )

        return AxiomTestResult(
            metric_name=metric_name,
            axiom_name="A1",
            axiom_label="Dummy",
            satisfies=delta > 0,
            test_description=(
                "Linear model; 1 dummy patch (weight=0). "
                "Good e=0 on dummy; bad e=0.5 on dummy."
            ),
            value_satisfying=score_good,
            value_violating=score_bad,
            delta=delta,
        )

    # ------------------------------------------------------------------
    # A2 (Completeness) test
    # ------------------------------------------------------------------

    def test_a2(
        self,
        metric_fn: Callable,
        metric_name: str,
        higher_is_better: bool,
    ) -> AxiomTestResult:
        """
        Setup: completeness-satisfying attribution vs. 2× scaled version.

        Theorem T1 predicts that F1–F4 will *not* distinguish them
        (``satisfies=False`` is the expected result for fidelity metrics).

        - Good attribution: Σ e_p = f(x) − f(baseline)  (A2 satisfied)
        - Bad attribution:  Σ e_p = 2 · Δf              (A2 violated)
        """
        N = self.n
        weights = self.rng.uniform(0.1, 1.0, N)

        if _TORCH_AVAILABLE:
            import torch
            model = LinearPatchModel(weights)
            x = torch.ones(1, N)
            x_baseline = torch.zeros(1, N)
            delta_f = float(model(x) - model(x_baseline))
        else:
            model = None
            x = None
            delta_f = float(weights.sum())

        e_sat = weights / weights.sum() * delta_f   # sum = Δf  (A2 satisfied)
        e_vio = e_sat * 2.0                          # sum = 2Δf (A2 violated)

        score_sat = metric_fn(e_sat, x, model)
        score_vio = metric_fn(e_vio, x, model)
        delta = (
            (score_sat - score_vio)
            if higher_is_better
            else (score_vio - score_sat)
        )

        return AxiomTestResult(
            metric_name=metric_name,
            axiom_name="A2",
            axiom_label="Completeness",
            satisfies=abs(delta) > 1e-6,
            test_description=(
                "Linear model; good e sums to Δf; bad e sums to 2×Δf."
            ),
            value_satisfying=score_sat,
            value_violating=score_vio,
            delta=delta,
            note=(
                "satisfies=False is expected for F1–F4 (Theorem T1). "
                "A2 insensitivity is a documented benchmark limitation."
            ),
        )

    # ------------------------------------------------------------------
    # A3 (Symmetry) test
    # ------------------------------------------------------------------

    def test_a3(
        self,
        metric_fn: Callable,
        metric_name: str,
        higher_is_better: bool,
    ) -> AxiomTestResult:
        """
        Setup: model with equal weights at p1 and p2.

        - Symmetric attribution  (A3 satisfied): e[p1] = e[p2] = 0.5
        - Asymmetric attribution (A3 violated):  e[p1] = 1.0, e[p2] = 0.0

        **Anti-alignment (Theorem T6):** C1–C3 will show ``delta < 0`` —
        they *reward* the A3-violating (asymmetric) attribution.
        """
        N = self.n
        weights = np.zeros(N)
        weights[0] = 0.5
        weights[1] = 0.5
        weights[2:] = self.rng.uniform(0.0, 0.2, N - 2)

        if _TORCH_AVAILABLE:
            import torch
            model = LinearPatchModel(weights)
            x = torch.ones(1, N)
        else:
            model = None
            x = None

        e_sym  = weights.copy(); e_sym[0] = 0.5; e_sym[1] = 0.5
        e_asym = weights.copy(); e_asym[0] = 1.0; e_asym[1] = 0.0

        score_sym  = metric_fn(e_sym,  x, model)
        score_asym = metric_fn(e_asym, x, model)
        delta = (
            (score_sym - score_asym)
            if higher_is_better
            else (score_asym - score_sym)
        )

        return AxiomTestResult(
            metric_name=metric_name,
            axiom_name="A3",
            axiom_label="Symmetry",
            satisfies=delta >= 0,
            test_description=(
                "Equal weights at p1, p2. "
                "Sym: e[p1]=e[p2]=0.5. Asym: e[p1]=1.0, e[p2]=0.0."
            ),
            value_satisfying=score_sym,
            value_violating=score_asym,
            delta=delta,
            note=(
                "For C1–C3: expected delta < 0 (anti-alignment — Theorem T6). "
                "For R1/R2: expected delta > 0 (indirect partial coupling)."
            ),
        )

    # ------------------------------------------------------------------
    # A4 (Linearity) test
    # ------------------------------------------------------------------

    def test_a4(
        self,
        metric_fn: Callable,
        metric_name: str,
        higher_is_better: bool,
    ) -> AxiomTestResult:
        """
        Setup: f = 0.5·g + 0.5·h.  Linear attribution = 0.5·w_g + 0.5·w_h.

        - Linear attribution    (A4 satisfied): e = 0.5·w_g + 0.5·w_h
        - Non-linear attribution (A4 violated): e = 0.7·w_g + 0.3·w_h
        """
        N = self.n
        wg = self.rng.uniform(0.0, 1.0, N)
        wh = self.rng.uniform(0.0, 1.0, N)
        wf = 0.5 * wg + 0.5 * wh

        if _TORCH_AVAILABLE:
            import torch
            model_f = LinearPatchModel(wf)
            x = torch.ones(1, N)
        else:
            model_f = None
            x = None

        e_linear     = 0.5 * wg + 0.5 * wh   # correct (A4 satisfied)
        e_nonlinear  = 0.7 * wg + 0.3 * wh   # wrong mixing ratio (A4 violated)

        score_lin    = metric_fn(e_linear,    x, model_f)
        score_nonlin = metric_fn(e_nonlinear, x, model_f)
        delta = (
            (score_lin - score_nonlin)
            if higher_is_better
            else (score_nonlin - score_lin)
        )

        return AxiomTestResult(
            metric_name=metric_name,
            axiom_name="A4",
            axiom_label="Linearity",
            satisfies=delta >= 0,
            test_description=(
                "f = 0.5·g + 0.5·h. "
                "Linear: e = 0.5·w_g + 0.5·w_h. "
                "Nonlinear: e = 0.7·w_g + 0.3·w_h."
            ),
            value_satisfying=score_lin,
            value_violating=score_nonlin,
            delta=delta,
        )

    # ------------------------------------------------------------------
    # Run all (metric × axiom) tests
    # ------------------------------------------------------------------

    def run_all(self) -> List[AxiomTestResult]:
        """
        Run all (metric, axiom) empirical tests.

        If ``metric_suite`` is ``None``, only C1–C3 standalone tests are
        executed.  F / L / R metrics require a ``MetricSuite`` instance
        (Phase 3 deliverable).

        Returns
        -------
        list of AxiomTestResult
        """
        results: List[AxiomTestResult] = []

        if self.ms is not None:
            all_metrics = self._get_all_metric_wrappers()
        else:
            all_metrics = self._get_complexity_metric_wrappers()

        for metric_name, metric_fn, higher_is_better in all_metrics:
            for test_fn in [self.test_a1, self.test_a2, self.test_a3, self.test_a4]:
                try:
                    result = test_fn(metric_fn, metric_name, higher_is_better)
                    results.append(result)
                except Exception as ex:
                    warnings.warn(
                        f"Test failed for ({metric_name}, {test_fn.__name__}): {ex}"
                    )
        return results

    def _get_complexity_metric_wrappers(self) -> List[tuple]:
        """C1–C3 metric wrappers (no MetricSuite required)."""
        from metrics.complexity import (
            gini_coefficient,
            attribution_entropy,
            effective_mass_ratio,
        )

        return [
            (
                "C1-Gini",
                lambda e, x, m: gini_coefficient(e),
                True,   # higher is better
            ),
            (
                "C2-Entropy",
                lambda e, x, m: attribution_entropy(e)["entropy_norm"],
                False,  # lower is better
            ),
            (
                "C3-EMR90",
                lambda e, x, m: effective_mass_ratio(e)["emr"],
                False,  # lower is better
            ),
        ]

    def _get_all_metric_wrappers(self) -> List[tuple]:
        """
        Full 13-metric wrapper list (requires self.ms MetricSuite).

        Toy GT mask for localization metrics: top quarter is GT region.
        """
        from metrics.complexity import (
            gini_coefficient,
            attribution_entropy,
            effective_mass_ratio,
        )

        gt_mask = np.zeros(self.n)
        gt_mask[: self.n // 4] = 1.0

        return [
            ("F1-InsertionAUC",      lambda e, x, m: self.ms.fidelity.insertion_auc(x, e),             True),
            ("F2-DeletionAUC",       lambda e, x, m: self.ms.fidelity.deletion_auc(x, e),              False),
            ("F3-Comprehensiveness", lambda e, x, m: self.ms.fidelity.comprehensiveness(x, e, k_frac=0.2), True),
            ("F4-LogOddsShift",      lambda e, x, m: self.ms.fidelity.log_odds_shift(x, e),             True),
            ("L1-IoU",               lambda e, x, m: self.ms.localization.iou(e, gt_mask),              True),
            ("L2-PointingGame",      lambda e, x, m: self.ms.localization.pointing_game(e, gt_mask),    True),
            ("L3-EnergyOnGT",        lambda e, x, m: self.ms.localization.energy_on_gt(e, gt_mask),     True),
            ("L4-CalibGap",          lambda e, x, m: self.ms.localization.calibration_gap(e, gt_mask),  True),
            ("R1-MaxSens",           lambda e, x, m: self.ms.robustness.max_sensitivity(e, x, m),       False),
            ("R2-AvgSens",           lambda e, x, m: self.ms.robustness.avg_sensitivity(e, x, m),       False),
            ("R3-ParamRand",         lambda e, x, m: self.ms.robustness.param_randomisation(e, x, m),   False),
            ("R4-LabelRand",         lambda e, x, m: self.ms.robustness.label_randomisation(e, x, m),   False),
            ("C1-Gini",              lambda e, x, m: gini_coefficient(e),                True),
            ("C2-Entropy",           lambda e, x, m: attribution_entropy(e)["entropy_norm"], False),
            ("C3-EMR90",             lambda e, x, m: effective_mass_ratio(e)["emr"],    False),
        ]

    # ------------------------------------------------------------------
    # Table builder
    # ------------------------------------------------------------------

    def build_satisfaction_table(self, results: List[AxiomTestResult]) -> str:
        """
        Generate a markdown-formatted axiom satisfaction table.

        Uses the four-symbol system: ✓ / ∼ / ✗ / ⊗.
        Anti-aligned C1–C3 × A3 entries are marked ∼†.
        """
        index: Dict[str, Dict[str, AxiomTestResult]] = defaultdict(dict)
        for r in results:
            index[r.metric_name][r.axiom_name] = r

        axioms = ["A1", "A2", "A3", "A4"]
        labels = {
            "A1": "Dummy (D)",
            "A2": "Completeness (CE)",
            "A3": "Symmetry (S)",
            "A4": "Linearity (L)",
        }

        header    = "| Metric | " + " | ".join(labels[a] for a in axioms) + " |"
        separator = "|--------|" + "---|" * 4

        rows = [header, separator]
        for metric in sorted(index.keys()):
            cells = []
            for axiom in axioms:
                if axiom in index[metric]:
                    r = index[metric][axiom]
                    symbol = "✓" if r.satisfies else "✗"
                    # Override for known structural cases
                    if axiom == "A3" and metric.startswith("C") and r.delta < 0:
                        symbol = "∼†"    # anti-alignment (Theorem T6)
                    cells.append(symbol)
                else:
                    cells.append("?")
            rows.append(f"| {metric:<24} | " + " | ".join(cells) + " |")

        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Standalone verification functions
# ---------------------------------------------------------------------------

def verify_completeness(
    explainer_fn: Callable,
    model: "torch.nn.Module",
    x: "torch.Tensor",
    target_class: int,
    x_baseline: "torch.Tensor",
    tolerance: float = 0.05,
) -> dict:
    """
    Verify whether an explainer satisfies Axiom A2 (Completeness) for one sample.

    A2: |Σ_p e_p − (f_c(x) − f_c(x_baseline))| < tolerance

    Parameters
    ----------
    explainer_fn : callable
        ``fn(x, target_class)`` → attribution tensor ``(H_p, W_p)`` or ``(N,)``.
    model : torch.nn.Module
        Trained ViT in eval mode.
    x : torch.Tensor
        Input image ``(3, H, W)``.
    target_class : int
        Class index c.
    x_baseline : torch.Tensor
        Baseline image ``(3, H, W)``.  Use the per-channel dataset mean.
    tolerance : float
        Absolute error threshold for CE check (default 0.05).

    Returns
    -------
    dict
        ``'attribution_sum'``    — Σ e_p
        ``'model_difference'``   — f_c(x) − f_c(baseline)
        ``'completeness_error'`` — |Σ e_p − Δf_c|
        ``'satisfies_ce'``       — bool
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for verify_completeness().")

    import torch

    model.eval()
    with torch.no_grad():
        probs_x        = torch.softmax(model(x.unsqueeze(0)),        dim=-1)[0, target_class].item()
        probs_baseline = torch.softmax(model(x_baseline.unsqueeze(0)), dim=-1)[0, target_class].item()

    model_difference = probs_x - probs_baseline
    attribution = explainer_fn(x, target_class)         # (H_p, W_p) or (N,)
    attribution_sum = float(attribution.sum().item())
    completeness_error = abs(attribution_sum - model_difference)

    return {
        "attribution_sum":    attribution_sum,
        "model_difference":   model_difference,
        "completeness_error": completeness_error,
        "satisfies_ce":       completeness_error < tolerance,
    }


def run_completeness_verification(
    explainers: Dict[str, Callable],
    model: "torch.nn.Module",
    test_loader,
    x_baseline: "torch.Tensor",
    n_samples: int = 100,
    tolerance: float = 0.05,
    output_path: str = "results/completeness_verification.json",
) -> dict:
    """
    Run completeness verification for all explainers over ``n_samples`` images.

    Saves results to JSON and returns a summary dict.

    Returns
    -------
    dict
        Keyed by method name:
        ``{'mean_completeness_error': float, 'std_completeness_error': float,
           'pct_satisfying_ce': float, 'n_samples': int}``
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required.")

    results: Dict[str, dict] = defaultdict(lambda: {"errors": [], "satisfies": []})

    for i, (x, y, *_) in enumerate(test_loader):
        if i >= n_samples:
            break
        x_img  = x[0]
        target = y[0].item()

        for method_name, explainer_fn in explainers.items():
            result = verify_completeness(
                explainer_fn, model, x_img, target, x_baseline, tolerance=tolerance
            )
            results[method_name]["errors"].append(result["completeness_error"])
            results[method_name]["satisfies"].append(result["satisfies_ce"])

    summary: Dict[str, dict] = {}
    for method_name, data in results.items():
        errors   = np.array(data["errors"])
        satisfies = np.array(data["satisfies"])
        summary[method_name] = {
            "mean_completeness_error": float(errors.mean()),
            "std_completeness_error":  float(errors.std()),
            "pct_satisfying_ce":       float(satisfies.mean() * 100),
            "n_samples":               len(errors),
        }

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Completeness verification saved to {output_path}")
    return summary


def generate_completeness_error_table(
    completeness_results: dict,
    latex: bool = True,
) -> str:
    """
    Generate Table T3: completeness error table for the paper.

    Columns: Method | Mean |CE| Error | Std | % Satisfying CE

    Parameters
    ----------
    completeness_results : dict
        Output of :func:`run_completeness_verification`.
    latex : bool
        If ``True``, return a LaTeX tabular string (booktabs style).
        Otherwise, return a tab-separated text table.
    """
    header = ["Method", "Mean |CE| Error", "Std", "% Satisfying CE"]
    rows = []

    for method, data in sorted(
        completeness_results.items(),
        key=lambda item: item[1]["mean_completeness_error"],
    ):
        rows.append([
            method,
            f"{data['mean_completeness_error']:.4f}",
            f"{data['std_completeness_error']:.4f}",
            f"{data['pct_satisfying_ce']:.1f}%",
        ])

    if latex:
        lines = [
            r"\begin{tabular}{llll}",
            r"\toprule",
            " & ".join(header) + r" \\",
            r"\midrule",
        ]
        for row in rows:
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)
    else:
        return "\n".join(
            ["\t".join(header)] + ["\t".join(row) for row in rows]
        )


def verify_dummy_axiom(
    explainer_fn: Callable,
    model: "torch.nn.Module",
    x: "torch.Tensor",
    target_class: int,
    patch_size: int = 16,
    threshold_confidence_change: float = 0.01,
) -> dict:
    """
    Estimate whether an explainer approximately satisfies A1 (Dummy).

    Step 1: Identify "approximately dummy" patches by masking each patch
            individually and measuring |Δf_c|.  Patches with
            ``|Δf_c| < threshold`` are declared dummy.
    Step 2: Measure what fraction of total attribution the explainer assigns
            to those dummy patches.

    Returns
    -------
    dict
        ``'n_dummy_patches'``              — count of approximately-dummy patches
        ``'n_total_patches'``              — N
        ``'dummy_fraction_of_patches'``    — n_dummy / N
        ``'mean_attribution_to_dummy'``    — mean attribution over dummy patches
        ``'mean_attribution_to_nondummy'`` — mean attribution over causal patches
        ``'dummy_attribution_ratio'``      — fraction of total mass on dummy patches
        ``'satisfies_dummy_approx'``       — bool (True if ratio < 10%)
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required.")

    import torch

    model.eval()
    H_p = W_p = 224 // patch_size
    N = H_p * W_p

    with torch.no_grad():
        original_conf = torch.softmax(
            model(x.unsqueeze(0)), dim=-1
        )[0, target_class].item()

    # Mean fill for masking
    x_mean = x.mean(dim=[1, 2], keepdim=True).expand_as(x)
    confidence_changes = []

    for p_idx in range(N):
        p_row = p_idx // W_p
        p_col = p_idx % W_p
        r0 = p_row * patch_size; r1 = r0 + patch_size
        c0 = p_col * patch_size; c1 = c0 + patch_size

        x_masked = x.clone()
        x_masked[:, r0:r1, c0:c1] = x_mean[:, r0:r1, c0:c1]

        with torch.no_grad():
            masked_conf = torch.softmax(
                model(x_masked.unsqueeze(0)), dim=-1
            )[0, target_class].item()

        confidence_changes.append(abs(original_conf - masked_conf))

    confidence_changes = np.array(confidence_changes)
    dummy_mask = confidence_changes < threshold_confidence_change

    attribution = explainer_fn(x, target_class)
    attribution_flat = np.clip(
        np.array(attribution.flatten().detach().cpu().numpy()
                 if hasattr(attribution, "detach") else attribution.flatten()),
        0.0, None,
    )
    total_attribution = attribution_flat.sum()
    n_dummy = int(dummy_mask.sum())

    if n_dummy == 0 or total_attribution == 0:
        mean_dummy_att = 0.0
        dummy_ratio    = 0.0
    else:
        mean_dummy_att = float(attribution_flat[dummy_mask].mean())
        dummy_ratio    = float(attribution_flat[dummy_mask].sum() / total_attribution)

    mean_nondummy_att = (
        float(attribution_flat[~dummy_mask].mean())
        if (~dummy_mask).sum() > 0 else 0.0
    )

    return {
        "n_dummy_patches":              n_dummy,
        "n_total_patches":              N,
        "dummy_fraction_of_patches":    float(n_dummy / N),
        "mean_attribution_to_dummy":    mean_dummy_att,
        "mean_attribution_to_nondummy": mean_nondummy_att,
        "dummy_attribution_ratio":      dummy_ratio,
        "satisfies_dummy_approx":       dummy_ratio < 0.10,
    }


def verify_rollout_dummy_violation(
    model: "torch.nn.Module",
    patch_size: int = 16,
    device: str = "cpu",
) -> dict:
    """
    Verify Theorem T3 empirically: Attention Rollout assigns non-zero
    attribution to provably informationally irrelevant (constant) patches.

    Constructs a synthetic input where the right half is set to the channel
    mean (constant = informationally irrelevant).  Measures the minimum
    attribution assigned by Rollout vs. GradCAM.

    Requires ``explainers.rollout.AttentionRolloutExplainer`` and
    ``explainers.gradcam.GradCAMExplainer`` to be available (Phase 3).

    Returns
    -------
    dict
        ``'rollout_min_dummy_attribution'``  — must be > 0 for T3 to hold
        ``'gradcam_min_dummy_attribution'``
        ``'theoretical_rollout_floor'``      — 0.5^L
        ``'theorem_3_verified'``             — bool
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required.")

    import torch

    try:
        from explainers.rollout import AttentionRolloutExplainer
        from explainers.gradcam import GradCAMExplainer
    except ImportError:
        warnings.warn(
            "explainers.rollout / explainers.gradcam not available. "
            "Verify Theorem T3 manually after Phase 3 explainer implementation.",
            ImportWarning,
        )
        return {
            "rollout_min_dummy_attribution": None,
            "gradcam_min_dummy_attribution": None,
            "theoretical_rollout_floor":     0.5 ** 12,
            "theorem_3_verified":            None,
        }

    H = W = 224
    N = (H // patch_size) ** 2

    x = torch.rand(3, H, W, device=device)
    x[:, :, W // 2:] = x.mean()    # right half → constant (dummy)
    target_class = 0

    rollout_explainer = AttentionRolloutExplainer(model, patch_size=patch_size)
    gradcam_explainer = GradCAMExplainer(model, patch_size=patch_size)

    rollout_att = rollout_explainer.explain(x, target_class).flatten().cpu().numpy()
    gradcam_att = gradcam_explainer.explain(x, target_class).flatten().cpu().numpy()

    H_p = W_p = H // patch_size
    dummy_indices = np.array([
        p_row * W_p + p_col
        for p_row in range(H_p)
        for p_col in range(W_p)
        if p_col >= W_p // 2
    ])

    rollout_dummy_min = float(rollout_att[dummy_indices].min())
    gradcam_dummy_min = float(np.clip(gradcam_att[dummy_indices], 0, None).min())
    L = 12
    theoretical_floor = 0.5 ** L

    print("\nVerification Test V4: Rollout Dummy Violation (Theorem T3)")
    print(f"  Rollout min attribution to dummy patches:  {rollout_dummy_min:.6f}")
    print(f"  GradCAM min attribution to dummy patches:  {gradcam_dummy_min:.6f}")
    print(f"  Theoretical rollout floor (0.5^{L}):       {theoretical_floor:.6f}")
    print(f"  Theorem T3 verified:                       {rollout_dummy_min > 0}")

    return {
        "rollout_min_dummy_attribution": rollout_dummy_min,
        "gradcam_min_dummy_attribution": gradcam_dummy_min,
        "theoretical_rollout_floor":     theoretical_floor,
        "theorem_3_verified":            bool(rollout_dummy_min > 0),
    }


# ---------------------------------------------------------------------------
# Figure F1 — Axiom Satisfaction Heatmap
# ---------------------------------------------------------------------------

def generate_axiom_satisfaction_heatmap(
    output_path: str = "figures/axiom_satisfaction.pdf",
) -> None:
    """
    Generate Figure F1: the 15×4 axiom satisfaction heatmap for the paper.

    Colour scheme:
        mediumseagreen  — ✓  (rewards compliance)
        gold            — ∼  (partially sensitive)
        white           — ✗  (insensitive)
        steelblue       — ⊗  (designed to test axiom)

    The heatmap is based on the *theoretical* satisfaction table in Section 3
    of the Task 2.5 spec and matches Table T1 in the paper.
    """
    import os
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = [
        "F1 — Insertion AUC",
        "F2 — Deletion AUC",
        "F3 — Comprehensiveness",
        "F4 — Log-odds shift",
        "L1 — IoU with GT",
        "L2 — Pointing Game",
        "L3 — Energy on GT",
        "L4 — Calibration Gap",
        "R1 — Max-Sensitivity",
        "R2 — Avg-Sensitivity",
        "R3 — Param. Rand.",
        "R4 — Label Rand.",
        "C1 — Gini",
        "C2 — Attribution Entropy",
        "C3 — Eff. Mass Ratio",
    ]
    axioms = ["Dummy (A1)", "Completeness (A2)", "Symmetry (A3)", "Linearity (A4)"]

    # Encoding: 2=✓ (green), 1=∼ (gold), 0=✗ (white), -1=⊗ (steelblue)
    # Note: C1/C2/C3 × A3 encoded as 1 (gold) with dagger footnote
    data = [
        [1,  0, 0, 0],   # F1
        [1,  0, 0, 0],   # F2
        [2,  1, 0, 0],   # F3
        [2,  1, 0, 0],   # F4
        [1,  0, 0, 0],   # L1
        [0,  0, 0, 0],   # L2
        [2,  0, 0, 0],   # L3
        [2,  0, 0, 0],   # L4
        [0,  0, 1, 0],   # R1
        [0,  0, 1, 0],   # R2
        [-1, 0, 0, 0],   # R3
        [-1, 0, 0, 0],   # R4
        [0,  0, 1, 0],   # C1  (∼†)
        [0,  0, 1, 0],   # C2  (∼†)
        [0,  0, 1, 0],   # C3  (∼†)
    ]
    symbols = [
        ["∼",  "✗", "✗",  "✗"],   # F1
        ["∼",  "✗", "✗",  "✗"],   # F2
        ["✓",  "∼", "✗",  "✗"],   # F3
        ["✓",  "∼", "✗",  "✗"],   # F4
        ["∼",  "✗", "✗",  "✗"],   # L1
        ["✗",  "✗", "✗",  "✗"],   # L2
        ["✓",  "✗", "✗",  "✗"],   # L3
        ["✓",  "✗", "✗",  "✗"],   # L4
        ["✗",  "✗", "∼",  "✗"],   # R1
        ["✗",  "✗", "∼",  "✗"],   # R2
        ["⊗",  "✗", "✗",  "✗"],   # R3
        ["⊗",  "✗", "✗",  "✗"],   # R4
        ["✗",  "✗", "∼†", "✗"],   # C1
        ["✗",  "✗", "∼†", "✗"],   # C2
        ["✗",  "✗", "∼†", "✗"],   # C3
    ]

    data_np = np.array(data)
    cmap = mcolors.ListedColormap(["white", "gold", "mediumseagreen"])
    fig, ax = plt.subplots(figsize=(7.5, 9))

    standard = np.clip(data_np, 0, 2)
    ax.imshow(standard, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    for i, (row_data, row_sym) in enumerate(zip(data_np, symbols)):
        for j, (val, sym) in enumerate(zip(row_data, row_sym)):
            if val == -1:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="steelblue", alpha=0.75,
                ))
                ax.text(
                    j, i, "⊗", ha="center", va="center",
                    fontsize=12, color="white", fontweight="bold",
                )
            else:
                colour = "black" if val > 0 else "lightgray"
                ax.text(
                    j, i, sym, ha="center", va="center",
                    fontsize=11, color=colour,
                )

    ax.set_xticks(range(len(axioms)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(axioms, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(metrics, fontsize=9)
    ax.set_title(
        "Axiom Sensitivity Table — 15 Metrics × 4 Axioms\n"
        "(† anti-aligned: higher score rewards axiom violation)",
        fontsize=10,
    )

    legend_elements = [
        mpatches.Patch(facecolor="mediumseagreen", label="✓  Rewards compliance"),
        mpatches.Patch(facecolor="gold",           label="∼  Partially sensitive"),
        mpatches.Patch(facecolor="white", edgecolor="gray", label="✗  Insensitive"),
        mpatches.Patch(facecolor="steelblue",      label="⊗  Designed to test axiom"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), fontsize=8, title="Legend",
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
