"""
test_axiom_verifier.py
======================
Unit tests for metrics/axiom_verifier.py — AxiomVerifier, toy models,
and standalone verification functions.

Run with::

    pytest tests/test_axiom_verifier.py -v

Tests that require torch skip gracefully when torch is not installed.
"""

import sys
import os
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.axiom_verifier import (
    AxiomTestResult,
    AxiomVerifier,
    _TORCH_AVAILABLE,
)
from metrics.complexity import (
    gini_coefficient,
    attribution_entropy,
    effective_mass_ratio,
)

# Skip marker for torch-dependent tests
requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not installed",
)


# ===========================================================================
# Toy Models
# ===========================================================================

@requires_torch
class TestLinearPatchModel:
    """LinearPatchModel: output = Σ w_i · x_i."""

    def test_forward_ones(self):
        """All-ones input → output equals sum of weights."""
        import torch
        from metrics.axiom_verifier import LinearPatchModel

        weights = np.array([0.5, 1.0, 1.5, 2.0])
        model = LinearPatchModel(weights)
        x = torch.ones(1, 4)
        out = model(x)
        assert abs(float(out) - 5.0) < 1e-5

    def test_forward_zeros(self):
        """All-zeros input → output = 0."""
        import torch
        from metrics.axiom_verifier import LinearPatchModel

        weights = np.array([1.0, 2.0, 3.0])
        model = LinearPatchModel(weights)
        x = torch.zeros(1, 3)
        out = model(x)
        assert abs(float(out) - 0.0) < 1e-5

    def test_dummy_patch_zero_weight(self):
        """Patch with weight=0 contributes nothing to output."""
        import torch
        from metrics.axiom_verifier import LinearPatchModel

        weights = np.array([1.0, 0.0, 2.0])
        model = LinearPatchModel(weights)

        x_with    = torch.tensor([[1.0, 5.0, 1.0]])
        x_without = torch.tensor([[1.0, 0.0, 1.0]])
        # Output should be identical regardless of dummy-patch value
        assert abs(float(model(x_with)) - float(model(x_without))) < 1e-5

    def test_batch_inference(self):
        """Accepts batched input (B, N)."""
        import torch
        from metrics.axiom_verifier import LinearPatchModel

        weights = np.array([1.0, 2.0])
        model = LinearPatchModel(weights)
        x = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
        out = model(x)
        assert out.shape == (2,)
        assert abs(float(out[0]) - 3.0) < 1e-5
        assert abs(float(out[1]) - 8.0) < 1e-5


@requires_torch
class TestXORInteractionModel:
    """XORInteractionModel: outputs 1.0 / 0.5 / 0.0 based on patch presence."""

    def test_both_present(self):
        import torch
        from metrics.axiom_verifier import XORInteractionModel

        model = XORInteractionModel(p1=0, p2=1)
        x = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        out = model(x)
        assert abs(float(out[0]) - 1.0) < 1e-5

    def test_one_present(self):
        import torch
        from metrics.axiom_verifier import XORInteractionModel

        model = XORInteractionModel(p1=0, p2=1)
        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        out = model(x)
        assert abs(float(out[0]) - 0.5) < 1e-5

    def test_none_present(self):
        import torch
        from metrics.axiom_verifier import XORInteractionModel

        model = XORInteractionModel(p1=0, p2=1)
        x = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        out = model(x)
        assert abs(float(out[0]) - 0.0) < 1e-5


# ===========================================================================
# AxiomVerifier — C1–C3 standalone tests
# ===========================================================================

class TestAxiomVerifierC1C3:
    """Tests that can run without a MetricSuite (C1–C3 only)."""

    def setup_method(self):
        self.verifier = AxiomVerifier(metric_suite=None, n_patches=16, seed=42)

    # ------------------------------------------------------------------
    # A1 (Dummy) — C1 Gini
    # ------------------------------------------------------------------

    def test_a1_gini(self):
        """Gini does NOT reward A1 compliance (axiom-agnostic — Theorem T5)."""
        result = self.verifier.test_a1(
            lambda e, x, m: gini_coefficient(e),
            "C1-Gini",
            higher_is_better=True,
        )
        assert isinstance(result, AxiomTestResult)
        assert result.axiom_name  == "A1"
        assert result.metric_name == "C1-Gini"
        # Gini is insensitive to axiom violations — satisfies may be either value
        # depending on the random weights; we just check it runs cleanly
        assert isinstance(result.satisfies, bool)

    # ------------------------------------------------------------------
    # A2 (Completeness) — C1 Gini
    # ------------------------------------------------------------------

    def test_a2_gini_scale_invariant(self):
        """Gini is scale-invariant → insensitive to 2× scaling → satisfies=False."""
        result = self.verifier.test_a2(
            lambda e, x, m: gini_coefficient(e),
            "C1-Gini",
            higher_is_better=True,
        )
        # Gini is scale-invariant, so 2× scaling produces the same Gini
        # delta should be ≈ 0 → satisfies = False
        assert abs(result.delta) < 0.1, (
            f"Gini should be near-scale-invariant; delta={result.delta:.4f}"
        )

    # ------------------------------------------------------------------
    # A3 (Symmetry) — C1 Gini  ← Anti-alignment (Theorem T6)
    # ------------------------------------------------------------------

    def test_a3_gini_antialignment(self):
        """Theorem T6: Gini rewards A3 violations (anti-aligned, delta < 0)."""
        result = self.verifier.test_a3(
            lambda e, x, m: gini_coefficient(e),
            "C1-Gini",
            higher_is_better=True,
        )
        assert result.axiom_name == "A3"
        # Anti-alignment: asym attribution has higher Gini → delta < 0 (sym scores lower)
        assert result.delta < 0, (
            f"Expected anti-alignment (delta < 0), got delta={result.delta:.4f}"
        )
        # satisfies should be False (symmetric e scores lower, not higher)
        assert result.satisfies is False

    def test_a3_entropy_antialignment(self):
        """Entropy anti-alignment: sym attribution has HIGHER entropy (less sparse)."""
        result = self.verifier.test_a3(
            lambda e, x, m: attribution_entropy(e)["entropy_norm"],
            "C2-Entropy",
            higher_is_better=False,   # lower entropy = sparser = better
        )
        # For lower-is-better: delta = (score_violating - score_satisfying).
        # sym has HIGHER entropy (worse score) → delta < 0 → satisfies = False.
        # This is the anti-alignment (Theorem T6): metric rewards A3 violation.
        assert result.axiom_name == "A3"
        assert result.satisfies is False   # anti-alignment confirmed: does NOT reward compliance

    def test_a3_emr_antialignment(self):
        """EMR anti-alignment: asymmetric attribution has lower (better) EMR."""
        result = self.verifier.test_a3(
            lambda e, x, m: effective_mass_ratio(e)["emr"],
            "C3-EMR",
            higher_is_better=False,
        )
        assert result.axiom_name == "A3"
        # Same logic as entropy: delta < 0 → satisfies = False (does NOT reward compliance).
        # This is the anti-alignment: the metric rewards the A3-violating attribution.
        assert result.satisfies is False

    # ------------------------------------------------------------------
    # A4 (Linearity) — C1 Gini
    # ------------------------------------------------------------------

    def test_a4_gini(self):
        """Gini is insensitive to A4 (linearity); result is either value."""
        result = self.verifier.test_a4(
            lambda e, x, m: gini_coefficient(e),
            "C1-Gini",
            higher_is_better=True,
        )
        assert result.axiom_name == "A4"
        assert isinstance(result.satisfies, bool)

    # ------------------------------------------------------------------
    # run_all() on complexity metrics
    # ------------------------------------------------------------------

    def test_run_all_returns_12_results(self):
        """run_all() with C1–C3 only → 3 metrics × 4 axioms = 12 results."""
        results = self.verifier.run_all()
        assert len(results) == 12

    def test_run_all_unique_combinations(self):
        """All (metric_name, axiom_name) pairs must be unique."""
        results = self.verifier.run_all()
        pairs = [(r.metric_name, r.axiom_name) for r in results]
        assert len(pairs) == len(set(pairs)), "Duplicate (metric, axiom) entries"

    def test_run_all_axiom_names(self):
        """All four axioms must appear in the results."""
        results = self.verifier.run_all()
        axiom_names = {r.axiom_name for r in results}
        assert axiom_names == {"A1", "A2", "A3", "A4"}

    # ------------------------------------------------------------------
    # build_satisfaction_table()
    # ------------------------------------------------------------------

    def test_build_satisfaction_table(self):
        """build_satisfaction_table() returns a non-empty markdown string."""
        results = self.verifier.run_all()
        table = self.verifier.build_satisfaction_table(results)
        assert isinstance(table, str)
        assert len(table) > 0
        assert "Dummy" in table       # column header is "Dummy (D)"
        assert "Symmetry" in table    # column header is "Symmetry (S)"

    def test_anti_alignment_dagger_in_table(self):
        """∼† marker appears for C1/C2/C3 × A3 in the satisfaction table."""
        results = self.verifier.run_all()
        table = self.verifier.build_satisfaction_table(results)
        assert "∼†" in table, (
            "Expected anti-alignment marker '∼†' for C1–C3 × A3 in table"
        )


# ===========================================================================
# AxiomTestResult dataclass
# ===========================================================================

class TestAxiomTestResult:
    def test_to_dict_keys(self):
        """to_dict() returns all required keys."""
        r = AxiomTestResult(
            metric_name="C1-Gini",
            axiom_name="A3",
            axiom_label="Symmetry",
            satisfies=False,
            test_description="Test",
            value_satisfying=0.5,
            value_violating=0.8,
            delta=-0.3,
            note="Anti-alignment",
        )
        d = r.to_dict()
        required = {
            "metric_name", "axiom_name", "axiom_label",
            "satisfies", "test_description",
            "value_satisfying", "value_violating", "delta", "note",
        }
        assert required == set(d.keys())

    def test_values_preserved(self):
        """to_dict() values match constructor arguments."""
        r = AxiomTestResult(
            metric_name="C2-Entropy",
            axiom_name="A1",
            axiom_label="Dummy",
            satisfies=True,
            test_description="desc",
            value_satisfying=0.3,
            value_violating=0.7,
            delta=-0.4,
        )
        d = r.to_dict()
        assert d["metric_name"] == "C2-Entropy"
        assert d["satisfies"]   is True
        assert d["delta"]       == pytest.approx(-0.4)


# ===========================================================================
# Theorem T6 — explicit canonical test from spec §4 (verify_completeness)
# ===========================================================================

class TestTheoremT6Canonical:
    """
    The canonical Theorem T6 test as listed in Task 2.5 §4:

        e_sym[0]  = 0.5, e_sym[1]  = 0.5  (A3 satisfied)
        e_asym[0] = 1.0, e_asym[1] = 0.0  (A3 violated)
        Same total mass (1.0).
        Gini(e_asym) > Gini(e_sym)  □
    """

    def test_t6_gini_symmetry_antialignment(self):
        N = 196
        e_sym  = np.zeros(N); e_sym[0]  = 0.5; e_sym[1]  = 0.5
        e_asym = np.zeros(N); e_asym[0] = 1.0; e_asym[1] = 0.0

        assert abs(e_sym.sum() - e_asym.sum()) < 1e-9  # same total mass

        g_sym  = gini_coefficient(e_sym)
        g_asym = gini_coefficient(e_asym)

        assert g_asym > g_sym, (
            f"T6: Gini(asym)={g_asym:.4f} should exceed Gini(sym)={g_sym:.4f}"
        )
