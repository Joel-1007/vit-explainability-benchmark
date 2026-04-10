"""
test_complexity.py
==================
Unit tests for metrics/complexity.py — C1 (Gini), C2 (Entropy), C3 (EMR).

Run with::

    pytest tests/test_complexity.py -v

All 28 tests must pass.  Expected runtime: < 1 second.
"""

import math
import sys
import os

import numpy as np
import pytest

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.complexity import (
    ComplexityMetrics,
    ComplexityResult,
    attribution_entropy,
    effective_mass_ratio,
    effective_mass_ratio_multi,
    gini_coefficient,
    normalise_attribution,
    run_sanity_check,
)


# ===========================================================================
# C1 — Gini Coefficient
# ===========================================================================

class TestGiniCoefficient:
    """Section 2.6 of the Task 2.4 specification."""

    def test_uniform(self):
        """Uniform distribution → Gini = 0."""
        e = np.ones(196)
        assert abs(gini_coefficient(e) - 0.0) < 1e-9

    def test_one_hot(self):
        """All mass on one patch → Gini = (N-1)/N."""
        N = 196
        e = np.zeros(N)
        e[0] = 1.0
        expected = (N - 1.0) / N
        assert abs(gini_coefficient(e) - expected) < 1e-9

    def test_all_zeros(self):
        """Zero map → Gini = 0 by convention."""
        e = np.zeros(196)
        assert gini_coefficient(e) == 0.0

    def test_scale_invariant(self):
        """Scaling the map must not change Gini."""
        e = np.array([1.0, 2.0, 3.0, 4.0])
        g1 = gini_coefficient(e)
        g2 = gini_coefficient(e * 100.0)
        assert abs(g1 - g2) < 1e-9

    def test_two_equal_patches(self):
        """Two equal patches, rest zero: Gini in valid range."""
        N = 100
        e = np.zeros(N)
        e[0] = 1.0
        e[1] = 1.0
        result = gini_coefficient(e)
        assert 0.0 <= result <= 1.0

    def test_range(self):
        """Gini must always be in [0, 1] for any non-negative input."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            e = np.abs(rng.standard_normal(196))
            g = gini_coefficient(e)
            assert 0.0 <= g <= 1.0, f"Gini out of range: {g}"

    def test_higher_concentration_higher_gini(self):
        """More concentrated maps must have strictly higher Gini."""
        e_uniform = np.ones(196)
        e_medium  = np.zeros(196); e_medium[:20] = 1.0
        e_sparse  = np.zeros(196); e_sparse[0]  = 1.0
        assert (
            gini_coefficient(e_uniform)
            < gini_coefficient(e_medium)
            < gini_coefficient(e_sparse)
        )

    def test_consistent_across_patch_sizes(self):
        """One-hot vector → Gini > 0.99 for all model patch sizes."""
        for N in [196, 256, 784]:
            e = np.zeros(N); e[0] = 1.0
            g = gini_coefficient(e)
            assert g > 0.99, f"N={N}: expected Gini~1, got {g}"

    def test_signed_input_handled(self):
        """Negative values should be clipped (not cause errors)."""
        e = np.array([-1.0, 0.5, 0.3, 0.2])
        g = gini_coefficient(e)   # should not raise; clips negatives to 0
        assert 0.0 <= g <= 1.0


# ===========================================================================
# C2 — Attribution Entropy
# ===========================================================================

class TestAttributionEntropy:
    """Section 3.5 of the Task 2.4 specification."""

    def test_uniform(self):
        """Uniform attribution → maximum entropy (norm = 1.0)."""
        N = 196
        e = np.ones(N)
        result = attribution_entropy(e)
        assert abs(result["entropy_norm"] - 1.0) < 1e-9
        assert abs(result["entropy_raw"] - math.log(N)) < 1e-9

    def test_one_hot(self):
        """All mass on one patch → entropy = 0."""
        N = 196
        e = np.zeros(N); e[42] = 1.0
        result = attribution_entropy(e)
        assert abs(result["entropy_raw"]  - 0.0) < 1e-9
        assert abs(result["entropy_norm"] - 0.0) < 1e-9

    def test_two_equal(self):
        """Two equal patches → H = log(2)."""
        N = 196
        e = np.zeros(N); e[0] = 1.0; e[1] = 1.0
        result = attribution_entropy(e)
        assert abs(result["entropy_raw"] - math.log(2)) < 1e-9

    def test_zero_map(self):
        """Zero map → treated as maximum entropy (flags uninformative explanation)."""
        e = np.zeros(196)
        result = attribution_entropy(e)
        assert result["entropy_norm"] == 1.0

    def test_scale_invariant(self):
        """Entropy depends only on relative magnitudes, not absolute scale."""
        e = np.array([1.0, 2.0, 3.0, 0.5])
        r1 = attribution_entropy(e)
        r2 = attribution_entropy(e * 1000.0)
        assert abs(r1["entropy_raw"] - r2["entropy_raw"]) < 1e-9

    def test_norm_range(self):
        """Normalised entropy must be in [0, 1] for any input."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            e = np.abs(rng.standard_normal(196))
            result = attribution_entropy(e)
            assert 0.0 <= result["entropy_norm"] <= 1.0

    def test_concentration_decreases_entropy(self):
        """More concentrated maps must have lower entropy."""
        e_diffuse = np.ones(196)
        e_medium  = np.zeros(196); e_medium[:20] = 1.0
        e_sparse  = np.zeros(196); e_sparse[0]  = 1.0
        h_d = attribution_entropy(e_diffuse)["entropy_norm"]
        h_m = attribution_entropy(e_medium)["entropy_norm"]
        h_s = attribution_entropy(e_sparse)["entropy_norm"]
        assert h_d > h_m > h_s


# ===========================================================================
# C3 — Effective Mass Ratio
# ===========================================================================

class TestEffectiveMassRatio:
    """Section 4.4 of the Task 2.4 specification."""

    def test_one_hot(self):
        """All mass on one patch → only 1 patch needed → EMR = 1/N."""
        N = 196
        e = np.zeros(N); e[0] = 1.0
        result = effective_mass_ratio(e, threshold=0.9)
        assert result["k_star"] == 1
        assert abs(result["emr"] - 1.0 / N) < 1e-9

    def test_uniform(self):
        """Uniform attribution → need threshold fraction of all patches."""
        N = 100
        e = np.ones(N)
        result = effective_mass_ratio(e, threshold=0.9)
        assert result["k_star"] == 90
        assert abs(result["emr"] - 0.90) < 1e-9

    def test_zero_map(self):
        """Zero map → worst case (EMR = 1.0)."""
        e = np.zeros(196)
        result = effective_mass_ratio(e)
        assert result["emr"] == 1.0

    def test_threshold_monotone(self):
        """Higher threshold → more patches needed (k_star non-decreasing)."""
        rng = np.random.default_rng(99)
        e = np.abs(rng.standard_normal(196))
        r50 = effective_mass_ratio(e, threshold=0.5)
        r90 = effective_mass_ratio(e, threshold=0.9)
        r95 = effective_mass_ratio(e, threshold=0.95)
        assert r50["k_star"] <= r90["k_star"] <= r95["k_star"]

    def test_range(self):
        """EMR must be in [1/N, 1.0] for any non-zero input."""
        rng = np.random.default_rng(42)
        N = 196
        for _ in range(100):
            e = np.abs(rng.standard_normal(N))
            result = effective_mass_ratio(e, threshold=0.9)
            assert 1.0 / N - 1e-9 <= result["emr"] <= 1.0 + 1e-9

    def test_scale_invariant(self):
        """EMR must be scale-invariant."""
        e = np.array([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        r1 = effective_mass_ratio(e, threshold=0.9)
        r2 = effective_mass_ratio(e * 1000.0, threshold=0.9)
        assert r1["k_star"] == r2["k_star"]

    def test_multi_thresholds_single_pass(self):
        """effective_mass_ratio_multi returns one dict per threshold."""
        e = np.abs(np.random.RandomState(7).randn(196))
        results = effective_mass_ratio_multi(e, thresholds=[0.5, 0.9, 0.95])
        assert set(results.keys()) == {0.5, 0.9, 0.95}
        for t, res in results.items():
            assert res["threshold"] == t
            assert 0 < res["emr"] <= 1.0


# ===========================================================================
# ComplexityMetrics class
# ===========================================================================

class TestComplexityMetricsClass:
    """Integration tests for the unified ComplexityMetrics class."""

    def setup_method(self):
        self.cm = ComplexityMetrics()

    def test_compute_returns_result(self):
        """compute() returns a ComplexityResult with all fields."""
        e = np.abs(np.random.default_rng(0).standard_normal(196))
        result = self.cm.compute(e)
        assert isinstance(result, ComplexityResult)
        assert 0.0 <= result.gini <= 1.0
        assert 0.0 <= result.entropy_norm <= 1.0
        assert 0.0 <= result.emr_90 <= 1.0
        assert result.k_star_90 >= 1

    def test_to_dict_flat(self):
        """to_dict() returns a flat dict with the required keys."""
        e = np.ones(196)
        result = self.cm.compute(e)
        d = result.to_dict()
        required_keys = {
            "gini", "entropy_raw", "entropy_norm", "n_patches",
            "emr_50", "emr_90", "emr_95", "k_star_90",
            "model_name", "explainer_name", "sample_id",
        }
        assert required_keys == set(d.keys())

    def test_n_patches_correct(self):
        """n_patches should equal the input length."""
        for N in [196, 256, 784]:
            e = np.ones(N)
            result = self.cm.compute(e)
            assert result.n_patches == N

    def test_2d_input_accepted(self):
        """2-D (H_p, W_p) attribution maps are accepted."""
        e_2d = np.abs(np.random.default_rng(3).standard_normal((14, 14)))
        result = self.cm.compute(e_2d)
        assert result.n_patches == 196

    def test_signed_input_abs_taken(self):
        """Signed attribution maps are handled via abs() internally."""
        e_pos = np.array([1.0, 2.0, 3.0, 4.0])
        e_neg = np.array([-1.0, -2.0, -3.0, -4.0])
        r_pos = self.cm.compute(e_pos)
        r_neg = self.cm.compute(e_neg)
        # abs(e_neg) == e_pos → identical results
        assert abs(r_pos.gini - r_neg.gini) < 1e-9

    def test_metadata_propagated(self):
        """Metadata fields are propagated to ComplexityResult."""
        e = np.ones(196)
        result = self.cm.compute(
            e, model_name="vit_b16", explainer_name="rollout", sample_id="img_0"
        )
        assert result.model_name     == "vit_b16"
        assert result.explainer_name == "rollout"
        assert result.sample_id      == "img_0"

    def test_compute_batch_list(self):
        """compute_batch() accepts a list of arrays."""
        maps = [np.abs(np.random.default_rng(i).standard_normal(196)) for i in range(5)]
        results = self.cm.compute_batch(maps)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, ComplexityResult)

    def test_compute_batch_as_dict(self):
        """compute_batch_as_dict() returns lists keyed by metric name."""
        maps = [np.abs(np.random.default_rng(i).standard_normal(196)) for i in range(4)]
        d = self.cm.compute_batch_as_dict(maps)
        assert "gini" in d
        assert len(d["gini"]) == 4

    def test_aggregate(self):
        """aggregate() returns mean/std/median for all scalar fields."""
        maps = [np.abs(np.random.default_rng(i).standard_normal(196)) for i in range(10)]
        results = self.cm.compute_batch(maps)
        agg = self.cm.aggregate(results)
        assert "gini_mean"  in agg
        assert "emr_90_std" in agg
        assert "n_patches"  in agg

    def test_required_thresholds_validated(self):
        """Constructor raises if required thresholds are missing."""
        with pytest.raises(ValueError, match="must include"):
            ComplexityMetrics(emr_thresholds=[0.5, 0.9])   # missing 0.95

    def test_warn_on_zero(self):
        """All-zero map triggers RuntimeWarning when warn_on_zero=True."""
        cm = ComplexityMetrics(warn_on_zero=True)
        with pytest.warns(RuntimeWarning, match="all zeros"):
            cm.compute(np.zeros(196))

    def test_no_warn_if_disabled(self):
        """No warning when warn_on_zero=False."""
        cm = ComplexityMetrics(warn_on_zero=False)
        # Should not raise or warn
        cm.compute(np.zeros(196))


# ===========================================================================
# Normalise Attribution  (shared utility)
# ===========================================================================

class TestNormaliseAttribution:
    def test_minmax_range(self):
        e = np.array([1.0, 3.0, 2.0, 5.0])
        out = normalise_attribution(e, mode="minmax")
        assert abs(out.min() - 0.0) < 1e-9
        assert abs(out.max() - 1.0) < 1e-9

    def test_minmax_constant_map(self):
        e = np.ones(10)
        out = normalise_attribution(e, mode="minmax")
        assert np.all(out == 0.0)

    def test_softmax_sums_to_one(self):
        e = np.abs(np.random.default_rng(1).standard_normal(100))
        out = normalise_attribution(e, mode="softmax")
        assert abs(out.sum() - 1.0) < 1e-9

    def test_softmax_all_positive(self):
        e = np.abs(np.random.default_rng(2).standard_normal(100))
        out = normalise_attribution(e, mode="softmax")
        assert np.all(out > 0)

    def test_percentile_clips_outlier(self):
        e = np.ones(100); e[0] = 1000.0    # single outlier
        out = normalise_attribution(e, mode="percentile")
        assert out.max() <= 1.0 + 1e-9

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            normalise_attribution(np.ones(10), mode="invalid")


# ===========================================================================
# Theorem T6 — Gini anti-alignment with A3 (Symmetry)
# ===========================================================================

class TestTheoremT6:
    """
    Theorem T6: Gini (C1) rewards A3 (Symmetry) violations.

    Breaking symmetry between two equal-contribution patches by concentrating
    all mass on one increases Gini, even though total mass is preserved.
    """

    def test_gini_symmetry_antialignment(self):
        """Gini increases when symmetry is broken between equal patches."""
        N = 196
        e_sym  = np.zeros(N); e_sym[0]  = 0.5; e_sym[1]  = 0.5
        e_asym = np.zeros(N); e_asym[0] = 1.0; e_asym[1] = 0.0

        # Same total mass
        assert abs(e_sym.sum() - e_asym.sum()) < 1e-9

        g_sym  = gini_coefficient(e_sym)
        g_asym = gini_coefficient(e_asym)
        assert g_asym > g_sym, (
            f"T6: Gini(asym)={g_asym:.4f} should exceed Gini(sym)={g_sym:.4f}"
        )

    def test_entropy_symmetry_alignment(self):
        """Entropy decreases when symmetry is broken (opposite direction to Gini)."""
        N = 196
        e_sym  = np.zeros(N); e_sym[0]  = 0.5; e_sym[1]  = 0.5
        e_asym = np.zeros(N); e_asym[0] = 1.0; e_asym[1] = 0.0

        h_sym  = attribution_entropy(e_sym)["entropy_norm"]
        h_asym = attribution_entropy(e_asym)["entropy_norm"]
        assert h_sym > h_asym, (
            f"T6 entropy: H_norm(sym)={h_sym:.4f} should exceed H_norm(asym)={h_asym:.4f}"
        )

    def test_emr_symmetry_alignment(self):
        """EMR decreases when symmetry is broken."""
        N = 196
        e_sym  = np.zeros(N); e_sym[0]  = 0.5; e_sym[1]  = 0.5
        e_asym = np.zeros(N); e_asym[0] = 1.0; e_asym[1] = 0.0

        emr_sym  = effective_mass_ratio(e_sym,  threshold=0.9)["emr"]
        emr_asym = effective_mass_ratio(e_asym, threshold=0.9)["emr"]
        assert emr_sym >= emr_asym, (
            f"T6 EMR: emr(sym)={emr_sym:.4f} should be >= emr(asym)={emr_asym:.4f}"
        )


# ===========================================================================
# Sanity check (expected orderings)
# ===========================================================================

class TestSanityCheck:
    """Validates that the sanity check script runs and assertions pass."""

    def test_run_sanity_check(self):
        """run_sanity_check() must complete without assertion errors."""
        run_sanity_check(N=196)   # prints output; raises AssertionError on failure
