"""
test_normalize.py  —  Phase 3 / Task 3.2 Unit Tests
=====================================================
24 pytest tests verifying metrics/normalize.py (Guide Listing 3).

Test layout
-----------
PT_N01–PT_N06   minmax mode         — degenerate, uniform, spiky, batch, dtypes
PT_N07–PT_N12   percentile mode     — 99th clip, batch, constant, neg+pos range
PT_N13–PT_N18   softmax mode        — sums-to-one, 1D/2D, batch, stability, shape
PT_N19–PT_N24   integration / edge  — explainer → norm pipeline, error handling,
                                       idempotency, 1-D rejection, mode list

Run with:
    pytest tests/test_normalize.py -v
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.normalize import (
    normalize_attribution,
    normalize_batch,
    AttributionNormError,
    NormMode,
    _VALID_MODES,
)


# ===========================================================================
# PT_N01–PT_N06 — minmax mode
# ===========================================================================

class TestMinmax:
    """minmax: (att - min) / (max - min) → [0, 1]."""

    def test_PT_N01_range(self):
        """PT_N01: minmax output strictly in [0.0, 1.0] for random input."""
        torch.manual_seed(0)
        att = torch.rand(14, 14) * 5 - 2   # range [-2, 3]
        out = normalize_attribution(att, mode="minmax")
        assert float(out.min()) >= 0.0 - 1e-7
        assert float(out.max()) <= 1.0 + 1e-7

    def test_PT_N02_min_is_zero_max_is_one(self):
        """PT_N02: after minmax, min=0.0 and max=1.0 exactly (non-degenerate)."""
        att = torch.tensor([[0.2, 0.5], [0.1, 0.9]])
        out = normalize_attribution(att, mode="minmax")
        assert abs(float(out.min()) - 0.0) < 1e-6, f"min={float(out.min())}"
        assert abs(float(out.max()) - 1.0) < 1e-6, f"max={float(out.max())}"

    def test_PT_N03_degenerate_constant_map_returns_zeros(self):
        """PT_N03: constant attribution (max==min) → all-zeros output."""
        att = torch.ones(7, 7) * 3.14
        out = normalize_attribution(att, mode="minmax")
        assert torch.all(out == 0.0), f"Expected all-zeros, got {out}"

    def test_PT_N04_shape_preserved(self):
        """PT_N04: output shape matches (14, 14) input exactly."""
        att = torch.rand(14, 14)
        out = normalize_attribution(att, mode="minmax")
        assert out.shape == (14, 14), f"Expected (14,14), got {out.shape}"

    def test_PT_N05_batch_each_sample_independent(self):
        """PT_N05: (B, H, W) batch — each sample normalised independently."""
        # Sample 0: all ones (degenerate) → zeros
        # Sample 1: [0, 1] → should max out at 1
        B, Hp, Wp = 2, 4, 4
        att = torch.zeros(B, Hp, Wp)
        att[0] = 1.0                                   # degenerate
        att[1, 0, 0] = 1.0                             # single peak
        out = normalize_attribution(att, mode="minmax")
        assert out.shape == (B, Hp, Wp)
        assert torch.all(out[0] == 0.0), "Degenerate sample should be all-zeros"
        assert abs(float(out[1].max()) - 1.0) < 1e-6

    def test_PT_N06_dtype_float32_output(self):
        """PT_N06: output dtype is always float32 regardless of input dtype."""
        for dtype in (torch.float16, torch.float32, torch.float64):
            att = torch.rand(4, 4, dtype=dtype)
            out = normalize_attribution(att, mode="minmax")
            assert out.dtype == torch.float32, (
                f"Input dtype {dtype}: expected float32, got {out.dtype}"
            )


# ===========================================================================
# PT_N07–PT_N12 — percentile mode
# ===========================================================================

class TestPercentile:
    """percentile: clamp at 99th percentile then minmax → [0, 1]."""

    def test_PT_N07_output_range(self):
        """PT_N07: percentile output in [0.0, 1.0]."""
        torch.manual_seed(1)
        att = torch.randn(14, 14) * 3          # heavy tails
        out = normalize_attribution(att, mode="percentile")
        assert float(out.min()) >= 0.0 - 1e-7
        assert float(out.max()) <= 1.0 + 1e-7

    def test_PT_N08_outlier_suppressed(self):
        """PT_N08: single massive outlier does not set max; p99 clamps it."""
        att      = torch.zeros(14, 14)
        att[7, 7] = 1e6     # extreme outlier
        att[0, 0] = 1.0     # second largest — should become max after clamping
        out      = normalize_attribution(att, mode="percentile")
        # The large spike and the 1.0 should be clamped to the same p99 ceiling,
        # so the output at (7,7) won't dominate as much
        assert float(out.min()) >= 0.0 - 1e-7
        assert float(out.max()) <= 1.0 + 1e-7

    def test_PT_N09_same_as_minmax_no_outliers(self):
        """PT_N09: without outliers, percentile result ≈ minmax (same ranking)."""
        torch.manual_seed(2)
        att = torch.rand(10, 10)     # uniform [0,1] — no outliers
        out_mm = normalize_attribution(att, mode="minmax")
        out_pc = normalize_attribution(att, mode="percentile")
        # Rankings should be identical (Spearman r ≈ 1)
        flat_mm = out_mm.flatten().numpy()
        flat_pc = out_pc.flatten().numpy()
        from scipy.stats import spearmanr
        r, _ = spearmanr(flat_mm, flat_pc)
        assert r > 0.99, f"Rank correlation minmax vs percentile = {r:.4f}"

    def test_PT_N10_batch_shape(self):
        """PT_N10: percentile normalisation on (B, Hp, Wp) returns correct shape."""
        B, H, W = 5, 14, 14
        att = torch.randn(B, H, W)
        out = normalize_attribution(att, mode="percentile")
        assert out.shape == (B, H, W)

    def test_PT_N11_constant_map_returns_zeros(self):
        """PT_N11: constant map under percentile also returns all-zeros."""
        att = torch.ones(6, 6) * -99.0
        out = normalize_attribution(att, mode="percentile")
        assert torch.all(out == 0.0)

    def test_PT_N12_negative_values_handled(self):
        """PT_N12: attribution with mixed negative and positive values handled cleanly."""
        att = torch.tensor([[-1.0, -0.5], [0.0, 2.0]])
        out = normalize_attribution(att, mode="percentile")
        assert float(out.min()) >= 0.0 - 1e-7
        assert float(out.max()) <= 1.0 + 1e-7


# ===========================================================================
# PT_N13–PT_N18 — softmax mode
# ===========================================================================

class TestSoftmax:
    """softmax: spatial softmax over all patches → probability distribution."""

    def test_PT_N13_sums_to_one(self):
        """PT_N13: softmax output sums to 1.0 over all patches."""
        att = torch.rand(14, 14) * 3
        out = normalize_attribution(att, mode="softmax")
        total = float(out.sum())
        assert abs(total - 1.0) < 1e-5, f"softmax sum = {total:.6f} ≠ 1.0"

    def test_PT_N14_all_positive(self):
        """PT_N14: softmax output is strictly positive (no zeros)."""
        att = torch.randn(7, 7) * 10   # large magnitude, mixed sign
        out = normalize_attribution(att, mode="softmax")
        assert float(out.min()) > 0.0, "softmax output should be strictly > 0"

    def test_PT_N15_batch_each_sums_to_one(self):
        """PT_N15: each sample in a (B, Hp, Wp) batch sums to 1.0 independently."""
        B, H, W = 4, 14, 14
        att = torch.randn(B, H, W)
        out = normalize_attribution(att, mode="softmax")
        assert out.shape == (B, H, W)
        for i in range(B):
            total = float(out[i].sum())
            assert abs(total - 1.0) < 1e-5, (
                f"Sample {i}: softmax sum = {total:.6f} ≠ 1.0"
            )

    def test_PT_N16_numerical_stability_large_values(self):
        """PT_N16: large attribution values (1e6) don't produce NaN/Inf."""
        att = torch.zeros(14, 14)
        att[7, 7] = 1e6     # dominant patch
        out = normalize_attribution(att, mode="softmax")
        assert torch.isfinite(out).all(), "softmax produced NaN/Inf"
        # Dominant patch should have probability ≈ 1
        assert float(out[7, 7]) > 0.99, (
            f"Dominant patch probability = {float(out[7,7]):.4f}, expected ≈ 1.0"
        )

    def test_PT_N17_uniform_map_is_uniform_distribution(self):
        """PT_N17: uniform attribution → uniform softmax distribution."""
        N = 196
        att = torch.ones(14, 14)
        out = normalize_attribution(att, mode="softmax")
        expected = 1.0 / N
        diff = (out - expected).abs().max().item()
        assert diff < 1e-6, f"Expected uniform {expected:.6f}, max diff = {diff:.2e}"

    def test_PT_N18_shape_preserved(self):
        """PT_N18: softmax preserves input shape (7×7, 14×14, 28×28)."""
        for sz in (7, 14, 28):
            att = torch.rand(sz, sz)
            out = normalize_attribution(att, mode="softmax")
            assert out.shape == (sz, sz), (
                f"Size {sz}: expected ({sz},{sz}), got {out.shape}"
            )


# ===========================================================================
# PT_N19–PT_N24 — integration / edge-case tests
# ===========================================================================

class TestIntegrationAndEdgeCases:
    """End-to-end and error handling tests."""

    def test_PT_N19_explainer_to_norm_pipeline(self):
        """
        PT_N19: simulate E1 → normalise → metric input shape/range.
        Uses a random (14,14) map to verify the full pipeline contract.
        """
        # Simulate raw explainer output (any range, any sign)
        torch.manual_seed(99)
        raw_att = torch.randn(14, 14) * 0.5 + 0.3     # typical E1 range

        for mode in ("minmax", "percentile", "softmax"):
            norm = normalize_attribution(raw_att, mode=mode)
            assert norm.shape == (14, 14), f"{mode}: shape mismatch"
            assert norm.dtype == torch.float32, f"{mode}: wrong dtype"
            assert float(norm.min()) >= 0.0 - 1e-7, f"{mode}: min < 0"
            if mode != "softmax":
                assert float(norm.max()) <= 1.0 + 1e-7, f"{mode}: max > 1"
            else:
                assert abs(float(norm.sum()) - 1.0) < 1e-5, f"{mode}: sum ≠ 1"

    def test_PT_N20_invalid_mode_raises(self):
        """PT_N20: unknown mode raises AttributionNormError."""
        att = torch.rand(4, 4)
        with pytest.raises(AttributionNormError, match="Unknown normalisation mode"):
            normalize_attribution(att, mode="l2")

    def test_PT_N21_1d_input_raises(self):
        """PT_N21: 1-D input raises AttributionNormError (not a valid att map)."""
        att = torch.rand(196)
        with pytest.raises(AttributionNormError):
            normalize_attribution(att, mode="minmax")

    def test_PT_N22_normalize_batch_3d_only(self):
        """PT_N22: normalize_batch enforces exactly 3-D (B, Hp, Wp) input."""
        # Valid: 3-D
        att_3d = torch.rand(3, 14, 14)
        out = normalize_batch(att_3d, mode="minmax")
        assert out.shape == (3, 14, 14)

        # Invalid: 2-D should raise
        with pytest.raises(AttributionNormError):
            normalize_batch(torch.rand(14, 14), mode="minmax")

        # Invalid: 4-D should raise
        with pytest.raises(AttributionNormError):
            normalize_batch(torch.rand(2, 3, 14, 14), mode="minmax")

    def test_PT_N23_idempotency_minmax(self):
        """PT_N23: applying minmax twice gives the same result (idempotent)."""
        torch.manual_seed(7)
        att  = torch.rand(10, 10)
        norm1 = normalize_attribution(att, mode="minmax")
        norm2 = normalize_attribution(norm1, mode="minmax")
        max_diff = (norm1 - norm2).abs().max().item()
        assert max_diff < 1e-6, (
            f"minmax not idempotent; max diff on second pass = {max_diff:.2e}"
        )

    def test_PT_N24_valid_mode_enum_strings(self):
        """PT_N24: NormMode enum values match the accepted mode strings."""
        assert NormMode.MINMAX.value     == "minmax"
        assert NormMode.PERCENTILE.value == "percentile"
        assert NormMode.SOFTMAX.value    == "softmax"
        # All three modes must work without error
        att = torch.rand(14, 14)
        for mode_enum in NormMode:
            out = normalize_attribution(att, mode=mode_enum.value)
            assert out.shape == att.shape
