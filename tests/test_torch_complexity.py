"""
test_torch_complexity.py
========================
Task 2.4 — 7 PyTorch-specific unit tests for metrics/complexity.py.

These tests exercise the GPU-native batch functions that are NOT covered by
the numpy-based test_complexity.py suite:

  PT_C1  gini_batch_torch        — output shape (B,)
  PT_C2  gini_batch_torch        — values match numpy reference (correctness)
  PT_C3  entropy_batch_torch     — normalised output in [0, 1]
  PT_C4  entropy_batch_torch     — uniform map → entropy_norm = 1.0
  PT_C5  emr_batch_torch         — one-hot map → EMR = 1/N
  PT_C6  ComplexityMetrics.compute_batch  accepts torch.Tensor (B, H_p, W_p)
  PT_C7  downsample_attribution  output shape and value range

Run with::

    pytest tests/test_torch_complexity.py -v

All 7 tests require PyTorch and are skipped gracefully when it is absent.
Expected runtime: < 2 seconds (CPU only).
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Skip guard — all tests in this file require torch
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="PyTorch not installed")

from metrics.complexity import (
    ComplexityMetrics,
    gini_batch_torch,
    entropy_batch_torch,
    emr_batch_torch,
    gini_coefficient,
    downsample_attribution,
)


# ===========================================================================
# PT_C1 — gini_batch_torch output shape
# ===========================================================================

class TestGiniBatchTorchShape:
    """PT_C1: gini_batch_torch returns a (B,) tensor for any valid batch."""

    def test_shape_standard(self):
        """Standard ViT-B/16 patch grid: B=4, 14×14 patches."""
        B, H_p, W_p = 4, 14, 14
        att = torch.rand(B, H_p, W_p).abs()
        result = gini_batch_torch(att)
        assert result.shape == (B,), (
            f"Expected shape ({B},), got {result.shape}"
        )

    def test_shape_flat_input(self):
        """Accepts (B, N) flattened input."""
        B, N = 8, 196
        att = torch.rand(B, N).abs()
        result = gini_batch_torch(att)
        assert result.shape == (B,)

    def test_dtype_float32(self):
        """Output dtype is float32."""
        att = torch.rand(2, 196).abs()
        result = gini_batch_torch(att)
        assert result.dtype == torch.float32


# ===========================================================================
# PT_C2 — gini_batch_torch values match numpy reference
# ===========================================================================

class TestGiniBatchTorchValues:
    """PT_C2: vectorised torch Gini must match numpy scalar implementation."""

    def test_matches_numpy_random(self):
        """Gini values agree with gini_coefficient() to within 1e-5."""
        rng = np.random.default_rng(42)
        B = 10
        maps_np = np.abs(rng.standard_normal((B, 196)))
        maps_t  = torch.tensor(maps_np, dtype=torch.float32)

        torch_result = gini_batch_torch(maps_t).numpy()
        numpy_result = np.array([gini_coefficient(maps_np[i]) for i in range(B)])

        np.testing.assert_allclose(
            torch_result, numpy_result, atol=1e-5,
            err_msg="gini_batch_torch diverges from numpy reference"
        )

    def test_one_hot_gini_equals_n_minus_1_over_n(self):
        """One-hot vector → Gini = (N-1)/N, both numpy and torch agree."""
        N = 196
        e = np.zeros(N); e[0] = 1.0
        expected = (N - 1.0) / N

        e_t = torch.tensor(e, dtype=torch.float32).unsqueeze(0)
        torch_val = float(gini_batch_torch(e_t)[0])
        numpy_val = gini_coefficient(e)

        assert abs(torch_val - expected) < 1e-5, f"torch: {torch_val:.6f} ≠ {expected:.6f}"
        assert abs(numpy_val - expected) < 1e-5, f"numpy: {numpy_val:.6f} ≠ {expected:.6f}"

    def test_uniform_gini_is_zero(self):
        """Uniform attribution → Gini = 0, from torch batch function."""
        e_t = torch.ones(1, 196)
        result = float(gini_batch_torch(e_t)[0])
        assert abs(result - 0.0) < 1e-5, f"Uniform Gini should be 0, got {result:.6f}"


# ===========================================================================
# PT_C3 — entropy_batch_torch range
# ===========================================================================

class TestEntropyBatchTorchRange:
    """PT_C3: normalised entropy must lie in [0, 1] for all valid inputs."""

    def test_range_random_maps(self):
        """100 random maps: all have entropy_norm in [0.0, 1.0]."""
        torch.manual_seed(7)
        maps = torch.rand(100, 196).abs()
        result = entropy_batch_torch(maps, normalise=True)
        assert result.shape == (100,)
        assert float(result.min()) >= -1e-6, f"Min entropy < 0: {float(result.min()):.6f}"
        assert float(result.max()) <= 1.0 + 1e-6, f"Max entropy > 1: {float(result.max()):.6f}"

    def test_range_one_hot(self):
        """
        Strongly spiked map (1 patch = 1e6, rest = 1) → entropy_norm near 0.

        Note: entropy_batch_torch uses softmax normalisation internally.
        A literal one-hot torch tensor (one 1.0, rest 0.0) gives softmax
        probability ≈ uniform (sum(exp(0)) dominates), so entropy stays
        high.  We use a large spike to force near-zero entropy instead.
        """
        N = 196
        e = torch.ones(1, N)          # start uniform
        e[0, 0] = 1e6                 # one patch has hugely dominant weight
        result = float(entropy_batch_torch(e, normalise=True)[0])
        assert result < 0.01, f"Strongly spiked map entropy_norm should ≈ 0, got {result:.4f}"


# ===========================================================================
# PT_C4 — entropy_batch_torch uniform map
# ===========================================================================

class TestEntropyBatchTorchUniform:
    """PT_C4: uniform map → maximum normalised entropy = 1.0."""

    def test_uniform_entropy_norm_is_one(self):
        """Uniform attribution → H_norm = 1.0 (worst-case parsimony)."""
        N = 196
        e = torch.ones(1, N)
        result = float(entropy_batch_torch(e, normalise=True)[0])
        assert abs(result - 1.0) < 1e-4, (
            f"Uniform map entropy_norm should be 1.0, got {result:.6f}"
        )

    def test_batch_of_uniforms(self):
        """All uniform → all entropy_norm values = 1.0."""
        B, N = 5, 196
        maps = torch.ones(B, N)
        result = entropy_batch_torch(maps, normalise=True)
        assert result.shape == (B,)
        for i, val in enumerate(result):
            assert abs(float(val) - 1.0) < 1e-4, (
                f"Sample {i}: expected 1.0, got {float(val):.6f}"
            )


# ===========================================================================
# PT_C5 — emr_batch_torch one-hot
# ===========================================================================

class TestEmrBatchTorchOneHot:
    """PT_C5: one-hot attribution requires only 1 patch → EMR = 1/N."""

    def test_one_hot_emr(self):
        """Single-patch attribution → EMR(0.9) = 1/N."""
        N = 196
        e = torch.zeros(1, N); e[0, 0] = 1.0
        result = float(emr_batch_torch(e, alpha=0.9)[0])
        expected = 1.0 / N
        assert abs(result - expected) < 1e-5, (
            f"One-hot EMR should be {expected:.6f}, got {result:.6f}"
        )

    def test_uniform_emr(self):
        """Uniform attribution → EMR(0.9) = 0.9 (need 90% of patches)."""
        N = 100
        e = torch.ones(1, N)
        result = float(emr_batch_torch(e, alpha=0.9)[0])
        assert abs(result - 0.9) < 0.02, (
            f"Uniform EMR should ≈ 0.9, got {result:.4f}"
        )

    def test_batch_shape(self):
        """emr_batch_torch returns (B,) for batch input."""
        B, H_p, W_p = 6, 14, 14
        maps = torch.rand(B, H_p, W_p).abs()
        result = emr_batch_torch(maps, alpha=0.9)
        assert result.shape == (B,), f"Expected ({B},), got {result.shape}"


# ===========================================================================
# PT_C6 — ComplexityMetrics.compute_batch accepts torch.Tensor
# ===========================================================================

class TestComputeBatchTorchTensor:
    """PT_C6: ComplexityMetrics.compute_batch() must accept a (B, H_p, W_p) tensor."""

    def setup_method(self):
        self.cm = ComplexityMetrics()

    def test_accepts_3d_tensor(self):
        """(B, H_p, W_p) torch.Tensor input produces B ComplexityResult objects."""
        B, H_p, W_p = 4, 14, 14
        maps = torch.rand(B, H_p, W_p).abs()
        results = self.cm.compute_batch(maps)
        assert len(results) == B, f"Expected {B} results, got {len(results)}"

    def test_tensor_values_valid(self):
        """All complexity values from torch.Tensor batch are in legal ranges."""
        maps = torch.rand(3, 14, 14).abs()
        results = self.cm.compute_batch(maps)
        for r in results:
            assert 0.0 <= r.gini        <= 1.0, f"Gini={r.gini:.4f} out of range"
            assert 0.0 <= r.entropy_norm <= 1.0, f"H_norm={r.entropy_norm:.4f} out of range"
            assert 0.0 <= r.emr_90      <= 1.0, f"EMR90={r.emr_90:.4f} out of range"
            assert r.n_patches == 196,          f"n_patches={r.n_patches} ≠ 196"

    def test_matches_per_sample_compute(self):
        """compute_batch result matches individual compute() calls."""
        torch.manual_seed(0)
        maps = torch.rand(3, 14, 14).abs()
        batch_results = self.cm.compute_batch(maps)
        for i in range(3):
            single = self.cm.compute(maps[i])
            assert abs(batch_results[i].gini - single.gini) < 1e-6, (
                f"Sample {i}: batch gini={batch_results[i].gini:.6f} ≠ "
                f"single gini={single.gini:.6f}"
            )


# ===========================================================================
# PT_C7 — downsample_attribution shape and value range
# ===========================================================================

class TestDownsampleAttribution:
    """PT_C7: downsample_attribution for cross-architecture alignment."""

    def test_output_shape_14x14(self):
        """DINO-ViT-B/8 (28×28) maps downsampled to ViT-B/16 (14×14) grid."""
        att = torch.rand(28, 28)
        result = downsample_attribution(att, target_size=14)
        assert result.shape == (14, 14), (
            f"Expected (14,14), got {result.shape}"
        )

    def test_output_shape_7x7(self):
        """Swin-B (7×7) alignment: downsample 14×14 → 7×7."""
        att = torch.rand(14, 14)
        result = downsample_attribution(att, target_size=7)
        assert result.shape == (7, 7), (
            f"Expected (7,7), got {result.shape}"
        )

    def test_nonnegative_output(self):
        """Downsampled positive attribution stays non-negative."""
        att = torch.rand(28, 28).abs()
        result = downsample_attribution(att, target_size=14)
        assert float(result.min()) >= -1e-6, (
            f"Negative values after bilinear downsample: min={float(result.min()):.6f}"
        )

    def test_rejects_non_2d(self):
        """3-D input should raise AssertionError."""
        att = torch.rand(3, 14, 14)
        with pytest.raises(AssertionError):
            downsample_attribution(att, target_size=7)
