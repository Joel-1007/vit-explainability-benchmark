"""
test_complexity.py  —  Task 2.4 Unit Tests
==========================================
12 unit tests for ComplexityMetrics (C1–C3) across four categories:

Category A — Bounds (all metrics in legal range, all 3 keys returned)
  C01  random_gini_in_0_1         Gini ∈ [0, 1] for 20 random maps
  C02  random_sparsity_in_0_1     Sparsity ∈ [0, 1] for 20 random maps
  C03  random_effres_in_0_1       EffRes ∈ (0, 1] for 20 random maps
  C04  compute_all_keys           compute_all() returns exactly 3 keys

Category B — Perfect Complexity (single-pixel / highly concentrated maps)
  C05  single_pixel_gini          Single-pixel map → Gini ≈ 1.0
  C06  single_pixel_sparsity      Single-pixel map → Sparsity = 1.0
  C07  single_pixel_effres        Single-pixel map → EffRes ≈ 0 (1 pixel / HW)

Category C — Zero Complexity (uniform maps)
  C08  uniform_gini_is_zero       Uniform map → Gini = 0.0
  C09  uniform_sparsity           Uniform map → Sparsity ≈ k_fraction
  C10  uniform_effres_is_one      Uniform map → EffRes = 1.0 (full BBox)

Category D — Edge cases and contracts
  C11  all_zero_map               All-zero attribution → no crash, legal range
  C12  k_fraction_validation      k_fraction ≤ 0 or > 1 → ValueError

Run with:
    python tests/test_complexity.py
    # or
    pytest tests/test_complexity.py -v
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.complexity import ComplexityMetrics

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
_TESTS: list[tuple[str, callable]] = []


def _register(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


cm_default = ComplexityMetrics(k_fraction=0.05, threshold=0.5)


# ===========================================================================
# CATEGORY A: Bounds
# ===========================================================================

@_register
def C01_random_gini_in_0_1():
    """Gini must be in [0, 1] for any random map."""
    torch.manual_seed(0)
    for _ in range(20):
        att = torch.randn(14, 14)
        g   = cm_default.gini_coefficient(att)
        assert 0.0 <= g <= 1.0 + 1e-6, f"Gini out of [0,1]: {g}"
    print("C01 ✓  random_gini_in_0_1")


@_register
def C02_random_sparsity_in_0_1():
    """Sparsity must be in [0, 1] for any random map."""
    torch.manual_seed(1)
    for _ in range(20):
        att = torch.randn(14, 14)
        s   = cm_default.sparsity(att)
        assert 0.0 <= s <= 1.0 + 1e-6, f"Sparsity out of [0,1]: {s}"
    print("C02 ✓  random_sparsity_in_0_1")


@_register
def C03_random_effres_in_0_1():
    """EffRes must be in (0, 1] for any random map."""
    torch.manual_seed(2)
    for _ in range(20):
        att = torch.randn(14, 14)
        er  = cm_default.effective_resolution(att)
        assert 0.0 < er <= 1.0 + 1e-6, f"EffRes out of (0,1]: {er}"
    print("C03 ✓  random_effres_in_0_1")


@_register
def C04_compute_all_keys():
    """compute_all() must return exactly the three expected keys."""
    torch.manual_seed(3)
    att    = torch.randn(14, 14)
    result = cm_default.compute_all(att)
    expected = {"gini", "sparsity", "effective_resolution"}
    assert set(result.keys()) == expected, (
        f"compute_all keys mismatch: {set(result.keys())} vs {expected}"
    )
    print(
        f"C04 ✓  compute_all_keys  "
        f"(Gini={result['gini']:.3f}, "
        f"Sparsity={result['sparsity']:.3f}, "
        f"EffRes={result['effective_resolution']:.3f})"
    )


# ===========================================================================
# CATEGORY B: Perfect Complexity (single-pixel map)
# ===========================================================================

@_register
def C05_single_pixel_gini():
    """
    A map with all attribution on ONE pixel has maximum Gini ≈ 1.0.

    With n pixels and Lorenz formulation, a single-pixel map (spike)
    approaches Gini = 1 - 1/n as n → ∞.  For 14×14 = 196 pixels,
    Gini = 1 - 1/196 ≈ 0.995.  We threshold at > 0.99.
    """
    att = torch.zeros(14, 14)
    att[7, 7] = 1.0   # single spike

    g = cm_default.gini_coefficient(att)
    assert g > 0.99, f"Single-pixel map Gini should be > 0.99, got {g:.4f}"
    print(f"C05 ✓  single_pixel_gini  (Gini={g:.4f})")


@_register
def C06_single_pixel_sparsity():
    """
    A map with all attribution on ONE pixel has Sparsity = 1.0.

    The single non-zero pixel (if k ≥ 1) captures all the mass.
    """
    # k = ceil(0.05 * 196) = ceil(9.8) = 10 pixels → still covers the spike
    att = torch.zeros(14, 14)
    att[7, 7] = 100.0

    s = cm_default.sparsity(att)
    assert abs(s - 1.0) < 1e-5, f"Single-pixel map Sparsity should be 1.0, got {s:.6f}"
    print(f"C06 ✓  single_pixel_sparsity  (Sparsity={s:.4f})")


@_register
def C07_single_pixel_effres():
    """
    A map with attribution on ONE pixel has EffRes = 1/(H*W).

    After normalisation, the single pixel is the only one ≥ 0.5.
    Its BBox is 1 pixel, so EffRes = 1 / (14*14) ≈ 0.0051.
    """
    att       = torch.zeros(14, 14)
    att[7, 7] = 1.0

    er      = cm_default.effective_resolution(att)
    expected = 1.0 / (14 * 14)
    assert abs(er - expected) < 1e-4, (
        f"Single-pixel EffRes should be ≈ {expected:.4f}, got {er:.6f}"
    )
    print(f"C07 ✓  single_pixel_effres  (EffRes={er:.4f})")


# ===========================================================================
# CATEGORY C: Zero Complexity (uniform maps)
# ===========================================================================

@_register
def C08_uniform_gini_is_zero():
    """
    A perfectly uniform attribution map has Gini = 0.0.

    All pixels equal → Lorenz curve is the diagonal → Gini = 0.
    """
    att = torch.ones(14, 14)
    g   = cm_default.gini_coefficient(att)
    assert abs(g) < 1e-5, f"Uniform map Gini should be 0.0, got {g:.6f}"
    print(f"C08 ✓  uniform_gini_is_zero  (Gini={g:.6f})")


@_register
def C09_uniform_sparsity():
    """
    A perfectly uniform map has Sparsity = 0.0.

    Sparsity shifts the attribution to non-negative values before computing
    the top-k fraction: ẽ = e - min(e).  For a uniform map, ẽ = 0 everywhere.
    Since total mass = 0, there is no mass to concentrate, so Sparsity = 0.

    This is correct behaviour: a uniform map is the least sparse possible
    (all pixels equally attributed), not the most.  The metric should NOT
    return k/n here — that would only hold if the raw values were already
    non-negative and non-zero.
    """
    att = torch.ones(28, 28)
    s   = cm_default.sparsity(att)
    assert abs(s) < 1e-5, (
        f"Uniform map Sparsity should be 0.0 (no mass after shift), got {s:.6f}"
    )
    print(f"C09 ✓  uniform_sparsity  (Sparsity={s:.4f} — correctly 0.0 for uniform map)")


@_register
def C10_uniform_effres_is_one():
    """
    A perfectly uniform map has EffRes = 1.0.

    After min-max normalisation, a constant map returns all zeros
    (constant guard returns zeros in _minmax_norm equivalent).
    ComplexityMetrics.effective_resolution handles constant maps by
    returning 1.0 (worst case — diffuse attribution).
    """
    att = torch.ones(14, 14)
    er  = cm_default.effective_resolution(att)
    assert abs(er - 1.0) < 1e-5, f"Uniform map EffRes should be 1.0, got {er:.6f}"
    print(f"C10 ✓  uniform_effres_is_one  (EffRes={er:.4f})")


# ===========================================================================
# CATEGORY D: Edge cases and contracts
# ===========================================================================

@_register
def C11_all_zero_map():
    """
    All-zero attribution map must not crash and must return legal values.

    Gini → 0 (all equal at zero), Sparsity → 0 (no mass),
    EffRes → 1.0 (constant map treated as diffuse).
    """
    att = torch.zeros(14, 14)
    try:
        g  = cm_default.gini_coefficient(att)
        s  = cm_default.sparsity(att)
        er = cm_default.effective_resolution(att)

        assert 0.0 <= g  <= 1.0 + 1e-6, f"Gini out of range: {g}"
        assert 0.0 <= s  <= 1.0 + 1e-6, f"Sparsity out of range: {s}"
        assert 0.0 <  er <= 1.0 + 1e-6, f"EffRes out of range: {er}"

        print(
            f"C11 ✓  all_zero_map  "
            f"(Gini={g:.3f}, Sparsity={s:.3f}, EffRes={er:.3f})"
        )
    except Exception as exc:
        raise AssertionError(f"All-zero map caused exception: {exc}") from exc


@_register
def C12_k_fraction_validation():
    """
    ComplexityMetrics(k_fraction ≤ 0) and (k_fraction > 1) must raise ValueError.
    """
    raised_zero = False
    try:
        ComplexityMetrics(k_fraction=0.0)
    except ValueError:
        raised_zero = True
    assert raised_zero, "Should raise ValueError for k_fraction=0.0"

    raised_neg = False
    try:
        ComplexityMetrics(k_fraction=-0.1)
    except ValueError:
        raised_neg = True
    assert raised_neg, "Should raise ValueError for k_fraction < 0"

    raised_over = False
    try:
        ComplexityMetrics(k_fraction=1.5)
    except ValueError:
        raised_over = True
    assert raised_over, "Should raise ValueError for k_fraction > 1"

    print("C12 ✓  k_fraction_validation")


# ===========================================================================
# Runner
# ===========================================================================

def _run_all_tests() -> None:
    passed = 0
    failed = 0
    print(f"\n{'='*60}")
    print("Task 2.4 — ComplexityMetrics Unit Tests")
    print(f"{'='*60}\n")

    for name, fn in _TESTS:
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
            print()

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print(f"{'='*60}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all_tests()
