"""
test_localization.py  —  Task 2.2 §4 Unit Tests
================================================
12 unit tests covering L1–L4 across three categories:

Category A — Random input → chance-level output
  T01  random_iou_not_one
  T02  random_pg_within_bounds
  T03  random_egt_within_bounds

Category B — Perfect input → maximum output
  T04  perfect_iou_is_one
  T05  perfect_pg_is_one
  T06  perfect_egt_is_one
  T07  perfect_calibgap_positive

Category C — GT-misaligned input → minimum output
  T08  misaligned_iou_is_zero
  T09  misaligned_pg_is_zero
  T10  misaligned_egt_near_zero

Category D — Edge cases and contracts
  T11  constant_att_map_iou
  T12  calibgap_empty_list_raises

Run with:
    pytest tests/test_localization.py -v
    # or without pytest:
    python tests/test_localization.py
"""

from __future__ import annotations

import math
import random
import sys
import traceback
from pathlib import Path

import torch

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.localization import LocalizationMetrics

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------
_TESTS: list[tuple[str, callable]] = []


def _register(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


lm = LocalizationMetrics(thresholds=[0.25, 0.50, 0.75], seed=42)

# ===========================================================================
# CATEGORY A: Random input → chance-level output
# ===========================================================================

@_register
def T01_random_iou_not_one():
    """
    A random attribution map over a random GT mask must NOT achieve perfect
    mIoU (= 1.0) with overwhelming probability.

    We repeat 20 independent trials and assert none achieves mIoU = 1.0.
    """
    torch.manual_seed(0)
    for _ in range(20):
        att  = torch.rand(14, 14)
        mask = (torch.rand(14, 14) > 0.5).float()
        result = lm.multi_threshold_iou(att, mask)
        assert result["miou"] < 1.0, (
            f"Random map achieved mIoU=1.0 — suspicious.  "
            f"Thresholds: {result}"
        )
    print("T01 ✓  random_iou_not_one")


@_register
def T02_random_pg_within_bounds():
    """
    Pointing game on random inputs must return exactly 0.0 or 1.0 — never
    an intermediate value.  Also checks that over 100 trials the result
    is not consistently 1.0 (which would suggest a bug in tie-breaking).
    """
    torch.manual_seed(1)
    hits = 0
    N    = 100
    for _ in range(N):
        att  = torch.rand(14, 14)
        mask = (torch.rand(14, 14) > 0.7).float()   # ~30% foreground
        pg   = lm.pointing_game(att, mask)
        assert pg in (0.0, 1.0), f"PG must be 0 or 1, got {pg}"
        hits += int(pg)
    # With 30% foreground, expected hit rate ≈ 0.30 ± sampling noise.
    # Assert it's not trivially 100% (which would indicate a bug).
    assert hits < N, f"All {N} random PG trials hit — bug in argmax/tie-breaking?"
    print(f"T02 ✓  random_pg_within_bounds  (hits={hits}/{N})")


@_register
def T03_random_egt_within_bounds():
    """
    EGT must always lie in [0, 1] for any input.
    """
    torch.manual_seed(2)
    for _ in range(50):
        att  = torch.randn(28, 28)        # DINO-scale: 28×28
        mask = (torch.rand(224, 224) > 0.6).float()
        e    = lm.egt(att, mask)
        assert 0.0 <= e <= 1.0 + 1e-6, (
            f"EGT out of [0,1]: {e}"
        )
    print("T03 ✓  random_egt_within_bounds")


# ===========================================================================
# CATEGORY B: Perfect input → maximum output
# ===========================================================================

@_register
def T04_perfect_iou_is_one():
    """
    When the attribution map is 1.0 inside the GT mask and 0.0 outside,
    multi_threshold_iou must return mIoU = 1.0 at all thresholds.

    This tests that the min-max normalisation and binarisation pipeline
    is end-to-end correct.
    """
    gt   = torch.zeros(14, 14)
    gt[4:10, 4:10] = 1.0                 # 6×6 square foreground
    att  = gt.clone()                    # perfect attribution
    result = lm.multi_threshold_iou(att, gt)

    assert abs(result["miou"] - 1.0) < 1e-5, (
        f"Perfect map should give mIoU=1.0, got {result['miou']:.6f}"
    )
    for τ in [0.25, 0.50, 0.75]:
        key = f"iou@{τ:.2f}"
        assert abs(result[key] - 1.0) < 1e-5, (
            f"{key} should be 1.0 for perfect map, got {result[key]}"
        )
    print("T04 ✓  perfect_iou_is_one")


@_register
def T05_perfect_pg_is_one():
    """
    When the peak of the attribution map lies strictly within the GT mask,
    pointing_game must return 1.0.
    """
    gt  = torch.zeros(14, 14)
    gt[7, 7] = 1.0                       # single foreground pixel
    att = torch.zeros(14, 14)
    att[7, 7] = 10.0                     # peak exactly at GT pixel
    pg  = lm.pointing_game(att, gt)
    assert pg == 1.0, f"Expected PG=1.0, got {pg}"
    print("T05 ✓  perfect_pg_is_one")


@_register
def T06_perfect_egt_is_one():
    """
    When all attribution mass is inside the GT mask, EGT must equal 1.0.

    We set att = +∞ inside GT and -∞ outside, so softmax concentrates
    100% of mass on GT pixels.
    """
    gt  = torch.zeros(14, 14, dtype=torch.float32)
    gt[3:8, 3:8]  = 1.0                 # 5×5 foreground

    att = torch.full((14, 14), -1e9)    # effectively zero softmax mass
    att[3:8, 3:8] = +1e9               # all mass inside GT

    e   = lm.egt(att, gt)
    assert abs(e - 1.0) < 1e-4, (
        f"Perfect EGT should be 1.0, got {e:.6f}"
    )
    print("T06 ✓  perfect_egt_is_one")


@_register
def T07_perfect_calibgap_positive():
    """
    CalibGap must be > 0 when correct-prediction attributions perfectly
    overlap GT and incorrect-prediction attributions perfectly miss GT.
    """
    gt_in  = torch.zeros(14, 14)
    gt_in[5:9, 5:9] = 1.0              # foreground region

    # Correct predictions: att peaks inside GT
    att_perfect = torch.zeros(14, 14)
    att_perfect[5:9, 5:9] = 10.0

    # Incorrect predictions: att peaks outside GT
    att_wrong = torch.zeros(14, 14)
    att_wrong[0:2, 0:2] = 10.0         # top-left corner — outside GT

    atts_c = [att_perfect] * 5
    atts_i = [att_wrong]   * 5
    masks  = [gt_in] * 10

    gap = lm.calibration_gap(atts_c, atts_i, gt_masks=masks)
    assert gap > 0.0, (
        f"CalibGap should be positive for perfect-vs-wrong split, got {gap:.4f}"
    )
    print(f"T07 ✓  perfect_calibgap_positive  (gap={gap:.4f})")


# ===========================================================================
# CATEGORY C: GT-misaligned input → minimum output
# ===========================================================================

@_register
def T08_misaligned_iou_is_zero():
    """
    When attribution is entirely outside the GT mask, mIoU must be 0.0
    at τ = 0.75 (strict threshold means only high-attribution pixels are
    predicted foreground).
    """
    gt  = torch.zeros(14, 14)
    gt[0:5, 0:5] = 1.0                  # top-left foreground

    att = torch.zeros(14, 14)
    att[9:14, 9:14] = 1.0              # bottom-right — no overlap with GT

    result = lm.multi_threshold_iou(att, gt)
    # At τ=0.75, predicted mask covers bottom-right; GT is top-left.
    assert result["iou@0.75"] == 0.0, (
        f"Misaligned IoU@0.75 should be 0.0, got {result['iou@0.75']}"
    )
    print("T08 ✓  misaligned_iou_is_zero")


@_register
def T09_misaligned_pg_is_zero():
    """
    When the attribution peak is strictly outside the GT mask, PG = 0.0.
    """
    gt  = torch.zeros(14, 14)
    gt[3:8, 3:8] = 1.0                  # centre foreground

    att = torch.zeros(14, 14)
    att[0, 0]    = 10.0                # peak at top-left corner (outside GT)

    pg = lm.pointing_game(att, gt)
    assert pg == 0.0, f"Expected PG=0.0 for misaligned peak, got {pg}"
    print("T09 ✓  misaligned_pg_is_zero")


@_register
def T10_misaligned_egt_near_zero():
    """
    When attribution mass is concentrated entirely outside the GT mask,
    EGT must be close to 0.0.
    """
    gt  = torch.zeros(14, 14)
    gt[6:10, 6:10] = 1.0               # 4×4 foreground in centre-right

    att = torch.full((14, 14), -1e9)
    att[0:2, 0:2] = +1e9               # all mass in top-left (outside GT)

    e = lm.egt(att, gt)
    assert e < 0.01, (
        f"Misaligned EGT should be ~0, got {e:.6f}"
    )
    print(f"T10 ✓  misaligned_egt_near_zero  (egt={e:.2e})")


# ===========================================================================
# CATEGORY D: Edge cases and contracts
# ===========================================================================

@_register
def T11_constant_att_map_iou():
    """
    A constant attribution map (all equal values) is the degenerate case:
    min-max normalisation produces 0/0.  The implementation must handle
    this gracefully by returning mIoU = 0.0 rather than NaN or an exception.
    """
    gt  = torch.zeros(14, 14)
    gt[4:10, 4:10] = 1.0
    att = torch.ones(14, 14)            # all-constant map

    try:
        result = lm.multi_threshold_iou(att, gt)
        v = result["miou"]
        assert not math.isnan(v), f"Constant map produced NaN mIoU"
        assert 0.0 <= v <= 1.0,   f"Constant map mIoU out of [0,1]: {v}"
        print(f"T11 ✓  constant_att_map_iou  (miou={v:.4f})")
    except Exception as e:
        raise AssertionError(
            f"Constant attribution map caused an exception: {e}"
        ) from e


@_register
def T12_calibgap_empty_list_raises():
    """
    calibration_gap() must raise ValueError when either input list is empty.
    This enforces the contract that CalibGap is undefined without samples
    in at least one partition.
    """
    att  = torch.rand(14, 14)
    mask = (torch.rand(14, 14) > 0.5).float()

    raised_correct = False
    try:
        lm.calibration_gap([], [att], gt_masks=[mask])
    except ValueError:
        raised_correct = True
    assert raised_correct, "Should raise ValueError when atts_correct is empty"

    raised_incorrect = False
    try:
        lm.calibration_gap([att], [], gt_masks=[mask])
    except ValueError:
        raised_incorrect = True
    assert raised_incorrect, "Should raise ValueError when atts_incorrect is empty"

    print("T12 ✓  calibgap_empty_list_raises")


# ===========================================================================
# Runner
# ===========================================================================

def _run_all_tests() -> None:
    passed = 0
    failed = 0
    print(f"\n{'='*60}")
    print("Task 2.2 §4 — LocalizationMetrics Unit Tests")
    print(f"{'='*60}\n")

    for name, fn in _TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:
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
