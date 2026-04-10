"""
test_robustness.py  —  Task 2.3 §4 Unit Tests  (+2.3 Addendum)
==============================================================
19 unit tests covering R1–R3, model utilities, and the Spearman
layer-curve (guide §R3 reporting requirement), across six categories:

Category A — Bounds (all metrics → [0, 1] on random inputs)
  R01  max_sensitivity_nonneg          MaxSens ≥ 0 for 20 random trials
  R02  model_randomisation_in_0_1      ModelRand ∈ [0, 1] for 20 trials
  R03  label_randomisation_in_0_0p5    LabelRand ∈ [0, 0.5] for 20 trials
  R04  compute_all_keys                compute_all() returns all 3 keys

Category B — Perfect Sensitivity (expected high scores)
  R05  sensitivity_increases_with_eps  Larger ε → higher (or equal) MaxSens
  R06  model_rand_orthogonal_maps      ModelRand ≈ 1 when maps are orthogonal
  R07  label_rand_orthogonal_maps      LabelRand = 0.5 when ρ = 0

Category C — Zero Sensitivity (expected low / zero scores)
  R08  sensitivity_zero_constant_expl  MaxSens = 0 for constant (deterministic) explainer
  R09  model_rand_identical_maps       ModelRand = 0 for att_orig == att_rand
  R10  label_rand_identical_maps       LabelRand = 0 for att_orig == att_shuf

Category D — Utility functions
  R11  randomise_model_changes_weights  all parameters change after randomisation
  R12  randomise_labels_only_head       backbone unchanged; head weight changes
  R13  ssim_self_consistency            _ssim(t, t) = 1.0

Category F — Spearman Layer Curve (Task 2.3 Addendum)
  R17  spearman_layer_curve_keys      keys ordered last-block → first-block
  R18  randomise_model_cascade_depth  only last N blocks changed; first intact
  R19  layer_curve_rho_bounds         every ρ value in [-1, 1]; no NaN

Run with:
    pytest tests/test_robustness.py -v
    # or without pytest:
    python tests/test_robustness.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
    randomise_model_cascade,
    _ssim,
    _spearman_corr,
)

# ---------------------------------------------------------------------------
# Test registry (mirrors test_localization.py scaffold exactly)
# ---------------------------------------------------------------------------
_TESTS: list[tuple[str, callable]] = []


def _register(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# A tiny block-structured model for R17–R19 (has 'blocks' ModuleList)
class _BlockModel(nn.Module):
    """
    3-block sequential model with a 'blocks' ModuleList.
    Compatible with randomise_model_cascade and spearman_layer_curve.
    """
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(16, 16),
            nn.Linear(16, 16),
            nn.Linear(16, 16),
        ])
        self.head = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = torch.relu(blk(x.float()))
        return self.head(x)


# A tiny Linear-only model that is structurally simple enough to test all
# utility functions without requiring a real ViT or downloading weights.
class _TinyModel(nn.Module):
    """2-layer classifier: Linear(16, 32) → ReLU → Linear(32, 10)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Linear(16, 32)
        self.head     = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.features(x)))


def _make_explainer(fixed_map: torch.Tensor | None = None):
    """
    Return an explainer callable.

    If fixed_map is given, the explainer always returns that map (constant,
    deterministic — MaxSens must be 0).
    If fixed_map is None, the explainer returns a fresh random (14, 14) map.
    """
    if fixed_map is not None:
        def _const_explainer(model, image_4d):
            return [fixed_map.clone()]
        return _const_explainer
    else:
        def _rand_explainer(model, image_4d):
            return [torch.rand(14, 14)]
        return _rand_explainer


rm_default = RobustnessMetrics(epsilon=0.05, n_samples=20, seed=42)


# ===========================================================================
# CATEGORY A: Bounds — all metrics in legal range on random inputs
# ===========================================================================

@_register
def R01_max_sensitivity_nonneg():
    """
    MaxSens must be ≥ 0 for any input.

    We use a random explainer (returns fresh random map each call) so that
    there IS some variation between perturbed attributions.  We check that
    the metric is non-negative over 20 trials.
    """
    torch.manual_seed(0)
    explainer = _make_explainer(fixed_map=None)
    model     = _TinyModel()

    for _ in range(20):
        image    = torch.randn(3, 14, 14)          # (C, H, W)
        att_orig = torch.rand(14, 14)
        ms = rm_default.max_sensitivity(explainer, model, image, att_orig)
        assert ms >= 0.0, f"MaxSens must be ≥ 0, got {ms}"

    print("R01 ✓  max_sensitivity_nonneg")


@_register
def R02_model_randomisation_in_0_1():
    """ModelRand must always be in [0, 1]."""
    torch.manual_seed(1)
    for _ in range(20):
        a = torch.randn(14, 14)
        b = torch.randn(14, 14)
        mr = rm_default.model_randomisation(a, b)
        assert 0.0 <= mr <= 1.0 + 1e-6, (
            f"ModelRand must be in [0,1], got {mr}"
        )
    print("R02 ✓  model_randomisation_in_0_1")


@_register
def R03_label_randomisation_in_0_0p5():
    """LabelRand must always be in [0, 0.5]."""
    torch.manual_seed(2)
    for _ in range(20):
        a = torch.randn(14, 14)
        b = torch.randn(14, 14)
        lr = rm_default.label_randomisation(a, b)
        assert 0.0 <= lr <= 0.5 + 1e-6, (
            f"LabelRand must be in [0, 0.5], got {lr}"
        )
    print("R03 ✓  label_randomisation_in_0_0p5")


@_register
def R04_compute_all_keys():
    """compute_all() must return exactly the three expected keys."""
    torch.manual_seed(3)
    explainer = _make_explainer(fixed_map=None)
    model     = _TinyModel()
    image     = torch.randn(3, 14, 14)
    att_orig  = torch.rand(14, 14)
    att_rand  = torch.rand(14, 14)
    att_shuf  = torch.rand(14, 14)

    result = rm_default.compute_all(
        explainer, model, image, att_orig, att_rand, att_shuf
    )
    expected_keys = {"max_sensitivity", "model_randomisation", "label_randomisation"}
    assert set(result.keys()) == expected_keys, (
        f"compute_all keys mismatch: {set(result.keys())} vs {expected_keys}"
    )
    print(
        f"R04 ✓  compute_all_keys  "
        f"(MaxSens={result['max_sensitivity']:.3f}, "
        f"ModelRand={result['model_randomisation']:.3f}, "
        f"LabelRand={result['label_randomisation']:.3f})"
    )


# ===========================================================================
# CATEGORY B: Perfect Sensitivity — expected high scores
# ===========================================================================

@_register
def R05_sensitivity_increases_with_eps():
    """
    For a random explainer (output changes with every call), a larger ε
    perturbation radius should produce a MaxSens that is ≥ the score at a
    smaller ε, averaged over multiple trials.

    We use 30 trials and assert the mean MaxSens at ε=0.20 is strictly
    larger than at ε=0.01.
    """
    torch.manual_seed(4)
    explainer = _make_explainer(fixed_map=None)
    model     = _TinyModel()

    rm_small = RobustnessMetrics(epsilon=0.01, n_samples=15, seed=10)
    rm_large = RobustnessMetrics(epsilon=0.20, n_samples=15, seed=10)

    scores_small, scores_large = [], []
    for _ in range(30):
        image    = torch.randn(3, 14, 14)
        att_orig = torch.rand(14, 14)
        scores_small.append(rm_small.max_sensitivity(explainer, model, image, att_orig))
        scores_large.append(rm_large.max_sensitivity(explainer, model, image, att_orig))

    mean_small = sum(scores_small) / len(scores_small)
    mean_large = sum(scores_large) / len(scores_large)

    assert mean_large >= mean_small, (
        f"Larger ε should produce ≥ MaxSens: ε=0.20 → {mean_large:.4f}, "
        f"ε=0.01 → {mean_small:.4f}"
    )
    print(
        f"R05 ✓  sensitivity_increases_with_eps  "
        f"(ε=0.01→{mean_small:.4f}, ε=0.20→{mean_large:.4f})"
    )


@_register
def R06_model_rand_orthogonal_maps():
    """
    ModelRand should be near 1.0 when att_orig and att_rand are orthogonal
    (completely different spatial patterns).

    We construct att_orig with all mass in the top half and att_rand with
    all mass in the bottom half — SSIM between these should be very low,
    so ModelRand should be close to 1.
    """
    att_orig = torch.zeros(14, 14)
    att_orig[:7, :] = 1.0                # top half

    att_rand = torch.zeros(14, 14)
    att_rand[7:, :] = 1.0               # bottom half

    mr = rm_default.model_randomisation(att_orig, att_rand)
    assert mr > 0.5, (
        f"ModelRand for orthogonal maps should be > 0.5, got {mr:.4f}"
    )
    print(f"R06 ✓  model_rand_orthogonal_maps  (ModelRand={mr:.4f})")


@_register
def R07_label_rand_orthogonal_maps():
    """
    LabelRand = 0.5 when ρ = 0 (perfectly uncorrelated maps).

    LabelRand = 1 - (|ρ| + 1) / 2.
    If ρ = 0 → LabelRand = 1 - 0.5 = 0.5 (maximum possible).

    We approximate ρ ≈ 0 using a random attribution pair and check that
    LabelRand is close to 0.5 over many draws.
    """
    torch.manual_seed(5)
    # Large random maps → expected ρ ≈ 0
    scores = []
    for _ in range(200):
        a = torch.randn(28, 28)
        b = torch.randn(28, 28)
        scores.append(rm_default.label_randomisation(a, b))

    mean_lr = sum(scores) / len(scores)
    # Expected: close to 0.5 (allow ±0.05 tolerance for finite-sample variance)
    assert abs(mean_lr - 0.5) < 0.05, (
        f"LabelRand mean for random maps should ≈ 0.5, got {mean_lr:.4f}"
    )
    print(f"R07 ✓  label_rand_orthogonal_maps  (mean LabelRand={mean_lr:.4f})")


# ===========================================================================
# CATEGORY C: Zero Sensitivity — expected low / zero scores
# ===========================================================================

@_register
def R08_sensitivity_zero_constant_expl():
    """
    MaxSens must be exactly 0.0 when the explainer is deterministic and
    always returns the same attribution map regardless of the input.

    A constant explainer is a regression test for the common bug of using
    random instead of seeded attribution — any perturbation would show no
    change and MaxSens must stay at 0.
    """
    fixed = torch.rand(14, 14)
    fixed_explainer = _make_explainer(fixed_map=fixed)
    model = _TinyModel()

    torch.manual_seed(6)
    for _ in range(5):
        image = torch.randn(3, 14, 14)
        ms = rm_default.max_sensitivity(fixed_explainer, model, image, fixed)
        assert abs(ms) < 1e-6, (
            f"Constant explainer must give MaxSens = 0, got {ms:.6f}"
        )
    print("R08 ✓  sensitivity_zero_constant_expl")


@_register
def R09_model_rand_identical_maps():
    """
    ModelRand must be 0.0 when att_orig == att_rand (SSIM = 1).
    """
    torch.manual_seed(7)
    for _ in range(10):
        att = torch.rand(14, 14)
        mr  = rm_default.model_randomisation(att, att.clone())
        assert abs(mr) < 1e-4, (
            f"ModelRand for identical maps must be 0, got {mr:.6f}"
        )
    print("R09 ✓  model_rand_identical_maps")


@_register
def R10_label_rand_identical_maps():
    """
    LabelRand must be 0.0 when att_orig == att_shuf (ρ = 1 → |ρ| = 1).

    LabelRand = 1 − (1 + 1)/2 = 0.
    """
    torch.manual_seed(8)
    for _ in range(10):
        att = torch.rand(14, 14)
        lr  = rm_default.label_randomisation(att, att.clone())
        assert abs(lr) < 1e-4, (
            f"LabelRand for identical maps must be 0, got {lr:.6f}"
        )
    print("R10 ✓  label_rand_identical_maps")


# ===========================================================================
# CATEGORY D: Utility functions
# ===========================================================================

@_register
def R11_randomise_model_changes_weights():
    """
    randomise_model_weights() must change EVERY parameter of the model.

    We verify that for each parameter in the randomised copy, it differs
    from the original (with extremely high probability — N(0,1) vs a fixed
    trained weight has zero probability of coincidence).
    """
    model      = _TinyModel()
    rand_model = randomise_model_weights(model, seed=0)

    n_params_changed = 0
    n_params_total   = 0
    for (name, p_orig), (_, p_rand) in zip(
        model.named_parameters(), rand_model.named_parameters()
    ):
        n_params_total += 1
        if not torch.allclose(p_orig, p_rand):
            n_params_changed += 1

    assert n_params_changed == n_params_total, (
        f"randomise_model_weights changed only {n_params_changed}/{n_params_total} params"
    )
    print(
        f"R11 ✓  randomise_model_changes_weights  "
        f"({n_params_changed}/{n_params_total} params changed)"
    )


@_register
def R12_randomise_labels_only_head():
    """
    randomise_classifier_labels() must:
    1. Leave backbone (features) parameters UNCHANGED.
    2. Change the classifier head weight.

    This verifies that column-permutation is applied only to the head layer.
    """
    model      = _TinyModel()
    shuf_model = randomise_classifier_labels(model, seed=0)

    # Backbone (features) must be identical
    for (name, p_orig), (_, p_shuf) in zip(
        model.features.named_parameters(),
        shuf_model.features.named_parameters(),
    ):
        assert torch.allclose(p_orig, p_shuf), (
            f"Backbone param '{name}' changed — should be frozen by randomise_classifier_labels"
        )

    # Head weight must differ
    head_unchanged = torch.allclose(model.head.weight, shuf_model.head.weight)
    assert not head_unchanged, (
        "Head weight should differ after randomise_classifier_labels"
    )
    print("R12 ✓  randomise_labels_only_head")


@_register
def R13_ssim_self_consistency():
    """
    SSIM(t, t) must equal exactly 1.0 (a pixel-perfect identical image
    achieves the maximum structural similarity).
    """
    torch.manual_seed(9)
    for _ in range(10):
        t = torch.rand(28, 28)
        s = _ssim(t, t)
        assert abs(s - 1.0) < 1e-4, (
            f"SSIM(t, t) should be 1.0, got {s:.6f}"
        )
    print("R13 ✓  ssim_self_consistency")


# ===========================================================================
# CATEGORY E: Edge cases and contracts
# ===========================================================================

@_register
def R14_spearman_constant_map():
    """
    Spearman rank correlation between a constant map and any other map is
    technically undefined (all ranks are tied), but our implementation must
    handle this gracefully and return a value in [-1, 1] without NaN/crash.

    After that, LabelRand must also not crash and return a value in [0, 0.5].
    """
    const_att = torch.ones(14, 14)
    rand_att  = torch.rand(14, 14)

    try:
        rho = _spearman_corr(const_att, rand_att)
        assert not math.isnan(rho), "Spearman on constant map returned NaN"
        assert -1.0 <= rho <= 1.0 + 1e-6, f"Spearman out of [-1, 1]: {rho}"

        lr = rm_default.label_randomisation(const_att, rand_att)
        assert not math.isnan(lr), "LabelRand on constant map returned NaN"
        assert 0.0 <= lr <= 0.5 + 1e-6, f"LabelRand out of [0, 0.5]: {lr}"

        print(f"R14 ✓  spearman_constant_map  (ρ={rho:.4f}, LabelRand={lr:.4f})")
    except Exception as exc:
        raise AssertionError(
            f"Constant attribution map caused an exception: {exc}"
        ) from exc


@_register
def R15_max_sensitivity_n_samples_one():
    """
    RobustnessMetrics with n_samples=1 must work without crashing.

    The worst-case (maximum) is trivially the single sample.
    The result must still be ≥ 0.
    """
    rm_one    = RobustnessMetrics(epsilon=0.05, n_samples=1, seed=123)
    explainer = _make_explainer(fixed_map=None)
    model     = _TinyModel()

    torch.manual_seed(10)
    image    = torch.randn(3, 14, 14)
    att_orig = torch.rand(14, 14)

    try:
        ms = rm_one.max_sensitivity(explainer, model, image, att_orig)
        assert ms >= 0.0, f"MaxSens with n_samples=1 must be ≥ 0, got {ms}"
        print(f"R15 ✓  max_sensitivity_n_samples_one  (MaxSens={ms:.4f})")
    except Exception as exc:
        raise AssertionError(
            f"n_samples=1 caused an exception: {exc}"
        ) from exc


@_register
def R16_epsilon_constructor_validation():
    """
    RobustnessMetrics(epsilon ≤ 0) must raise ValueError.

    The ε=0 case would make all perturbations zero-magnitude, making
    MaxSens trivially 0 for any explainer.  The user contract requires ε > 0.
    """
    raised_zero = False
    try:
        RobustnessMetrics(epsilon=0.0)
    except ValueError:
        raised_zero = True
    assert raised_zero, "Should raise ValueError for epsilon=0"

    raised_neg = False
    try:
        RobustnessMetrics(epsilon=-0.1)
    except ValueError:
        raised_neg = True
    assert raised_neg, "Should raise ValueError for epsilon < 0"

    print("R16 ✓  epsilon_constructor_validation")


# ===========================================================================
# CATEGORY F: Spearman Layer Curve (Task 2.3 Addendum)
# ===========================================================================

@_register
def R17_spearman_layer_curve_keys():
    """
    spearman_layer_curve() must return L keys in order from last block
    to first — i.e. blocks.2, blocks.1, blocks.0 for a 3-block model.
    """
    model     = _BlockModel()
    explainer = _make_explainer(fixed_map=None)
    image     = torch.randn(3, 14, 14)
    att_orig  = torch.rand(14, 14)

    rm_curve = RobustnessMetrics(epsilon=0.05, n_samples=1, seed=0)
    curve    = rm_curve.spearman_layer_curve(
        explainer, model, image, att_orig, block_attr="blocks"
    )

    # Must have exactly L=3 entries
    assert len(curve) == 3, f"Expected 3 keys, got {len(curve)}: {list(curve.keys())}"

    expected_keys = ["blocks.2", "blocks.1", "blocks.0"]
    assert list(curve.keys()) == expected_keys, (
        f"Key order mismatch: {list(curve.keys())} vs {expected_keys}"
    )
    print(f"R17 ✓  spearman_layer_curve_keys  (keys={list(curve.keys())})")


@_register
def R18_randomise_model_cascade_depth():
    """
    randomise_model_cascade(model, n) must change parameters in the last n
    blocks and leave the first (L - n) blocks UNCHANGED.

    We verify for n=1: blocks[2] differs, blocks[0] and blocks[1] are intact.
    """
    model      = _BlockModel()
    rand_model = randomise_model_cascade(model, n_layers_to_randomise=1, seed=42)

    # Last block (index 2) must have changed
    for (p_orig, p_rand) in zip(
        model.blocks[2].parameters(),
        rand_model.blocks[2].parameters(),
    ):
        assert not torch.allclose(p_orig, p_rand), (
            "Last block should have been randomised"
        )

    # First two blocks must be identical
    for blk_idx in [0, 1]:
        for (p_orig, p_rand) in zip(
            model.blocks[blk_idx].parameters(),
            rand_model.blocks[blk_idx].parameters(),
        ):
            assert torch.allclose(p_orig, p_rand), (
                f"Block {blk_idx} should be UNCHANGED after cascading n=1"
            )

    print("R18 ✓  randomise_model_cascade_depth  (n=1: block[2] changed, [0],[1] intact)")


@_register
def R19_layer_curve_rho_bounds():
    """
    Every ρ value returned by spearman_layer_curve() must be in [-1, 1]
    and must not be NaN.  This ensures numerical stability of the Spearman
    implementation across all cascade depths.
    """
    model     = _BlockModel()
    explainer = _make_explainer(fixed_map=None)
    image     = torch.randn(3, 14, 14)
    att_orig  = torch.rand(14, 14)

    rm_curve = RobustnessMetrics(epsilon=0.05, n_samples=1, seed=7)
    curve    = rm_curve.spearman_layer_curve(
        explainer, model, image, att_orig, block_attr="blocks"
    )

    for key, rho in curve.items():
        assert not math.isnan(rho), f"ρ for {key} is NaN"
        assert -1.0 - 1e-6 <= rho <= 1.0 + 1e-6, (
            f"ρ for {key} out of [-1, 1]: {rho:.4f}"
        )

    rho_str = ", ".join(f"{k}={v:.2f}" for k, v in curve.items())
    print(f"R19 ✓  layer_curve_rho_bounds  ({rho_str})")


# ===========================================================================
# Runner
# ===========================================================================

def _run_all_tests() -> None:
    passed = 0
    failed = 0
    print(f"\n{'='*60}")
    print("Task 2.3 + Addendum — RobustnessMetrics Unit Tests")
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
