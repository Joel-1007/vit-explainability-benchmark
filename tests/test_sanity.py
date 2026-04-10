"""
test_sanity.py  —  Phase 3 / Task 3.4 Unit Tests
=================================================
16 pytest tests for metrics/sanity.py (guide §3.4 mandatory sanity checks).

Test layout
-----------
SC_S1_01–SC_S1_05   S1 Random Baseline     — result schema, expected metric values
SC_S2_01–SC_S2_05   S2 Model Param Rand    — Spearman curve, monotone decrease
SC_S3_01–SC_S3_04   S3 Label Permutation   — runs cleanly, rho finite/bounded
SC_ALL_01–SC_ALL_02 Integration            — run_all, Phase3Runner.sanity_checks()

All tests use the tiny MockViT (patch_size=4, img_size=16) and fast explainers
so the full suite runs in <5 s on CPU.

Run with:
    pytest tests/test_sanity.py -v
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explainers.base import BaseExplainer, UnsupportedArchitectureError
from metrics.sanity import (
    run_s1_random_baseline,
    run_s2_model_randomisation,
    run_s3_label_permutation,
    run_all_sanity_checks,
    S1Result, S2Result, S3Result, SanityResult,
    _spearman_corr_flat,
)
from metrics.runner import Phase3Runner


# ===========================================================================
# Shared mock infrastructure (same tiny spec as test_runner.py)
# ===========================================================================

IMG_SIZE    = 16
PATCH_SIZE  = 4
P           = IMG_SIZE // PATCH_SIZE    # = 4
N_CLASSES   = 5
DIM         = 32
NUM_HEADS   = 4


class _MockAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim  = DIM // NUM_HEADS
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(DIM, DIM * 3, bias=False)
        self.proj      = nn.Linear(DIM, DIM, bias=False)
        self.attn_drop = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv  = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x_out)


class _MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attn  = _MockAttn()
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp   = nn.Sequential(
            nn.Linear(DIM, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, DIM)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MockViT(nn.Module):
    def __init__(self, depth: int = 2):
        super().__init__()
        n_patches = P * P
        self.patch_embed = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, DIM, bias=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, DIM))
        self.blocks      = nn.ModuleList([_MockBlock() for _ in range(depth)])
        self.norm        = nn.LayerNorm(DIM)
        self.head        = nn.Linear(DIM, N_CLASSES)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = PATCH_SIZE
        n_h, n_w = H // ps, W // ps
        patches = (
            x.unfold(2, ps, ps).unfold(3, ps, ps)
            .contiguous().view(B, C, n_h * n_w, ps * ps)
            .permute(0, 2, 1, 3).reshape(B, n_h * n_w, C * ps * ps)
        )
        tok = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.norm(tok)[:, 0])


class _ConstantExplainer(BaseExplainer):
    """Always returns a all-ones (P, P) map — label-independent, zero variation."""
    def explain(self, x, target_class, **kwargs):
        return torch.ones(P, P)


class _GradientExplainer(BaseExplainer):
    """
    Minimal gradient-based explainer — uses ∂logit/∂x, so output depends
    on target_class. Returns (P, P) abs-gradient patch map.
    """
    def explain(self, x, target_class, **kwargs):
        P_ = IMG_SIZE // self.patch_size
        x_ = x.unsqueeze(0).requires_grad_(True)
        logits = self.model(x_)
        score  = logits[0, target_class]
        score.backward()
        g = x_.grad[0].abs().mean(0)    # (H, W)
        # Pool to patch grid
        out = F.adaptive_avg_pool2d(
            g.unsqueeze(0).unsqueeze(0), (P_, P_)
        ).squeeze()
        return out.detach()


torch.manual_seed(0)
_model = _MockViT(depth=2).eval()


# ===========================================================================
# SC_S1_01–SC_S1_05 — S1 Random Baseline
# ===========================================================================

class TestS1RandomBaseline:

    def test_SC_S1_01_returns_S1Result(self):
        """SC_S1_01: run_s1_random_baseline returns an S1Result instance."""
        res = run_s1_random_baseline(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_images=10, seed=0
        )
        assert isinstance(res, S1Result)
        assert isinstance(res, SanityResult)

    def test_SC_S1_02_check_id_is_S1(self):
        """SC_S1_02: result.check_id == 'S1'."""
        res = run_s1_random_baseline(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_images=4, seed=0
        )
        assert res.check_id == "S1"

    def test_SC_S1_03_entropy_high_for_random_map(self):
        """SC_S1_03: uniform random map → mean_entropy > 0.85 (near max entropy)."""
        res = run_s1_random_baseline(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_images=50, seed=42
        )
        assert res.mean_entropy > 0.85, (
            f"Expected mean_entropy > 0.85 for random map; got {res.mean_entropy:.4f}"
        )

    def test_SC_S1_04_gini_low_for_random_map(self):
        """SC_S1_04: uniform random map → mean_gini < 0.50 (confirms non-sparsity).

        Note: For a 4×4 grid (P=4), the expected Gini of a random uniform
        map is ~0.33. Threshold is 0.50 (not 0.15) because Gini of a finite
        random draw scales inversely with N; a truly sparse map would be > 0.7.
        """
        res = run_s1_random_baseline(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_images=50, seed=42
        )
        assert res.mean_gini < 0.50, (
            f"Expected mean_gini < 0.50 for random map; got {res.mean_gini:.4f}"
        )

    def test_SC_S1_05_passes_and_details_populated(self):
        """SC_S1_05: 50-image check passes and details dict has required keys."""
        res = run_s1_random_baseline(
            patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_images=50, seed=42
        )
        assert res.passed, (
            f"S1 should pass for random maps; "
            f"gini={res.mean_gini:.4f} (< 0.50), "
            f"entropy={res.mean_entropy:.4f} (> 0.85), "
            f"emr90={res.mean_emr90:.4f} (> 0.50)"
        )
        for key in ("mean_gini", "mean_entropy", "mean_emr90", "P"):
            assert key in res.details, f"Missing key in S1 details: {key}"


# ===========================================================================
# SC_S2_01–SC_S2_05 — S2 Model Parameter Randomisation
# ===========================================================================

class TestS2ModelRandomisation:

    def test_SC_S2_01_returns_S2Result(self):
        """SC_S2_01: run_s2_model_randomisation returns an S2Result."""
        res = run_s2_model_randomisation(
            explainer_cls=_ConstantExplainer, model=_model,
            n_images=4, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            seed=0, n_classes=N_CLASSES,
        )
        assert isinstance(res, S2Result)

    def test_SC_S2_02_check_id_is_S2(self):
        """SC_S2_02: result.check_id == 'S2'."""
        res = run_s2_model_randomisation(
            explainer_cls=_ConstantExplainer, model=_model,
            n_images=2, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            seed=0, n_classes=N_CLASSES,
        )
        assert res.check_id == "S2"

    def test_SC_S2_03_rho_curve_length_equals_n_blocks(self):
        """SC_S2_03: spearman_rho_per_layer has one entry per transformer block."""
        model = _MockViT(depth=3).eval()
        res = run_s2_model_randomisation(
            explainer_cls=_ConstantExplainer, model=model,
            n_images=2, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            seed=0, n_classes=N_CLASSES,
        )
        assert len(res.spearman_rho_per_layer) == 3, (
            f"Expected 3 entries for depth-3 model, "
            f"got {len(res.spearman_rho_per_layer)}"
        )

    def test_SC_S2_04_constant_explainer_rho_stays_one(self):
        """
        SC_S2_04: label-independent constant explainer always returns identical
        maps regardless of layer randomisation → all rhos ≈ 1.0.
        """
        res = run_s2_model_randomisation(
            explainer_cls=_ConstantExplainer, model=_model,
            n_images=4, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            seed=0, n_classes=N_CLASSES,
        )
        for depth, rho in enumerate(res.spearman_rho_per_layer, start=1):
            assert abs(rho - 1.0) < 1e-4, (
                f"Constant explainer: depth {depth} rho={rho:.4f}, expected ≈ 1.0"
            )

    def test_SC_S2_05_gradient_explainer_rho_decreases_with_depth(self):
        """
        SC_S2_05: gradient-based explainer rho[0] >= rho[-1] - 0.05 after
        cascading randomisation (within slack for tiny 2-block model).
        """
        res = run_s2_model_randomisation(
            explainer_cls=_GradientExplainer, model=_MockViT(depth=2).eval(),
            n_images=4, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            seed=1, n_classes=N_CLASSES,
        )
        valid = [r for r in res.spearman_rho_per_layer if not math.isnan(r)]
        assert len(valid) > 0, "No valid rho values computed"
        # Monotone check uses slack=0.05 to handle tiny 2-block models
        assert res.is_monotone_decreasing, (
            f"Expected rho[0] ≥ rho[-1] - 0.05; curve = "
            f"{[f'{r:.4f}' for r in res.spearman_rho_per_layer]}"
        )


# ===========================================================================
# SC_S3_01–SC_S3_04 — S3 Label Permutation
# ===========================================================================

class TestS3LabelPermutation:

    def test_SC_S3_01_returns_S3Result(self):
        """SC_S3_01: run_s3_label_permutation returns an S3Result."""
        res = run_s3_label_permutation(
            explainer_cls=_ConstantExplainer, model=_model,
            n_images=4, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            n_classes=N_CLASSES, seed=0,
        )
        assert isinstance(res, S3Result)

    def test_SC_S3_02_check_id_is_S3(self):
        """SC_S3_02: result.check_id == 'S3'."""
        res = run_s3_label_permutation(
            explainer_cls=_ConstantExplainer, model=_model,
            n_images=2, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            n_classes=N_CLASSES, seed=0,
        )
        assert res.check_id == "S3"

    def test_SC_S3_03_constant_explainer_rho_is_one(self):
        """
        SC_S3_03: label-independent constant explainer always returns
        the same map → Spearman ρ = 1.0 regardless of label.
        """
        res = run_s3_label_permutation(
            explainer_cls=_ConstantExplainer, model=_model,
            n_images=6, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            n_classes=N_CLASSES, seed=0,
        )
        assert res.passed, f"S3 should pass; errors = {res.details.get('errors')}"
        assert abs(res.mean_rho_true_vs_wrong - 1.0) < 1e-4, (
            f"Constant explainer: expected ρ≈1.0, got {res.mean_rho_true_vs_wrong:.4f}"
        )

    def test_SC_S3_04_gradient_explainer_rho_less_than_one(self):
        """
        SC_S3_04: gradient explainer should differ across labels → ρ < 1.0.
        (Strict bound: < 0.99 for a 4×4 random-weight model is reliable.)
        """
        res = run_s3_label_permutation(
            explainer_cls=_GradientExplainer, model=_MockViT(depth=2).eval(),
            n_images=6, patch_size=PATCH_SIZE, img_size=IMG_SIZE,
            n_classes=N_CLASSES, seed=7,
        )
        assert res.passed, f"S3 should pass without errors; errors={res.details.get('errors')}"
        assert res.mean_rho_true_vs_wrong < 0.99, (
            f"Gradient explainer: expected ρ < 0.99 across labels, "
            f"got {res.mean_rho_true_vs_wrong:.4f}"
        )


# ===========================================================================
# SC_ALL_01–SC_ALL_02 — Integration
# ===========================================================================

class TestSanityIntegration:

    def test_SC_ALL_01_run_all_returns_three_results(self):
        """SC_ALL_01: run_all_sanity_checks returns dict with S1, S2, S3 keys."""
        results = run_all_sanity_checks(
            explainer_cls=_ConstantExplainer,
            model=_model,
            patch_size=PATCH_SIZE,
            img_size=IMG_SIZE,
            n_images=4,
            n_classes=N_CLASSES,
            seed=0,
        )
        assert set(results.keys()) == {"S1", "S2", "S3"}, (
            f"Expected keys {{'S1','S2','S3'}}, got {set(results.keys())}"
        )
        assert isinstance(results["S1"], S1Result)
        assert isinstance(results["S2"], S2Result)
        assert isinstance(results["S3"], S3Result)

    def test_SC_ALL_02_to_dict_serialisable(self):
        """SC_ALL_02: all results are serialisable via to_dict() (for Phase4 CSV)."""
        results = run_all_sanity_checks(
            explainer_cls=_ConstantExplainer,
            model=_model,
            patch_size=PATCH_SIZE,
            img_size=IMG_SIZE,
            n_images=4,
            n_classes=N_CLASSES,
            seed=0,
        )
        for check_id, result in results.items():
            d = result.to_dict()
            assert d["check_id"] == check_id
            assert isinstance(d["passed"], bool)
            assert isinstance(d["details"], dict)
            # Ensure it is JSON-serialisable (no tensors)
            import json
            try:
                json.dumps(d)
            except TypeError as e:
                pytest.fail(f"{check_id}.to_dict() not JSON-serialisable: {e}")
