"""
test_explainers.py  —  Phase 3 Task 3.1 Unit Tests
====================================================
27 pytest tests across 8 categories, all using a fast MockViT (no timm download).

Category A — Shape (7 tests)
    Each explainer returns exactly (P, P) where P = img_size // patch_size.

Category B — No NaN / Inf (7 tests)
    torch.isfinite(output).all() must hold for every explainer.

Category C — Batch consistency (3 tests)
    explain_batch result must match looped explain calls (E1, E2, E3).

Category D — Spatial variation (3 tests)
    Output std > 0 for non-trivial inputs (E1, E5, E6 specifically).

Category E — Swin-B UnsupportedArchitectureError (2 tests)
    E1 (RawAttention) and E2 (Rollout) raise the right exception.

Category F — RISE-specific (2 tests)
    Non-negative output; shape correctness at small resolution.

Category G — LIME-specific (2 tests)
    Finite output; regression produces S=P² coefficients.

Category H — DIME placeholder (1 test)
    DIMEExplainer.explain() raises NotImplementedError with a message
    that references BENCHMARK.md §3.1.

Run with:
    pytest tests/test_explainers.py -v
    # or directly:
    python tests/test_explainers.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainers import (
    RawAttentionExplainer,
    AttentionRolloutExplainer,
    GradCAMExplainer,
    CheferLRPExplainer,
    RISEExplainer,
    LIMEExplainer,
    DIMEExplainer,
    UnsupportedArchitectureError,
)

# ===========================================================================
# Mock models
# ===========================================================================

IMG_SIZE    = 16    # tiny test image (fast)
PATCH_SIZE  = 4     # 4×4 patches → P = 4 → output (4, 4)
P           = IMG_SIZE // PATCH_SIZE   # = 4
N_PATCHES   = P * P                   # = 16
N_TOKENS    = N_PATCHES + 1           # +1 for CLS
DIM         = 32
NUM_HEADS   = 4
NUM_CLASSES = 5


class _MockAttn(nn.Module):
    """Minimal self-attention with explicit attn_drop hook point."""

    def __init__(self, dim: int = DIM, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Identity()   # hook point: input = attention weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)   # ← hooked here

        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x_out)


class _MockBlock(nn.Module):
    def __init__(self, dim: int = DIM, num_heads: int = NUM_HEADS) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _MockAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MockViT(nn.Module):
    """Minimal ViT that supports all CLS-based explainers (E1–E4)."""

    def __init__(
        self,
        patch_size:  int = PATCH_SIZE,
        img_size:    int = IMG_SIZE,
        dim:         int = DIM,
        num_heads:   int = NUM_HEADS,
        depth:       int = 2,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        n_patches       = (img_size // patch_size) ** 2

        self.patch_embed = nn.Linear(3 * patch_size * patch_size, dim, bias=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks      = nn.ModuleList([_MockBlock(dim, num_heads) for _ in range(depth)])
        self.norm        = nn.LayerNorm(dim)
        self.head        = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size
        n_h, n_w = H // ps, W // ps

        # Manual patch extraction: (B, n_patches, C*ps*ps)
        patches = (
            x.unfold(2, ps, ps).unfold(3, ps, ps)
            .contiguous()
            .view(B, C, n_h * n_w, ps * ps)
            .permute(0, 2, 1, 3)
            .reshape(B, n_h * n_w, C * ps * ps)
        )
        tok = self.patch_embed(patches)                          # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)                  # (B, 1, D)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed     # (B, N+1, D)

        for blk in self.blocks:
            tok = blk(tok)

        tok = self.norm(tok)
        return self.head(tok[:, 0])   # CLS logits


class _MockSwinB(nn.Module):
    """
    Minimal Swin-B mock: has 'layers' (not 'blocks'), no CLS token.
    Used to verify UnsupportedArchitectureError.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dim: int = DIM) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self.head   = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        feat = x.float().mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, DIM)
        for ly in self.layers:
            feat = F.relu(ly(feat))
        return self.head(feat)


# ---------------------------------------------------------------------------
# Shared module-level instances (created once, reused across all tests)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
_vit   = _MockViT().eval()
_swin  = _MockSwinB().eval()
_img   = torch.rand(3, IMG_SIZE, IMG_SIZE)
_label = 2


# ===========================================================================
# CATEGORY A: Shape — each explainer returns (P, P)
# ===========================================================================

def test_E01_raw_attention_shape():
    """E01: RawAttentionExplainer output shape is (P, P)."""
    e   = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"


def test_E02_rollout_shape():
    """E02: AttentionRolloutExplainer output shape is (P, P)."""
    e   = AttentionRolloutExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"


def test_E03_gradcam_shape():
    """E03: GradCAMExplainer output shape is (P, P)."""
    e   = GradCAMExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"


def test_E04_chefer_lrp_shape():
    """E04: CheferLRPExplainer output shape is (P, P)."""
    e   = CheferLRPExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"


def test_E05_rise_shape():
    """E05: RISEExplainer output shape is (P, P)."""
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=50, chunk_size=10, seed=0)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"


def test_E06_lime_shape():
    """E06: LIMEExplainer output shape is (P, P)."""
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=20, seed=0)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"


def test_E07_dime_shape_not_tested_pending():
    """E07: DIMEExplainer raises NotImplementedError — shape test skipped with note."""
    e = DIMEExplainer(_vit, patch_size=PATCH_SIZE)
    with pytest.raises(NotImplementedError):
        e.explain(_img, _label)


# ===========================================================================
# CATEGORY B: No NaN / Inf
# ===========================================================================

def test_E08_raw_attention_no_nan():
    """E08: RawAttentionExplainer output contains no NaN or Inf."""
    e   = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"RawAttention output contains NaN/Inf: {out}"


def test_E09_rollout_no_nan():
    """E09: AttentionRolloutExplainer output contains no NaN or Inf."""
    e   = AttentionRolloutExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"Rollout output contains NaN/Inf: {out}"


def test_E10_gradcam_no_nan():
    """E10: GradCAMExplainer output contains no NaN or Inf."""
    e   = GradCAMExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"GradCAM output contains NaN/Inf: {out}"


def test_E11_chefer_lrp_no_nan():
    """E11: CheferLRPExplainer output contains no NaN or Inf."""
    e   = CheferLRPExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"CheferLRP output contains NaN/Inf: {out}"


def test_E12_rise_no_nan():
    """E12: RISEExplainer output contains no NaN or Inf."""
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=30, chunk_size=10, seed=0)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"RISE output contains NaN/Inf: {out}"


def test_E13_lime_no_nan():
    """E13: LIMEExplainer output contains no NaN or Inf."""
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=20, seed=0)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"LIME output contains NaN/Inf: {out}"


def test_E14_dime_no_nan_pending():
    """E14: Not tested — DIMEExplainer is a pending placeholder (documented skip)."""
    pytest.skip("DIME is a placeholder pending guide inconsistency resolution")


# ===========================================================================
# CATEGORY C: Batch consistency (explain_batch ≡ loop over explain)
# ===========================================================================

def _batch_consistency_check(explainer, atol: float = 1e-4) -> None:
    torch.manual_seed(7)
    B   = 3
    xs  = torch.rand(B, 3, IMG_SIZE, IMG_SIZE)
    cls = torch.randint(0, NUM_CLASSES, (B,))

    batch_out = explainer.explain_batch(xs, cls)       # (B, P, P)
    loop_out  = torch.stack([
        explainer.explain(xs[i], int(cls[i].item()))
        for i in range(B)
    ])
    max_diff = (batch_out - loop_out).abs().max().item()
    assert max_diff < atol, (
        f"explain_batch vs loop max diff = {max_diff:.2e} > {atol}"
    )


def test_E15_raw_attention_batch_consistency():
    """E15: RawAttentionExplainer.explain_batch matches looped explain."""
    e = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    _batch_consistency_check(e)


def test_E16_rollout_batch_consistency():
    """E16: AttentionRolloutExplainer.explain_batch matches looped explain."""
    e = AttentionRolloutExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    _batch_consistency_check(e)


def test_E17_gradcam_batch_consistency():
    """E17: GradCAMExplainer.explain_batch matches looped explain (loose tol)."""
    e = GradCAMExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    _batch_consistency_check(e, atol=1e-3)  # gradient methods: slightly looser


# ===========================================================================
# CATEGORY D: Spatial variation (output std > 0)
# ===========================================================================

def test_E18_raw_attention_has_variation():
    """E18: RawAttentionExplainer output has spatial variation (std > 0)."""
    torch.manual_seed(42)
    e   = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(torch.rand(3, IMG_SIZE, IMG_SIZE), _label)
    assert out.std() > 0, "RawAttention output is constant (zero variation)"


def test_E19_rise_has_variation():
    """E19: RISEExplainer output has spatial variation (std > 0)."""
    torch.manual_seed(43)
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=100, chunk_size=20, seed=0)
    out = e.explain(torch.rand(3, IMG_SIZE, IMG_SIZE), _label)
    assert out.std() > 0, "RISE output is constant (zero variation)"


def test_E20_lime_has_variation():
    """E20: LIMEExplainer output has spatial variation (std > 0)."""
    torch.manual_seed(44)
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=50, seed=1)
    out = e.explain(torch.rand(3, IMG_SIZE, IMG_SIZE), _label)
    assert out.std() > 0, "LIME output is constant (zero variation)"


# ===========================================================================
# CATEGORY E: Swin-B raises UnsupportedArchitectureError
# ===========================================================================

def test_E21_raw_attention_swin_raises():
    """E21: RawAttentionExplainer raises UnsupportedArchitectureError on Swin-B."""
    e = RawAttentionExplainer(_swin, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    with pytest.raises(UnsupportedArchitectureError):
        e.explain(_img, _label)


def test_E22_rollout_swin_raises():
    """E22: AttentionRolloutExplainer raises UnsupportedArchitectureError on Swin-B."""
    e = AttentionRolloutExplainer(_swin, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    with pytest.raises(UnsupportedArchitectureError):
        e.explain(_img, _label)


# ===========================================================================
# CATEGORY F: RISE-specific
# ===========================================================================

def test_E23_rise_nonnegative():
    """E23: RISE saliency = weighted sum of masked probabilities → output ≥ 0."""
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=50, chunk_size=10, seed=5)
    out = e.explain(_img, _label)
    assert out.min().item() >= -1e-6, f"RISE output has negative values: min={out.min():.6f}"


def test_E24_rise_mask_count_attribute():
    """E24: RISEExplainer pre-generates the correct number of masks."""
    n = 80
    e = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=n, chunk_size=10, seed=0)
    assert e._masks.shape[0] == n, (
        f"Expected {n} masks, got {e._masks.shape[0]}"
    )


# ===========================================================================
# CATEGORY G: LIME-specific
# ===========================================================================

def test_E25_lime_output_finite():
    """E25: LIME coefficients must be finite for any non-degenerate input."""
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=30, seed=2)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"LIME output not finite: {out}"


def test_E26_lime_coef_count():
    """E26: LIME must return exactly P² coefficients reshaped to (P, P)."""
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=30, seed=3)
    out = e.explain(_img, _label)
    assert out.numel() == P * P, (
        f"LIME should return {P*P} coefficients, returned {out.numel()}"
    )


# ===========================================================================
# CATEGORY H: DIME placeholder
# ===========================================================================

def test_E27_dime_placeholder_documented():
    """
    E27: DIMEExplainer must:
    1. Import without error.
    2. Raise NotImplementedError with a message mentioning BENCHMARK.md.
    3. Have is_resolved == False.
    """
    e = DIMEExplainer(_vit, patch_size=PATCH_SIZE)

    assert not DIMEExplainer.is_resolved, (
        "DIMEExplainer.is_resolved should be False until the guide inconsistency is resolved"
    )

    error_msg = ""
    with pytest.raises(NotImplementedError) as exc_info:
        e.explain(_img, _label)
    error_msg = str(exc_info.value)

    assert "BENCHMARK.md" in error_msg, (
        "DIMEExplainer error message should reference BENCHMARK.md"
    )
    assert len(error_msg) > 50, "Error message should be informative"


# ===========================================================================
# Standalone runner (python tests/test_explainers.py)
# ===========================================================================

_TESTS = [
    ("E01", test_E01_raw_attention_shape),
    ("E02", test_E02_rollout_shape),
    ("E03", test_E03_gradcam_shape),
    ("E04", test_E04_chefer_lrp_shape),
    ("E05", test_E05_rise_shape),
    ("E06", test_E06_lime_shape),
    ("E07", test_E07_dime_shape_not_tested_pending),
    ("E08", test_E08_raw_attention_no_nan),
    ("E09", test_E09_rollout_no_nan),
    ("E10", test_E10_gradcam_no_nan),
    ("E11", test_E11_chefer_lrp_no_nan),
    ("E12", test_E12_rise_no_nan),
    ("E13", test_E13_lime_no_nan),
    ("E14", test_E14_dime_no_nan_pending),
    ("E15", test_E15_raw_attention_batch_consistency),
    ("E16", test_E16_rollout_batch_consistency),
    ("E17", test_E17_gradcam_batch_consistency),
    ("E18", test_E18_raw_attention_has_variation),
    ("E19", test_E19_rise_has_variation),
    ("E20", test_E20_lime_has_variation),
    ("E21", test_E21_raw_attention_swin_raises),
    ("E22", test_E22_rollout_swin_raises),
    ("E23", test_E23_rise_nonnegative),
    ("E24", test_E24_rise_mask_count_attribute),
    ("E25", test_E25_lime_output_finite),
    ("E26", test_E26_lime_coef_count),
    ("E27", test_E27_dime_placeholder_documented),
]


def _run_all_tests() -> None:
    passed = failed = skipped = 0

    print(f"\n{'='*62}")
    print("Phase 3 Task 3.1 — Explainer Interface Tests")
    print(f"{'='*62}\n")

    for name, fn in _TESTS:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except pytest.skip.Exception:
            print(f"  – {name}  [SKIP]")
            skipped += 1
        except Exception:
            print(f"  ✗ {name}  FAILED")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*62}")
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed")
    print(f"{'='*62}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all_tests()
