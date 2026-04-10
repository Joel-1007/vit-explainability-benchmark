"""
test_explainers.py  —  Phase 3 Task 3.1 Unit Tests
====================================================
27 tests across 8 categories, all using a fast MockViT (no timm download).

Category A — Shape (7 tests, E01–E07)
    Each explainer returns exactly (P, P) where P = img_size // patch_size.

Category B — No NaN / Inf (7 tests, E08–E14)
    torch.isfinite(output).all() must hold for every explainer.

Category C — Batch consistency (3 tests, E15–E17)
    explain_batch result must match looped explain calls (E1, E2, E3).

Category D — Spatial variation (3 tests, E18–E20)
    Output std > 0 for non-trivial inputs (E1, E5, E6 specifically).

Category E — Swin-B UnsupportedArchitectureError (2 tests, E21–E22)
    E1 (RawAttention) and E2 (Rollout) raise the right exception.

Category F — RISE-specific (2 tests, E23–E24)
    Non-negative output; shape correctness at small resolution.

Category G — LIME-specific (2 tests, E25–E26)
    Finite output; regression produces S=P² coefficients.

Category H — DIME placeholder (1 test, E27)
    DIMEExplainer.explain() raises NotImplementedError with a message
    that references BENCHMARK.md §3.1.

Run with:
    python tests/test_explainers.py
    # or
    pytest tests/test_explainers.py -v
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

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

IMG_SIZE   = 16    # tiny test image (fast)
PATCH_SIZE = 4     # 4×4 patches → P = 4 → output (4, 4)
P          = IMG_SIZE // PATCH_SIZE   # = 4
N_PATCHES  = P * P                   # = 16
N_TOKENS   = N_PATCHES + 1           # +1 for CLS
DIM        = 32
NUM_HEADS  = 4
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
# Shared instances
# ---------------------------------------------------------------------------
torch.manual_seed(0)
_vit   = _MockViT().eval()
_swin  = _MockSwinB().eval()
_img   = torch.rand(3, IMG_SIZE, IMG_SIZE)
_label = 2


# ===========================================================================
# Test registry
# ===========================================================================
_TESTS: list[tuple[str, callable]] = []


def _register(fn):
    _TESTS.append((fn.__name__, fn))
    return fn


# ===========================================================================
# CATEGORY A: Shape — each explainer returns (P, P)
# ===========================================================================

@_register
def E01_raw_attention_shape():
    e   = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"
    print(f"E01 ✓  raw_attention_shape  {out.shape}")


@_register
def E02_rollout_shape():
    e   = AttentionRolloutExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"
    print(f"E02 ✓  rollout_shape  {out.shape}")


@_register
def E03_gradcam_shape():
    e   = GradCAMExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"
    print(f"E03 ✓  gradcam_shape  {out.shape}")


@_register
def E04_chefer_lrp_shape():
    e   = CheferLRPExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"
    print(f"E04 ✓  chefer_lrp_shape  {out.shape}")


@_register
def E05_rise_shape():
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=50, chunk_size=10, seed=0)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"
    print(f"E05 ✓  rise_shape  {out.shape}")


@_register
def E06_lime_shape():
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=20, seed=0)
    out = e.explain(_img, _label)
    assert out.shape == (P, P), f"Expected ({P},{P}), got {out.shape}"
    print(f"E06 ✓  lime_shape  {out.shape}")


@_register
def E07_dime_shape_not_tested_pending():
    """DIMEExplainer raises NotImplementedError — shape test skipped with note."""
    e = DIMEExplainer(_vit, patch_size=PATCH_SIZE)
    raised = False
    try:
        e.explain(_img, _label)
    except NotImplementedError:
        raised = True
    assert raised, "DIMEExplainer.explain() should raise NotImplementedError"
    print("E07 ✓  dime_raises_not_implemented  [shape test pending resolution]")


# ===========================================================================
# CATEGORY B: No NaN / Inf
# ===========================================================================

@_register
def E08_raw_attention_no_nan():
    e   = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"RawAttention output contains NaN/Inf: {out}"
    print("E08 ✓  raw_attention_no_nan")


@_register
def E09_rollout_no_nan():
    e   = AttentionRolloutExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"Rollout output contains NaN/Inf: {out}"
    print("E09 ✓  rollout_no_nan")


@_register
def E10_gradcam_no_nan():
    e   = GradCAMExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"GradCAM output contains NaN/Inf: {out}"
    print("E10 ✓  gradcam_no_nan")


@_register
def E11_chefer_lrp_no_nan():
    e   = CheferLRPExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"CheferLRP output contains NaN/Inf: {out}"
    print("E11 ✓  chefer_lrp_no_nan")


@_register
def E12_rise_no_nan():
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=30, chunk_size=10, seed=0)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"RISE output contains NaN/Inf: {out}"
    print("E12 ✓  rise_no_nan")


@_register
def E13_lime_no_nan():
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=20, seed=0)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"LIME output contains NaN/Inf: {out}"
    print("E13 ✓  lime_no_nan")


@_register
def E14_dime_no_nan_pending():
    """Not tested — DIMEExplainer is a pending placeholder."""
    print("E14 –  dime_no_nan  [SKIP — pending resolution of DIME guide inconsistency]")


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


@_register
def E15_raw_attention_batch_consistency():
    e = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    _batch_consistency_check(e)
    print("E15 ✓  raw_attention_batch_consistency")


@_register
def E16_rollout_batch_consistency():
    e = AttentionRolloutExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    _batch_consistency_check(e)
    print("E16 ✓  rollout_batch_consistency")


@_register
def E17_gradcam_batch_consistency():
    e = GradCAMExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    _batch_consistency_check(e, atol=1e-3)  # gradient methods: slightly looser
    print("E17 ✓  gradcam_batch_consistency")


# ===========================================================================
# CATEGORY D: Spatial variation (output std > 0)
# ===========================================================================

@_register
def E18_raw_attention_has_variation():
    torch.manual_seed(42)
    e   = RawAttentionExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    out = e.explain(torch.rand(3, IMG_SIZE, IMG_SIZE), _label)
    assert out.std() > 0, "RawAttention output is constant (zero variation)"
    print(f"E18 ✓  raw_attention_has_variation  (std={out.std():.4f})")


@_register
def E19_rise_has_variation():
    torch.manual_seed(43)
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=100, chunk_size=20, seed=0)
    out = e.explain(torch.rand(3, IMG_SIZE, IMG_SIZE), _label)
    assert out.std() > 0, "RISE output is constant (zero variation)"
    print(f"E19 ✓  rise_has_variation  (std={out.std():.4f})")


@_register
def E20_lime_has_variation():
    torch.manual_seed(44)
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=50, seed=1)
    out = e.explain(torch.rand(3, IMG_SIZE, IMG_SIZE), _label)
    assert out.std() > 0, "LIME output is constant (zero variation)"
    print(f"E20 ✓  lime_has_variation  (std={out.std():.4f})")


# ===========================================================================
# CATEGORY E: Swin-B raises UnsupportedArchitectureError
# ===========================================================================

@_register
def E21_raw_attention_swin_raises():
    e = RawAttentionExplainer(_swin, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    raised = False
    try:
        e.explain(_img, _label)
    except UnsupportedArchitectureError:
        raised = True
    assert raised, "RawAttentionExplainer should raise for Swin-B"
    print("E21 ✓  raw_attention_swin_raises  UnsupportedArchitectureError")


@_register
def E22_rollout_swin_raises():
    e = AttentionRolloutExplainer(_swin, patch_size=PATCH_SIZE, img_size=IMG_SIZE)
    raised = False
    try:
        e.explain(_img, _label)
    except UnsupportedArchitectureError:
        raised = True
    assert raised, "AttentionRolloutExplainer should raise for Swin-B"
    print("E22 ✓  rollout_swin_raises  UnsupportedArchitectureError")


# ===========================================================================
# CATEGORY F: RISE-specific
# ===========================================================================

@_register
def E23_rise_nonnegative():
    """RISE saliency = weighted sum of masked probabilities → must be ≥ 0."""
    e   = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=50, chunk_size=10, seed=5)
    out = e.explain(_img, _label)
    assert out.min().item() >= -1e-6, f"RISE output has negative values: min={out.min():.6f}"
    print(f"E23 ✓  rise_nonnegative  (min={out.min():.4f})")


@_register
def E24_rise_mask_count_attribute():
    """RISEExplainer stores pre-generated masks with the correct count."""
    n = 80
    e = RISEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_masks=n, chunk_size=10, seed=0)
    assert e._masks.shape[0] == n, (
        f"Expected {n} masks, got {e._masks.shape[0]}"
    )
    print(f"E24 ✓  rise_mask_count_attribute  ({n} masks stored)")


# ===========================================================================
# CATEGORY G: LIME-specific
# ===========================================================================

@_register
def E25_lime_output_finite():
    """LIME coefficients must be finite for any non-degenerate input."""
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=30, seed=2)
    out = e.explain(_img, _label)
    assert torch.isfinite(out).all(), f"LIME output not finite: {out}"
    print("E25 ✓  lime_output_finite")


@_register
def E26_lime_coef_count():
    """LIME must return exactly S = P² coefficients reshaped to (P, P)."""
    e   = LIMEExplainer(_vit, patch_size=PATCH_SIZE, img_size=IMG_SIZE, n_samples=30, seed=3)
    out = e.explain(_img, _label)
    assert out.numel() == P * P, (
        f"LIME should return {P*P} coefficients, returned {out.numel()}"
    )
    print(f"E26 ✓  lime_coef_count  ({out.numel()} = {P}×{P})")


# ===========================================================================
# CATEGORY H: DIME placeholder
# ===========================================================================

@_register
def E27_dime_placeholder_documented():
    """
    DIMEExplainer must:
    1. Import without error.
    2. Raise NotImplementedError with a message mentioning BENCHMARK.md.
    3. Have is_resolved == False.
    """
    e = DIMEExplainer(_vit, patch_size=PATCH_SIZE)

    assert not DIMEExplainer.is_resolved, (
        "DIMEExplainer.is_resolved should be False until the guide inconsistency is resolved"
    )

    error_msg = ""
    try:
        e.explain(_img, _label)
    except NotImplementedError as exc:
        error_msg = str(exc)

    assert "BENCHMARK.md" in error_msg, (
        "DIMEExplainer error message should reference BENCHMARK.md"
    )
    assert len(error_msg) > 50, "Error message should be informative"

    print(
        "E27 ✓  dime_placeholder_documented  "
        "(is_resolved=False, NotImplementedError raised, BENCHMARK.md referenced)"
    )


# ===========================================================================
# Runner
# ===========================================================================

def _run_all_tests() -> None:
    passed = 0
    failed = 0
    skipped = 0

    print(f"\n{'='*62}")
    print("Phase 3 Task 3.1 — Explainer Interface Tests")
    print(f"{'='*62}\n")

    for name, fn in _TESTS:
        try:
            fn()
            passed += 1
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
            print()

    print(f"\n{'='*62}")
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print(f"  Note: E14 is a documented skip (DIME pending)")
    print(f"{'='*62}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all_tests()
