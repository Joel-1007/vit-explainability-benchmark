"""
Tests for Phase 4 factor analysis, ablations A1–A4, the explainer ablation
variants, the PGD fix, and an end-to-end run with --ablations. Fast: synthetic
data with planted structure + tiny random timm models.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase4_ablations as abl                      # noqa: E402
import phase4_correlation_analysis as corr          # noqa: E402

from tests.test_eval_datasets import make_cub, make_voc  # noqa: E402


# ---------------------------------------------------------------------------
# Factor analysis
# ---------------------------------------------------------------------------

def _two_factor_df(n=400, seed=0):
    """Localization metrics share latent f1; complexity metrics share latent f2."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    e = lambda: rng.normal(scale=0.3, size=n)       # noqa: E731
    return pd.DataFrame({
        "dataset": "d", "model": "m", "explainer": rng.choice(["a", "b", "c"], n),
        "sample_id": range(n), "label": 0, "pred": 0, "correct": rng.integers(0, 2, n),
        "miou": f1 + e(), "pointing_game": f1 + e(), "egt": f1 + e(),
        "gini": f2 + e(), "entropy_norm": -f2 + e(), "emr_90": -f2 + e(),
        "max_sensitivity": np.where(np.arange(n) < 10, rng.normal(size=n), np.nan),
    })


def test_factor_analysis_recovers_planted_structure(tmp_path):
    out = tmp_path / "fa.csv"
    res = corr.run_factor_analysis(_two_factor_df(), str(out))
    assert res["n_factors"] == 2
    assert res["top3_pct"] > 80
    assert "max_sensitivity" in res["dropped"]               # sparse → dropped at sample level
    L = res["loadings"].drop(index="rotated_variance_pct")
    f_loc = L.loc["miou", ["F1", "F2"]].abs().idxmax()
    f_cmp = "F2" if f_loc == "F1" else "F1"
    for m in ["miou", "pointing_game", "egt"]:
        assert abs(L.loc[m, f_loc]) > 0.9 and abs(L.loc[m, f_cmp]) < 0.15
    for m in ["gini", "entropy_norm", "emr_90"]:
        assert abs(L.loc[m, f_cmp]) > 0.9 and abs(L.loc[m, f_loc]) < 0.15
    assert np.sign(L.loc["gini", f_cmp]) == -np.sign(L.loc["entropy_norm", f_cmp])
    var = pd.read_csv(out)
    assert var.Cumulative_Variance_Pct.iloc[-1] == pytest.approx(100)
    assert var.Retained.sum() == 2
    assert (tmp_path / "factor_loadings_varimax.csv").exists()


def test_varimax_is_orthogonal_rotation():
    rng = np.random.default_rng(1)
    L = rng.normal(size=(8, 3))
    R = corr.varimax(L)
    np.testing.assert_allclose(R @ R.T, L @ L.T, atol=1e-8)   # communalities preserved


def test_varimax_reaches_brute_force_optimum():
    """Two factors: compare with an exhaustive search over rotation angles."""
    rng = np.random.default_rng(2)
    L = rng.normal(size=(7, 2))
    crit = lambda M: ((M ** 2) ** 2).sum() - ((M ** 2).sum(0) ** 2).sum() / M.shape[0]  # noqa: E731
    rot = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])         # noqa: E731
    best = max(crit(L @ rot(t)) for t in np.linspace(0, np.pi / 2, 20001))
    assert crit(corr.varimax(L, normalize=False)) == pytest.approx(best, rel=1e-6)


def test_factor_analysis_combination_level_and_insufficient_data(tmp_path):
    df = _two_factor_df(n=30)
    df["explainer"] = [f"e{i}" for i in range(30)]           # 30 combinations
    res = corr.run_factor_analysis(df, str(tmp_path / "fa.csv"), level="combination",
                                   min_coverage=0.0)
    assert res["n_factors"] >= 2
    with pytest.raises(ValueError, match="Not enough data"):
        corr.run_factor_analysis(df.head(3), str(tmp_path / "fa2.csv"))


# ---------------------------------------------------------------------------
# Ablations (synthetic data with planted effects)
# ---------------------------------------------------------------------------

def test_cohens_d_formula_and_labels():
    a, b = np.array([1.0, 2, 3, 4]), np.array([0.0, 1, 2, 3])
    assert abl.compute_cohens_d(a, b) == pytest.approx(1 / np.sqrt(np.var(a, ddof=1)))
    assert abl.compute_cohens_d(np.ones(3), np.ones(3)) == 0.0
    assert abl.interpret_cohens_d(0.1) == "Negligible" and abl.interpret_cohens_d(-0.9) == "Large"


def _ablation_df(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for model in ["vit_b16", "mae_b16"]:
        for exp in ["raw_attention", "raw_attention_patch", "raw_attention_last3", "rollout"]:
            for i in range(n):
                base = rng.uniform(0.3, 0.7)
                shift = {"raw_attention_patch": -0.2, "rollout": 0.1}.get(exp, 0.0)
                if model == "mae_b16":
                    shift += 0.15
                comp = rng.uniform(0, 1)
                rows.append({
                    "dataset": "cub200", "model": model, "explainer": exp, "sample_id": i,
                    "label": 0, "pred": 0, "correct": 1,
                    "miou": base + shift + rng.normal(scale=0.02),
                    "pointing_game": float(rng.integers(0, 2)), "egt": base,
                    "comprehensiveness@0.20": comp,
                    "comprehensiveness@0.20[zero]": comp + 0.3 + rng.normal(scale=0.01),
                    "comprehensiveness@0.20[blur]": comp + rng.normal(scale=0.01),
                })
    return pd.DataFrame(rows)


def test_ablations_detect_planted_effects(tmp_path):
    df = _ablation_df()
    a1 = abl.run_ablation_A1(df)
    r = a1[(a1.metric == "miou") & (a1.model == "vit_b16")].iloc[0]
    assert r.variant_A == "CLS token" and r.cohens_d > 0.8 and r.wilcoxon_p < 1e-3

    a2 = abl.run_ablation_A2(df)
    roll = a2[(a2.variant_A == "all layers (rollout)") & (a2.metric == "miou")
              & (a2.model == "vit_b16")].iloc[0]
    last3 = a2[(a2.variant_A == "last 3 layers") & (a2.metric == "miou")
               & (a2.model == "vit_b16")].iloc[0]
    assert roll.cohens_d > 0.3 and abs(last3.cohens_d) < 0.8

    a3 = abl.run_ablation_A3(df, str(tmp_path))
    z = a3[(a3.variant_A == "zero mask") & (a3.explainer == "raw_attention")
           & (a3.model == "vit_b16")].iloc[0]
    b = a3[(a3.variant_A == "blur mask") & (a3.explainer == "raw_attention")
           & (a3.model == "vit_b16")].iloc[0]
    assert z.mean_A - z.mean_B == pytest.approx(0.3, abs=0.02) and z.wilcoxon_p < 1e-3
    assert abs(b.mean_A - b.mean_B) < 0.01
    sens = pd.read_csv(tmp_path / "ablation_A3_masking_sensitivity.csv")
    assert (sens.masking_sensitivity > 0.1).all()          # zero shifts by 0.3

    a4 = abl.run_ablation_A4(df)
    m = a4[(a4.metric == "miou") & (a4.explainer == "raw_attention")].iloc[0]
    assert m.variant_A == "MAE" and m.mean_A - m.mean_B == pytest.approx(0.15, abs=0.02)


def test_ablations_skip_missing_inputs(tmp_path, capsys):
    df = _ablation_df()
    only_main = df[df.explainer.isin(["rollout"]) & (df.model == "vit_b16")]
    only_main = only_main.drop(columns=[c for c in df.columns if "[" in c])
    assert abl.run_ablation_A1(only_main) is None
    assert abl.run_ablation_A3(only_main, str(tmp_path)) is None
    assert abl.run_ablation_A4(only_main) is None
    assert "run run_phase3.py --ablations" in capsys.readouterr().out
    csv = tmp_path / "r.csv"
    only_main.to_csv(csv, index=False)
    with pytest.raises(SystemExit, match="no ablation could be computed"):
        abl.main(["--results_csv", str(csv), "--output_dir", str(tmp_path / "o")])


def test_bonferroni_scales_p_values():
    t = pd.DataFrame({"wilcoxon_p": [0.01, 0.2, 0.6]})
    assert abl._bonferroni(t).p_bonferroni.tolist() == pytest.approx([0.03, 0.6, 1.0])


# ---------------------------------------------------------------------------
# Explainer ablation variants + DeiT prefix tokens
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def deit():
    import timm
    torch.manual_seed(0)
    return timm.create_model("deit_tiny_distilled_patch16_224", pretrained=False,
                             num_classes=5).eval()


def _all_attn(model, x):
    from explainers.base import _capture_attn_weights, _get_timm_blocks
    n = len(_get_timm_blocks(model))
    return [A[0].mean(0) for A in _capture_attn_weights(model, x[None], list(range(n)))]


def test_attention_explainers_work_on_deit(deit):
    from explainers import AttentionRolloutExplainer, CheferLRPExplainer, RawAttentionExplainer
    x = torch.randn(3, 224, 224)
    for E in (RawAttentionExplainer, AttentionRolloutExplainer, CheferLRPExplainer):
        assert E(deit).explain(x, 1).shape == (14, 14)


def test_raw_attention_variants_match_definitions(deit):
    from explainers import (RawAttentionAllLayersExplainer, RawAttentionExplainer,
                            RawAttentionLast3Explainer, RawPatchAttentionExplainer)
    x = torch.randn(3, 224, 224)
    A = _all_attn(deit, x)                                   # per-layer (T, T), T = 198
    P = 2                                                    # DeiT: CLS + distillation
    torch.testing.assert_close(RawAttentionExplainer(deit).explain(x, 0).flatten(), A[-1][0, P:])
    torch.testing.assert_close(RawAttentionLast3Explainer(deit).explain(x, 0).flatten(),
                               torch.stack(A[-3:]).mean(0)[0, P:])
    torch.testing.assert_close(RawAttentionAllLayersExplainer(deit).explain(x, 0).flatten(),
                               torch.stack(A).mean(0)[0, P:])
    torch.testing.assert_close(RawPatchAttentionExplainer(deit).explain(x, 0).flatten(),
                               A[-1][P:, P:].mean(0))


# ---------------------------------------------------------------------------
# PGD in normalised input space
# ---------------------------------------------------------------------------

def test_pgd_respects_pixel_budget_with_normalised_inputs():
    import timm
    from utils.eval_datasets import IMAGENET_MEAN as M, IMAGENET_STD as S
    from utils.pgd import pgd_attack
    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5).eval()
    m, s = torch.tensor(M).view(3, 1, 1), torch.tensor(S).view(3, 1, 1)
    pix = torch.rand(3, 224, 224)
    adv = pgd_attack(model, (pix - m) / s, 1, epsilon=0.03, alpha=0.01, steps=3, mean=M, std=S)
    adv_pix = adv * s + m
    assert (adv_pix - pix).abs().max() <= 0.03 + 1e-5
    assert adv_pix.min() >= -1e-5 and adv_pix.max() <= 1 + 1e-5
    assert all(p.grad is None for p in model.parameters())   # no gradient leakage
    with pytest.raises(ValueError):
        pgd_attack(model, pix, 1, mean=M)


# ---------------------------------------------------------------------------
# End-to-end: run_phase3 --ablations (+ MAE, adversarial) → all Phase 4 scripts
# ---------------------------------------------------------------------------

def test_end_to_end_ablations_and_factor_analysis(tmp_path):
    import timm
    import run_phase3

    def tiny(name, num_classes=0, pretrained=True, **_):
        torch.manual_seed(0 if name == "vit_b16" else 1)   # different "pre-training"
        return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                                 num_classes=num_classes).eval()

    data = tmp_path / "data"
    make_cub(data / "cub200")
    make_voc(data / "voc2012")
    ckpt = tmp_path / "ckpt"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_phase3, "load_model", tiny)
        run_phase3.main([
            "--data-root", str(data), "--datasets", "cub200", "pascal_voc",
            "--models", "vit_b16", "mae_b16", "--explainers", "gradcam", "--ablations",
            "--metrics", "localization", "complexity", "fidelity", "robustness", "adversarial",
            "--allow-random-head", "--checkpoint-dir", str(ckpt),
            "--robustness-subset", "1", "--robustness-samples", "1",
            "--batch-size", "2", "--num-workers", "0", "--device", "cpu",
        ])
    df = pd.read_csv(ckpt / "aggregated_results.csv")
    assert set(df.explainer) == {"gradcam", "raw_attention", "rollout", "raw_attention_patch",
                                 "raw_attention_last3", "raw_attention_all"}
    assert set(df.model) == {"vit_b16", "mae_b16"}
    assert "comprehensiveness@0.20[zero]" in df and "sufficiency@0.50[blur]" in df
    assert df.groupby(["dataset", "model", "explainer"]).pgd_robustness.count().eq(1).all()
    assert df.pgd_robustness.dropna().between(0, 1).all()

    out = tmp_path / "p4"
    for script in ["phase4_correlation_analysis.py", "phase4_interaction_analysis.py",
                   "phase4_ablations.py"]:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / script),
                            "--results_csv", str(ckpt / "aggregated_results.csv"),
                            "--output_dir", str(out)],
                           capture_output=True, text=True, cwd=tmp_path)
        assert r.returncode == 0, script + "\n" + r.stdout + r.stderr

    for f in ["factor_analysis_results.csv", "factor_loadings_varimax.csv",
              "ablation_A1.csv", "ablation_A2.csv", "ablation_A3.csv", "ablation_A4.csv",
              "ablation_A3_masking_sensitivity.csv", "ablation_summary.csv"]:
        assert (out / f).exists(), f
    corr_m = pd.read_csv(out / "inter_metric_correlations.csv", index_col=0)
    assert not any("[" in c for c in corr_m.index)                  # ablation cols excluded
    ranks = pd.read_csv(out / "explainer_rankings.csv")
    assert set(ranks.explainer) == {"gradcam", "raw_attention", "rollout"}   # main only
    summary = pd.read_csv(out / "ablation_summary.csv")
    assert set(summary.ablation) == {"A1", "A2", "A3", "A4"}


# ---------------------------------------------------------------------------
# Integrated Gradients (E8) — axioms hold up to the Riemann-sum error
# ---------------------------------------------------------------------------

class _PatchMasked(torch.nn.Module):
    """Wraps a model so the top-left 16x16 patch never influences the output."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        x = x.clone()
        x[..., :16, :16] = 0.0
        return self.inner(x)


@pytest.mark.parametrize("arch", ["vit_tiny_patch16_224", "swin_tiny_patch4_window7_224"])
def test_integrated_gradients_completeness_converges(arch):
    import timm
    from explainers import IntegratedGradientsExplainer
    torch.manual_seed(0)
    m = timm.create_model(arch, pretrained=False, num_classes=5).eval()
    x = torch.randn(3, 224, 224)
    gaps = []
    for steps in (10, 100):
        e = IntegratedGradientsExplainer(m, n_steps=steps)
        phi = e.explain(x, 1)
        with torch.no_grad():
            f = m(torch.stack([x, torch.zeros_like(x)]))[:, 1]
        assert phi.shape == (14, 14)
        assert float(phi.sum()) == pytest.approx(float(f[0] - f[1]), abs=abs(e.last_completeness_gap) + 1e-4)
        gaps.append(abs(e.last_completeness_gap))
    assert gaps[1] < gaps[0] and gaps[1] < 0.01


def test_integrated_gradients_dummy_patch_gets_zero():
    import timm
    from explainers import IntegratedGradientsExplainer
    torch.manual_seed(0)
    m = _PatchMasked(timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=5)).eval()
    phi = IntegratedGradientsExplainer(m, n_steps=20).explain(torch.randn(3, 224, 224), 0)
    assert phi[0, 0] == 0.0 and phi.abs().sum() > 0
