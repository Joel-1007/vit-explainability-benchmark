"""
Fast tests for step 3: per-sample metrics in Phase3Runner, aggregation into
aggregated_results.csv / combination_summary.csv, and the Phase 4 scripts
running on that output. Tiny random models + fake datasets; no downloads.
"""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

from metrics.aggregate import (
    NON_METRIC_COLUMNS, aggregate, correctness_localisation_gap, load_per_sample, summarise,
)
from metrics.robustness import permute_classifier_head, randomise_classifier_labels
from metrics.sample_metrics import SampleMetricComputer, metric_direction
from tests.test_eval_datasets import make_cub, make_voc

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,d", [
    ("miou", 1), ("iou@0.25", 1), ("egt", 1), ("gini", 1), ("entropy_norm", -1),
    ("emr_90", -1), ("sufficiency@0.10", -1), ("comprehensiveness@0.50", 1),
    ("log_odds_drop@0.20", 1), ("max_sensitivity", -1), ("model_randomisation", 1),
    ("head_permutation", 1), ("pred", None), ("whatever", None),
])
def test_metric_direction(name, d):
    assert metric_direction(name) == d


def test_unknown_metric_family_rejected():
    with pytest.raises(ValueError, match="Unknown metric families"):
        SampleMetricComputer(families=["fidelity", "vibes"])


class _Wrapper(nn.Module):
    """Mimics model_zoo wrappers: classifier lives at .backbone.head."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


@pytest.mark.parametrize("arch", ["vit_tiny_patch16_224", "swin_tiny_patch4_window7_224"])
def test_head_permutation_finds_wrapped_heads(arch):
    import timm
    model = _Wrapper(timm.create_model(arch, pretrained=False, num_classes=7))
    assert randomise_classifier_labels is permute_classifier_head      # old name kept
    shuf = permute_classifier_head(model, seed=0)
    w0 = [p for n, p in model.named_parameters() if n.endswith("head.weight") or n.endswith("head.fc.weight")]
    w1 = [p for n, p in shuf.named_parameters() if n.endswith("head.weight") or n.endswith("head.fc.weight")]
    assert len(w0) == 1 and not torch.equal(w0[0], w1[0])
    assert torch.equal(w0[0].sort(dim=1).values, w1[0].sort(dim=1).values)  # permuted only


def test_correctness_localisation_gap_basic():
    df = pd.DataFrame({"label": 0, "egt_lift": [0.8, 0.6, 0.2, 0.4], "correct": [1, 1, 0, 0]})
    clg, lo, hi, n = correctness_localisation_gap(df, min_n=2)
    assert clg == pytest.approx(0.7 - 0.3) and n == 1 and lo <= clg <= hi
    assert math.isnan(correctness_localisation_gap(df[df.correct == 1], min_n=2)[0])


def test_aggregate_rejects_checkpoints_without_metrics(tmp_path):
    (tmp_path / "a-b-c.pkl").write_bytes(pickle.dumps({"key": "a-b-c", "labels": [0]}))
    with pytest.raises(ValueError, match="no per-sample metrics"):
        load_per_sample(tmp_path)


def test_summarise_means_and_counts():
    df = pd.DataFrame({
        "dataset": ["d"] * 4, "model": ["m"] * 4, "explainer": ["e"] * 4,
        "sample_id": range(4), "label": [0] * 4, "pred": [0, 0, 1, 1],
        "correct": [1, 1, 0, 0], "egt": [0.8, 0.6, 0.2, 0.4],
        "max_sensitivity": [0.5, math.nan, math.nan, math.nan],
    })
    s = summarise(df).iloc[0]
    assert s.n_samples == 4 and s.accuracy == 0.5 and s.egt == pytest.approx(0.5)
    assert math.isnan(s.correctness_localisation_gap)    # 2 incorrect < min 10 per class
    assert s.n_max_sensitivity == 1 and s.max_sensitivity == 0.5


# ---------------------------------------------------------------------------
# End-to-end: run_phase3 → metrics → CSVs → Phase 4 scripts
# ---------------------------------------------------------------------------

def _tiny_model(name, num_classes=0, pretrained=True, **_):
    import timm
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False,
                             num_classes=num_classes).eval()


def _run(data: Path, ckpt: Path) -> None:
    import run_phase3
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_phase3, "load_model", _tiny_model)
        run_phase3.main([
            "--data-root", str(data), "--datasets", "cub200", "pascal_voc",
            "--models", "vit_b16", "--explainers", "gradcam", "rollout", "raw_attention",
            "--allow-random-head", "--checkpoint-dir", str(ckpt),
            "--robustness-subset", "1", "--robustness-samples", "2",
            "--batch-size", "2", "--num-workers", "0", "--device", "cpu",
        ])


@pytest.fixture(scope="module")
def phase3_output(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("phase3")
    data = tmp / "data"
    make_cub(data / "cub200")
    make_voc(data / "voc2012")
    _run(data, tmp / "ckpt")
    return tmp / "ckpt"


def test_phase3_writes_metrics_and_csvs(phase3_output):
    ckpt = phase3_output
    res = pickle.loads((ckpt / "cub200-vit_b16-gradcam.pkl").read_bytes())
    assert (res["dataset"], res["model"], res["explainer"]) == ("cub200", "vit_b16", "gradcam")
    assert len(res["metrics"]) == res["n_samples"] == 3
    assert res["metric_config"]["robustness_subset"] == 1

    df = pd.read_csv(ckpt / "aggregated_results.csv")
    assert len(df) == (3 + 3) * 3          # (3 CUB + 3 VOC (image, class) units) x 3 explainers
    assert set(df.explainer) == {"gradcam", "rollout", "raw_attention"}
    for col in ["miou", "pointing_game", "egt", "gini", "entropy_norm", "emr_90",
                "sufficiency@0.10", "comprehensiveness@0.50", "log_odds_drop@0.20",
                "max_sensitivity", "model_randomisation", "head_permutation"]:
        assert col in df.columns, col
    assert df.miou.between(0, 1).all() and df.gini.between(0, 1).all()
    assert df.entropy_norm.between(0, 1).all() and df.emr_90.between(0, 1).all()
    assert df["comprehensiveness@0.50"].between(-1, 1).all()
    assert set(df.correct.unique()) <= {0, 1}
    # robustness only on the first image of each combination
    per_combo = df.groupby(["dataset", "explainer"]).max_sensitivity.count()
    assert (per_combo == 1).all()
    assert (df.max_sensitivity.dropna() >= 0).all()
    assert df.head_permutation.dropna().between(0, 0.5).all()

    summ = pd.read_csv(ckpt / "combination_summary.csv")
    assert len(summ) == 6 and "correctness_localisation_gap" in summ and "accuracy" in summ
    assert (summ.n_max_sensitivity == 1).all()


def test_phase4_scripts_run_on_real_output(phase3_output, tmp_path):
    csv = phase3_output / "aggregated_results.csv"
    out = tmp_path / "p4"
    for script in ["phase4_correlation_analysis.py", "phase4_interaction_analysis.py"]:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script),
             "--results_csv", str(csv), "--output_dir", str(out)],
            capture_output=True, text=True, cwd=tmp_path,
        )
        assert r.returncode == 0, r.stdout + r.stderr
    corr = pd.read_csv(out / "inter_metric_correlations.csv", index_col=0)
    assert not set(corr.index) & set(NON_METRIC_COLUMNS)
    fa = pd.read_csv(out / "factor_analysis_results.csv")          # computed from the data
    assert fa.Cumulative_Variance_Pct.iloc[-1] == pytest.approx(100)
    assert fa.Eigenvalue.tolist() != [3.1, 1.8, 1.1]                  # not the old placeholder

    ranks = pd.read_csv(out / "explainer_rankings.csv")
    # rank 1 must be the best explainer under each metric's direction
    df = pd.read_csv(csv)
    for metric in ["max_sensitivity", "miou"]:
        means = df[df.dataset == "cub200"].groupby("explainer")[metric].mean()
        best = means.idxmin() if metric_direction(metric) < 0 else means.idxmax()
        r1 = ranks[(ranks.dataset == "cub200") & (ranks.metric == metric) & (ranks["rank"] == 1)]
        assert r1.explainer.iloc[0] == best
    assert (out / "dataset_concordance.csv").exists()


def test_rerun_resumes_and_gives_identical_results(phase3_output):
    """A second run skips every checkpointed combination and re-aggregates identically."""
    before = pd.read_csv(phase3_output / "aggregated_results.csv")
    mtimes = {p.name: p.stat().st_mtime_ns for p in phase3_output.glob("*.pkl")}
    _run(phase3_output.parent / "data", phase3_output)
    assert {p.name: p.stat().st_mtime_ns for p in phase3_output.glob("*.pkl")} == mtimes
    pd.testing.assert_frame_equal(before, pd.read_csv(phase3_output / "aggregated_results.csv"))
