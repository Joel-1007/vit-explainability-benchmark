"""
Tests for the experimental-risk fixes: fine-tuning seeds, explainer
compatibility reporting, NIH patient leakage, class-level summaries,
paired statistics with Holm correction, and the dataset audit.
Fast: fake on-disk datasets and tiny random timm models.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tests.test_eval_datasets import make_cub, make_imagenet_s, make_nih, make_voc

ROOT = Path(__file__).resolve().parents[1]


def _tiny(arch="vit_tiny_patch16_224"):
    import timm

    def factory(name, num_classes=0, pretrained=True, **_):
        torch.manual_seed(0)
        return timm.create_model(arch, pretrained=False, num_classes=num_classes).eval()
    return factory


# ---------------------------------------------------------------------------
# Fine-tuning seeds
# ---------------------------------------------------------------------------

def test_checkpoint_name_and_find_weights_by_seed(tmp_path):
    from run_phase3 import find_weights
    from training.trainer import checkpoint_name
    assert checkpoint_name("vit_b16", "cub200", 3, seed=2, date="20260101") == \
        "vit_b16_ft_cub200_seed2_20260101_ep003.pth"
    for name in ["vit_b16_ft_cub200_20260101_ep010.pth",
                 "vit_b16_ft_cub200_seed0_20260101_ep010.pth",
                 "vit_b16_ft_cub200_seed1_20260101_ep005.pth",
                 "vit_b16_ft_cub200_seed1_20260101_ep010.pth"]:
        (tmp_path / name).write_bytes(b"")
    assert find_weights(str(tmp_path), "vit_b16", "cub200").name == "vit_b16_ft_cub200_20260101_ep010.pth"
    assert find_weights(str(tmp_path), "vit_b16", "cub200", seed=1).name.endswith("seed1_20260101_ep010.pth")
    assert find_weights(str(tmp_path), "vit_b16", "cub200", seed=7) is None


def test_set_seed_reproducible():
    from training.trainer import set_seed
    set_seed(3); a = (torch.rand(3), np.random.rand(3))
    set_seed(3); b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.fixture(scope="module")
def seeded_run(tmp_path_factory):
    """Two seeds of vit_b16 + one Swin (unsupported rollout) on fake CUB."""
    import timm
    import run_phase3
    tmp = tmp_path_factory.mktemp("seeds")
    make_cub(tmp / "data" / "cub200")
    weights = tmp / "w"
    weights.mkdir()
    for seed in (0, 1):
        torch.manual_seed(100 + seed)
        m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=200)
        torch.save({"model_state_dict": m.state_dict()},
                   weights / f"vit_b16_ft_cub200_seed{seed}_20260101_ep001.pth")
        torch.manual_seed(200 + seed)
        s = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=200)
        torch.save({"model_state_dict": s.state_dict()},
                   weights / f"swin_b_ft_cub200_seed{seed}_20260101_ep001.pth")

    def factory(name, num_classes=0, pretrained=True, **_):
        arch = "swin_tiny_patch4_window7_224" if name == "swin_b" else "vit_tiny_patch16_224"
        return timm.create_model(arch, pretrained=False, num_classes=num_classes).eval()

    ckpt = tmp / "ckpt"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_phase3, "load_model", factory)
        run_phase3.main([
            "--data-root", str(tmp / "data"), "--datasets", "cub200",
            "--models", "vit_b16", "swin_b", "--finetune-seeds", "0", "1",
            "--explainers", "gradcam", "rollout", "--weights-dir", str(weights),
            "--metrics", "localization", "complexity", "fidelity",
            "--checkpoint-dir", str(ckpt), "--num-workers", "0", "--device", "cpu",
        ])
    return ckpt


def test_seed_results_and_seed_summary(seeded_run):
    df = pd.read_csv(seeded_run / "aggregated_results.csv")
    assert set(df.seed.dropna().astype(int)) == {0, 1}
    assert set(df.model) == {"vit_b16", "swin_b"}
    seeds = pd.read_csv(seeded_run / "seed_summary.csv")
    row = seeds[(seeds.model == "vit_b16") & (seeds.explainer == "gradcam")].iloc[0]
    assert row.n_seeds == 2 and "miou_seed_std" in seeds and row.miou_seed_std >= 0
    summ = pd.read_csv(seeded_run / "combination_summary.csv")
    assert len(summ[(summ.model == "vit_b16") & (summ.explainer == "gradcam")]) == 2   # one row per seed


def test_unsupported_combinations_reported_not_scored(seeded_run):
    cov = pd.read_csv(seeded_run / "coverage.csv")
    swin_roll = cov[(cov.model == "swin_b") & (cov.explainer == "rollout")]
    assert len(swin_roll) == 2 and (swin_roll.status == "unsupported").all()
    assert swin_roll.reason.str.contains("CLS").all()
    assert (cov[cov.explainer == "gradcam"].status == "ok").all()
    summ = pd.read_csv(seeded_run / "combination_summary.csv")
    assert summ[(summ.model == "swin_b") & (summ.explainer == "rollout")].empty   # absent, not 0
    from metrics.runner import Phase3Runner
    assert Phase3Runner.combination_done(seeded_run, "cub200-swin_b@seed0-rollout")


def test_coverage_marks_missing(tmp_path):
    from metrics.aggregate import coverage
    (tmp_path / "run_metadata.json").write_text(json.dumps({"args": {
        "datasets": ["cub200"], "models": ["vit_b16"], "explainers": ["gradcam", "lime"],
        "finetune_seeds": None}}))
    (tmp_path / "cub200-vit_b16-gradcam.pkl").write_bytes(b"")
    cov = coverage(tmp_path).set_index("explainer")
    assert cov.loc["gradcam", "status"] == "ok" and cov.loc["lime", "status"] == "missing"


# ---------------------------------------------------------------------------
# Class imbalance: class-level metrics and macro averages
# ---------------------------------------------------------------------------

def test_class_summary_and_macro():
    from metrics.aggregate import class_summary, summarise
    # class 0: 90 samples with miou 0.9; class 1: 10 samples with miou 0.1
    df = pd.DataFrame({
        "dataset": "d", "model": "m", "seed": pd.array([pd.NA] * 100, dtype="Int64"),
        "explainer": "e", "sample_id": range(100), "label": [0] * 90 + [1] * 10,
        "pred": 0, "correct": [1] * 90 + [0] * 10, "miou": [0.9] * 90 + [0.1] * 10,
    })
    s = summarise(df).iloc[0]
    assert s.miou == pytest.approx(0.82)                 # pooled, dominated by class 0
    assert s.miou_macro == pytest.approx(0.5)            # every class weighs the same
    assert s.accuracy == pytest.approx(0.9) and s.accuracy_macro == pytest.approx(0.5)
    assert s.n_classes == 2
    cs = class_summary(df).set_index("label")
    assert cs.loc[1, "n"] == 10 and cs.loc[1, "class_share"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# NIH patient leakage
# ---------------------------------------------------------------------------

def test_nih_excludes_patients_seen_in_training(tmp_path):
    from utils.eval_datasets import NIHBBoxEval, nih_patient_id
    make_nih(tmp_path)                       # box images: patients 00000001, 00000002
    assert nih_patient_id("00000013_005.png") == "00000013"
    # patient 00000001 has a *different* image in the training list
    (tmp_path / "train_val_list.txt").write_text("00000001_007.png\n00000099_000.png\n")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = NIHBBoxEval(tmp_path)
    assert {s[0] for s in ds.samples} == {"00000002_000.png"}
    assert ds.leakage == {"checked": True, "excluded_patients": ["00000001"], "excluded_samples": 2}
    assert any("excluded 2 box samples" in str(x.message) for x in w)


def test_nih_require_split(tmp_path):
    from utils.eval_datasets import NIHBBoxEval
    make_nih(tmp_path)
    assert NIHBBoxEval(tmp_path).leakage["checked"] is False
    with pytest.raises(FileNotFoundError):
        NIHBBoxEval(tmp_path, require_split=True)


# ---------------------------------------------------------------------------
# Paired statistics + Holm correction
# ---------------------------------------------------------------------------

def test_holm_and_effect_sizes():
    from metrics.stats import cohens_d, holm, paired_dz, wilcoxon_p
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([0.01, np.nan])[1])
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = a - 0.3 + rng.normal(0, 0.1, 200)          # strongly paired shift
    assert paired_dz(a, b) > 2 and 0 < cohens_d(a, b) < 0.5   # pairing matters
    assert wilcoxon_p(a, b) < 1e-10 and wilcoxon_p(a, a) == 1.0


def test_compare_explainers_finds_planted_winner(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    for exp, shift in [("gradcam", 0.2), ("rollout", 0.0), ("rise", 0.0)]:
        for i in range(60):
            base = rng.uniform(0.2, 0.6)
            rows.append({"dataset": "d", "model": "m", "explainer": exp, "sample_id": i,
                         "label": 0, "pred": 0, "correct": 1,
                         "miou": base + shift, "max_sensitivity": base})
    csv = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    out = tmp_path / "o"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_explainers.py"),
                        "--results_csv", str(csv), "--output_dir", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = pd.read_csv(out / "explainer_pairwise.csv")
    miou = t[t.metric == "miou"]
    assert len(miou) == 3                                       # 3 choose 2
    g_vs_r = miou[(miou.explainer_A == "gradcam") & (miou.explainer_B == "rollout")].iloc[0]
    assert g_vs_r.better == "gradcam" and g_vs_r.p_holm_family < 0.05
    assert (t.p_holm_global >= t.wilcoxon_p).all() and (t.p_holm_family >= t.wilcoxon_p).all()


# ---------------------------------------------------------------------------
# Dataset audit
# ---------------------------------------------------------------------------

def _run_audit(data, out, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "audit_datasets.py"),
                           "--data-root", str(data), "--output_dir", str(out), *extra],
                          capture_output=True, text=True)


def test_audit_reports_population_labels_and_annotations(tmp_path):
    data = tmp_path / "data"
    make_cub(data / "cub200"); make_voc(data / "voc2012")
    make_imagenet_s(data / "imagenet_s50"); make_nih(data / "nih_chestxray")
    (data / "nih_chestxray" / "train_val_list.txt").write_text("00000002_009.png\n")
    r = _run_audit(data, tmp_path / "a")
    # fake data has 3/200 CUB classes and 2/50 ImageNet-S classes → blocking issues
    assert r.returncode == 1 and "classes have no evaluation sample" in r.stdout
    rep = {d["dataset"]: d for d in json.loads((tmp_path / "a" / "dataset_audit.json").read_text())}
    assert rep["cub200"]["n_samples"] == 3 and rep["cub200"]["n_classes_present"] == 3
    assert rep["pascal_voc"]["n_samples"] == 3            # (image, class) units
    assert any("without any labelled object" in w for w in rep["pascal_voc"]["warnings"])
    assert rep["nih_chestxray"]["patient_leakage"]["excluded_samples"] == 1
    assert rep["imagenet_s50"]["empty_gt_fraction"] == 0.0
    assert 0 < rep["imagenet_s50"]["ignore_fraction_mean"] < 0.1
    counts = pd.read_csv(tmp_path / "a" / "dataset_class_counts.csv")
    assert set(counts[counts.dataset == "pascal_voc"]["class"]) == {"person", "aeroplane", "dog"}


def test_audit_passes_on_complete_dataset_and_flags_empty_masks(tmp_path):
    from PIL import Image
    data = tmp_path / "data"
    make_nih(data / "nih_chestxray")
    (data / "nih_chestxray" / "train_val_list.txt").write_text("00000099_000.png\n")
    r = _run_audit(data, tmp_path / "a", "--datasets", "nih_chestxray")
    assert r.returncode == 0, r.stdout + r.stderr
    # ImageNet-S with an all-background mask → blocking issue
    make_imagenet_s(data / "imagenet_s50")
    m = next((data / "imagenet_s50" / "validation").rglob("*.png"))
    Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(m)
    r = _run_audit(data, tmp_path / "b", "--datasets", "imagenet_s50")
    assert r.returncode == 1 and "empty GT mask" in r.stdout
