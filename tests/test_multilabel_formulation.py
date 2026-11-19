"""
Stage 3 tests, part 2: the multi-label formulation used consistently in
inference, fidelity, RISE/LIME, PGD, correctness and L4 — and an end-to-end
fine-tune → thresholds → evaluation run on fake VOC/NIH data with tiny models.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from utils.tasks import (MULTI, SINGLE, class_log_odds, class_score, is_correct, pgd_loss,
                         set_task)


class _Lin(torch.nn.Module):
    """Linear classifier on channel means — tiny, differentiable, deterministic."""
    def __init__(self, C=4, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(3, C)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach())
        return self.lin(x.mean(dim=(2, 3)))


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def test_class_score_and_log_odds_by_task():
    z = torch.tensor([[2.0, -1.0, 0.5], [0.0, 3.0, -2.0]])
    y = torch.tensor([0, 1])
    torch.testing.assert_close(class_score(z, y, SINGLE), torch.softmax(z, -1)[[0, 1], [0, 1]])
    torch.testing.assert_close(class_score(z, y, MULTI), torch.sigmoid(z)[[0, 1], [0, 1]])
    torch.testing.assert_close(class_log_odds(z, y, MULTI), z[[0, 1], [0, 1]])   # exact logit
    p = torch.softmax(z, -1)[[0, 1], [0, 1]]
    torch.testing.assert_close(class_log_odds(z, y, SINGLE), torch.log(p / (1 - p)))


def test_is_correct_by_task():
    z = torch.tensor([[2.0, 0.1], [0.2, -3.0]])
    y = torch.tensor([1, 0])
    assert is_correct(z, y, SINGLE).tolist() == [False, True]                # argmax
    t = torch.tensor([0.6, 0.5])        # t indexed by each unit's own class
    # unit 0: class 1, σ(0.1)=0.525 ≥ t[1]=0.5 → detected; unit 1: class 0, σ(0.2)=0.55 < t[0]=0.6
    assert is_correct(z, y, MULTI, t).tolist() == [True, False]
    assert is_correct(z, y, MULTI, torch.tensor([0.6, 0.6])).tolist() == [False, False]
    with pytest.raises(ValueError):
        is_correct(z, y, MULTI)


def test_pgd_loss_multilabel_pushes_target_logit_down():
    from utils.pgd import pgd_attack
    m = set_task(_Lin(C=3), MULTI)
    x = torch.rand(3, 16, 16)
    adv = pgd_attack(m, x, target=1, epsilon=0.1, alpha=0.05, steps=5, random_start=False, task=MULTI)
    with torch.no_grad():
        assert m(adv[None])[0, 1] < m(x[None])[0, 1]
    assert adv.min() >= 0 and adv.max() <= 1
    z = torch.tensor([[0.3, -0.2]])
    assert pgd_loss(z, 0, MULTI).item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], torch.ones(1)).item())


# ---------------------------------------------------------------------------
# Fidelity, RISE, LIME use the task's score
# ---------------------------------------------------------------------------

def test_fidelity_multilabel_uses_sigmoid_and_logit_difference():
    from metrics.fidelity import FidelityMetrics
    m = set_task(_Lin(C=4), MULTI)
    x = torch.rand(2, 3, 32, 32)
    y = torch.tensor([1, 3])
    att = torch.rand(2, 4, 4)
    f = FidelityMetrics(mask_mode="zero", k_fractions=(0.25,))
    res = f.compute_all(m, x, y, att)
    mask = f._generate_mask(att, 0.25)
    with torch.no_grad():
        z0, zk, zd = m(x), m(f._apply_mask(x, mask, True)), m(f._apply_mask(x, mask, False))
    idx = torch.arange(2)
    torch.testing.assert_close(res["sufficiency@0.25"], torch.sigmoid(z0)[idx, y] - torch.sigmoid(zk)[idx, y])
    torch.testing.assert_close(res["comprehensiveness@0.25"], torch.sigmoid(z0)[idx, y] - torch.sigmoid(zd)[idx, y])
    torch.testing.assert_close(res["log_odds_drop@0.25"], z0[idx, y] - zd[idx, y])
    # single-label path unchanged: softmax
    set_task(m, SINGLE)
    res_s = f.compute_all(m, x, y, att)
    torch.testing.assert_close(res_s["comprehensiveness@0.25"],
                               torch.softmax(z0, -1)[idx, y] - torch.softmax(zd, -1)[idx, y])


@pytest.mark.parametrize("task", [SINGLE, MULTI])
def test_rise_weights_masks_by_task_score(task):
    from explainers import RISEExplainer
    m = set_task(_Lin(C=3), task)
    e = RISEExplainer(m, patch_size=16, n_masks=6, chunk_size=6)
    x = torch.rand(3, 224, 224)
    sal = e.explain(x, 2)
    assert sal.shape == (14, 14) and torch.isfinite(sal).all()
    # recompute with the OTHER activation: maps must differ (the task matters)
    other = MULTI if task == SINGLE else SINGLE
    sal_other = RISEExplainer(set_task(_Lin(C=3), other), patch_size=16, n_masks=6, chunk_size=6).explain(x, 2)
    assert not torch.allclose(sal, sal_other)


def test_lime_uses_sigmoid_for_multilabel(monkeypatch):
    from explainers import LIMEExplainer
    import explainers.lime as lime_mod
    calls = []
    real = lime_mod.class_score
    monkeypatch.setattr(lime_mod, "class_score", lambda z, c, t: calls.append(t) or real(z, c, t))
    LIMEExplainer(set_task(_Lin(C=3), MULTI), patch_size=16, n_samples=3).explain(torch.rand(3, 224, 224), 1)
    assert calls and set(calls) == {MULTI}


# ---------------------------------------------------------------------------
# Correctness in per-sample metrics; GT cropped out → NaN
# ---------------------------------------------------------------------------

def test_sample_metrics_multilabel_correctness_and_cropped_gt():
    import timm
    from explainers import GradCAMExplainer
    from metrics.sample_metrics import SampleMetricComputer
    torch.manual_seed(0)
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=3).eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        s = torch.sigmoid(model(x))
    y = torch.tensor([0, 2])
    thr = torch.tensor([float(s[0, 0]) + 1e-3, 0.0, float(s[1, 2]) - 1e-3])   # unit 0 missed, unit 1 detected
    set_task(model, MULTI, thr, "validation_f1")
    mask = torch.zeros(2, 224, 224, dtype=torch.long)
    mask[0, 50:120, 50:120] = 1                       # unit 1 has an empty (cropped-out) GT
    bound = SampleMetricComputer(families=["localization"]).bind(GradCAMExplainer(model))
    rows = bound.compute_batch(x, y, mask, torch.rand(2, 14, 14))
    assert [r["correct"] for r in rows] == [0, 1]
    assert rows[0]["score"] == pytest.approx(float(s[0, 0]), abs=1e-6)
    assert 0 < rows[0]["gt_area"] < 1 and rows[0]["egt_lift"] == pytest.approx(rows[0]["egt"] - rows[0]["gt_area"])
    assert rows[1]["gt_area"] == 0 and math.isnan(rows[1]["egt"]) and math.isnan(rows[1]["miou"])
    assert bound.threshold_source == "validation_f1"
    # no thresholds on a multi-label model → flagged default
    set_task(model, MULTI, None)
    b2 = SampleMetricComputer(families=["localization"]).bind(GradCAMExplainer(model))
    b2.compute_batch(x, y, mask, torch.rand(2, 14, 14))
    assert b2.threshold_source == "default_0.5_UNVALIDATED"


# ---------------------------------------------------------------------------
# L4 — correctness-conditional localisation gap
# ---------------------------------------------------------------------------

def _units(label, n_ok, n_bad, egt_ok, egt_bad, area_ok, area_bad, rng):
    rows = []
    for ok, n, egt, area in ((1, n_ok, egt_ok, area_ok), (0, n_bad, egt_bad, area_bad)):
        for _ in range(n):
            a = area + rng.normal(0, 0.01)
            e = egt + rng.normal(0, 0.01)
            rows.append({"label": label, "correct": ok, "egt": e, "gt_area": a,
                         "egt_lift": e - a, "score": rng.uniform()})
    return rows


def test_l4_removes_object_size_confound():
    """Errors on small objects: raw EGT gap is large, area-adjusted CLG is ~0."""
    from metrics.aggregate import correctness_localisation_gap
    rng = np.random.default_rng(0)
    # identical lift over chance (0.3) whether correct or not; incorrect units have small objects
    df = pd.DataFrame(_units(0, 30, 30, 0.8, 0.4, 0.5, 0.1, rng)
                      + _units(1, 30, 30, 0.8, 0.4, 0.5, 0.1, rng))
    raw_gap = df[df.correct == 1].egt.mean() - df[df.correct == 0].egt.mean()
    clg, lo, hi, n = correctness_localisation_gap(df)
    assert raw_gap == pytest.approx(0.4, abs=0.02)
    assert abs(clg) < 0.02 and lo <= 0 <= hi and n == 2


def test_l4_within_class_and_eligibility():
    from metrics.aggregate import correctness_localisation_gap, egt_score_spearman
    rng = np.random.default_rng(1)
    df = pd.DataFrame(_units(0, 20, 20, 0.7, 0.5, 0.2, 0.2, rng)       # gap 0.2
                      + _units(1, 20, 20, 0.6, 0.6, 0.2, 0.2, rng)     # gap 0
                      + _units(2, 50, 3, 0.9, 0.1, 0.2, 0.2, rng))     # < 10 incorrect → ineligible
    clg, lo, hi, n = correctness_localisation_gap(df)
    assert n == 2 and clg == pytest.approx(0.1, abs=0.02)             # equal-weight class mean
    assert lo < clg < hi
    assert math.isnan(correctness_localisation_gap(df, min_n=60)[0])
    # threshold-free companion: perfect within-class monotone relation → ρ = 1
    df2 = pd.DataFrame({"label": [0] * 12, "egt_lift": np.arange(12.0), "score": np.arange(12.0) / 20,
                        "correct": [1] * 12})
    assert egt_score_spearman(df2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End to end: fine-tune (thresholds frozen) → evaluate (VOC units, thresholds used)
# ---------------------------------------------------------------------------

def _voc_full(root: Path):
    from tests.test_multilabel_training import _voc_xml
    for d in ("JPEGImages", "Annotations", "SegmentationClass", "ImageSets/Main", "ImageSets/Segmentation"):
        (root / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ids = [f"2009_{i:06d}" for i in range(16)]
    for k, i in enumerate(ids):
        Image.fromarray(rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)).save(root / "JPEGImages" / f"{i}.jpg")
        _voc_xml(root / "Annotations" / f"{i}.xml", [("person", 0), ("dog", k % 2), ("cat", 0)][: 1 + k % 3])
    seg_ids = ids[12:]
    for i in seg_ids:
        seg = np.zeros((60, 80), np.uint8)
        seg[5:30, 5:40] = 15          # person
        seg[35:55, 45:75] = 12        # dog
        im = Image.fromarray(seg, mode="P"); im.putpalette([0, 0, 0] * 256)
        im.save(root / "SegmentationClass" / f"{i}.png")
    (root / "ImageSets/Main/train.txt").write_text("\n".join(ids[:8]) + "\n")
    (root / "ImageSets/Main/val.txt").write_text("\n".join(ids[8:]) + "\n")
    (root / "ImageSets/Segmentation/val.txt").write_text("\n".join(seg_ids) + "\n")


def _tiny_factory(name, num_classes=0, pretrained=True, **_):
    import timm
    torch.manual_seed(0)
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)


def test_finetune_then_evaluate_multilabel_end_to_end(tmp_path):
    import run_phase3
    import scripts.finetune as ft
    data = tmp_path / "data"
    _voc_full(data / "voc2012")
    weights = tmp_path / "w"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ft, "load_model", _tiny_factory)
        ckpt = ft.main(["--model", "vit_b16", "--dataset", "pascal_voc", "--data-root", str(data),
                        "--seed", "0", "--save-dir", str(weights), "--epochs", "1",
                        "--batch-size", "4", "--num-workers", "0", "--no-amp"])
    ckpt = Path(ckpt)
    assert "_seed0_" in ckpt.name and ckpt.exists()
    meta = torch.load(ckpt, weights_only=True)["metadata"]
    assert meta["task"] == "multi" and meta["selection"] == "final_epoch"
    assert len(meta["decision_thresholds"]) == 20 and "macro_auroc" in meta["val_metrics"]
    side = json.loads(Path(str(ckpt) + ".thresholds.json").read_text())
    assert side["decision_thresholds"] == meta["decision_thresholds"]
    assert side["rule"].startswith("s_c >= t_c")

    out = tmp_path / "ckpt"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_phase3, "load_model", _tiny_factory)
        run_phase3.main(["--data-root", str(data), "--datasets", "pascal_voc", "--models", "vit_b16",
                         "--finetune-seeds", "0", "--explainers", "gradcam",
                         "--metrics", "localization", "fidelity", "--weights-dir", str(weights),
                         "--checkpoint-dir", str(out), "--num-workers", "0", "--device", "cpu"])
    res = pickle.loads((out / "pascal_voc-vit_b16@seed0-gradcam.pkl").read_bytes())
    assert res["task_type"] == "multi" and res["threshold_source"].startswith("validation_f1")
    assert res["decision_thresholds"] == pytest.approx(meta["decision_thresholds"])
    assert res["labels"] == [11, 14] * 4                          # dog, person per seg image
    t = torch.tensor(meta["decision_thresholds"])
    for lab, row in zip(res["labels"], res["metrics"]):
        assert row["correct"] == int(row["score"] >= float(t[lab]))
    summ = pd.read_csv(out / "combination_summary.csv")
    assert summ.task_type.iloc[0] == "multi" and "correctness_localisation_gap" in summ


def test_multilabel_checkpoint_without_thresholds_is_refused(tmp_path):
    import run_phase3
    w = tmp_path / "w"; w.mkdir()
    m = _tiny_factory("vit_b16", num_classes=20)
    torch.save({"model_state_dict": m.state_dict(), "metadata": {"task": "multi"}},
               w / "vit_b16_pascal_voc.pth")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(run_phase3, "load_model", _tiny_factory)
        with pytest.raises(ValueError, match="without validation thresholds"):
            run_phase3.load_finetuned_model("vit_b16", "pascal_voc", str(w))
        torch.save({"model_state_dict": m.state_dict(), "metadata": {"task": "single"}},
                   w / "vit_b16_pascal_voc.pth")
        with pytest.raises(ValueError, match="trained as single-label"):
            run_phase3.load_finetuned_model("vit_b16", "pascal_voc", str(w))


def test_finetune_nih_uses_class_weights(tmp_path):
    import scripts.finetune as ft
    from tests.test_multilabel_training import make_nih_train
    root = tmp_path / "data" / "nih_chestxray"
    names = make_nih_train(root)
    for n in names:
        Image.new("L", (32, 32), 100).save(root / "images" / n)
    captured = {}
    real = ft.fine_tune

    def spy(model, tl, vl, config):
        captured.update(config)
        return real(model, tl, vl, config)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ft, "load_model", _tiny_factory)
        mp.setattr(ft, "fine_tune", spy)
        ft.main(["--model", "vit_b16", "--dataset", "nih_chestxray", "--data-root", str(tmp_path / "data"),
                 "--save-dir", str(tmp_path / "w"), "--epochs", "1", "--batch-size", "8",
                 "--max-steps", "1", "--num-workers", "0", "--no-amp"])
    pw = captured["pos_weight"]
    assert pw.shape == (14,) and torch.isfinite(pw).all() and captured["task"] == "multi"
