"""
Stage 3 tests, part 1: multi-label training components, validated in
isolation — loss, label construction, Mixup, thresholds, training splits.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from utils.eval_datasets import NIH_CLASSES, VOC_CLASSES


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def test_masked_bce_matches_reference_and_ignores_masked_entries():
    from training.loss import MaskedBCEWithLogits
    torch.manual_seed(0)
    z, y = torch.randn(4, 5), (torch.rand(4, 5) > 0.5).float()
    w = torch.ones(4, 5); w[0, 1] = 0; w[2, 3] = 0
    pw = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.5])
    loss = MaskedBCEWithLogits(pw)(z, y, w)
    ref = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw, reduction="none")
    assert loss.item() == pytest.approx((ref * w).sum().item() / w.sum().item(), rel=1e-6)
    # an ignored entry has no influence: changing its logit leaves the loss unchanged
    z2 = z.clone(); z2[0, 1] = 50.0
    assert MaskedBCEWithLogits(pw)(z2, y, w).item() == pytest.approx(loss.item(), rel=1e-6)
    # unweighted call equals plain mean BCE
    assert MaskedBCEWithLogits()(z, y).item() == pytest.approx(
        F.binary_cross_entropy_with_logits(z, y).item(), rel=1e-6)


def test_build_loss_by_task():
    from timm.loss import SoftTargetCrossEntropy
    from training.loss import MaskedBCEWithLogits, build_loss
    tr, va = build_loss("single")
    assert isinstance(tr, SoftTargetCrossEntropy) and va.label_smoothing == 0.1
    tr, va = build_loss("multi", pos_weight=torch.ones(3))
    assert isinstance(tr, MaskedBCEWithLogits) and tr is va
    with pytest.raises(ValueError):
        build_loss("pascal_voc")


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _voc_xml(path: Path, objects):
    objs = "".join(f"<object><name>{n}</name><difficult>{int(d)}</difficult></object>"
                   for n, d in objects)
    path.write_text(f"<annotation>{objs}</annotation>")


def test_voc_labels_difficult_only_classes_are_ignored(tmp_path):
    from utils.train_datasets import voc_labels
    _voc_xml(tmp_path / "a.xml", [("person", 0), ("dog", 1), ("cat", 0), ("cat", 1)])
    y, w = voc_labels(tmp_path / "a.xml")
    P, D, C = (VOC_CLASSES.index(n) for n in ("person", "dog", "cat"))
    assert y[P] == 1 and y[C] == 1 and y[D] == 0
    assert w[D] == 0                         # difficult-only → ignored
    assert w[C] == 1 and w[P] == 1           # non-difficult present → used
    assert w.sum() == len(VOC_CLASSES) - 1 and y.sum() == 2


def make_voc_train(root: Path):
    for d in ("JPEGImages", "Annotations", "ImageSets/Main", "ImageSets/Segmentation"):
        (root / d).mkdir(parents=True, exist_ok=True)
    ids = [f"2008_{i:06d}" for i in range(8)]
    for i in ids:
        Image.new("RGB", (40, 30)).save(root / "JPEGImages" / f"{i}.jpg")
        _voc_xml(root / "Annotations" / f"{i}.xml", [("person", 0), ("car", int(i.endswith("3")))])
    (root / "ImageSets/Main/train.txt").write_text("\n".join(ids[:4]) + "\n")
    (root / "ImageSets/Main/val.txt").write_text("\n".join(ids[4:]) + "\n")
    (root / "ImageSets/Segmentation/val.txt").write_text(f"{ids[6]}\n{ids[7]}\n")   # eval ⊂ Main/val
    return ids


def test_voc_splits_exclude_evaluation_images(tmp_path):
    from utils.train_datasets import voc_items
    ids = make_voc_train(tmp_path)
    train, val = voc_items(tmp_path)
    assert [p.stem for p, *_ in train] == ids[:4]
    assert [p.stem for p, *_ in val] == ids[4:6]                 # seg-val removed
    # leakage is a hard error
    (tmp_path / "ImageSets/Segmentation/val.txt").write_text(f"{ids[0]}\n")
    with pytest.raises(ValueError, match="evaluation"):
        voc_items(tmp_path)


def make_nih_train(root: Path, n_patients=40):
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows, names = ["Image Index,Finding Labels"], []
    for p in range(1, n_patients + 1):
        for k in range(2):
            n = f"{p:08d}_{k:03d}.png"
            names.append(n)
            lab = "No Finding" if p % 3 == 0 else ("Effusion|Mass" if p % 3 == 1 else "Hernia")
            rows.append(f"{n},{lab}")
    (root / "Data_Entry_2017_v2.csv").write_text("\n".join(rows) + "\n")
    (root / "train_val_list.txt").write_text("\n".join(names) + "\n")
    (root / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n00000005_000.png,Mass,1,1,5,5,,,\n")
    return names


def test_nih_labels_and_patient_disjoint_split(tmp_path):
    from utils.eval_datasets import nih_patient_id
    from utils.train_datasets import nih_items, positive_weights, _ImageListDataset
    make_nih_train(tmp_path)
    train, val = nih_items(tmp_path)
    tp = {nih_patient_id(p.name) for p, *_ in train}
    vp = {nih_patient_id(p.name) for p, *_ in val}
    assert tp and vp and not (tp & vp)                           # patient-disjoint
    assert "00000005" not in tp | vp                             # bbox patient excluded
    assert len(vp) == round(0.1 * 39)
    by = {p.name: y for p, y, _ in train + val}
    y = by["00000004_000.png"]                                   # p % 3 == 1 → Effusion|Mass
    assert y[NIH_CLASSES.index("Effusion")] == 1 and y[NIH_CLASSES.index("Mass")] == 1 and y.sum() == 2
    assert by["00000003_001.png"].sum() == 0                     # No Finding → all zeros
    pw = positive_weights(_ImageListDataset(train, None, "multi"))
    n_h = sum(by[p.name][NIH_CLASSES.index("Hernia")] for p, *_ in train)
    assert pw[NIH_CLASSES.index("Hernia")] == pytest.approx((len(train) - n_h) / n_h)


def test_cub_train_split_holds_out_and_avoids_test(tmp_path):
    from tests.test_eval_datasets import make_cub
    from utils.train_datasets import cub200_items
    make_cub(tmp_path)                       # img_1 of each class is training, img_0 test
    train, val = cub200_items(tmp_path)
    names = [p.name for p, _ in train + val]
    assert all(n == "img_1.jpg" for n in names)                  # no test image
    assert sorted(y for _, y in train + val) == [0, 1, 2]        # eval class order


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def test_multilabel_mixup_mixes_targets_and_masks():
    from training.mixup import MultiLabelMixup, build_mixup_fn
    np.random.seed(0)
    x = torch.rand(4, 3, 8, 8)
    y = torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
    w = torch.ones(4, 3); w[0, 2] = 0
    xm, ym, wm = MultiLabelMixup()(x, y, w)
    # recover λ from the images and check targets use the same λ
    lam = float(((xm[0] - x[3]) / (x[0] - x[3])).median())
    torch.testing.assert_close(ym, lam * y + (1 - lam) * y.flip(0))
    assert ym.min() >= 0 and ym.max() <= 1
    assert wm[0, 2] == 0 and wm[3, 2] == 0                     # ignored in either → ignored
    assert torch.equal(wm, w * w.flip(0))
    assert isinstance(build_mixup_fn(3, "multi"), MultiLabelMixup)
    # identity when disabled
    xi, yi, wi = MultiLabelMixup(alpha=0)(x, y, w)
    assert torch.equal(xi, x) and torch.equal(yi, y)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_f1_optimal_thresholds():
    from training.thresholds import f1_optimal_thresholds
    s = np.array([[0.9, 0.2], [0.8, 0.3], [0.4, 0.1], [0.3, 0.6], [0.2, 0.05]])
    y = np.array([[1, 0], [1, 0], [0, 0], [1, 0], [0, 0]])
    t, info = f1_optimal_thresholds(s, y)
    # class 0: brute force over candidate thresholds
    best = max(((2 * ((s[:, 0] >= c) & (y[:, 0] == 1)).sum()) /
                ((s[:, 0] >= c).sum() + y[:, 0].sum()), c) for c in s[:, 0])
    assert info["best_f1"][0] == pytest.approx(best[0]) and t[0] == pytest.approx(0.3)
    assert info["fallback_classes"] == [1] and t[1] == 0.5        # no validation positives
    # ignored entries do not count
    w = np.ones_like(s); w[3, 0] = 0
    t2, info2 = f1_optimal_thresholds(s, y, w)
    assert t2[0] == pytest.approx(0.8) and info2["n_val_positives"][0] == 2
