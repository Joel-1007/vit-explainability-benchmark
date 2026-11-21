"""
train_datasets.py — fine-tuning datasets, consistent with utils/eval_datasets.py.

Class orders are shared with evaluation (a mismatch would silently explain the
wrong class). Every split is checked for overlap with the evaluation set, and
model-selection / threshold data never comes from the evaluation set.

Splits
------
CUB-200-2011  train  : official train split minus a 10% per-class hold-out
              val    : that hold-out (monitoring only — final-epoch selection)
              eval   : official test split (utils/eval_datasets)
ImageNet-S-50 train  : <root>/train/<wnid>/*.JPEG (ImageNet-1k training images of
                       the 50 classes) minus a 10% per-class hold-out
              val    : that hold-out
              eval   : <root>/validation (ImageNet-S masks)
PASCAL VOC    train  : ImageSets/Main/train.txt, image-level labels from the XML
                       annotations (multi-label)
              val    : ImageSets/Main/val.txt MINUS ImageSets/Segmentation/val.txt
                       (threshold estimation + monitoring)
              eval   : ImageSets/Segmentation/val.txt — asserted disjoint from both
NIH CXR-14    train  : train_val_list.txt, 90% of PATIENTS (fixed split, seed 0)
              val    : the other 10% of patients (thresholds + monitoring)
              eval   : BBox_List_2017.csv images; any patient in train_val_list is
                       excluded at evaluation time (utils/eval_datasets.NIHBBoxEval)

Multi-label targets
-------------------
Items are (image, y, w): y ∈ {0,1}^C multi-hot, w ∈ {0,1}^C per-class loss
weights. VOC: y_c = 1 if any non-difficult object of class c; w_c = 0 if class
c occurs ONLY as "difficult" objects (ignored — the standard VOC
classification protocol). NIH: labels from Data_Entry_2017(_v2).csv
"Finding Labels"; "No Finding" = all zeros; w = 1.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.eval_datasets import NIH_CLASSES, VOC_CLASSES, nih_patient_id
from utils.tasks import MULTI, SINGLE, TASK_TYPE

HOLDOUT_FRACTION = 0.10
SPLIT_SEED = 0           # fixed: identical splits for every fine-tuning seed
NIH_LABEL_FILES = ("Data_Entry_2017_v2.csv", "Data_Entry_2017.csv", "Data_Entry_2017_v2020.csv")


def _holdout(groups: Sequence, fraction: float = HOLDOUT_FRACTION,
             seed: int = SPLIT_SEED) -> np.ndarray:
    """Boolean hold-out mask: ``fraction`` of the items of every group (≥1 if ≥2 items)."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    hold = np.zeros(len(groups), bool)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        k = max(1, int(round(fraction * len(idx))))
        hold[rng.choice(idx, size=k, replace=False)] = True
    return hold


class _ImageListDataset(Dataset):
    """(path, target[, weight]) items → (transform(image), target[, weight])."""

    def __init__(self, items: List[tuple], transform: Optional[Callable], task: str) -> None:
        self.items, self.transform, self.task = items, transform, task

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        path, *rest = self.items[i]
        img = Image.open(path).convert("RGB")
        x = self.transform(img) if self.transform else img
        if self.task == SINGLE:
            return x, int(rest[0])
        y, w = rest
        return x, torch.as_tensor(y, dtype=torch.float32), torch.as_tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Single-label
# ---------------------------------------------------------------------------

def cub200_items(root: Path) -> Tuple[List[tuple], List[tuple]]:
    classes = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
    cls_idx = {c: i for i, c in enumerate(classes)}
    id_to_path = dict(l.split() for l in (root / "images.txt").read_text().splitlines() if l.strip())
    train_ids = [i for i, flag in (l.split() for l in
                 (root / "train_test_split.txt").read_text().splitlines() if l.strip()) if flag == "1"]
    items = [(root / "images" / id_to_path[i], cls_idx[id_to_path[i].split("/")[0]])
             for i in sorted(train_ids, key=int)]
    hold = _holdout([y for _, y in items])
    return ([it for it, h in zip(items, hold) if not h], [it for it, h in zip(items, hold) if h])


def imagenet_s50_items(root: Path) -> Tuple[List[tuple], List[tuple]]:
    eval_classes = sorted(p.name for p in (root / "validation").iterdir() if p.is_dir())
    train_classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if train_classes != eval_classes:
        raise ValueError("ImageNet-S-50: train/ and validation/ class folders differ "
                         f"({len(train_classes)} vs {len(eval_classes)})")
    items = [(p, y) for y, wnid in enumerate(train_classes)
             for p in sorted((root / "train" / wnid).glob("*.JPEG"))]
    eval_names = {p.name for p in (root / "validation").rglob("*.JPEG")}
    overlap = {p.name for p, _ in items} & eval_names
    if overlap:
        raise ValueError(f"ImageNet-S-50: {len(overlap)} training images are evaluation images")
    hold = _holdout([y for _, y in items])
    return ([it for it, h in zip(items, hold) if not h], [it for it, h in zip(items, hold) if h])


# ---------------------------------------------------------------------------
# Multi-label
# ---------------------------------------------------------------------------

def voc_labels(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-hot y and weights w from one VOC annotation (difficult-only → ignored)."""
    C = len(VOC_CLASSES)
    pos, difficult = np.zeros(C, bool), np.zeros(C, bool)
    for obj in ET.parse(xml_path).getroot().iter("object"):
        c = VOC_CLASSES.index(obj.findtext("name").strip())
        if obj.findtext("difficult", "0").strip() == "1":
            difficult[c] = True
        else:
            pos[c] = True
    w = np.where(difficult & ~pos, 0.0, 1.0)
    return pos.astype(np.float32), w.astype(np.float32)


def _ids(path: Path) -> List[str]:
    return [l.split()[0] for l in path.read_text().splitlines() if l.strip()]


def voc_items(root: Path) -> Tuple[List[tuple], List[tuple]]:
    seg_val = set(_ids(root / "ImageSets" / "Segmentation" / "val.txt"))
    train_ids = _ids(root / "ImageSets" / "Main" / "train.txt")
    val_ids = [i for i in _ids(root / "ImageSets" / "Main" / "val.txt") if i not in seg_val]
    leak = set(train_ids) & seg_val
    if leak:
        raise ValueError(f"VOC: {len(leak)} Main/train images are segmentation-val (evaluation) images")
    if set(train_ids) & set(val_ids):
        raise ValueError("VOC: Main/train and Main/val overlap")

    def items(ids):
        return [(root / "JPEGImages" / f"{i}.jpg", *voc_labels(root / "Annotations" / f"{i}.xml"))
                for i in ids]
    return items(train_ids), items(val_ids)


def nih_items(root: Path) -> Tuple[List[tuple], List[tuple]]:
    import pandas as pd

    entry = next((root / n for n in NIH_LABEL_FILES
                  if (root / n).exists()), None)
    if entry is None:
        raise FileNotFoundError(f"{root}: expected one of {', '.join(NIH_LABEL_FILES)}")
    df = pd.read_csv(entry, usecols=["Image Index", "Finding Labels"])
    labels = dict(zip(df["Image Index"], df["Finding Labels"]))
    names = _ids(root / "train_val_list.txt")
    bbox_patients = set()
    if (root / "BBox_List_2017.csv").exists():
        bbox_patients = {nih_patient_id(n) for n in
                         pd.read_csv(root / "BBox_List_2017.csv")["Image Index"]}
    names = [n for n in names if nih_patient_id(n) not in bbox_patients]   # guard

    def target(name):
        y = np.zeros(len(NIH_CLASSES), np.float32)
        for f in str(labels[name]).split("|"):
            if f != "No Finding":
                y[NIH_CLASSES.index(f)] = 1.0
        return y

    patients = sorted({nih_patient_id(n) for n in names})
    rng = np.random.default_rng(SPLIT_SEED)
    val_patients = set(rng.choice(patients, size=max(1, int(round(HOLDOUT_FRACTION * len(patients)))),
                                  replace=False))
    ones = np.ones(len(NIH_CLASSES), np.float32)
    items = [(root / "images" / n, target(n), ones) for n in names]
    is_val = [nih_patient_id(n) in val_patients for n in names]
    return ([it for it, v in zip(items, is_val) if not v], [it for it, v in zip(items, is_val) if v])


_ITEMS = {"cub200": ("cub200", cub200_items), "imagenet_s50": ("imagenet_s50", imagenet_s50_items),
          "pascal_voc": ("voc2012", voc_items), "nih_chestxray": ("nih_chestxray", nih_items)}


def build_train_val(name: str, data_root: str | Path, train_tf: Callable,
                    val_tf: Callable) -> Tuple[Dataset, Dataset]:
    """(train, val) datasets for fine-tuning ``name``; val never overlaps evaluation."""
    subdir, fn = _ITEMS[name]
    train, val = fn(Path(data_root) / subdir)
    task = TASK_TYPE[name]
    return _ImageListDataset(train, train_tf, task), _ImageListDataset(val, val_tf, task)


def positive_weights(ds: Dataset) -> torch.Tensor:
    """BCE pos_weight_c = N_neg / N_pos over valid (w = 1) training labels."""
    y = np.stack([it[1] for it in ds.items])
    w = np.stack([it[2] for it in ds.items])
    pos = (y * w).sum(0)
    neg = ((1 - y) * w).sum(0)
    return torch.as_tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32)
