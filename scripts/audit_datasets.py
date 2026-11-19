"""
Audit the evaluation protocol of each dataset, using exactly the loaders the
benchmark uses (utils/eval_datasets.py).

Per dataset it reports:
  • evaluation population — split used, number of samples, what was excluded
    (VOC images with no foreground, NIH patients that leak from training)
  • class labels — expected vs present classes, per-class counts, imbalance
    ratio (largest / smallest class)
  • annotation availability — fraction of samples with an empty GT mask,
    mean GT area fraction, mean ignore-region fraction (on up to
    --mask-samples samples, spread over the dataset)

Writes ``dataset_audit.json`` (summary + issues) and ``dataset_class_counts.csv``
to --output_dir. Exit code 1 if any dataset has a blocking issue (missing
classes, samples without GT, or a loader error).

Usage:
    python scripts/audit_datasets.py --data-root /data --output_dir results/audit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.eval_datasets import (  # noqa: E402
    DATASET_CLASSES, IGNORE, NIH_CLASSES, NUM_CLASSES, VOC_CLASSES, spread_indices,
)

SPLIT_DESCRIPTION = {
    "cub200": "official test split (train_test_split.txt == 0)",
    "pascal_voc": "VOC2012 segmentation val (ImageSets/Segmentation/val.txt); dominant-class label",
    "imagenet_s50": "ImageNet-S-50 validation split with pixel masks",
    "nih_chestxray": "BBox_List_2017.csv images, minus patients in train_val_list.txt",
}


def _labels(ds) -> List[int]:
    s = ds.samples
    if not s:
        return []
    if isinstance(s[0][1], int) and len(s[0]) == 2:      # CUB, VOC: (id, label)
        return [x[1] for x in s]
    if len(s[0]) == 3 and isinstance(s[0][2], int):        # ImageNet-S: (img, mask, label)
        return [x[2] for x in s]
    return [x[1] for x in s]                               # NIH: (img, label, boxes)


def _class_name(name: str, label: int) -> str:
    if name == "pascal_voc":
        return VOC_CLASSES[label]
    if name == "nih_chestxray":
        return NIH_CLASSES[label]
    return str(label)


def audit_one(name: str, root: Path, mask_samples: int) -> Dict:
    subdir, cls = DATASET_CLASSES[name]
    path = root / subdir
    rep: Dict = {"dataset": name, "path": str(path), "split": SPLIT_DESCRIPTION[name],
                 "issues": [], "warnings": []}
    if not path.exists():
        rep["status"] = "absent"
        return rep
    try:
        ds = cls(path)
    except Exception as e:                                 # loader refuses the data
        rep["status"] = "error"
        rep["issues"].append(f"loader error: {e}")
        return rep

    labels = _labels(ds)
    counts = pd.Series(labels).value_counts().sort_index()
    expected = NUM_CLASSES[name]
    rep.update({
        "status": "ok",
        "n_samples": len(ds),
        "n_classes_expected": expected,
        "n_classes_present": int(counts.size),
        "class_count_min": int(counts.min()) if len(counts) else 0,
        "class_count_median": float(counts.median()) if len(counts) else 0.0,
        "class_count_max": int(counts.max()) if len(counts) else 0,
        "imbalance_ratio": float(counts.max() / counts.min()) if len(counts) else float("nan"),
        "class_counts": {_class_name(name, int(k)): int(v) for k, v in counts.items()},
    })
    if name == "nih_chestxray":
        # Boxes exist for 8 of the 14 findings by design
        rep["warnings"].append("NIH boxes cover only 8 of 14 findings; per-finding n is small")
        lk = getattr(ds, "leakage", {})
        rep["patient_leakage"] = lk
        if not lk.get("checked"):
            rep["warnings"].append("train_val_list.txt missing: patient leakage NOT checked")
        elif lk.get("excluded_samples"):
            rep["warnings"].append(
                f"excluded {lk['excluded_samples']} samples from "
                f"{len(lk['excluded_patients'])} patients present in train_val_list")
    elif counts.size < expected:
        missing = sorted(set(range(expected)) - set(int(k) for k in counts.index))
        rep["issues"].append(f"{len(missing)} classes have no evaluation sample: "
                             f"{[_class_name(name, m) for m in missing][:10]}")
    if name == "pascal_voc" and getattr(ds, "skipped", None):
        rep["warnings"].append(f"{len(ds.skipped)} images without any labelled object skipped")
    if rep["imbalance_ratio"] > 10:
        rep["warnings"].append(f"class imbalance ratio {rep['imbalance_ratio']:.1f} "
                               "(report class-level / macro metrics)")

    # Annotation statistics on a class-spread subset
    idx = spread_indices(len(ds), mask_samples) if mask_samples else range(len(ds))
    empty, area, ignore = 0, [], []
    for i in idx:
        _, _, m = ds._load(i)
        valid = m != IGNORE
        fg = (m == 1) & valid
        if not fg.any():
            empty += 1
        area.append(fg.sum() / max(valid.sum(), 1))
        ignore.append(1 - valid.mean())
    n_checked = len(area)
    rep.update({
        "masks_checked": n_checked,
        "empty_gt_fraction": empty / n_checked if n_checked else float("nan"),
        "gt_area_fraction_mean": float(np.mean(area)) if area else float("nan"),
        "ignore_fraction_mean": float(np.mean(ignore)) if ignore else float("nan"),
    })
    if empty:
        rep["issues"].append(f"{empty}/{n_checked} checked samples have an empty GT mask")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit evaluation splits, labels and annotations")
    p.add_argument("--data-root", required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASET_CLASSES), choices=list(DATASET_CLASSES))
    p.add_argument("--mask-samples", type=int, default=500,
                   help="Masks to inspect per dataset (spread over classes; 0 = all).")
    p.add_argument("--output_dir", default="results/audit")
    args = p.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    reports = [audit_one(n, Path(args.data_root), args.mask_samples) for n in args.datasets]

    rows = [{"dataset": r["dataset"], "class": c, "n": n}
            for r in reports for c, n in r.get("class_counts", {}).items()]
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "dataset_class_counts.csv"), index=False)
    summary = [{k: v for k, v in r.items() if k != "class_counts"} for r in reports]
    with open(os.path.join(args.output_dir, "dataset_audit.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    blocking = False
    for r in reports:
        line = f"[{r['dataset']}] {r['status']}"
        if r["status"] == "ok":
            line += (f" — {r['n_samples']} samples, {r['n_classes_present']}/"
                     f"{r['n_classes_expected']} classes, imbalance {r['imbalance_ratio']:.1f}, "
                     f"empty GT {r['empty_gt_fraction']:.1%}")
        print(line)
        for w in r["warnings"]:
            print(f"    warning: {w}")
        for i in r["issues"]:
            print(f"    ISSUE:   {i}")
            blocking = True
        if r["status"] == "error":
            blocking = True
    print(f"Wrote {args.output_dir}/dataset_audit.json and dataset_class_counts.csv")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
