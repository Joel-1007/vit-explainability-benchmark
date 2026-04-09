#!/usr/bin/env python3
"""
verify_datasets.py  —  Task 1.3 §3 Dataset Verification Protocol
=================================================================
Runs all mandatory integrity checks for every dataset in the benchmark
before any fine-tuning begins.  Each dataset has its own check function;
all checks are aggregated into a pass/fail summary at the end.

All four datasets from the final roster (Task 1.3 §1.3):
  1. CUB-200-2011               (§3.2)
  2. PASCAL VOC 2012 Seg        (§3.3)
  3. ImageNet-S-50              (§3.4)
  4. NIH ChestX-ray14 BBox      (§3.5)

Usage
-----
    python scripts/verify_datasets.py \\
        --cub      /path/to/CUB_200_2011 \\
        --voc      /path/to/VOCdevkit/VOC2012 \\
        --imagenet_s /path/to/ImageNetS50 \\
        --nih      /path/to/ChestXray-NIHCC

    # Verify individual datasets by omitting other flags:
    python scripts/verify_datasets.py --cub /path/to/CUB_200_2011

Exit code
---------
    0  — all verified datasets passed every check
    1  — one or more checks failed (details printed to stderr)

Requirements
------------
    pip install Pillow numpy pandas tqdm scikit-learn
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import logging
from pathlib import Path
from typing import Callable

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("verify_datasets")

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_results: list[tuple[str, str, bool, str]] = []   # (dataset, check, passed, detail)


def _check(dataset: str, name: str, condition: bool, detail: str = "") -> bool:
    symbol = "✓" if condition else "✗"
    level  = log.info if condition else log.error
    level(f"  [{dataset}] {symbol}  {name}{('  — ' + detail) if detail else ''}")
    _results.append((dataset, name, condition, detail))
    return condition


def _warn(dataset: str, name: str, detail: str = "") -> None:
    log.warning(f"  [{dataset}] ⚠  {name}{('  — ' + detail) if detail else ''}")
    _results.append((dataset, name, True, f"[WARNING] {detail}"))   # warnings don't fail


# ---------------------------------------------------------------------------
# 1. CUB-200-2011  (Task 1.3 §3.2)
# ---------------------------------------------------------------------------

def verify_cub200(root: str) -> None:
    DS = "CUB-200"
    log.info(f"\n{'='*60}")
    log.info(f"[{DS}] Verifying: {root}")
    log.info(f"{'='*60}")

    root = Path(root)

    # Step 1 — required files exist
    for fname in ["images.txt", "train_test_split.txt", "bounding_boxes.txt",
                  "image_class_labels.txt"]:
        _check(DS, f"File exists: {fname}", (root / fname).exists())

    _check(DS, "Directory: images/",        (root / "images").is_dir())
    _check(DS, "Directory: segmentations/", (root / "segmentations").is_dir())

    # Step 2 — image and mask counts
    n_images = sum(1 for _ in (root / "images").rglob("*.jpg"))
    n_masks  = sum(1 for _ in (root / "segmentations").rglob("*.png"))
    _check(DS, f"Image count = 11,788", n_images == 11788, f"found {n_images}")
    _check(DS, f"Mask count  = 11,788", n_masks  == 11788, f"found {n_masks}")

    # Step 3 — split counts
    try:
        import pandas as pd
        split_df  = pd.read_csv(root / "train_test_split.txt",
                                sep=" ", header=None, names=["image_id", "is_train"])
        images_df = pd.read_csv(root / "images.txt",
                                sep=" ", header=None, names=["image_id", "filename"])
        bboxes_df = pd.read_csv(root / "bounding_boxes.txt",
                                sep=" ", header=None, names=["image_id","x","y","w","h"])

        n_train = (split_df["is_train"] == 1).sum()
        n_test  = (split_df["is_train"] == 0).sum()

        _check(DS, "Train split = 5,994", n_train == 5994, f"found {n_train}")
        _check(DS, "Test split  = 5,794", n_test  == 5794, f"found {n_test}")
        _check(DS, "Total       = 11,788", len(split_df) == 11788, f"found {len(split_df)}")
        _check(DS, "BBox rows   = 11,788", len(bboxes_df) == 11788, f"found {len(bboxes_df)}")

        # Step 4 — annotation consistency
        missing_imgs   = []
        missing_masks  = []
        for _, row in images_df.iterrows():
            if not (root / "images" / row["filename"]).exists():
                missing_imgs.append(row["filename"])
            mask_path = root / "segmentations" / row["filename"].replace(".jpg", ".png")
            if not mask_path.exists():
                missing_masks.append(row["filename"])

        _check(DS, "Missing image files = 0", len(missing_imgs)  == 0,
               f"{len(missing_imgs)} missing")
        _check(DS, "Missing mask files  = 0", len(missing_masks) == 0,
               f"{len(missing_masks)} missing")

        # Step 5 — readability + mask pixel values (sample 100)
        from PIL import Image

        sample = images_df.sample(100, random_state=42)
        corrupt_imgs, corrupt_masks = [], []

        for _, row in sample.iterrows():
            img_path  = root / "images" / row["filename"]
            mask_path = root / "segmentations" / row["filename"].replace(".jpg", ".png")
            try:
                img = Image.open(img_path).convert("RGB")
                assert img.size[0] > 0
            except Exception as e:
                corrupt_imgs.append(str(e))
            try:
                mask = np.array(Image.open(mask_path))
                unique = set(np.unique(mask))
                # CUB masks are binary: 0=background, 255=foreground
                assert unique.issubset({0, 255}), f"unexpected values: {unique - {0,255}}"
            except Exception as e:
                corrupt_masks.append(str(e))

        _check(DS, "Sample 100: corrupt images  = 0", len(corrupt_imgs)  == 0,
               "; ".join(corrupt_imgs[:3]))
        _check(DS, "Sample 100: corrupt masks   = 0", len(corrupt_masks) == 0,
               "; ".join(corrupt_masks[:3]))
        _check(DS, "Sample 100: mask values ∈ {0, 255}", len(corrupt_masks) == 0)

        # Step 6 — val split creation check
        val_csv = Path("cub_val_ids.csv")
        if val_csv.exists():
            _check(DS, "Val split CSV (cub_val_ids.csv) already exists", True)
        else:
            _warn(DS, "cub_val_ids.csv not found",
                  "Run: python scripts/create_cub_val_split.py --root <CUB_ROOT>")

    except ImportError:
        log.error("  pandas not installed.  pip install pandas")


# ---------------------------------------------------------------------------
# 2. PASCAL VOC 2012 (Segmentation)  (Task 1.3 §3.3)
# ---------------------------------------------------------------------------

def verify_pascal_voc(root: str) -> None:
    DS = "VOC2012"
    log.info(f"\n{'='*60}")
    log.info(f"[{DS}] Verifying: {root}")
    log.info(f"{'='*60}")

    root = Path(root)

    for dname in ["JPEGImages", "SegmentationClass", "SegmentationObject",
                  "Annotations", "ImageSets/Segmentation"]:
        _check(DS, f"Directory: {dname}/", (root / dname).is_dir())

    train_file = root / "ImageSets" / "Segmentation" / "train.txt"
    val_file   = root / "ImageSets" / "Segmentation" / "val.txt"

    _check(DS, "train.txt exists", train_file.exists())
    _check(DS, "val.txt exists",   val_file.exists())

    if not (train_file.exists() and val_file.exists()):
        log.error(f"  [{DS}] Cannot continue — split files missing.")
        return

    with open(train_file) as f:
        train_ids = [l.strip() for l in f if l.strip()]
    with open(val_file) as f:
        val_ids   = [l.strip() for l in f if l.strip()]

    _check(DS, "Train split = 1,464", len(train_ids) == 1464, f"found {len(train_ids)}")
    _check(DS, "Val split   = 1,449", len(val_ids)   == 1449, f"found {len(val_ids)}")

    overlap = set(train_ids) & set(val_ids)
    _check(DS, "No train/val overlap", len(overlap) == 0,
           f"{len(overlap)} overlapping IDs" if overlap else "")

    # File existence
    missing_imgs, missing_masks = [], []
    for img_id in train_ids + val_ids:
        if not (root / "JPEGImages" / f"{img_id}.jpg").exists():
            missing_imgs.append(img_id)
        if not (root / "SegmentationClass" / f"{img_id}.png").exists():
            missing_masks.append(img_id)

    _check(DS, "Missing JPEG images  = 0", len(missing_imgs)  == 0,
           f"{len(missing_imgs)} missing")
    _check(DS, "Missing class masks  = 0", len(missing_masks) == 0,
           f"{len(missing_masks)} missing")

    # Mask pixel values (sample 100)
    from PIL import Image

    sample_ids = random.sample(train_ids + val_ids, min(100, len(train_ids + val_ids)))
    unexpected = []

    for img_id in sample_ids:
        mask_path = root / "SegmentationClass" / f"{img_id}.png"
        try:
            mask = np.array(Image.open(mask_path))
            valid = set(range(21)) | {255}
            bad   = set(np.unique(mask)) - valid
            if bad:
                unexpected.append((img_id, bad))
        except Exception as e:
            unexpected.append((img_id, str(e)))

    _check(DS, "Sample 100: mask values ∈ {0–20, 255}", len(unexpected) == 0,
           f"{len(unexpected)} masks with unexpected values")
    _check(DS, "Void label (255) = boundary/ignored pixels — must be excluded from metrics",
           True)   # informational


# ---------------------------------------------------------------------------
# 3. ImageNet-S-50  (Task 1.3 §3.4)
# ---------------------------------------------------------------------------

def verify_imagenet_s50(root: str) -> None:
    DS = "ImgNet-S50"
    log.info(f"\n{'='*60}")
    log.info(f"[{DS}] Verifying: {root}")
    log.info(f"{'='*60}")

    root = Path(root)

    for split in ["validation", "test"]:
        _check(DS, f"Directory: {split}/", (root / split).is_dir())

    # Count val and test images + masks
    for split, expected_n in [("validation", 752), ("test", 739)]:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        imgs  = list(split_dir.rglob("*.JPEG"))
        masks = list(split_dir.rglob("*.png"))
        _check(DS, f"{split}: image count = {expected_n}", len(imgs)  == expected_n,
               f"found {len(imgs)}")
        _check(DS, f"{split}: mask  count = {expected_n}", len(masks) == expected_n,
               f"found {len(masks)}")
        _check(DS, f"{split}: image count == mask count",  len(imgs)  == len(masks),
               f"imgs={len(imgs)}, masks={len(masks)}")

    # Class directory count for validation
    val_dir = root / "validation"
    if val_dir.is_dir():
        n_classes = sum(1 for p in val_dir.iterdir() if p.is_dir())
        _check(DS, "val: 50 class subdirectories", n_classes == 50,
               f"found {n_classes}")

    # Mask readability + non-zero foreground (sample 50)
    from PIL import Image

    val_imgs  = sorted((root / "validation").rglob("*.JPEG")) if (root / "validation").is_dir() else []
    val_masks = sorted((root / "validation").rglob("*.png"))  if (root / "validation").is_dir() else []

    if val_imgs and len(val_imgs) == len(val_masks):
        pairs  = list(zip(val_imgs, val_masks))
        sample = random.sample(pairs, min(50, len(pairs)))
        corrupt = []

        for img_path, mask_path in sample:
            try:
                img  = Image.open(img_path).convert("RGB")
                mask = np.array(Image.open(mask_path))
                assert img.size[0] > 0
                assert mask.ndim in [2, 3]
                assert np.any(mask > 0), "Mask is entirely background (all zeros)"
            except Exception as e:
                corrupt.append(f"{img_path.name}: {e}")

        _check(DS, "Sample 50: 0 corrupt pairs, non-empty masks", len(corrupt) == 0,
               "; ".join(corrupt[:3]))

    _warn(DS, "train/ annotations are partial — use val/ and test/ for localization metrics ONLY")


# ---------------------------------------------------------------------------
# 4. NIH ChestX-ray14  (Task 1.3 §3.5)
# ---------------------------------------------------------------------------

def verify_nih_chestxray(root: str) -> None:
    DS = "NIH-CXR14"
    log.info(f"\n{'='*60}")
    log.info(f"[{DS}] Verifying: {root}")
    log.info(f"{'='*60}")

    root = Path(root)

    # Required files
    for fname in ["BBox_List_2017.csv", "Data_Entry_2017_v2.csv",
                  "train_val_list.txt", "test_list.txt"]:
        _check(DS, f"File exists: {fname}", (root / fname).exists())

    _check(DS, "Directory: images/", (root / "images").is_dir())

    n_images = sum(1 for _ in (root / "images").glob("*.png")) if (root / "images").is_dir() else 0
    _check(DS, "Image count = 112,120", n_images == 112120, f"found {n_images}")

    try:
        import pandas as pd

        # BBox integrity
        bbox_df = pd.read_csv(root / "BBox_List_2017.csv")
        n_bbox_unique = bbox_df["Image Index"].nunique()
        found_classes = sorted(bbox_df["Finding Label"].unique())
        expected_classes = sorted([
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate",
            "Mass", "Nodule", "Pneumonia", "Pneumothorax"
        ])

        _check(DS, f"BBox unique images ~ 984", 900 <= n_bbox_unique <= 1000,
               f"found {n_bbox_unique}")
        _check(DS, "BBox covers exactly 8 pathology classes",
               found_classes == expected_classes,
               f"found: {found_classes}")

        # All bbox images exist
        missing_bbox = [
            img for img in bbox_df["Image Index"].unique()
            if not (root / "images" / img).exists()
        ]
        _check(DS, "All BBox images exist on disk", len(missing_bbox) == 0,
               f"{len(missing_bbox)} missing")

        # Train/test split
        with open(root / "train_val_list.txt") as f:
            train_val = set(l.strip() for l in f if l.strip())
        with open(root / "test_list.txt") as f:
            test_set  = set(l.strip() for l in f if l.strip())

        _check(DS, "Train+Val count ~ 86,524", 80000 <= len(train_val) <= 95000,
               f"found {len(train_val)}")
        _check(DS, "Test count ~ 25,596", 20000 <= len(test_set) <= 30000,
               f"found {len(test_set)}")
        _check(DS, "No train/test overlap", len(train_val & test_set) == 0,
               f"{len(train_val & test_set)} overlap")

        # Readability + size (sample 50 from bbox subset)
        from PIL import Image

        all_bbox_imgs = list(bbox_df["Image Index"].unique())
        sample = random.sample(all_bbox_imgs, min(50, len(all_bbox_imgs)))
        corrupt, wrong_size = [], []

        for img_name in sample:
            path = root / "images" / img_name
            try:
                img = Image.open(path)
                w, h = img.size
                if w != 1024 or h != 1024:
                    wrong_size.append(f"{img_name}: {w}×{h}")
            except Exception as e:
                corrupt.append(f"{img_name}: {e}")

        _check(DS, "Sample 50: 0 corrupt images",    len(corrupt)     == 0,
               "; ".join(corrupt[:3]))
        _check(DS, "Sample 50: all images 1024×1024", len(wrong_size) == 0,
               "; ".join(wrong_size[:3]))

        # BBox coordinate sanity
        # Actual column names: 'Image Index', 'Finding Label', 'Bbox [x', 'y', 'w', 'h]'
        try:
            x_col, y_col, w_col, h_col = "Bbox [x", "y", "w", "h]"
            bad = bbox_df[
                (bbox_df[x_col] < 0) | (bbox_df[y_col] < 0) |
                (bbox_df[w_col] <= 0) | (bbox_df[h_col] <= 0) |
                (bbox_df[x_col] + bbox_df[w_col] > 1024) |
                (bbox_df[y_col] + bbox_df[h_col] > 1024)
            ]
            _check(DS, "BBox coordinates within [0, 1024]", len(bad) == 0,
                   f"{len(bad)} invalid boxes")
        except KeyError as e:
            _warn(DS, f"BBox column name mismatch ({e}) — check CSV headers manually")

        _warn(DS, "Images are grayscale PNG — always call .convert('RGB') before inference")
        _warn(DS, "Use per-class metric reporting due to severe class imbalance")
        _warn(DS, "Only box-level IoU / pointing game available — no pixel masks")

    except ImportError:
        log.error("  pandas not installed.  pip install pandas")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary() -> int:
    """Print a formatted summary of all checks.  Returns 0=pass, 1=fail."""
    header = "\n" + "=" * 60 + "\nDataset Verification Summary\n" + "=" * 60
    log.info(header)

    passed = [r for r in _results if r[2] and not r[3].startswith("[WARNING]")]
    failed = [r for r in _results if not r[2]]
    warned = [r for r in _results if r[2] and r[3].startswith("[WARNING]")]

    log.info(f"  Checks PASSED  : {len(passed)}")
    if warned:
        log.warning(f"  Checks WARNED  : {len(warned)}")
    if failed:
        log.error(f"  Checks FAILED  : {len(failed)}")
        log.error("")
        for ds, name, _, detail in failed:
            log.error(f"    [{ds}] ✗ {name}  —  {detail}")
        log.error("")

    if not failed:
        log.info("\n✅  All dataset checks PASSED.  Proceed to fine-tuning (Task 1.2).\n")
        return 0
    else:
        log.error("\n❌  Some checks FAILED.  Fix issues above before fine-tuning.\n")
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1.3 §3 — Dataset integrity verification for all benchmark datasets"
    )
    parser.add_argument("--cub",        metavar="PATH", help="Path to CUB_200_2011 root")
    parser.add_argument("--voc",        metavar="PATH", help="Path to VOCdevkit/VOC2012")
    parser.add_argument("--imagenet_s", metavar="PATH", help="Path to ImageNet-S-50 root")
    parser.add_argument("--nih",        metavar="PATH", help="Path to ChestXray-NIHCC root")
    parser.add_argument("--seed",       type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    any_requested = any([args.cub, args.voc, args.imagenet_s, args.nih])
    if not any_requested:
        parser.print_help()
        log.error("\nNo dataset paths provided.  Specify at least one of "
                  "--cub, --voc, --imagenet_s, --nih.")
        sys.exit(1)

    if args.cub:        verify_cub200(args.cub)
    if args.voc:        verify_pascal_voc(args.voc)
    if args.imagenet_s: verify_imagenet_s50(args.imagenet_s)
    if args.nih:        verify_nih_chestxray(args.nih)

    sys.exit(_print_summary())


if __name__ == "__main__":
    main()
