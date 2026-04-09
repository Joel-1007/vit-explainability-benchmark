#!/usr/bin/env python3
"""
create_cub_val_split.py  —  Task 1.3 §3.2 Step 6
===================================================
CUB-200-2011 does not ship an official validation split.
This script creates a stratified 10% val split from the official
training set (5,994 images) and saves two CSV files:

    cub_train_ids.csv   — ~5,394 image IDs for training
    cub_val_ids.csv     — ~600 image IDs for validation

Stratification is over the 200 class labels so that every class
has proportional representation in both train and val.

The random_state is fixed at 42 for reproducibility — all
benchmark participants must use this exact split.

Usage
-----
    python scripts/create_cub_val_split.py --root /path/to/CUB_200_2011

    # Custom output directory:
    python scripts/create_cub_val_split.py \\
        --root /path/to/CUB_200_2011 \\
        --out_dir /path/to/output

Requirements
------------
    pip install pandas scikit-learn

Output
------
    <out_dir>/cub_train_ids.csv   — columns: image_id
    <out_dir>/cub_val_ids.csv     — columns: image_id

Commit both CSV files to the project repository so every
team member and reviewer uses the identical split.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("create_cub_val_split")


def main(cub_root: str, out_dir: str, val_frac: float = 0.10, seed: int = 42) -> None:
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        log.error(f"Missing dependency: {e}.  pip install pandas scikit-learn")
        sys.exit(1)

    root    = Path(cub_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load official split and class labels
    # -----------------------------------------------------------------------
    split_file  = root / "train_test_split.txt"
    labels_file = root / "image_class_labels.txt"

    for p in (split_file, labels_file):
        if not p.exists():
            log.error(f"Required file not found: {p}")
            log.error("Ensure CUB-200-2011 is fully extracted before running this script.")
            sys.exit(1)

    split_df = pd.read_csv(
        split_file,
        sep=" ", header=None,
        names=["image_id", "is_train"],
    )
    labels_df = pd.read_csv(
        labels_file,
        sep=" ", header=None,
        names=["image_id", "class_id"],
    )

    # Merge and filter to official train set only
    train_df = (
        split_df[split_df["is_train"] == 1]
        .merge(labels_df, on="image_id")
    )

    log.info(f"Official train set size : {len(train_df):,}")
    log.info(f"Number of classes       : {train_df['class_id'].nunique()}")

    expected_total = 5994
    if len(train_df) != expected_total:
        log.warning(
            f"Expected {expected_total} train images, found {len(train_df)}. "
            "Proceeding anyway — verify with verify_datasets.py."
        )

    # -----------------------------------------------------------------------
    # Stratified split
    # -----------------------------------------------------------------------
    train_ids, val_ids = train_test_split(
        train_df["image_id"].values,
        test_size=val_frac,
        stratify=train_df["class_id"].values,
        random_state=seed,
    )

    log.info(f"Train after split : {len(train_ids):,}  ({len(train_ids)/len(train_df)*100:.1f}%)")
    log.info(f"Val after split   : {len(val_ids):>5,}  ({len(val_ids)/len(train_df)*100:.1f}%)")

    # Expected: ~5394 train / ~600 val
    if len(val_ids) < 550 or len(val_ids) > 650:
        log.warning(
            f"Val size {len(val_ids)} is outside the expected range 550–650. "
            "Check that val_frac and random_state are correct."
        )

    # -----------------------------------------------------------------------
    # Verify class coverage
    # -----------------------------------------------------------------------
    val_df        = train_df[train_df["image_id"].isin(val_ids)]
    val_classes   = val_df["class_id"].nunique()
    train_classes = train_df[train_df["image_id"].isin(train_ids)]["class_id"].nunique()

    log.info(f"Classes in train split : {train_classes}")
    log.info(f"Classes in val   split : {val_classes}")

    if val_classes < 200:
        log.warning(
            f"Only {val_classes}/200 classes represented in val split. "
            "This is unexpected for a 10% stratified split — "
            "some very small classes may have been rounded to 0 val images."
        )

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    train_csv = out_dir / "cub_train_ids.csv"
    val_csv   = out_dir / "cub_val_ids.csv"

    pd.DataFrame({"image_id": train_ids}).to_csv(train_csv, index=False)
    pd.DataFrame({"image_id": val_ids}).to_csv(val_csv,   index=False)

    log.info(f"Saved: {train_csv}  ({len(train_ids):,} rows)")
    log.info(f"Saved: {val_csv}    ({len(val_ids):,} rows)")
    log.info("")
    log.info("✅  CUB-200-2011 val split created.")
    log.info("    Commit both CSV files to the repository so all")
    log.info("    benchmark participants use the identical split.")
    log.info("")
    log.info("    Parameters used:")
    log.info(f"      val_frac     = {val_frac}")
    log.info(f"      random_state = {seed}")
    log.info(f"      dataset      = CUB-200-2011")
    log.info(f"      root         = {root}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Task 1.3 §3.2 Step 6 — Create a stratified 10%% val split "
            "from the CUB-200-2011 official training set."
        )
    )
    parser.add_argument(
        "--root",
        required=True,
        metavar="PATH",
        help="Path to the CUB_200_2011 root directory "
             "(must contain train_test_split.txt and image_class_labels.txt).",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        metavar="PATH",
        help="Output directory for cub_train_ids.csv and cub_val_ids.csv "
             "(default: current directory).",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.10,
        help="Fraction of training set to use for validation (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42 — DO NOT CHANGE for benchmark).",
    )
    args = parser.parse_args()
    main(
        cub_root=args.root,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )
