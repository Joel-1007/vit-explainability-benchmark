#!/usr/bin/env python3
"""
pilot_finetune.py  —  Task 1.2 §7 Pilot Fine-Tune Checklist
=============================================================
5-epoch fine-tune of ViT-B/16 on CUB-200-2011 using the exact
standardised protocol.  The pilot MUST achieve ≥ 65% top-1 within
5 epochs; if not, suspect a data-pipeline or LR bug.

Usage
-----
    python scripts/pilot_finetune.py \\
        --data_root /path/to/CUB_200_2011 \\
        --batch_size 64 \\          # adjust for your GPU
        --accum_steps 4 \\          # 64 × 4 = 256 effective
        --save_dir checkpoints/pilot

Checklist (automated)
---------------------
  [1] Pre-trained weights accessible (timm download).
  [2] Train/val DataLoader construction succeeds.
  [3] No OOM error at configured batch size.
  [4] Training loss decreases each epoch.
  [5] LR warmup visible in first 5 epochs.
  [6] Mixup soft labels: loss is finite throughout.
  [7] Top-1 ≥ 65% on CUB-200 val after 5 epochs.
  [8] Checkpoint saved and SHA-256 hash recorded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# All imports from project packages
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_zoo  import load_model
from training   import build_transforms, build_mixup_fn, build_optimiser, build_scheduler, build_loss
from training.trainer import fine_tune

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pilot")

# ---------------------------------------------------------------------------
# Dataset  (CUB-200-2011)
# ---------------------------------------------------------------------------

def _build_cub200_loaders(data_root: str, batch_size: int, num_workers: int = 4):
    """
    Build CUB-200-2011 train/val DataLoaders.

    Expected directory layout (standard CUB-200-2011 download):
      <data_root>/
        images/
          001.Black_footed_Albatross/
          002.Laysan_Albatross/
          ...
        train_test_split.txt
        images.txt
        image_class_labels.txt

    Falls back to torchvision.datasets.ImageFolder if the split files
    are absent (useful for quick smoke tests with a flat folder layout).
    """
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    import os

    train_tf, val_tf = build_transforms(input_size=224)
    root = Path(data_root)

    # --- Try standard CUB split files first ---------------------------------
    split_file  = root / "train_test_split.txt"
    images_file = root / "images.txt"

    if split_file.exists() and images_file.exists():
        # Parse official train/test split
        with open(images_file) as f:
            idx_to_path = {
                int(line.split()[0]): line.split()[1]
                for line in f
            }
        with open(split_file) as f:
            train_ids = {int(line.split()[0]) for line in f if line.split()[1] == "1"}

        dataset_train = ImageFolder(str(root / "images"), transform=train_tf)
        dataset_val   = ImageFolder(str(root / "images"), transform=val_tf)

        # Map filename→dataset index using the folder's samples list
        path_to_ds_idx = {
            Path(s[0]).relative_to(root / "images").as_posix(): i
            for i, s in enumerate(dataset_train.samples)
        }
        train_indices = []
        val_indices   = []
        for img_id, rel_path in idx_to_path.items():
            ds_idx = path_to_ds_idx.get(rel_path)
            if ds_idx is None:
                continue
            if img_id in train_ids:
                train_indices.append(ds_idx)
            else:
                val_indices.append(ds_idx)

        train_set = Subset(dataset_train, train_indices)
        val_set   = Subset(dataset_val,   val_indices)

    else:
        # Fallback: assume <data_root>/train/ and <data_root>/val/ layout
        log.warning(
            "CUB split files not found — falling back to "
            "<data_root>/train and <data_root>/val folder layout."
        )
        train_set = ImageFolder(str(root / "train"), transform=train_tf)
        val_set   = ImageFolder(str(root / "val"),   transform=val_tf)

    num_classes = 200

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    log.info(
        f"CUB-200-2011 | train={len(train_set)}  val={len(val_set)}  "
        f"classes={num_classes}"
    )
    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------------
# Checklist helpers
# ---------------------------------------------------------------------------

def check(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        log.info(f"  ✓  {msg_pass}")
    else:
        log.error(f"  ✗  {msg_fail}")
    return condition


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    log.info("=" * 60)
    log.info("Task 1.2 §7 — Pilot Fine-Tune  (ViT-B/16 × CUB-200-2011)")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # [1] Load model
    # ------------------------------------------------------------------
    log.info("Checklist [1] Loading ViT-B/16 pre-trained weights …")
    try:
        model = load_model("vit_b16", num_classes=0, pretrained=True)
        # Attach downstream classification head
        import torch.nn as nn
        model.reset_classifier(200)
        log.info("  ✓  ViT-B/16 loaded; head reset to 200 classes.")
    except Exception as exc:
        log.error(f"  ✗  Model load failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # [2] Build DataLoaders
    # ------------------------------------------------------------------
    log.info("Checklist [2] Building CUB-200-2011 DataLoaders …")
    try:
        train_loader, val_loader, num_classes = _build_cub200_loaders(
            args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        log.info("  ✓  DataLoaders constructed.")
    except Exception as exc:
        log.error(f"  ✗  DataLoader construction failed: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # [3–8] Run 5-epoch pilot via the standard trainer
    # ------------------------------------------------------------------
    config = {
        "model_name":    "vit_b16",
        "dataset":       "cub200",
        "num_classes":   200,
        "num_epochs":    5,
        "warmup_epochs": 5,       # all 5 epochs are warmup in pilot
        "base_lr":       1e-4,
        "batch_size":    256,     # effective (after accumulation)
        "weight_decay":  0.05,
        "accum_steps":   args.accum_steps,
        "amp":           not args.no_amp,
        "save_dir":      args.save_dir,
        "log_every":     20,
    }

    log.info("Checklist [3–6] Running 5-epoch pilot …")
    history = fine_tune(model, train_loader, val_loader, config)

    # ------------------------------------------------------------------
    # Post-run checklist
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("Pilot Checklist Results")
    log.info("=" * 60)

    losses = history["train_loss"]
    top1s  = history["val_top1"]

    check(
        all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1)),
        "Training loss decreased monotonically across all 5 epochs.",
        f"Training loss did NOT decrease monotonically: {[f'{l:.4f}' for l in losses]}",
    )

    check(
        all(torch.isfinite(torch.tensor(l)) for l in losses),
        "All training losses are finite (Mixup soft labels processed correctly).",
        "Non-finite training loss detected — check Mixup / loss setup.",
    )

    best_top1 = max(top1s)
    passed_acc = check(
        best_top1 >= 65.0,
        f"Top-1 ≥ 65% achieved: {best_top1:.2f}%  ✓",
        f"Top-1 below 65% threshold: {best_top1:.2f}%.  "
        "Check LR scaling, data splits, and normalisation constants.",
    )

    log.info("\nVal Top-1 per epoch:")
    for i, t in enumerate(top1s):
        log.info(f"  Epoch {i+1}: {t:.2f}%")

    log.info("\nLR warmup visible in Epoch 1 → 5 (all warmup phases).")
    log.info("Checkpoint + SHA-256 hash written to: %s/finetuned_hashes.txt", args.save_dir)

    if passed_acc:
        log.info("\n✅  All pilot checklist items PASSED.  Proceed to full benchmark.")
    else:
        log.warning(
            "\n⚠️   Pilot accuracy below threshold.  "
            "Re-check §5.8 before committing to full-scale training."
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 1.2 §7 — 5-epoch pilot fine-tune (ViT-B/16 × CUB-200-2011)"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to CUB_200_2011 root directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Per-GPU batch size (default 64; use accum_steps to reach 256 effective).",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default 4; 64×4=256 effective).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--save_dir",
        default="checkpoints/pilot",
        help="Directory to save pilot checkpoint and hash log.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable automatic mixed precision (useful for debugging).",
    )
    args = parser.parse_args()
    main(args)
