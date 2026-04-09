"""
loss.py — Loss functions (Task 1.2 §5.7)

Training  : SoftTargetCrossEntropy  (required when Mixup is active —
            Mixup outputs soft/mixed labels, not integer class indices)
Validation : CrossEntropyLoss(label_smoothing=0.1)

NIH ChestX-ray14 special case
------------------------------
NIH ChestX-ray14 is a multi-label, class-imbalanced dataset (14 pathology
classes; 8 have GT bounding boxes for localization evaluation).
build_loss() returns a class-weighted BCEWithLogitsLoss for this dataset.
Class weights must be computed on the training set and passed in.

NOTE: CheXpert was originally planned but replaced by NIH ChestX-ray14
because CheXpert has no spatial GT annotations (Task 1.3 §1.2).

Usage
-----
from training.loss import build_loss

criterion_train, criterion_val = build_loss(dataset="cub200")
criterion_train, criterion_val = build_loss(
    dataset="nih_chestxray", class_weights=weights_tensor
)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy

# Datasets that need special loss handling
# "chexpert" kept as alias for backwards compatibility (replaced by nih_chestxray
# per Task 1.3 §1.2 — CheXpert has no spatial GT for localization metrics).
_MULTILABEL_DATASETS  = {"nih_chestxray", "chexpert"}
_STANDARD_DATASETS    = {"cub200", "pascal_voc", "imagenet", "imagenet_s50"}


def build_loss(
    dataset: str = "cub200",
    label_smoothing: float = 0.1,
    class_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, nn.Module]:
    """
    Return (train_criterion, val_criterion) for a given dataset.

    Parameters
    ----------
    dataset : str
        One of: 'cub200', 'pascal_voc', 'imagenet', 'imagenet_s50',
        'nih_chestxray'.  ('chexpert' accepted as deprecated alias.)
    label_smoothing : float
        Label smoothing for the validation criterion (default 0.1, §5.1).
    class_weights : Tensor | None
        Per-class weights for 'chexpert' class-imbalance correction.
        Shape: (num_classes,).  Ignored for standard datasets.

    Returns
    -------
    train_criterion : nn.Module
        Used inside the training loop (with Mixup-produced soft labels).
    val_criterion : nn.Module
        Used during validation (hard integer labels).
    """
    ds = dataset.lower().replace("-", "_")

    if ds in _MULTILABEL_DATASETS:
        # NIH ChestX-ray14 (or CheXpert alias) — multi-label binary classification
        # with severe class imbalance.  BCEWithLogitsLoss with per-class pos_weight.
        dataset_label = "NIH ChestX-ray14" if "nih" in ds else "CheXpert"
        if class_weights is None:
            print(
                f"[WARNING] No class_weights provided for {dataset_label}. "
                "Consider computing pos_weight from training set label frequencies:\n"
                "  pos_weight[c] = (N - N_pos_c) / N_pos_c"
            )
        train_criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        val_criterion   = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    elif ds in _STANDARD_DATASETS:
        # Standard single-label classification + Mixup compatibility
        train_criterion = SoftTargetCrossEntropy()          # works with soft labels
        val_criterion   = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Supported: {sorted(_MULTILABEL_DATASETS | _STANDARD_DATASETS)}\n"
            "Note: use 'nih_chestxray' for the medical domain dataset "
            "(CheXpert was superseded per Task 1.3 §1.2)."
        )

    return train_criterion, val_criterion
