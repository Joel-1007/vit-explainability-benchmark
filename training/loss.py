"""
loss.py — Loss functions (Task 1.2 §5.7), by task type (utils/tasks.py).

Single-label (CUB-200, ImageNet-S-50)
    train : SoftTargetCrossEntropy (Mixup + label smoothing 0.1 produce soft targets)
    val   : CrossEntropyLoss(label_smoothing=0.1)

Multi-label (PASCAL VOC, NIH ChestX-ray14)
    train/val : MaskedBCEWithLogits — mean over valid (image, class) entries,
                with per-entry weights w ∈ [0, 1] (0 = ignored, e.g. VOC
                difficult-only classes; fractional after Mixup is not used —
                see training/mixup.py) and optional per-class pos_weight.
    No label smoothing: smoothing positives in sparse, imbalanced multi-label
    targets shifts every class's operating point.
    pos_weight: NIH uses N_neg/N_pos per class from the training split
    (configs/nih_chestxray.yaml: class_weighted_loss); VOC uses none.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy

from utils.tasks import MULTI, SINGLE


class MaskedBCEWithLogits(nn.Module):
    """sum(w · BCE(z, y; pos_weight)) / sum(w)."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.register_buffer("pos_weight", None if pos_weight is None else pos_weight.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none", pos_weight=self.pos_weight)
        if weights is None:
            return loss.mean()
        return (loss * weights).sum() / weights.sum().clamp(min=1.0)


def build_loss(task: str = SINGLE, label_smoothing: float = 0.1,
               pos_weight: Optional[torch.Tensor] = None):
    """Return (train_criterion, val_criterion) for a task type."""
    if task == MULTI:
        crit = MaskedBCEWithLogits(pos_weight)
        return crit, crit
    if task == SINGLE:
        return SoftTargetCrossEntropy(), nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    raise ValueError(f"Unknown task {task!r}")
