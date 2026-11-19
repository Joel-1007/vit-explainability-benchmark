"""
mixup.py — Batch-level Mixup (Task 1.2 §5.4), by task type.

Single-label: timm Mixup, mixup_alpha=0.8, CutMix disabled, label_smoothing=0.1
(integer labels → smoothed one-hot → mixed).

Multi-label: ``MultiLabelMixup`` — the same recipe adapted to multi-hot targets:
    λ ~ Beta(0.8, 0.8), one λ per batch (timm "batch" mode), permutation =
    batch reversal (as in timm):
        x̃ = λ·x + (1−λ)·x_flip
        ỹ = λ·y + (1−λ)·y_flip           (soft multi-hot targets in [0, 1])
        w̃ = w · w_flip                    (an entry ignored in either image is
                                            ignored in the mix: its true mixed
                                            target is unknown)
    No label smoothing (see training/loss.py).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from timm.data.mixup import Mixup

from utils.tasks import MULTI

MIXUP_ALPHA = 0.8


class MultiLabelMixup:
    """Mixup for (images, multi-hot targets, entry weights)."""

    def __init__(self, alpha: float = MIXUP_ALPHA, prob: float = 1.0) -> None:
        self.alpha, self.prob = alpha, prob

    def __call__(self, x: torch.Tensor, y: torch.Tensor, w: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.alpha <= 0 or np.random.rand() > self.prob:
            return x, y, w
        lam = float(np.random.beta(self.alpha, self.alpha))
        xf, yf, wf = x.flip(0), y.flip(0), w.flip(0)
        return lam * x + (1 - lam) * xf, lam * y + (1 - lam) * yf, w * wf


def build_mixup_fn(num_classes: int, task: str = "single"):
    """Mixup for the task: timm Mixup (single-label) or MultiLabelMixup."""
    if task == MULTI:
        return MultiLabelMixup(alpha=MIXUP_ALPHA)
    return Mixup(
        mixup_alpha=MIXUP_ALPHA,
        cutmix_alpha=0.0,      # Disable CutMix — Mixup only (see §5.4)
        label_smoothing=0.1,
        num_classes=num_classes,
    )
