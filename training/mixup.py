"""
mixup.py — Batch-level Mixup (Task 1.2 §5.4)

Wraps timm's Mixup with the benchmark-standard settings:
    mixup_alpha  = 0.8
    cutmix_alpha = 0.0   (CutMix disabled — Mixup only)
    label_smoothing = 0.1
    
Usage
-----
from training.mixup import build_mixup_fn

mixup_fn = build_mixup_fn(num_classes=200)

for images, labels in train_loader:
    images, labels = mixup_fn(images, labels)
    ...
"""

from timm.data.mixup import Mixup


def build_mixup_fn(num_classes: int) -> Mixup:
    """
    Build a Mixup function configured per Task 1.2 §5.4.

    Parameters
    ----------
    num_classes : int
        Number of downstream classes (required for soft-label generation).

    Returns
    -------
    Mixup
        A callable that takes (images, labels) and returns mixed tensors.
    """
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=0.0,      # Disable CutMix — Mixup only (see §5.4)
        label_smoothing=0.1,
        num_classes=num_classes,
    )
