"""
transforms.py — Standardised data augmentation pipeline (Task 1.2 §5.4)

Pixel-space protocol: every transform returns RGB tensors in [0, 1]; the
model wrapper applies its own pre-training mean/std (model_zoo/normalize.py).

Training  : RandomResizedCrop (bicubic) + flip + colour jitter + RandAugment
            (M=9, N=2) + random erasing (p=0.25, "pixel" noise)
Validation: resize shorter side to 256 (bicubic) + centre-crop 224 — the same
            geometry as evaluation (utils/eval_datasets.PairTransform)

Random erasing. timm's "pixel" mode fills the erased box with N(0, 1) noise,
a distribution defined in ImageNet-normalised space. To keep exactly that
augmentation in the pixel-space protocol, ``PixelSpaceRandomErasing`` maps the
image to ImageNet-normalised coordinates, erases with timm, maps back, and
clamps to [0, 1]. The resulting noise is identical for every model, whatever
normalisation the model applies afterwards.

Mixup (α=0.8) is applied at batch level — see mixup.py.

Usage
-----
from training.transforms import build_transforms
train_tf, val_tf = build_transforms()
"""

from __future__ import annotations

import torch
from timm.data import create_transform
from timm.data.random_erasing import RandomErasing
from torchvision import transforms as T

from utils.eval_datasets import INPUT_SIZE, RESIZE_SIZE

# Reference statistics for the erasing-noise distribution only (not model input)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
_IDENTITY_MEAN = (0.0, 0.0, 0.0)
_IDENTITY_STD  = (1.0, 1.0, 1.0)

# RandAugment config string (M=9, N=2, magnitude std=0.5, increasing=True)
RAND_AUGMENT_STR = "rand-m9-n2-mstd0.5-inc1"
RE_PROB = 0.25


class PixelSpaceRandomErasing:
    """timm RandomErasing ("pixel" mode) applied in ImageNet-normalised space."""

    def __init__(self, probability: float = RE_PROB, max_count: int = 1) -> None:
        self.erase = RandomErasing(probability=probability, mode="pixel",
                                   max_count=max_count, device="cpu")
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.mean) / self.std
        z = self.erase(z)
        return (z * self.std + self.mean).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"PixelSpaceRandomErasing(p={self.erase.probability}, mode=pixel)"


def build_transforms(input_size: int = INPUT_SIZE):
    """
    Build training and validation transforms (both return [0, 1] pixels).

    Returns
    -------
    train_transform, val_transform
    """
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment=RAND_AUGMENT_STR,
        re_prob=0.0,                 # erasing added below, in pixel space
        interpolation="bicubic",
        mean=_IDENTITY_MEAN,
        std=_IDENTITY_STD,
    )
    train_transform = T.Compose([train_transform, PixelSpaceRandomErasing(RE_PROB)])

    val_transform = create_transform(
        input_size=input_size,
        is_training=False,
        interpolation="bicubic",
        crop_pct=input_size / RESIZE_SIZE,   # 224/256: same geometry as evaluation
        mean=_IDENTITY_MEAN,
        std=_IDENTITY_STD,
    )
    return train_transform, val_transform
