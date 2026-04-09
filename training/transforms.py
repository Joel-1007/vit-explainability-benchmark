"""
transforms.py — Standardised data augmentation pipeline (Task 1.2 §5.4)

Training  : RandAugment (M=9, N=2) + random erasing + bicubic resize
Validation: Centre-crop + bicubic resize
Both      : ImageNet mean/std normalisation

Mixup (α=0.8) is applied at batch level — see mixup.py.

Usage
-----
from training.transforms import build_transforms
train_tf, val_tf = build_transforms()
"""

from timm.data import create_transform

# ImageNet statistics — used for ALL models in the zoo
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# RandAugment config string (M=9, N=2, magnitude std=0.5, increasing=True)
RAND_AUGMENT_STR = "rand-m9-n2-mstd0.5-inc1"


def build_transforms(input_size: int = 224):
    """
    Build training and validation transforms.

    Parameters
    ----------
    input_size : int
        Square input resolution (default 224 for all zoo models).

    Returns
    -------
    train_transform, val_transform
    """
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment=RAND_AUGMENT_STR,
        re_prob=0.25,        # Random erasing probability
        re_mode="pixel",
        re_count=1,
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    val_transform = create_transform(
        input_size=input_size,
        is_training=False,
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    return train_transform, val_transform
