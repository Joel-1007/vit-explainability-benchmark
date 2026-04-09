"""
Training sub-package — Task 1.2 §5.5
"""

from .transforms import build_transforms
from .mixup       import build_mixup_fn
from .optimizer   import build_optimiser, build_scheduler
from .loss        import build_loss

__all__ = [
    "build_transforms",
    "build_mixup_fn",
    "build_optimiser",
    "build_scheduler",
    "build_loss",
]
