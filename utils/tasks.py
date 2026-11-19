"""
tasks.py — task formulation per dataset, and the ONE definition of a class score.

Task types
----------
single-label (softmax): CUB-200-2011, ImageNet-S-50
    the model outputs a categorical distribution; score s_c = softmax(z)_c;
    a prediction is correct iff argmax(z) = y.
multi-label (sigmoid) : PASCAL VOC 2012, NIH ChestX-ray14
    each class is an independent binary decision; score s_c = sigmoid(z_c);
    an (image, class c) evaluation unit is *detected* iff s_c ≥ t_c, where t_c
    is a per-class operating threshold estimated on the VALIDATION split
    (F1-maximising), frozen and stored with the checkpoint.

The thresholds are detection operating points, not probability calibration:
nothing here claims that s_c is a calibrated probability.

Every consumer (fidelity F1–F3, RISE, LIME, PGD, correctness for L4) calls
``class_score`` / ``class_log_odds`` so the formulation cannot drift between
components. Gradient-/attention-based explainers (GradCAM, Chefer LRP, IG)
use the class logit z_c, which is the class score's monotone pre-activation
under both formulations and needs no change.

The task type travels with the loaded model: ``set_task(model, task, thresholds)``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

SINGLE, MULTI = "single", "multi"

TASK_TYPE = {
    "cub200": SINGLE,
    "imagenet_s50": SINGLE,
    "pascal_voc": MULTI,
    "nih_chestxray": MULTI,
}


def set_task(model: torch.nn.Module, task: str,
             thresholds: Optional[Sequence[float]] = None,
             threshold_source: str = "none") -> torch.nn.Module:
    """Attach the task formulation (and frozen thresholds) to a model."""
    if task not in (SINGLE, MULTI):
        raise ValueError(f"task must be '{SINGLE}' or '{MULTI}', got {task!r}")
    model.task_type = task
    model.decision_thresholds = (None if thresholds is None
                                 else torch.as_tensor(thresholds, dtype=torch.float32))
    model.threshold_source = threshold_source
    return model


def task_of(model: torch.nn.Module) -> str:
    return getattr(model, "task_type", SINGLE)


def class_scores(logits: torch.Tensor, task: str) -> torch.Tensor:
    """(B, C) logits → (B, C) class scores (softmax or sigmoid)."""
    return torch.sigmoid(logits) if task == MULTI else F.softmax(logits, dim=-1)


def class_score(logits: torch.Tensor, targets, task: str) -> torch.Tensor:
    """Score of the target class per row. ``targets``: int or (B,) tensor."""
    s = class_scores(logits, task)
    if isinstance(targets, int):
        return s[:, targets]
    return s[torch.arange(s.shape[0], device=s.device), targets]


def class_log_odds(logits: torch.Tensor, targets, task: str, eps: float = 1e-7) -> torch.Tensor:
    """
    log(s / (1 − s)) of the target class. For sigmoid this is exactly the
    logit z_c (computed directly, no clamping); for softmax it is computed
    from the clamped probability.
    """
    if task == MULTI:
        if isinstance(targets, int):
            return logits[:, targets]
        return logits[torch.arange(logits.shape[0], device=logits.device), targets]
    p = class_score(logits, targets, task).clamp(eps, 1 - eps)
    return torch.log(p / (1 - p))


def is_correct(logits: torch.Tensor, targets: torch.Tensor, task: str,
               thresholds: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    single: argmax(z) == y.  multi: s_y ≥ t_y (needs thresholds; raises otherwise).
    Returns a (B,) bool tensor.
    """
    if task == SINGLE:
        return logits.argmax(dim=1) == targets
    if thresholds is None:
        raise ValueError("multi-label correctness needs per-class thresholds")
    t = thresholds.to(logits.device)[targets]
    return class_score(logits, targets, task) >= t


def pgd_loss(logits: torch.Tensor, target: int, task: str) -> torch.Tensor:
    """Loss the untargeted attack maximises: CE (single) / BCE vs label 1 (multi)."""
    if task == MULTI:
        z = logits[:, target]
        return F.binary_cross_entropy_with_logits(z, torch.ones_like(z))
    return F.cross_entropy(logits, torch.full((logits.shape[0],), target, device=logits.device))
