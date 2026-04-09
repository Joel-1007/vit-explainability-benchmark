"""
optimizer.py — AdamW optimiser + LambdaLR scheduler (Task 1.2 §5.5)

Protocol
--------
Optimiser  : AdamW  β₁=0.9, β₂=0.999, ε=1e-8
Weight decay : 0.05  (bias, LayerNorm params excluded)
LR scaling   : lr = base_lr × batch_size / 256  (linear rule)
Schedule     : linear warmup (5 epochs) → cosine annealing decay

Usage
-----
from training.optimizer import build_optimiser, build_scheduler

opt, effective_lr = build_optimiser(model, base_lr=1e-4, batch_size=256)
sched = build_scheduler(opt, num_epochs=50, warmup_epochs=5)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Parameter grouping
# ---------------------------------------------------------------------------
_NO_DECAY_KEYWORDS = ("bias", "norm", "layernorm", "ln_")


def _group_params(model: nn.Module, weight_decay: float):
    """
    Split parameters into two groups:
      - decay_params   : all parameters that receive weight decay
      - no_decay_params: bias, LayerNorm, and BN parameters

    This is the standard ViT fine-tuning convention (§5.1).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name.lower() for kw in _NO_DECAY_KEYWORDS):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------
def build_optimiser(
    model: nn.Module,
    base_lr: float = 1e-4,
    batch_size: int = 256,
    weight_decay: float = 0.05,
) -> tuple[AdamW, float]:
    """
    Build an AdamW optimiser with linear LR scaling.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimised.
    base_lr : float
        Base learning rate for batch size 256 (default 1e-4).
    batch_size : int
        Actual effective batch size (including gradient accumulation).
    weight_decay : float
        Weight decay coefficient (applied to non-bias/non-norm params only).

    Returns
    -------
    optimiser : AdamW
    effective_lr : float
        The scaled learning rate that was actually set.
    """
    # Linear LR scaling rule: lr = base_lr × batch_size / 256
    effective_lr = base_lr * batch_size / 256

    param_groups = _group_params(model, weight_decay)

    optimiser = AdamW(
        param_groups,
        lr=effective_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,   # group-level override will take precedence
    )

    return optimiser, effective_lr


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def build_scheduler(
    optimiser: AdamW,
    num_epochs: int,
    warmup_epochs: int = 5,
) -> LambdaLR:
    """
    Build a LambdaLR scheduler: linear warmup → cosine annealing.

    Parameters
    ----------
    optimiser : AdamW
        The optimiser to attach the scheduler to.
    num_epochs : int
        Total number of training epochs.
    warmup_epochs : int
        Number of initial epochs for linear warmup (default 5).

    Returns
    -------
    LambdaLR scheduler  (step per epoch)

    LR multiplier at epoch e
    ------------------------
    e  < warmup_epochs  →  (e+1) / warmup_epochs          [ramps 0→1]
    e >= warmup_epochs  →  0.5 × (1 + cos(π × progress))  [cosine decay]
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup from ~0 → 1
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimiser, lr_lambda=lr_lambda)
