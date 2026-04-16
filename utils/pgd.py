# utils/pgd.py
"""
Utility functions for Projected Gradient Descent (PGD) attacks.
Provides a simple PGD implementation used by the adversarial robustness metric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: int,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    steps: int = 10,
    random_start: bool = True,
) -> torch.Tensor:
    """Perform a PGD attack on a single image.

    Parameters
    ----------
    model: torch.nn.Module
        Model to attack (in eval mode).
    x: torch.Tensor
        Input image tensor of shape (C, H, W) in [0, 1].
    target: int
        Target class index for which to maximize loss.
    epsilon: float, default 0.03
        Maximum L-infinity perturbation.
    alpha: float, default 0.01
        Step size.
    steps: int, default 10
        Number of gradient steps.
    random_start: bool, default True
        If True, start from a random point within the epsilon ball.

    Returns
    -------
    torch.Tensor
        Adversarially perturbed image of same shape as ``x``.
    """
    model.eval()
    x_adv = x.clone().detach()
    if random_start:
        x_adv = x_adv + (torch.rand_like(x_adv) * 2 - 1) * epsilon
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        logits = model(x_adv.unsqueeze(0))
        loss = loss_fn(logits, torch.tensor([target], device=logits.device))
        loss.backward()
        grad = x_adv.grad.sign()
        x_adv = x_adv + alpha * grad
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        x_adv = x_adv.detach()
        x_adv.requires_grad = True
    return x_adv.detach()
