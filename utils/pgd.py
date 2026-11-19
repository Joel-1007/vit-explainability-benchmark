# utils/pgd.py
"""
Utility functions for Projected Gradient Descent (PGD) attacks.
Provides a simple PGD implementation used by the adversarial robustness metric.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from utils.tasks import pgd_loss


def pgd_attack(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: int,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    steps: int = 10,
    random_start: bool = True,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
    task: str = "single",
) -> torch.Tensor:
    """Untargeted L∞ PGD on a single image: maximise the loss of ``target``.

    Parameters
    ----------
    model: torch.nn.Module
        Model to attack (in eval mode).
    x: torch.Tensor
        Input image (C, H, W). In [0, 1] pixel space when ``mean``/``std``
        are None — the benchmark protocol (models normalise internally);
        otherwise already normalised as ``(pixel - mean) / std``.
    target: int
        Class whose loss is maximised (usually the true label), i.e. the
        attack pushes the prediction *away* from it: cross-entropy for
        single-label models, binary cross-entropy against "present" for
        multi-label models (utils/tasks.pgd_loss).
    epsilon, alpha: float
        L∞ radius and step size, always in [0, 1] pixel units (0.03 ≈ 8/255).
    steps: int
        Number of gradient steps.
    random_start: bool
        Start from a random point inside the epsilon ball.
    mean, std: per-channel normalisation of ``x``. When given, the attack
        runs in pixel space (so clipping to [0, 1] and epsilon refer to real
        pixel values) and the result is returned normalised again.

    Returns
    -------
    torch.Tensor
        Adversarial image, same shape and space as ``x``.
    """
    model.eval()
    if (mean is None) != (std is None):
        raise ValueError("Pass both mean and std, or neither.")

    if mean is not None:
        m = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        s = torch.as_tensor(std,  dtype=x.dtype, device=x.device).view(-1, 1, 1)
        to_model = lambda p: (p - m) / s          # noqa: E731
        x_pix = (x * s + m).clamp(0.0, 1.0)
    else:
        to_model = lambda p: p                    # noqa: E731
        x_pix = x

    x_pix = x_pix.detach()
    x_adv = x_pix.clone()
    if random_start:
        x_adv = x_adv + (torch.rand_like(x_adv) * 2 - 1) * epsilon
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = pgd_loss(model(to_model(x_adv).unsqueeze(0)), target, task)
            grad, = torch.autograd.grad(loss, x_adv)
        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_pix + epsilon), x_pix - epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return to_model(x_adv.detach())
