"""
normalize.py — model-specific input normalisation, applied INSIDE each wrapper.

Protocol: every model receives the same [0, 1] RGB pixel tensor (identical
geometry, identical perturbations); each wrapper then applies the mean/std its
backbone was pre-trained with (``model_zoo/specs.py``). Normalisation is part
of the model, not of the experiment, so perturbation-based explainers and
metrics (RISE, LIME, IG, fidelity baselines, R1 noise, PGD) are defined in
the same pixel space for every model.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PixelNormalize(nn.Module):
    """(x − mean) / std per channel; buffers are not saved in state_dict."""

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    def extra_repr(self) -> str:
        return f"mean={tuple(self.mean.flatten().tolist())}, std={tuple(self.std.flatten().tolist())}"
