"""
fidelity.py — Task 2.1 Fidelity Metrics
=======================================
Implements the fidelity metrics F1–F3 as defined in BENCHMARK.md.

F1  sufficiency           — f(x)_y* - f(x_keep)_y*
F2  comprehensiveness     — f(x)_y* - f(x_drop)_y*
F3  log_odds_drop         — log-odds change upon removing top-k patches
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_2d_batch(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure mask/attribution is shape (B, H, W).
    """
    t = t.float()
    if t.dim() == 2:
        return t.unsqueeze(0)
    elif t.dim() == 3:
        return t
    elif t.dim() == 4 and t.shape[1] == 1:
        return t.squeeze(1)
    else:
        raise ValueError(f"Unexpected attribution map shape: {t.shape}")

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FidelityMetrics:
    """
    Stateless collection of fidelity metrics F1–F3.

    Calculates changes in model confidence after masking input features
    determined to be important by an explanation method.

    Parameters
    ----------
    mask_mode : str
        'zero' to replace masked pixels with 0.
        'mean' to replace masked pixels with the image per-channel mean.
    k_fractions : Sequence[float]
        The fractions of top patches to select for the mask.
    """

    def __init__(
        self,
        mask_mode: str = "mean",
        k_fractions: Sequence[float] = (0.1, 0.2, 0.5),
    ) -> None:
        if mask_mode not in ("zero", "mean"):
            raise ValueError(f"Unknown mask_mode '{mask_mode}'. Must be 'zero' or 'mean'.")
        self.mask_mode = mask_mode
        self.k_fractions = tuple(k_fractions)

    def _generate_mask(self, att_map: torch.Tensor, k_frac: float) -> torch.Tensor:
        """
        Generate a binary mask keeping the top `k_frac` fraction of att_map pixels.

        Parameters
        ----------
        att_map : (B, H_a, W_a) raw attribution map.
        k_frac : float in (0, 1].

        Returns
        -------
        mask : (B, 1, H_a, W_a) boolean mask (1.0 for keep, 0.0 for drop).
        """
        att = _to_2d_batch(att_map)
        B, H_a, W_a = att.shape
        N = H_a * W_a
        k = max(1, int(N * k_frac))

        flat_att = att.view(B, N)
        # Find the threshold value for top k for each batch item
        topk_vals = torch.topk(flat_att, k, dim=-1).values
        thresh = topk_vals[:, -1].view(B, 1, 1)

        mask = (att >= thresh).unsqueeze(1).float()
        return mask

    def _apply_mask(self, x: torch.Tensor, mask_small: torch.Tensor, keep: bool = True) -> torch.Tensor:
        """
        Mask out regions of `x`.

        Parameters
        ----------
        x : (B, C, H, W)
        mask_small : (B, 1, H_a, W_a)
        keep : bool, if True retains the region where mask_small==1.
               If False, drops it.

        Returns
        -------
        masked_x : (B, C, H, W)
        """
        # Interpolate mask to original image size using nearest to preserve patches
        mask = F.interpolate(mask_small, size=(x.shape[2], x.shape[3]), mode="nearest")

        if not keep:
            mask = 1.0 - mask

        if self.mask_mode == "zero":
            baseline = torch.zeros_like(x)
        elif self.mask_mode == "mean":
            # Image per-channel mean (B, C, 1, 1)
            baseline = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

        return x * mask + baseline * (1.0 - mask)

    def _get_probs(self, model: torch.nn.Module, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities for target classes."""
        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        batch_idx = torch.arange(x.shape[0], device=x.device)
        return probs[batch_idx, targets]

    @torch.no_grad()
    def compute_all(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        targets: torch.Tensor,
        att_map: torch.Tensor,
        k_fractions: Sequence[float] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute F1, F2, and F3 in a single pass.

        Parameters
        ----------
        model : torch.nn.Module
        x : (B, C, H, W) input images
        targets : (B,) target class indices
        att_map : (B, H_a, W_a) raw attribution maps
        k_fractions : local override for self.k_fractions

        Returns
        -------
        results : Dict[str, torch.Tensor] containing (B,) tensors for each metric
            e.g., 'sufficiency@0.10': tensor([0.05, 0.12, ...])
        """
        model.eval()
        k_fracs = k_fractions if k_fractions is not None else self.k_fractions

        # Original probabilities
        p_orig = self._get_probs(model, x, targets)

        results = {}
        for k_frac in k_fracs:
            mask = self._generate_mask(att_map, k_frac)

            x_keep = self._apply_mask(x, mask, keep=True)
            x_drop = self._apply_mask(x, mask, keep=False)

            p_keep = self._get_probs(model, x_keep, targets)
            p_drop = self._get_probs(model, x_drop, targets)

            # F1 - Sufficiency: f(x) - f(x_keep)
            results[f"sufficiency@{k_frac:.2f}"] = (p_orig - p_keep).cpu()

            # F2 - Comprehensiveness: f(x) - f(x_drop)
            results[f"comprehensiveness@{k_frac:.2f}"] = (p_orig - p_drop).cpu()

            # F3 - Log Odds Drop
            eps = 1e-7
            p_orig_clamped = p_orig.clamp(min=eps, max=1.0 - eps)
            p_drop_clamped = p_drop.clamp(min=eps, max=1.0 - eps)

            log_odds_orig = torch.log(p_orig_clamped / (1.0 - p_orig_clamped))
            log_odds_drop = torch.log(p_drop_clamped / (1.0 - p_drop_clamped))

            results[f"log_odds_drop@{k_frac:.2f}"] = (log_odds_orig - log_odds_drop).cpu()

        return results

    @torch.no_grad()
    def sufficiency(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        targets: torch.Tensor,
        att_map: torch.Tensor,
        k_frac: float,
    ) -> torch.Tensor:
        """
        Standalone F1 Metric.
        Returns tensor of shape (B,).
        """
        model.eval()
        p_orig = self._get_probs(model, x, targets)
        mask = self._generate_mask(att_map, k_frac)
        x_keep = self._apply_mask(x, mask, keep=True)
        p_keep = self._get_probs(model, x_keep, targets)
        return (p_orig - p_keep).cpu()

    @torch.no_grad()
    def comprehensiveness(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        targets: torch.Tensor,
        att_map: torch.Tensor,
        k_frac: float,
    ) -> torch.Tensor:
        """
        Standalone F2 Metric.
        Returns tensor of shape (B,).
        """
        model.eval()
        p_orig = self._get_probs(model, x, targets)
        mask = self._generate_mask(att_map, k_frac)
        x_drop = self._apply_mask(x, mask, keep=False)
        p_drop = self._get_probs(model, x_drop, targets)
        return (p_orig - p_drop).cpu()

    @torch.no_grad()
    def log_odds_drop(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        targets: torch.Tensor,
        att_map: torch.Tensor,
        k_frac: float,
    ) -> torch.Tensor:
        """
        Standalone F3 Metric.
        Returns tensor of shape (B,).
        """
        model.eval()
        p_orig = self._get_probs(model, x, targets)
        mask = self._generate_mask(att_map, k_frac)
        x_drop = self._apply_mask(x, mask, keep=False)
        p_drop = self._get_probs(model, x_drop, targets)

        eps = 1e-7
        p_orig_clamped = p_orig.clamp(min=eps, max=1.0 - eps)
        p_drop_clamped = p_drop.clamp(min=eps, max=1.0 - eps)

        log_odds_orig = torch.log(p_orig_clamped / (1.0 - p_orig_clamped))
        log_odds_drop = torch.log(p_drop_clamped / (1.0 - p_drop_clamped))

        return (log_odds_orig - log_odds_drop).cpu()
