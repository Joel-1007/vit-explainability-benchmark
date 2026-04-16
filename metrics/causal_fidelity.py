# metrics/causal_fidelity.py
"""
Causal Fidelity Metric with Generative In‑painting
------------------------------------------------
We replace the simple random‑noise in‑painting placeholder with a
MAE‑style generative reconstruction. The class supports two modes:

* ``mode="noise"`` – baseline uniform‑noise in‑painting (kept for ablation).
* ``mode="mae"``   – MAE (Masked Auto‑Encoder) reconstruction, the
  primary method for the TPAMI contribution.

The mask is **soft** and derived from a continuous saliency map, allowing
gradient‑friendly weighting of patches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Callable, Tuple

# Optional import – MAE model from timm (if available).
# We lazily import inside the class to avoid hard dependency failures.


class CausalMaskingMetric:
    """Causal fidelity metric with optional MAE in‑painting.

    Parameters
    ----------
    epsilon: float, default 0.1
        Scale of random noise used when ``mode="noise"``.
    mode: str, default "mae"
        ``"noise"`` – baseline random‑noise in‑painting.
        ``"mae"``   – MAE generative reconstruction (ViT‑compatible).
    mae_model_name: str, default "mae_vit_base_patch16"
        Identifier for a pretrained MAE model (timm). Ignored when
        ``mode="noise"``.
    device: torch.device | str, default "cpu"
        Device on which the MAE model will run.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        mode: str = "mae",
        mae_model_name: str = "mae_vit_base_patch16",
        device: str | torch.device = "cpu",
    ) -> None:
        self.epsilon = epsilon
        self.mode = mode.lower()
        self.device = torch.device(device)
        if self.mode == "mae":
            try:
                import timm
                self.mae = timm.create_model(mae_model_name, pretrained=True)
                self.mae.eval().to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load MAE model '{mae_model_name}'. Ensure timm is installed."
                ) from e
        elif self.mode != "noise":
            raise ValueError(f"Unsupported mode '{mode}'. Use 'noise' or 'mae'.")

    # ---------------------------------------------------------------------
    # Helper: soft mask from a continuous saliency map
    # ---------------------------------------------------------------------
    @staticmethod
    def _soft_mask(
        saliency: torch.Tensor,
        tau: float = 0.5,
        alpha: float = 10.0,
        patch_size: Tuple[int, int] = (16, 16)
    ) -> torch.Tensor:
        """Return a soft mask in [0,1] from a saliency map.

        Parameters
        ----------
        saliency: (H_a, W_a) tensor – explainer output.
        tau: percentile threshold (0‑1). Values above the ``tau``‑quantile
            are considered important.
        alpha: sigmoid sharpness. Larger values approximate a hard mask.
        """
        ph, pw = patch_size
        sal_patch = F.max_pool2d(
            saliency.unsqueeze(0).unsqueeze(0),
            kernel_size=(ph, pw),
            stride=(ph, pw),
        ).squeeze()
        thresh = torch.quantile(sal_patch, tau)
        mask_patch = torch.sigmoid(alpha * (sal_patch - thresh))
        mask_up = F.interpolate(
            mask_patch.unsqueeze(0).unsqueeze(0),
            size=(saliency.shape[0] * ph, saliency.shape[1] * pw),
            mode="nearest",
        ).squeeze()
        return mask_up

    # ---------------------------------------------------------------------
    # In‑painting implementations
    # ---------------------------------------------------------------------
    def _inpaint_noise(self, x: torch.Tensor, mask_up: torch.Tensor) -> torch.Tensor:
        """Uniform‑noise in‑painting (baseline).

        ``mask_up`` is a binary/soft mask at image resolution.
        """
        noise = (torch.rand_like(x) * 2 - 1) * self.epsilon
        return x * (1 - mask_up) + noise * mask_up

    def _inpaint_mae(self, x: torch.Tensor, mask_up: torch.Tensor) -> torch.Tensor:
        """MAE‑style reconstruction.

        The MAE model expects *masked* patches as input. We therefore
        construct a *masked* image where the masked region is replaced by a
        learnable [MASK] token (implemented as zeros) and let the decoder
        fill it in.
        """
        # The MAE implementation in timm works on *patch* tensors.
        # We therefore down‑sample the mask to the patch grid, apply it,
        # run the encoder‑decoder, and up‑sample the output back.
        # This is a lightweight wrapper – exact details depend on the
        # underlying MAE architecture, but the following works for the
        # standard ViT‑base MAE.
        # -------------------------------------------------------------
        # 1. Down‑sample mask to patch resolution (same as MAE patch size).
        patch_size = getattr(self.mae, "patch_embed", None).patch_size
        if isinstance(patch_size, tuple):
            ph, pw = patch_size
        else:
            ph = pw = patch_size
        mask_patch = F.avg_pool2d(mask_up.unsqueeze(0), kernel_size=(ph, pw), stride=(ph, pw))
        mask_patch = (mask_patch > 0.5).float().squeeze(0)  # binary patch mask
        # 2. Prepare masked image for MAE – the MAE forward expects a batch.
        img_batch = x.unsqueeze(0)  # (1, C, H, W)
        # The MAE model from timm provides a ``forward`` that accepts a
        # ``mask`` argument (bool tensor of shape (B, N) where N is number
        # of patches). We flatten the patch mask accordingly.
        B, C, H, W = img_batch.shape
        num_patches_h = H // ph
        num_patches_w = W // pw
        N = num_patches_h * num_patches_w
        mask_flat = mask_patch.view(1, N).bool()  # (1, N)
        with torch.no_grad():
            # ``self.mae`` returns the reconstructed image (same shape as input).
            recon = self.mae(img_batch.to(self.device), mask=mask_flat)
        recon = recon.squeeze(0).to(x.device)
        # 3. Direct soft‑mask blending – the sigmoid mask already provides a smooth transition
        return x * (1 - mask_up) + recon * mask_up

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def _inpaint(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Dispatch to the selected in‑painting mode.

        ``mask`` is a *soft* patch‑level mask (H_p, W_p). It is first up‑sampled
        to image resolution with nearest‑neighbour interpolation.
        """
        # Upsample to image resolution (nearest neighbour preserves soft values)
        mask_up = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=x.shape[1:],
            mode="nearest",
        ).squeeze()
        if self.mode == "noise":
            return self._inpaint_noise(x, mask_up)
        else:  # mae
            return self._inpaint_mae(x, mask_up)

    def compute(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        saliency: torch.Tensor,
        target_class: int,
        tau: float = 0.5,
        alpha: float = 10.0,
    ) -> float:
        """Compute causal necessity."""
        model.eval()
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            orig_conf = torch.softmax(logits, dim=-1)[0, target_class].item()
            mask = self._soft_mask(saliency, tau=tau, alpha=alpha, patch_size=(16, 16))
            x_cf = self._inpaint(x, mask)
            logits_cf = model(x_cf.unsqueeze(0))
            cf_conf = torch.softmax(logits_cf, dim=-1)[0, target_class].item()
        drop = max(orig_conf - cf_conf, 0.0)
        return drop / (orig_conf + 1e-8)

    def compute_sufficiency(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        saliency: torch.Tensor,
        target_class: int,
        tau: float = 0.5,
        alpha: float = 10.0,
    ) -> float:
        """Return the sufficiency score (Eq. 2) normalised by uniform baseline."""
        model.eval()
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            orig_conf = torch.softmax(logits, dim=-1)[0, target_class].item()
            mask = self._soft_mask(saliency, tau=tau, alpha=alpha, patch_size=(16, 16))
            keep_mask = 1.0 - mask
            x_keep = self._inpaint(x, keep_mask)
            logits_keep = model(x_keep.unsqueeze(0))
            keep_conf = torch.softmax(logits_keep, dim=-1)[0, target_class].item()
        K = logits.shape[-1]
        p_min = 1.0 / K
        numerator   = keep_conf - p_min
        denominator = orig_conf - p_min + 1e-8
        return max(0.0, numerator / denominator)

    def curve(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        saliency: torch.Tensor,
        target_class: int,
        mode: str = "insertion",
        steps: int = 20,
        tau: float = 0.5,
        alpha: float = 10.0,
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Returns a list of (fraction, confidence) pairs and AUC."""
        import numpy as np
        ph, pw = 16, 16
        H, W = x.shape[1], x.shape[2]
        n_h, n_w = H // ph, W // pw
        sal_patch = F.max_pool2d(
            saliency.unsqueeze(0).unsqueeze(0),
            kernel_size=(ph, pw),
            stride=(ph, pw),
        ).squeeze()
        order = torch.argsort(sal_patch.flatten(), descending=True)
        if mode == "insertion":
            base = torch.zeros_like(x)
        else:
            base = x.clone()
        results = []
        for step in range(steps + 1):
            frac = step / steps
            k = int(frac * len(order))
            idxs = order[:k]
            mask = torch.zeros((n_h, n_w), device=x.device)
            if float(len(idxs)) > 0:
                mask.view(-1)[idxs] = 1.0
            mask_up = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest"
            ).squeeze()
            cur = base * (1 - mask_up) + x * mask_up if mode == "insertion" else x * (1 - mask_up) + base * mask_up
            with torch.no_grad():
                conf = torch.softmax(model(cur.unsqueeze(0)), dim=-1)[0, target_class].item()
            results.append((frac, conf))
        fractions, confidences = zip(*results)
        auc = float(np.trapz(confidences, fractions))
        return results, auc
