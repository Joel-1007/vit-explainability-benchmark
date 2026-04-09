"""
localization.py  —  Task 2.2 Localization Metrics  (Task 2.2 §2)
=================================================================
Implements the four localization metrics L1–L4 defined in Task 2.2 §1.

L1  multi_threshold_iou   — Mean IoU at τ ∈ {0.25, 0.50, 0.75}
L2  pointing_game         — Peak-attribution point inside GT mask
L3  egt                   — Energy-Ground-Truth (softmax-normalised)
L4  calibration_gap       — EGT(correct) − EGT(incorrect)

Design principles
-----------------
• All methods accept attribution maps of any spatial resolution and
  internally align them to the GT mask resolution via bilinear
  interpolation before metric computation.
• All inputs are torch.Tensor on any device; outputs are Python float
  (for L1), float (L2, L3), or float (L4).
• No global state — LocalizationMetrics is stateless; results are
  aggregated by the caller or BenchmarkRunner.
• Code style, import order, and docstring format match training/optimizer.py
  and all other Phase 1 modules.

Usage
-----
from metrics.localization import LocalizationMetrics

lm = LocalizationMetrics()

# L1
iou_dict = lm.multi_threshold_iou(att_map, gt_mask)
# → {'iou@0.25': 0.74, 'iou@0.50': 0.61, 'iou@0.75': 0.42, 'miou': 0.59}

# L2
pg = lm.pointing_game(att_map, gt_mask)         # → 1.0 or 0.0

# L3
e = lm.egt(att_map, gt_mask)                    # → float in [0, 1]

# L4
gap = lm.calibration_gap(atts_correct,
                          atts_incorrect)         # → float (can be negative)
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75)
_EPS = 1e-8   # numerical stability floor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """
    Squeeze a tensor to shape (H, W).

    Accepts:
      (H, W), (1, H, W), (1, 1, H, W), (B, 1, H, W) with B=1
    """
    t = t.float()
    while t.dim() > 2:
        if t.shape[0] == 1:
            t = t.squeeze(0)
        else:
            raise ValueError(
                f"Cannot reduce tensor of shape {tuple(t.shape)} to 2D — "
                "batch dimension > 1.  Pass one sample at a time."
            )
    return t


def _align(att: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly resize `att` (H_a × W_a) to match `ref` (H_r × W_r).

    If they already match, returns `att` unchanged (no copy).
    """
    if att.shape == ref.shape:
        return att
    return F.interpolate(
        att.unsqueeze(0).unsqueeze(0),   # (1, 1, H_a, W_a)
        size=ref.shape,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)              # (H_r, W_r)


def _normalise_softmax(att: torch.Tensor) -> torch.Tensor:
    """
    Flatten → softmax → reshape.

    Produces a probability distribution over all spatial positions.
    This is the normalisation scheme required by L3 (EGT) as specified
    in Task 2.2 §1.3.
    """
    flat = att.flatten()
    return flat.softmax(dim=0).reshape(att.shape)


def _binary_gt(gt: torch.Tensor) -> torch.Tensor:
    """
    Return a strict binary mask from `gt`.

    Handles:
      • Float masks in [0, 1]  — thresholded at 0.5
      • Integer masks          — any non-zero pixel is foreground
      • VOC void label (255)   — treated as background (excluded)
    """
    gt = _to_2d(gt)
    if gt.dtype == torch.float32 or gt.dtype == torch.float64:
        return (gt >= 0.5).bool()
    # integer mask: void=255 in VOC → treat as background
    valid = (gt > 0) & (gt != 255)
    return valid.bool()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LocalizationMetrics:
    """
    Stateless collection of localization metrics L1–L4.

    All public methods follow the same interface contract:

      att_map  : torch.Tensor, shape (H, W) or broadcastable to it.
                 Raw (un-normalised) attribution values — normalisation
                 is applied internally as required by each metric.
      gt_mask  : torch.Tensor, same or different spatial resolution.
                 Binary or integer segmentation mask / bounding-box mask.

    The class handles patch-resolution differences internally via bilinear
    upsampling from the ViT patch grid to the GT mask resolution.

    Parameters
    ----------
    thresholds : sequence of float
        Binarisation thresholds for L1 (default: 0.25, 0.50, 0.75).
    seed : int | None
        Random seed for tie-breaking in pointing_game (L2).
        Set to None for non-deterministic tie-breaking.
    """

    def __init__(
        self,
        thresholds: Sequence[float] = _DEFAULT_THRESHOLDS,
        seed: int | None = 42,
    ) -> None:
        self.thresholds = tuple(sorted(thresholds))
        self._rng       = random.Random(seed)

    # ------------------------------------------------------------------
    # L1 — Multi-Threshold IoU
    # ------------------------------------------------------------------

    def multi_threshold_iou(
        self,
        att_map: torch.Tensor,
        gt_mask: torch.Tensor,
        thresholds: Sequence[float] | None = None,
    ) -> Dict[str, float]:
        """
        L1 — Mean Intersection-over-Union at multiple binarisation thresholds.

        Formal definition (Task 2.2 §1.1)
        -----------------------------------
        For threshold τ, binarise the normalised attribution map:

            M̂_τ  =  {p : ẽ_p ≥ τ}

        where ẽ is the min-max normalised attribution map in [0, 1].

        IoU(M̂_τ, M_GT)  =  |M̂_τ ∩ M_GT|
                            ─────────────────
                            |M̂_τ ∪ M_GT|

        mIoU = mean over all τ in `thresholds`.

        Parameters
        ----------
        att_map    : (H_a, W_a) raw attribution map.
        gt_mask    : (H_r, W_r) ground-truth binary/integer mask.
        thresholds : override instance thresholds for this call.

        Returns
        -------
        dict with keys:
          'iou@{τ}'  for each threshold (e.g. 'iou@0.25')
          'miou'     — mean across all thresholds
        """
        τ_list = thresholds if thresholds is not None else self.thresholds

        att = _to_2d(att_map)
        gt  = _binary_gt(gt_mask)
        att = _align(att, gt)

        # Min-max normalise to [0, 1]
        a_min = att.min()
        a_max = att.max()
        if (a_max - a_min) < _EPS:
            # Constant map → can never overlap GT except trivially at τ=0
            result = {f"iou@{τ:.2f}": 0.0 for τ in τ_list}
            result["miou"] = 0.0
            return result

        norm = (att - a_min) / (a_max - a_min + _EPS)

        ious: list[float] = []
        result: Dict[str, float] = {}

        for τ in τ_list:
            pred = norm >= τ                      # boolean predicted mask
            inter = (pred & gt).sum().item()
            union = (pred | gt).sum().item()
            iou   = inter / (union + _EPS) if union > 0 else 0.0
            result[f"iou@{τ:.2f}"] = float(iou)
            ious.append(iou)

        result["miou"] = float(sum(ious) / len(ious)) if ious else 0.0
        return result

    # ------------------------------------------------------------------
    # L2 — Pointing Game
    # ------------------------------------------------------------------

    def pointing_game(
        self,
        att_map: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> float:
        """
        L2 — Pointing Game.

        Formal definition (Task 2.2 §1.2)
        -----------------------------------
        Let p* = argmax_p e_p  where e_p is the raw attribution at pixel p.

            PG = 1[ p* ∈ M_GT ]

        Tie-breaking: when multiple pixels share the maximum attribution
        value, one is drawn uniformly at random (seeded for reproducibility).
        This prevents the pathological case where a constant attribution map
        scores 1.0 by always picking position (0, 0) which may lie inside
        the GT mask.

        Parameters
        ----------
        att_map : (H_a, W_a)
        gt_mask : (H_r, W_r)

        Returns
        -------
        float — 1.0 (hit) or 0.0 (miss).
        """
        att = _to_2d(att_map)
        gt  = _binary_gt(gt_mask)
        att = _align(att, gt)

        max_val = att.max()
        # All positions where att == max (may be multiple ties)
        tied_positions = (att == max_val).nonzero(as_tuple=False)   # (K, 2)

        if tied_positions.shape[0] == 0:
            return 0.0

        if tied_positions.shape[0] == 1:
            row, col = tied_positions[0].tolist()
        else:
            # Tie-break: uniform random draw among tied positions
            idx      = self._rng.randint(0, tied_positions.shape[0] - 1)
            row, col = tied_positions[idx].tolist()

        hit = bool(gt[row, col].item())
        return 1.0 if hit else 0.0

    # ------------------------------------------------------------------
    # L3 — Energy-Ground-Truth (EGT)
    # ------------------------------------------------------------------

    def egt(
        self,
        att_map: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> float:
        """
        L3 — Energy-Ground-Truth (EGT).

        Formal definition (Task 2.2 §1.3)
        -----------------------------------
        Let ẽ_p = softmax(e)_p  be the spatially normalised attribution:

            ẽ_p = exp(e_p) / Σ_q exp(e_q)

        EGT  =  Σ_{p ∈ M_GT} ẽ_p

        Interpretation: the fraction of total attribution mass (under the
        softmax distribution) that falls inside the GT region.  EGT ∈ [0, 1].

        Notes
        -----
        Softmax normalisation is required rather than sum-normalisation to
        ensure that negative attribution values (e.g. from GradCAM) are
        handled correctly and that the distribution always sums to 1.

        Parameters
        ----------
        att_map : (H_a, W_a) — raw attribution values.
        gt_mask : (H_r, W_r)

        Returns
        -------
        float in [0, 1].
        """
        att = _to_2d(att_map)
        gt  = _binary_gt(gt_mask)
        att = _align(att, gt)

        norm = _normalise_softmax(att)   # (H, W), sums to 1.0
        return float(norm[gt].sum().item())

    # ------------------------------------------------------------------
    # L4 — Calibration Gap
    # ------------------------------------------------------------------

    def calibration_gap(
        self,
        atts_correct:   List[torch.Tensor],
        atts_incorrect: List[torch.Tensor],
        gt_masks:       List[torch.Tensor] | None = None,
    ) -> float:
        """
        L4 — Calibration Gap.

        Formal definition (Task 2.2 §1.4)
        -----------------------------------
        CalibGap  =  E[EGT | correct prediction]
                   − E[EGT | incorrect prediction]

        A positive CalibGap means the model allocates more attribution
        mass to the GT region when it classifies correctly — a necessary
        (though not sufficient) condition for faithful explanations.
        A negative or near-zero CalibGap indicates the explanation is
        insensitive to model correctness, which is a serious failure mode.

        Parameters
        ----------
        atts_correct   : list of (H, W) attribution maps for correctly
                         classified samples.
        atts_incorrect : list of (H, W) attribution maps for incorrectly
                         classified samples.
        gt_masks       : list of corresponding GT masks, one per sample
                         across both lists (correct first, then incorrect).
                         If None, all EGT values must already be computed
                         externally and passed as 1-element tensors.

        Returns
        -------
        float — CalibGap ∈ (−1, 1).  Positive is desirable.

        Raises
        ------
        ValueError  if either list is empty.
        """
        if not atts_correct:
            raise ValueError(
                "atts_correct is empty — need at least one correctly "
                "classified sample to compute CalibGap."
            )
        if not atts_incorrect:
            raise ValueError(
                "atts_incorrect is empty — need at least one incorrectly "
                "classified sample to compute CalibGap.  "
                "If all predictions are correct, CalibGap is undefined."
            )

        n_correct   = len(atts_correct)
        n_incorrect = len(atts_incorrect)

        if gt_masks is not None:
            # gt_masks: correct samples first, then incorrect
            assert len(gt_masks) == n_correct + n_incorrect, (
                f"gt_masks length ({len(gt_masks)}) must equal "
                f"len(atts_correct) + len(atts_incorrect) "
                f"({n_correct} + {n_incorrect} = {n_correct + n_incorrect})."
            )
            masks_c = gt_masks[:n_correct]
            masks_i = gt_masks[n_correct:]

            egt_c = [self.egt(a, m) for a, m in zip(atts_correct,   masks_c)]
            egt_i = [self.egt(a, m) for a, m in zip(atts_incorrect, masks_i)]
        else:
            # Caller pre-computed: each "att" is a scalar EGT value tensor
            egt_c = [float(a.item()) for a in atts_correct]
            egt_i = [float(a.item()) for a in atts_incorrect]

        mean_c = sum(egt_c) / n_correct
        mean_i = sum(egt_i) / n_incorrect
        return float(mean_c - mean_i)

    # ------------------------------------------------------------------
    # Patch-resolution GT mask alignment (public utility)
    # ------------------------------------------------------------------

    @staticmethod
    def align_patch_to_pixel(
        patch_attr: torch.Tensor,
        target_hw:  tuple[int, int],
        mode:       str = "bilinear",
    ) -> torch.Tensor:
        """
        Upsample a ViT patch-resolution attribution map to pixel resolution.

        ViT patch grids are typically:
          • 14×14 = 196 patches  (patch_size=16, input=224)
          • 28×28 = 784 patches  (patch_size=8,  input=224)
          • 16×16 = 256 patches  (patch_size=14, input=224)

        For CUB-200-2011 pixel masks the target is 224×224 or the original
        image size. For VOC/ImageNet-S the target is also 224×224 after resize.

        Parameters
        ----------
        patch_attr : (H_p, W_p) or (1, H_p, W_p) — patch-level attribution.
        target_hw  : (H_pixel, W_pixel) — desired output resolution.
        mode       : interpolation mode (default 'bilinear').

        Returns
        -------
        (H_pixel, W_pixel) float tensor.
        """
        attr = _to_2d(patch_attr)
        if tuple(attr.shape) == tuple(target_hw):
            return attr
        return F.interpolate(
            attr.unsqueeze(0).unsqueeze(0).float(),
            size=target_hw,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        ).squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # Convenience: compute all four metrics in a single call
    # ------------------------------------------------------------------

    def compute_all(
        self,
        att_map: torch.Tensor,
        gt_mask: torch.Tensor,
        thresholds: Sequence[float] | None = None,
    ) -> Dict[str, float]:
        """
        Compute L1, L2, and L3 in a single call for one sample.

        L4 (CalibGap) requires paired correct/incorrect lists and cannot
        be computed per-sample; use calibration_gap() directly.

        Parameters
        ----------
        att_map    : (H_a, W_a)
        gt_mask    : (H_r, W_r)
        thresholds : override instance thresholds

        Returns
        -------
        dict with all L1 keys + 'pointing_game' (L2) + 'egt' (L3).
        """
        result = self.multi_threshold_iou(att_map, gt_mask, thresholds)
        result["pointing_game"] = self.pointing_game(att_map, gt_mask)
        result["egt"]           = self.egt(att_map, gt_mask)
        return result
