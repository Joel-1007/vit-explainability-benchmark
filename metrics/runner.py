"""
runner.py  —  BenchmarkRunner  (Tasks 2.2–2.4 / integration)
=============================================================
Integrates LocalizationMetrics, ComplexityMetrics (optional), and
RobustnessMetrics (optional) into a dataset-level evaluation loop.

Responsibilities
----------------
• Iterate over a DataLoader of (image, gt_mask, label, pred_label) tuples.
• Call LocalizationMetrics.compute_all() for L1–L3 per sample.
• Partition samples into correct/incorrect for L4 (CalibGap).
• Aggregate per-class and macro-average results.
• Return a structured results dict compatible with the paper's Table format.

Usage
-----
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
)

# Localization only (Task 2.2, backward-compatible)
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_explainer,
)

# Localization + Robustness (Task 2.3)
rm = RobustnessMetrics(epsilon=0.05, n_samples=50)
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_explainer,
    robustness=rm,
    randomised_model=randomise_model_weights(model),
    label_randomised_model=randomise_classifier_labels(model),
)
results = runner.evaluate(model, val_loader, dataset_name="cub200")
# results["macro"]["miou"], results["macro"]["max_sensitivity"], etc.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch
from tqdm import tqdm

from .localization import LocalizationMetrics
from .complexity   import ComplexityMetrics
from .robustness   import RobustnessMetrics

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Dataset-level evaluation loop for localization metrics L1–L4
    and (optionally) robustness metrics R1–R3.

    Parameters
    ----------
    metrics : LocalizationMetrics
        Configured metrics instance (thresholds, seed).
    explainer : callable
        Signature: ``(model, image_batch) -> att_maps``
        where  ``att_maps``  is a list of (H, W) tensors, one per image.
    device : str | torch.device
        Device for model inference (default 'cuda' if available).
    robustness : RobustnessMetrics | None
        When provided, R1–R3 are computed per sample and included in
        ``results["macro"]`` under keys 'max_sensitivity',
        'model_randomisation', 'label_randomisation'.
        When None (default), R1–R3 are silently skipped and all
        existing behaviour is preserved unchanged.
    randomised_model : nn.Module | None
        Pre-computed weight-randomised copy of the model for R2.
        Required when robustness is not None.
    label_randomised_model : nn.Module | None
        Pre-computed label-randomised copy of the model for R3.
        Required when robustness is not None.
    """

    def __init__(
        self,
        metrics:   LocalizationMetrics,
        explainer: Callable,
        device:    str | torch.device | None = None,
        robustness:             Optional[RobustnessMetrics] = None,
        randomised_model:       Optional[torch.nn.Module]  = None,
        label_randomised_model: Optional[torch.nn.Module]  = None,
        complexity:             Optional[ComplexityMetrics] = None,
    ) -> None:
        self.metrics                = metrics
        self.explainer              = explainer
        self.device                 = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.robustness             = robustness
        self.randomised_model       = randomised_model
        self.label_randomised_model = label_randomised_model
        self.complexity             = complexity

        # Validate robustness dependency when robustness is requested
        if robustness is not None:
            if randomised_model is None:
                log.warning(
                    "BenchmarkRunner: robustness is set but randomised_model "
                    "is None.  R2 (ModelRand) will be skipped."
                )
            if label_randomised_model is None:
                log.warning(
                    "BenchmarkRunner: robustness is set but label_randomised_model "
                    "is None.  R3 (LabelRand) will be skipped."
                )

    # ------------------------------------------------------------------

    def evaluate(
        self,
        model:        torch.nn.Module,
        loader:       torch.utils.data.DataLoader,
        dataset_name: str = "unknown",
        max_batches:  int | None = None,
    ) -> Dict[str, Any]:
        """
        Run the full L1–L4 evaluation over a DataLoader.

        The DataLoader must yield tuples of:
            (images, gt_masks, labels)
        where
            images    : (B, C, H, W)
            gt_masks  : (B, H_m, W_m)  — integer or float masks
            labels    : (B,)            — integer class indices

        Parameters
        ----------
        model        : fine-tuned nn.Module (eval mode set internally).
        loader       : DataLoader yielding (images, gt_masks, labels).
        dataset_name : logged in results for traceability.
        max_batches  : stop after this many batches (None = full epoch).

        Returns
        -------
        dict with keys:
          'dataset'      : str
          'n_samples'    : int
          'n_correct'    : int
          'n_incorrect'  : int
          'macro'        : dict — macro-averaged L1–L4
          'per_metric'   : dict — all accumulated metric values
        """
        model.eval()
        model.to(self.device)

        # Accumulators
        all_metrics: Dict[str, List[float]] = defaultdict(list)
        atts_correct:   List[torch.Tensor] = []
        atts_incorrect: List[torch.Tensor] = []
        masks_correct:  List[torch.Tensor] = []
        masks_incorrect: List[torch.Tensor] = []

        n_samples  = 0
        n_correct  = 0

        with torch.no_grad():
            for batch_idx, (images, gt_masks, labels) in enumerate(
                tqdm(loader, desc=f"Evaluating [{dataset_name}]")
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                # --- Model forward pass ------------------------------------
                logits = model(images)
                preds  = logits.argmax(dim=1)

                # --- Attribution maps (one per image in batch) -------------
                att_maps: List[torch.Tensor] = self.explainer(model, images)

                # --- Per-sample metrics ------------------------------------
                for i in range(images.shape[0]):
                    att  = att_maps[i].cpu()
                    mask = gt_masks[i].cpu()

                    sample_metrics = self.metrics.compute_all(att, mask)
                    for k, v in sample_metrics.items():
                        all_metrics[k].append(v)

                    # --- Optional C1–C3 per sample -------------------------
                    if self.complexity is not None:
                        c_scores = self.complexity.compute_all(att)
                        for k, v in c_scores.items():
                            all_metrics[k].append(v)

                    # --- Optional R1–R3 per sample -------------------------
                    if self.robustness is not None:
                        single_img = images[i].cpu()   # (C, H, W)
                        # R1 — Max-Sensitivity
                        r1 = self.robustness.max_sensitivity(
                            self.explainer, model, single_img, att
                        )
                        all_metrics["max_sensitivity"].append(r1)

                        # R2 — Model Randomisation
                        if self.randomised_model is not None:
                            att_rand_list = self.explainer(
                                self.randomised_model,
                                images[i].unsqueeze(0).to(self.device),
                            )
                            att_rand = (
                                att_rand_list[0].cpu()
                                if isinstance(att_rand_list, (list, tuple))
                                else att_rand_list.cpu()
                            )
                            r2 = self.robustness.model_randomisation(att, att_rand)
                            all_metrics["model_randomisation"].append(r2)

                        # R3 — Label Randomisation
                        if self.label_randomised_model is not None:
                            att_shuf_list = self.explainer(
                                self.label_randomised_model,
                                images[i].unsqueeze(0).to(self.device),
                            )
                            att_shuf = (
                                att_shuf_list[0].cpu()
                                if isinstance(att_shuf_list, (list, tuple))
                                else att_shuf_list.cpu()
                            )
                            r3 = self.robustness.label_randomisation(att, att_shuf)
                            all_metrics["label_randomisation"].append(r3)

                    # Partition for L4
                    is_correct = (preds[i] == labels[i]).item()
                    if is_correct:
                        atts_correct.append(att)
                        masks_correct.append(mask)
                        n_correct += 1
                    else:
                        atts_incorrect.append(att)
                        masks_incorrect.append(mask)

                n_samples += images.shape[0]

        # --- L4: CalibGap ------------------------------------------------
        calib_gap: float | None = None
        if atts_correct and atts_incorrect:
            gt_all = masks_correct + masks_incorrect
            try:
                calib_gap = self.metrics.calibration_gap(
                    atts_correct,
                    atts_incorrect,
                    gt_masks=gt_all,
                )
                all_metrics["calibration_gap"].append(calib_gap)
            except ValueError as e:
                log.warning(f"CalibGap not computed: {e}")
        else:
            log.warning(
                "CalibGap skipped: need both correct and incorrect predictions. "
                f"correct={len(atts_correct)}, incorrect={len(atts_incorrect)}"
            )

        # --- Macro averages -----------------------------------------------
        macro = {
            k: (sum(v) / len(v) if v else float("nan"))
            for k, v in all_metrics.items()
        }

        log.info(
            f"[{dataset_name}] n={n_samples} | correct={n_correct} | "
            f"mIoU={macro.get('miou', float('nan')):.4f} | "
            f"PG={macro.get('pointing_game', float('nan')):.4f} | "
            f"EGT={macro.get('egt', float('nan')):.4f} | "
            f"CalibGap={macro.get('calibration_gap', float('nan')):.4f}"
            + (
                f" | Gini={macro.get('gini', float('nan')):.4f}"
                f" | Sparsity={macro.get('sparsity', float('nan')):.4f}"
                f" | EffRes={macro.get('effective_resolution', float('nan')):.4f}"
                if self.complexity is not None else ""
            )
            + (
                f" | MaxSens={macro.get('max_sensitivity', float('nan')):.4f}"
                f" | ModelRand={macro.get('model_randomisation', float('nan')):.4f}"
                f" | LabelRand={macro.get('label_randomisation', float('nan')):.4f}"
                if self.robustness is not None else ""
            )
        )

        return {
            "dataset":     dataset_name,
            "n_samples":   n_samples,
            "n_correct":   n_correct,
            "n_incorrect": n_samples - n_correct,
            "macro":       macro,
            "per_metric":  dict(all_metrics),
        }
