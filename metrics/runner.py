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

import argparse
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import torch
from tqdm import tqdm

from .localization import LocalizationMetrics
from .complexity   import ComplexityMetrics
from .robustness   import RobustnessMetrics
from .normalize    import normalize_attribution

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task 2.x BenchmarkRunner  (unchanged — backward-compatible)
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


# ---------------------------------------------------------------------------
# Phase3Runner  —  Guide Listing 4
# ---------------------------------------------------------------------------
# Orchestrates the full  dataset × model × explainer  matrix with:
#   • fixed random seed injection before every combination
#   • pickle-based checkpoint-and-resume (one .pkl per combination)
#   • normalisation via Task-3.2 normalize_attribution()
#   • tqdm progress bars at all three loop levels
# ---------------------------------------------------------------------------

class Phase3Runner:
    """
    Guide Listing 4 — top-level evaluation orchestrator for Phase 3.

    Iterates over all combinations of:
        datasets  × models × explainers
    producing one checkpointed ``results`` dict per combination.

    Parameters
    ----------
    models : dict[str, nn.Module]
        ``{'vit-b16': model_vit, 'swin-b': model_swin, ...}``
        Models are moved to ``device`` and put into eval mode at run time.
    explainers : dict[str, type[BaseExplainer]]
        ``{'rollout': AttentionRolloutExplainer, 'gradcam': GradCAMExplainer}``
        Classes (not instances) — instantiated fresh per model.
    datasets : dict[str, DataLoader]
        ``{'imagenet': loader_in, 'cub200': loader_cub}``
        Each DataLoader must yield 2- or 3-tuples: ``(x, y)`` or
        ``(x, y, gt_mask)``.
    device : torch.device | str
        Inference device.  Defaults to CUDA if available, else CPU.
    norm_mode : str
        Normalisation mode for raw attribution maps (default ``'minmax'``).
        Passed to :func:`normalize_attribution`.
    patch_size : int
        ViT patch size in pixels (default 16).

    Usage
    -----
    ::

        from metrics.runner import Phase3Runner
        from explainers import AttentionRolloutExplainer, GradCAMExplainer

        runner = Phase3Runner(
            models     = {'vit-b16': model},
            explainers = {'rollout': AttentionRolloutExplainer,
                          'gradcam': GradCAMExplainer},
            datasets   = {'imagenet': val_loader},
        )
        runner.run(checkpoint_dir='results/phase3', seed=42)

    Checkpoint layout::

        results/phase3/
            imagenet-vit-b16-rollout.pkl
            imagenet-vit-b16-gradcam.pkl
            imagenet-swin-b-gradcam.pkl    ← skipped if exists
    """

    def __init__(
        self,
        models:     Dict[str, "torch.nn.Module"],
        explainers: Dict[str, Type],
        datasets:   Dict[str, "torch.utils.data.DataLoader"],
        device:     "str | torch.device | None" = None,
        norm_mode:  str  = "minmax",
        patch_size: int  = 16,
    ) -> None:
        self.models     = models
        self.explainers = explainers
        self.datasets   = datasets
        self.device     = torch.device(device) if isinstance(device, str) else (
            device or (
                torch.device("cuda") if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.norm_mode  = norm_mode
        self.patch_size = patch_size

    # ------------------------------------------------------------------
    # Public: run
    # ------------------------------------------------------------------

    def run(
        self,
        checkpoint_dir: str,
        seed:           int = 42,
        max_batches:    Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation matrix with checkpoint-and-resume.

        Parameters
        ----------
        checkpoint_dir : str
            Directory for per-combination .pkl files.  Created if absent.
        seed           : int
            Master RNG seed (default 42).  Applied via ``torch.manual_seed``
            at the start of every combination for reproducibility.
        max_batches    : int | None
            If set, each combination processes at most this many batches.
            Useful for dry-runs (e.g. ``max_batches=1``).

        Returns
        -------
        dict
            ``{combination_key: results_dict}`` — results for every
            combination processed in this call (skipped ones excluded).
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)

        all_results: Dict[str, Any] = {}

        ds_iter  = tqdm(self.datasets.items(),  desc="Datasets",
                        position=0, leave=True)
        for ds_name, loader in ds_iter:
            ds_iter.set_description(f"Dataset: {ds_name}")

            mdl_iter = tqdm(self.models.items(), desc="Models",
                            position=1, leave=False)
            for model_name, model in mdl_iter:
                mdl_iter.set_description(f"Model: {model_name}")
                model.to(self.device).eval()

                exp_iter = tqdm(self.explainers.items(), desc="Explainers",
                                position=2, leave=False)
                for exp_name, ExplainerCls in exp_iter:
                    exp_iter.set_description(f"Explainer: {exp_name}")

                    key      = f"{ds_name}-{model_name}-{exp_name}"
                    ckpt_path = Path(checkpoint_dir) / f"{key}.pkl"

                    if ckpt_path.exists():
                        log.info(f"Skipping {key} — checkpoint found at {ckpt_path}")
                        print(f"  ↩  Skip: {key}")
                        continue

                    # Fresh seed + explainer instance per combination
                    torch.manual_seed(seed)
                    try:
                        explainer = ExplainerCls(
                            model, patch_size=self.patch_size
                        )
                    except Exception as exc:
                        log.warning(
                            f"Cannot instantiate {ExplainerCls.__name__} "
                            f"for {model_name}: {exc}  — skipping."
                        )
                        continue

                    results = self._run_combination(
                        key, explainer, loader, ckpt_path, max_batches
                    )
                    all_results[key] = results

        return all_results

    # ------------------------------------------------------------------
    # Internal: _run_combination
    # ------------------------------------------------------------------

    def _run_combination(
        self,
        key:         str,
        explainer:   Any,
        loader:      "torch.utils.data.DataLoader",
        ckpt_path:   Path,
        max_batches: Optional[int],
    ) -> Dict[str, Any]:
        """
        Run one (dataset, model, explainer) combination.

        Batch loop:
            x, y [, gt_mask] = next(loader)
            att  = explainer.explain_batch(x, y)   # (B, Hp, Wp)
            att  = normalize_attribution(att, mode) # [0, 1]
            store att + y [+ gt_mask] in results

        Checkpoint saved atomically at the end.

        Returns
        -------
        dict with keys:
            'key'           : str  combination identifier
            'n_samples'     : int  total samples processed
            'attributions'  : list[Tensor]  per-image (Hp,Wp) maps
            'labels'        : list[int]
            'gt_masks'      : list[Tensor | None]
            'norm_mode'     : str
        """
        attributions: List[torch.Tensor] = []
        labels_list:  List[int]          = []
        gt_masks_list: List[Any]         = []
        n_samples = 0

        with torch.no_grad():
            batch_iter = tqdm(loader, desc=key, position=3, leave=False)
            for batch_idx, batch in enumerate(batch_iter):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                # Unpack: support (x, y) and (x, y, gt_mask)
                if len(batch) == 2:
                    x, y = batch
                    gt_mask = None
                elif len(batch) >= 3:
                    x, y, gt_mask = batch[0], batch[1], batch[2]
                else:
                    raise ValueError(
                        f"DataLoader batch must have 2 or 3 elements; got {len(batch)}"
                    )

                x = x.to(self.device).float()
                y = y.to(self.device)

                # explain_batch → (B, Hp, Wp)
                att_batch = explainer.explain_batch(x, y)   # (B, Hp, Wp)

                # Normalise (Task 3.2)
                att_norm = normalize_attribution(
                    att_batch.cpu(), mode=self.norm_mode
                )                                            # (B, Hp, Wp)

                # Store per-image
                B = x.shape[0]
                for i in range(B):
                    attributions.append(att_norm[i])
                    labels_list.append(int(y[i].item()))
                    gt_masks_list.append(
                        gt_mask[i].cpu() if gt_mask is not None else None
                    )

                n_samples += B
                batch_iter.set_postfix(n=n_samples)

        results = {
            "key":          key,
            "n_samples":    n_samples,
            "attributions": attributions,
            "labels":       labels_list,
            "gt_masks":     gt_masks_list,
            "norm_mode":    self.norm_mode,
        }

        # Atomic write: write to tmp then rename
        tmp_path = ckpt_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.rename(ckpt_path)

        print(f"  ✓  Saved: {ckpt_path.name}  ({n_samples} samples)")
        log.info(f"Checkpoint saved: {ckpt_path}")
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
        """
        Load a previously saved combination checkpoint.

        Parameters
        ----------
        ckpt_path : str  Path to the .pkl file.

        Returns
        -------
        dict — same structure as returned by ``_run_combination``.
        """
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def list_checkpoints(checkpoint_dir: str) -> List[str]:
        """Return sorted list of .pkl filenames in checkpoint_dir."""
        return sorted(
            p.name for p in Path(checkpoint_dir).glob("*.pkl")
        )

    def combination_key(
        self, ds_name: str, model_name: str, exp_name: str
    ) -> str:
        """Canonical key string for one combination."""
        return f"{ds_name}-{model_name}-{exp_name}"


# ---------------------------------------------------------------------------
# CLI  —  python -m metrics.runner  (or  python metrics/runner.py)
# ---------------------------------------------------------------------------

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark-runner",
        description=(
            "Phase 3 BenchmarkRunner CLI — run the full dataset × model × "
            "explainer evaluation matrix with automatic checkpointing.\n\n"
            "Example:\n"
            "  python -m metrics.runner \\\n"
            "    --checkpoint-dir results/phase3 \\\n"
            "    --seed 42 \\\n"
            "    --max-batches 1      # dry-run: one batch per combination"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-dir", default="results/phase3",
        metavar="DIR",
        help="Directory for .pkl checkpoint files (created if absent). "
             "Default: results/phase3",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42).",
    )
    p.add_argument(
        "--max-batches", type=int, default=None,
        metavar="N",
        help="Process at most N batches per combination (dry-run mode).",
    )
    p.add_argument(
        "--norm-mode", choices=["minmax", "percentile", "softmax"],
        default="minmax",
        help="Attribution normalisation mode (default: minmax).",
    )
    p.add_argument(
        "--patch-size", type=int, default=16,
        metavar="P",
        help="ViT patch size in pixels (default: 16).",
    )
    p.add_argument(
        "--list-checkpoints", action="store_true",
        help="List existing checkpoints in --checkpoint-dir and exit.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Shorthand for --max-batches 1 (one batch per combination).",
    )
    return p


def _cli_main(argv: Optional[List[str]] = None) -> None:
    """Entry point for `python -m metrics.runner`."""
    parser = _build_cli_parser()
    args   = parser.parse_args(argv)

    if args.list_checkpoints:
        ckpts = Phase3Runner.list_checkpoints(args.checkpoint_dir)
        if ckpts:
            print(f"Checkpoints in '{args.checkpoint_dir}':")
            for c in ckpts:
                print(f"  {c}")
        else:
            print(f"No checkpoints found in '{args.checkpoint_dir}'.")
        return

    max_batches = 1 if args.dry_run else args.max_batches

    print(
        f"\nPhase3Runner CLI\n"
        f"  checkpoint-dir : {args.checkpoint_dir}\n"
        f"  seed           : {args.seed}\n"
        f"  norm-mode      : {args.norm_mode}\n"
        f"  max-batches    : {max_batches}\n"
        f"  patch-size     : {args.patch_size}\n"
    )
    print(
        "Note: call Phase3Runner(...).run() from Python to supply actual\n"
        "models, explainers, and datasets. The CLI shows all options and\n"
        "validates flags; dataset/model loading requires user-side Python.\n"
    )


if __name__ == "__main__":
    _cli_main()
