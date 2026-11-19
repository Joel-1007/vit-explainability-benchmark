#!/usr/bin/env python3
"""
run_phase3.py — Phase 3 Baseline Evaluation Pipeline
=====================================================
End-to-end script to run the full benchmark evaluation matrix:

    6 models × 6 explainers × 4 datasets

Wires up all model loaders (model_zoo), explainer classes (explainers),
and dataset loaders, then delegates to Phase3Runner for checkpointed,
resumable evaluation across all combinations.

Checkpointing
-------------
Each (dataset, model, explainer) combination is saved as a separate .pkl
file in --checkpoint-dir. If the script is interrupted and restarted, it
automatically skips already-completed combinations and resumes from where
it left off. This makes the benchmark robust to crashes, OOM errors, and
preemptions on shared GPU clusters.

Usage
-----
    # Full benchmark (all combinations, all data):
    python run_phase3.py --data-root /path/to/datasets

    # Dry run (1 batch per combination, for validation):
    python run_phase3.py --data-root /path/to/datasets --dry-run

    # Subset: specific models and explainers only:
    python run_phase3.py --data-root /path/to/datasets \\
        --models vit_b16 deit_b16 \\
        --explainers gradcam rollout \\
        --datasets imagenet_s50 cub200

    # Quick sanity check (2 batches, single model):
    python run_phase3.py --data-root /path/to/datasets \\
        --models vit_b16 --max-batches 2

Requirements
------------
    pip install -r requirements.txt
    Datasets must be downloaded and organized per configs/*.yaml specs.
    See scripts/verify_datasets.py for integrity checks.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
from model_zoo import load_model, MODEL_REGISTRY, ABLATION_MODEL_REGISTRY
from explainers import (
    RawAttentionExplainer,
    RawPatchAttentionExplainer,
    RawAttentionLast3Explainer,
    RawAttentionAllLayersExplainer,
    AttentionRolloutExplainer,
    GradCAMExplainer,
    CheferLRPExplainer,
    RISEExplainer,
    LIMEExplainer,
    IntegratedGradientsExplainer,
)
from metrics.runner import Phase3Runner
from metrics.sample_metrics import ALL_FAMILIES, FAMILIES, SampleMetricComputer
from metrics.fidelity import MASK_MODES
from metrics.aggregate import aggregate
from utils.eval_datasets import DATASET_CLASSES, NUM_CLASSES, build_eval_loader
from utils.tasks import TASK_TYPE, set_task

log = logging.getLogger("run_phase3")


# ═══════════════════════════════════════════════════════════════════════════
# Registry of available explainers (E1–E6; E7 DIME excluded — see BENCHMARK.md §3.1)
# ═══════════════════════════════════════════════════════════════════════════
EXPLAINER_REGISTRY: Dict[str, type] = {
    "raw_attention": RawAttentionExplainer,     # E1
    "rollout":       AttentionRolloutExplainer,  # E2
    "gradcam":       GradCAMExplainer,           # E3
    "chefer_lrp":    CheferLRPExplainer,         # E4
    "rise":          RISEExplainer,              # E5
    "lime":          LIMEExplainer,              # E6
}

# Phase 4 ablation variants — selectable, never part of the default matrix.
ABLATION_EXPLAINER_REGISTRY: Dict[str, type] = {
    "raw_attention_patch": RawPatchAttentionExplainer,      # A1 patch vs CLS
    "raw_attention_last3": RawAttentionLast3Explainer,      # A2 last 3 layers
    "raw_attention_all":   RawAttentionAllLayersExplainer,  # A2 all layers
}
# Optional main-matrix explainers (not run by default; select with --explainers)
OPTIONAL_EXPLAINER_REGISTRY: Dict[str, type] = {
    "integrated_gradients": IntegratedGradientsExplainer,   # E8, axiomatic reference
}
ALL_EXPLAINERS = {**EXPLAINER_REGISTRY, **OPTIONAL_EXPLAINER_REGISTRY,
                  **ABLATION_EXPLAINER_REGISTRY}
ALL_MODELS = [*MODEL_REGISTRY, *ABLATION_MODEL_REGISTRY]


# ═══════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════

def _setup_logging(log_file: str | None = None) -> None:
    """Configure logging to both console and (optionally) a log file."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root_logger.addHandler(console)

    # File handler (if requested)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)
        log.info(f"Logging to file: {log_file}")


# ═══════════════════════════════════════════════════════════════════════════
# Models: one fine-tuned model per (model, dataset)
# ═══════════════════════════════════════════════════════════════════════════

# Pixel size of one attribution-grid cell for each model (224 / P cells per side).
# Swin's native patch is 4 px (56x56); its maps are resampled to the same 14x14
# grid as the 16-px models.
MODEL_GRID_PATCH = {
    "vit_b16":       16,
    "deit_b16":      16,
    "beit_b16":      16,
    "swin_b":        16,
    "dino_vitb8":     8,
    "dinov2_vitb14": 14,
    "mae_b16":       16,   # ablation A4
}


SEED_TAG = "@seed"   # model name in checkpoint keys: "vit_b16@seed1"


def find_weights(weights_dir: str | None, model_name: str, dataset: str,
                 seed: int | None = None) -> Path | None:
    """
    Fine-tuned checkpoint for (model, dataset[, seed]) in ``weights_dir``:
    ``{model}_{dataset}[_seed{N}].pth`` if present, else the newest file with
    the trainer's naming ``{model}_ft_{dataset}[_seed{N}]_{YYYYMMDD}_ep{NNN}.pth``.
    Without ``seed``, seed-tagged files are ignored.
    """
    if not weights_dir:
        return None
    d = Path(weights_dir)
    tag = f"_seed{seed}" if seed is not None else ""
    exact = d / f"{model_name}_{dataset}{tag}.pth"
    if exact.exists():
        return exact
    matches = sorted(d.glob(f"{model_name}_ft_{dataset}{tag}_*.pth"))
    if seed is None:
        matches = [m for m in matches if "_seed" not in m.name]
    return matches[-1] if matches else None


def load_finetuned_model(
    model_name: str,
    dataset: str,
    weights_dir: str | None,
    allow_random_head: bool = False,
    seed: int | None = None,
) -> nn.Module:
    """
    Build ``model_name`` with a head sized for ``dataset`` and load its
    fine-tuned weights. Without weights the head is random, so explanations
    would describe an untrained classifier: refused unless allow_random_head.
    """
    num_classes = NUM_CLASSES[dataset]
    task = TASK_TYPE[dataset]
    ckpt = find_weights(weights_dir, model_name, dataset, seed)
    ckpt_data = (torch.load(ckpt, map_location="cpu", weights_only=True)
                 if ckpt is not None else None)
    meta = ckpt_data.get("metadata", {}) if isinstance(ckpt_data, dict) else {}
    # Build the architecture the checkpoint was trained with (e.g. BEiT head pooling)
    model = load_model(model_name, num_classes=num_classes, pretrained=True,
                       **meta.get("model_kwargs", {}))
    if ckpt is None:
        seed_txt = f" (seed {seed})" if seed is not None else ""
        msg = (f"No fine-tuned weights for {model_name} on {dataset}{seed_txt} in "
               f"--weights-dir={weights_dir!r}")
        if not allow_random_head:
            raise FileNotFoundError(msg + " (pass --allow-random-head for smoke tests only)")
        log.warning(f"  ⚠ {msg} — using a RANDOM {num_classes}-class head. "
                    "Results are NOT valid for reporting.")
        return set_task(model, task, None, "none (random head)")
    state = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state, strict=True)
    if meta.get("task") not in (None, task):
        raise ValueError(f"{ckpt.name}: trained as {meta.get('task')}-label, "
                         f"but {dataset} is evaluated as {task}-label")
    thresholds = meta.get("decision_thresholds")
    if task == "multi" and thresholds is None:
        msg = (f"{ckpt.name}: multi-label checkpoint without validation thresholds "
               "(fine-tune with scripts/finetune.py)")
        if not allow_random_head:
            raise ValueError(msg)
        log.warning(f"  ⚠ {msg} — correctness uses 0.5, NOT valid for reporting.")
    set_task(model, task, thresholds,
             "validation_f1 (frozen in checkpoint)" if thresholds is not None else "argmax")
    log.info(f"  ✓ {model_name}/{dataset}: loaded {ckpt.name}")
    return model

# ═══════════════════════════════════════════════════════════════════════════
# Run metadata (saved alongside checkpoints for reproducibility)
# ═══════════════════════════════════════════════════════════════════════════

def _save_run_metadata(args: argparse.Namespace, checkpoint_dir: str) -> None:
    """Save run configuration as JSON for reproducibility."""
    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "args": {
            "data_root": args.data_root,
            "models": args.models,
            "explainers": args.explainers,
            "datasets": args.datasets,
            "checkpoint_dir": args.checkpoint_dir,
            "seed": args.seed,
            "max_batches": args.max_batches,
            "max_samples": args.max_samples,
            "norm_mode": args.norm_mode,
            "patch_size": args.patch_size,
            "weights_dir": args.weights_dir,
            "finetune_seeds": args.finetune_seeds,
            "allow_random_head": args.allow_random_head,
            "metrics": args.metrics,
            "robustness_subset": args.robustness_subset,
            "robustness_samples": args.robustness_samples,
            "fidelity_ablation_modes": args.fidelity_ablation_modes,
            "ablations": args.ablations,
            "batch_size": args.batch_size,
            "dry_run": args.dry_run,
        },
    }
    meta_path = Path(checkpoint_dir) / "run_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Run metadata saved to {meta_path}")


def _generate_summary_csv(checkpoint_dir: str) -> None:
    """
    Scan all .pkl checkpoints and produce a summary CSV with one row
    per combination, listing n_samples and the normalisation mode used.
    """
    import csv

    ckpt_dir = Path(checkpoint_dir)
    pkl_files = sorted(ckpt_dir.glob("*.pkl"))
    if not pkl_files:
        log.info("No checkpoints to summarize.")
        return

    rows = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            rows.append({
                "combination": data.get("key", pkl_path.stem),
                "n_samples": data.get("n_samples", "?"),
                "norm_mode": data.get("norm_mode", "?"),
                "checkpoint_file": pkl_path.name,
            })
        except Exception as e:
            log.warning(f"Could not read checkpoint {pkl_path.name}: {e}")

    if rows:
        csv_path = ckpt_dir / "checkpoint_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Checkpoint summary CSV saved to {csv_path} ({len(rows)} combinations)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI & main
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_phase3",
        description=(
            "Phase 3 — Run the full ViT Explainability Benchmark evaluation matrix.\n\n"
            "Evaluates all combinations of models × explainers × datasets\n"
            "with checkpoint-and-resume support via Phase3Runner.\n\n"
            "Each (dataset, model, explainer) combination is saved as a .pkl\n"
            "checkpoint. If the script crashes or is interrupted, simply re-run\n"
            "with the same --checkpoint-dir to resume from where it left off.\n\n"
            "Example:\n"
            "  python run_phase3.py --data-root /data --dry-run\n"
            "  python run_phase3.py --data-root /data --models vit_b16 --explainers gradcam"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument(
        "--data-root", type=str, required=True,
        help="Root directory containing all dataset folders (cub200/, voc2012/, imagenet_s50/, nih_chestxray/).",
    )

    # Filters
    p.add_argument(
        "--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
        choices=ALL_MODELS,
        metavar="MODEL",
        help=f"Models to evaluate. Default: the main 6 ({', '.join(MODEL_REGISTRY.keys())}). "
             "Ablation A4 also needs: mae_b16.",
    )
    p.add_argument(
        "--explainers", nargs="+", default=list(EXPLAINER_REGISTRY.keys()),
        choices=list(ALL_EXPLAINERS),
        metavar="EXPLAINER",
        help=f"Explainers to evaluate. Default: E1–E6 ({', '.join(EXPLAINER_REGISTRY.keys())}). "
             f"Optional: {', '.join(OPTIONAL_EXPLAINER_REGISTRY)}. "
             f"Ablation variants: {', '.join(ABLATION_EXPLAINER_REGISTRY)}.",
    )
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CLASSES.keys()),
        choices=list(DATASET_CLASSES.keys()),
        metavar="DATASET",
        help=f"Datasets to evaluate. Default: all ({', '.join(DATASET_CLASSES.keys())})",
    )

    # Execution
    p.add_argument(
        "--checkpoint-dir", default="results/phase3",
        help="Directory for .pkl checkpoints (default: results/phase3).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42).",
    )
    p.add_argument(
        "--max-batches", type=int, default=None,
        help="Max batches per combination (None = full dataset).",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to load per dataset (useful for quick tests).",
    )
    p.add_argument(
        "--norm-mode", choices=["minmax", "percentile", "softmax"],
        default="minmax",
        help="Attribution normalisation mode (default: minmax).",
    )
    p.add_argument(
        "--patch-size", type=int, default=None,
        help="Override the attribution-grid patch size for ALL models. "
             "Default: per model (16; DINO-B/8 → 8; DINOv2-B/14 → 14).",
    )
    p.add_argument(
        "--weights-dir", type=str, default=None,
        help="Directory with fine-tuned checkpoints, named {model}_{dataset}.pth "
             "or {model}_ft_{dataset}_*.pth (trainer output). Required unless "
             "--allow-random-head.",
    )
    p.add_argument(
        "--finetune-seeds", type=int, nargs="+", default=None, metavar="SEED",
        help="Evaluate one fine-tuned checkpoint per seed ({model}_{dataset}_seed{N}.pth "
             "or {model}_ft_{dataset}_seed{N}_*.pth) to measure seed variance. "
             "Default: a single unseeded checkpoint per (model, dataset).",
    )
    p.add_argument(
        "--allow-random-head", action="store_true",
        help="Smoke tests only: if a fine-tuned checkpoint is missing, use the "
             "pre-trained backbone with a random head. Results are NOT valid.",
    )
    p.add_argument(
        "--metrics", nargs="+", default=list(FAMILIES), choices=list(ALL_FAMILIES),
        help=f"Metric families computed per sample (default: {' '.join(FAMILIES)}). "
             "'adversarial' adds PGD robustness on the robustness subset.",
    )
    p.add_argument(
        "--fidelity-ablation-modes", nargs="+", default=[], choices=list(MASK_MODES),
        help="Ablation A3: also compute fidelity with these masking baselines "
             "(columns suffixed [zero]/[blur]); the main columns use 'mean'.",
    )
    p.add_argument(
        "--ablations", action="store_true",
        help="Shortcut for ablations A1–A3: add the raw-attention variants to "
             "--explainers and --fidelity-ablation-modes zero blur. "
             "For A4 also pass --models vit_b16 mae_b16 (needs MAE weights).",
    )
    p.add_argument(
        "--robustness-subset", type=int, default=100,
        help="Compute R1–R3 on the first N images of each combination "
             "(default: 100; R1 re-runs the explainer --robustness-samples times "
             "per image). Use -1 for all images.",
    )
    p.add_argument(
        "--robustness-samples", type=int, default=20,
        help="Perturbations per image for R1 Max-Sensitivity (default: 20).",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Shorthand for --max-batches 1 --max-samples 64.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device for inference (default: auto-detect cuda/cpu).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for data loaders (default: 32).",
    )
    p.add_argument(
        "--log-file", type=str, default=None,
        help="Path to a log file. Logs are always printed to console; this adds file output.",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Ablation shortcut (before metadata is saved) ─────────────────────
    if args.ablations:
        for name in ["raw_attention", "rollout", *ABLATION_EXPLAINER_REGISTRY]:
            if name not in args.explainers:
                args.explainers.append(name)
        args.fidelity_ablation_modes = sorted({*args.fidelity_ablation_modes, "zero", "blur"})

    # ── Handle dry-run defaults ──────────────────────────────────────────
    if args.dry_run:
        args.max_batches = args.max_batches or 1
        args.max_samples = args.max_samples or 64

    # ── Logging ──────────────────────────────────────────────────────────
    log_file = args.log_file or os.path.join(args.checkpoint_dir, "run_phase3.log")
    _setup_logging(log_file)

    # ── Device ───────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Banner ───────────────────────────────────────────────────────────
    banner = (
        "\n" + "=" * 65 + "\n"
        "  ViT Explainability Benchmark — Phase 3 Evaluation Pipeline\n"
        + "=" * 65 + "\n"
        f"  Timestamp      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  Device         : {device}\n"
        f"  Models         : {', '.join(args.models)}\n"
        f"  Explainers     : {', '.join(args.explainers)}\n"
        f"  Datasets       : {', '.join(args.datasets)}\n"
        f"  Checkpoint dir : {args.checkpoint_dir}\n"
        f"  Log file       : {log_file}\n"
        f"  Seed           : {args.seed}\n"
        f"  Norm mode      : {args.norm_mode}\n"
        f"  Batch size     : {args.batch_size}\n"
        f"  Max batches    : {args.max_batches or 'all'}\n"
        f"  Max samples    : {args.max_samples or 'all'}\n"
        f"  Dry run        : {args.dry_run}\n"
        + "=" * 65 + "\n"
    )
    print(banner)
    log.info("Phase 3 evaluation starting...")

    # ── Save run metadata ────────────────────────────────────────────────
    _save_run_metadata(args, args.checkpoint_dir)

    explainers = {
        name: cls for name, cls in ALL_EXPLAINERS.items()
        if name in args.explainers
    }

    # ── Step 1: Load datasets ────────────────────────────────────────────
    log.info("[1/2] Loading datasets...")
    datasets: Dict[str, DataLoader] = {}
    for ds_name in args.datasets:
        try:
            loader = build_eval_loader(
                ds_name, args.data_root,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                num_workers=args.num_workers,
            )
            datasets[ds_name] = loader
            log.info(f"  ✓ {ds_name}: {len(loader.dataset)} samples, {len(loader)} batches")
        except Exception as e:
            log.error(f"  ✗ Failed to load {ds_name}: {e}")

    if not datasets:
        log.error("No datasets loaded successfully. Aborting.")
        sys.exit(1)

    total_combos = len(datasets) * len(args.models) * len(explainers)
    log.info(
        f"  Total combinations: {len(datasets)} datasets × "
        f"{len(args.models)} models × {len(explainers)} explainers = {total_combos}"
    )

    # ── Step 2: Evaluate one (dataset, model) pair at a time ─────────────
    # Each dataset has its own fine-tuned head, so models are loaded per
    # dataset, and freed before the next one to bound memory.
    log.info("[2/2] Running Phase3Runner...")
    t0 = time.time()
    results: Dict[str, Any] = {}
    failures: list[str] = []
    metric_computer = SampleMetricComputer(
        families=args.metrics,
        norm_mode=args.norm_mode,
        robustness_subset=None if args.robustness_subset < 0 else args.robustness_subset,
        robustness_samples=args.robustness_samples,
        fidelity_ablation_modes=args.fidelity_ablation_modes,
        seed=args.seed,
    )

    seeds = args.finetune_seeds or [None]
    runs = [(m, sd) for m in args.models for sd in seeds]
    for ds_name, loader in datasets.items():
        for base_model, ft_seed in runs:
            # Checkpoint keys carry the seed: "cub200-vit_b16@seed1-gradcam"
            model_name = base_model if ft_seed is None else f"{base_model}{SEED_TAG}{ft_seed}"
            keys = [f"{ds_name}-{model_name}-{e}" for e in explainers]
            if all(Phase3Runner.combination_done(args.checkpoint_dir, k) for k in keys):
                log.info(f"  ↩ {ds_name}/{model_name}: all explainers done — skipping")
                continue
            try:
                model = load_finetuned_model(
                    base_model, ds_name, args.weights_dir, args.allow_random_head,
                    seed=ft_seed,
                )
            except Exception as e:
                log.error(f"  ✗ {model_name}/{ds_name}: {e}")
                failures.append(f"{model_name}/{ds_name}")
                continue

            runner = Phase3Runner(
                models={model_name: model},
                explainers=explainers,
                datasets={ds_name: loader},
                device=device,
                norm_mode=args.norm_mode,
                patch_size=args.patch_size or MODEL_GRID_PATCH[base_model],
                metric_computer=metric_computer,
            )
            results.update(runner.run(
                checkpoint_dir=args.checkpoint_dir,
                seed=args.seed,
                max_batches=args.max_batches,
            ))
            del model, runner
            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    n_combos = len(results)

    # ── Generate summary CSV from all checkpoints ────────────────────────
    _generate_summary_csv(args.checkpoint_dir)

    # ── Aggregate per-sample metrics for Phase 4 ─────────────────────────
    agg_paths = None
    try:
        agg_paths = aggregate(args.checkpoint_dir)
        log.info(f"Aggregated results: {agg_paths[0]} and {agg_paths[1]}")
    except Exception as e:
        log.error(f"Aggregation failed: {e}")
        failures.append(f"aggregation ({e})")

    # ── Summary ──────────────────────────────────────────────────────────
    summary = (
        "\n" + "=" * 65 + "\n"
        "  Phase 3 Evaluation Complete!\n"
        + "=" * 65 + "\n"
        f"  Combinations evaluated this run : {n_combos}\n"
        f"  Total time                      : {elapsed:.1f}s ({elapsed/3600:.2f}h)\n"
        f"  Checkpoints saved to            : {args.checkpoint_dir}/\n"
        f"  Log file                        : {log_file}\n"
    )

    if results:
        summary += "\n  Results per combination:\n"
        for key, res in results.items():
            n = res.get("n_samples", 0)
            summary += f"    {key}: {n} samples\n"
    else:
        summary += "\n  (All combinations were already checkpointed — nothing new to run)\n"

    # List all checkpoints (including previously saved ones)
    existing_ckpts = Phase3Runner.list_checkpoints(args.checkpoint_dir)
    summary += f"\n  Total checkpoints on disk: {len(existing_ckpts)}/{total_combos}\n"

    if agg_paths:
        summary += f"\n  Per-sample results : {agg_paths[0]}\n  Summary            : {agg_paths[1]}\n"
    summary += (
        "\n  Next steps:\n"
        "    • Verify checkpoints: python -m metrics.runner --list-checkpoints\n"
        "    • Run Phase 4 analysis: python scripts/phase4_correlation_analysis.py "
        "--results_csv <checkpoint-dir>/aggregated_results.csv\n"
        + "=" * 65 + "\n"
    )

    if failures:
        summary += f"\n  FAILED: {', '.join(failures)}\n"

    print(summary)
    if failures:
        sys.exit(1)
    log.info(f"Phase 3 complete. {n_combos} new combinations, {elapsed:.1f}s elapsed.")


if __name__ == "__main__":
    main()
