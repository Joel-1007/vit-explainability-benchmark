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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
from model_zoo import load_model, MODEL_REGISTRY
from explainers import (
    RawAttentionExplainer,
    AttentionRolloutExplainer,
    GradCAMExplainer,
    CheferLRPExplainer,
    RISEExplainer,
    LIMEExplainer,
)
from metrics.runner import Phase3Runner

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
# Dataset loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_transform(input_size: int = 224) -> transforms.Compose:
    """Standard evaluation transform matching BENCHMARK.md §5.3."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _load_imagefolder_dataset(
    root: str,
    split: str = "val",
    input_size: int = 224,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> DataLoader:
    """Load an ImageFolder-style dataset with optional sample limit."""
    dataset = ImageFolder(
        os.path.join(root, split),
        transform=_get_transform(input_size),
    )
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


def load_cub200(
    data_root: str,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> DataLoader:
    """
    Load CUB-200-2011 test split as ImageFolder.

    Expected structure:
        {data_root}/cub200/images/001.Black_footed_Albatross/...
    """
    cub_root = os.path.join(data_root, "cub200")
    return _load_imagefolder_dataset(
        cub_root, split="images", batch_size=batch_size, max_samples=max_samples,
    )


def load_pascal_voc(
    data_root: str,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> DataLoader:
    """
    Load PASCAL VOC 2012 segmentation val split.

    Expected structure:
        {data_root}/voc2012/JPEGImages/...
        {data_root}/voc2012/ImageSets/Segmentation/val.txt
    """
    from PIL import Image
    from torch.utils.data import Dataset

    voc_root = os.path.join(data_root, "voc2012")
    transform = _get_transform(224)

    class VOCSegDataset(Dataset):
        """Minimal VOC dataset yielding (image, label) tuples."""
        def __init__(self, root, split="val", transform=None, max_samples=None):
            self.root = Path(root)
            self.transform = transform
            split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
            with open(split_file) as f:
                self.ids = [line.strip() for line in f if line.strip()]
            if max_samples is not None:
                self.ids = self.ids[:max_samples]

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_id = self.ids[idx]
            img = Image.open(self.root / "JPEGImages" / f"{img_id}.jpg").convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0  # class label not used; localization uses seg masks

    dataset = VOCSegDataset(voc_root, split="val", transform=transform, max_samples=max_samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False,
    )


def load_imagenet_s50(
    data_root: str,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> DataLoader:
    """
    Load ImageNet-S-50 validation split.

    Expected structure:
        {data_root}/imagenet_s50/validation/n01440764/...
    """
    ins_root = os.path.join(data_root, "imagenet_s50")
    return _load_imagefolder_dataset(
        ins_root, split="validation", batch_size=batch_size, max_samples=max_samples,
    )


def load_nih_chestxray(
    data_root: str,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> DataLoader:
    """
    Load NIH ChestX-ray14 test images.

    Expected structure:
        {data_root}/nih_chestxray/images/00000001_000.png
        {data_root}/nih_chestxray/test_list.txt
    """
    from PIL import Image
    from torch.utils.data import Dataset

    nih_root = Path(data_root) / "nih_chestxray"
    transform = _get_transform(224)

    class NIHChestXrayDataset(Dataset):
        """Minimal NIH CXR dataset yielding (image, label) tuples."""
        def __init__(self, root, transform=None, max_samples=None):
            self.root = Path(root)
            self.transform = transform
            list_file = self.root / "test_list.txt"
            with open(list_file) as f:
                self.image_ids = [line.strip() for line in f if line.strip()]
            if max_samples is not None:
                self.image_ids = self.image_ids[:max_samples]

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_name = self.image_ids[idx]
            img = Image.open(self.root / "images" / img_name).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, 0  # dummy label

    dataset = NIHChestXrayDataset(nih_root, transform=transform, max_samples=max_samples)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True, drop_last=False,
    )


DATASET_LOADERS = {
    "imagenet_s50":   load_imagenet_s50,
    "cub200":         load_cub200,
    "pascal_voc":     load_pascal_voc,
    "nih_chestxray":  load_nih_chestxray,
}


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_models(
    model_names: list[str],
    num_classes: int = 0,
) -> Dict[str, nn.Module]:
    """Load requested models from model_zoo."""
    models = {}
    for name in model_names:
        log.info(f"Loading model: {name}")
        try:
            model = load_model(name, num_classes=num_classes, pretrained=True)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            models[name] = model
            log.info(f"  ✓ {name} loaded ({n_params:.1f}M params)")
        except Exception as e:
            log.error(f"  ✗ Failed to load {name}: {e}")
    return models


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
            "num_classes": args.num_classes,
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
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=f"Models to evaluate. Default: all ({', '.join(MODEL_REGISTRY.keys())})",
    )
    p.add_argument(
        "--explainers", nargs="+", default=list(EXPLAINER_REGISTRY.keys()),
        choices=list(EXPLAINER_REGISTRY.keys()),
        metavar="EXPLAINER",
        help=f"Explainers to evaluate. Default: all ({', '.join(EXPLAINER_REGISTRY.keys())})",
    )
    p.add_argument(
        "--datasets", nargs="+", default=list(DATASET_LOADERS.keys()),
        choices=list(DATASET_LOADERS.keys()),
        metavar="DATASET",
        help=f"Datasets to evaluate. Default: all ({', '.join(DATASET_LOADERS.keys())})",
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
        "--patch-size", type=int, default=16,
        help="ViT patch size in pixels (default: 16).",
    )
    p.add_argument(
        "--num-classes", type=int, default=0,
        help="Number of downstream classes for model heads. 0 = feature extractor (default: 0).",
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

    # ── Step 1: Load models ──────────────────────────────────────────────
    log.info("[1/3] Loading models...")
    models = load_models(args.models, num_classes=args.num_classes)
    if not models:
        log.error("No models loaded successfully. Aborting.")
        sys.exit(1)
    log.info(f"  {len(models)}/{len(args.models)} models loaded successfully.")

    # ── Step 2: Load datasets ────────────────────────────────────────────
    log.info("[2/3] Loading datasets...")
    datasets: Dict[str, DataLoader] = {}
    for ds_name in args.datasets:
        log.info(f"  Loading dataset: {ds_name}")
        try:
            loader = DATASET_LOADERS[ds_name](
                args.data_root,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
            datasets[ds_name] = loader
            n = len(loader.dataset)
            log.info(f"  ✓ {ds_name}: {n} samples, {len(loader)} batches")
        except Exception as e:
            log.error(f"  ✗ Failed to load {ds_name}: {e}")
            log.error(f"    Make sure the dataset is at: {args.data_root}/{ds_name}")

    if not datasets:
        log.error("No datasets loaded successfully. Aborting.")
        sys.exit(1)
    log.info(f"  {len(datasets)}/{len(args.datasets)} datasets loaded successfully.")

    # ── Step 3: Build explainer registry (filtered) ──────────────────────
    explainers = {
        name: cls for name, cls in EXPLAINER_REGISTRY.items()
        if name in args.explainers
    }
    log.info(f"  Explainers registered: {', '.join(explainers.keys())}")

    # ── Combination count ────────────────────────────────────────────────
    total_combos = len(models) * len(explainers) * len(datasets)
    log.info(
        f"  Total combinations: {len(datasets)} datasets × "
        f"{len(models)} models × {len(explainers)} explainers = {total_combos}"
    )

    # ── Step 4: Run Phase3Runner ─────────────────────────────────────────
    log.info("[3/3] Starting Phase3Runner evaluation matrix...")
    t0 = time.time()

    runner = Phase3Runner(
        models=models,
        explainers=explainers,
        datasets=datasets,
        device=device,
        norm_mode=args.norm_mode,
        patch_size=args.patch_size,
    )

    results = runner.run(
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        max_batches=args.max_batches,
    )

    elapsed = time.time() - t0
    n_combos = len(results)

    # ── Generate summary CSV from all checkpoints ────────────────────────
    _generate_summary_csv(args.checkpoint_dir)

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

    summary += (
        "\n  Next steps:\n"
        "    • Verify checkpoints: python -m metrics.runner --list-checkpoints\n"
        "    • Run Phase 4 analysis: python scripts/phase4_correlation_analysis.py\n"
        + "=" * 65 + "\n"
    )

    print(summary)
    log.info(f"Phase 3 complete. {n_combos} new combinations, {elapsed:.1f}s elapsed.")


if __name__ == "__main__":
    main()
