#!/usr/bin/env python3
"""
finetune.py — standardised fine-tuning of one (model, dataset[, seed]).

Produces the checkpoint run_phase3.py evaluates:
    <save-dir>/<model>_ft_<dataset>[_seed<N>]_<YYYYMMDD>_ep<NNN>.pth
(+ ``.thresholds.json`` for multi-label datasets).

Protocol (BENCHMARK.md §1.2): AdamW, cosine schedule with warm-up, the
dataset's configs/<dataset>.yaml hyper-parameters, pixel-space inputs
(training/transforms.py), Mixup per task type (training/mixup.py), final-epoch
selection, validation-only thresholds for multi-label tasks. Splits and class
orders come from utils/train_datasets.py and never overlap the evaluation set.

Usage:
    python scripts/finetune.py --model vit_b16 --dataset pascal_voc \\
        --data-root /data --seed 0 --save-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_zoo import MODEL_SPECS, load_model           # noqa: E402
from training.transforms import build_transforms        # noqa: E402
from training.trainer import fine_tune, set_seed        # noqa: E402
from utils.eval_datasets import NUM_CLASSES             # noqa: E402
from utils.tasks import MULTI, TASK_TYPE                # noqa: E402
from utils.train_datasets import build_train_val, positive_weights  # noqa: E402


def main(argv=None) -> str:
    p = argparse.ArgumentParser(description="Fine-tune one model on one dataset")
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--dataset", required=True, choices=list(TASK_TYPE))
    p.add_argument("--data-root", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--config", default=None, help="Default: configs/<dataset>.yaml")
    p.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Effective batch size (default: config batch_size, 256)")
    p.add_argument("--accum-steps", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="Smoke tests only")
    p.add_argument("--no-pretrained", action="store_true", help="Smoke tests only")
    p.add_argument("--beit-pool", choices=["token", "avg"], default="token",
                   help="BEiT head: CLS token (frozen protocol) or native mean pooling "
                        "(one-off recorded diagnostic only; stored in the checkpoint)")
    args = p.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = args.config or os.path.join(root, "configs", f"{args.dataset}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        set_seed(args.seed)                     # before the head is initialised
    task = TASK_TYPE[args.dataset]
    num_classes = NUM_CLASSES[args.dataset]
    model_kwargs = {"global_pool": args.beit_pool} if args.model == "beit_b16" else {}
    model = load_model(args.model, num_classes=num_classes, pretrained=not args.no_pretrained,
                       **model_kwargs)

    train_tf, val_tf = build_transforms()
    train_ds, val_ds = build_train_val(args.dataset, args.data_root, train_tf, val_tf)

    accum = args.accum_steps or cfg.get("accum_steps", 1)
    effective_bs = args.batch_size or cfg.get("batch_size", 256)
    loader_bs = max(1, effective_bs // accum)
    g = torch.Generator().manual_seed(args.seed if args.seed is not None else 0)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=loader_bs, shuffle=True, drop_last=True, generator=g,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=loader_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    config = {
        "model_name": args.model, "dataset": args.dataset, "task": task,
        "num_classes": num_classes, "seed": args.seed,
        "num_epochs": args.epochs or cfg.get("num_epochs", 30),
        "warmup_epochs": min(cfg.get("warmup_epochs", 5), args.epochs or cfg.get("num_epochs", 30)),
        "base_lr": float(cfg.get("base_lr", 1e-4)), "weight_decay": cfg.get("weight_decay", 0.05),
        "batch_size": loader_bs * accum, "accum_steps": accum,
        "amp": cfg.get("amp", True) and not args.no_amp,
        "save_dir": args.save_dir, "max_steps": args.max_steps,
        "model_kwargs": model_kwargs,
    }
    if task == MULTI and cfg.get("class_weighted_loss", False):
        config["pos_weight"] = positive_weights(train_ds)

    history = fine_tune(model, train_loader, val_loader, config)
    print(json.dumps({"checkpoint": history["checkpoint"],
                      "final_val": history["val_metrics"][-1],
                      "n_train": len(train_ds), "n_val": len(val_ds)}, indent=2, default=float))
    return history["checkpoint"]


if __name__ == "__main__":
    main()
