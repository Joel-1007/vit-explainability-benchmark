"""
trainer.py — Standardised fine-tuning training loop (Task 1.2 §5.5 / §5.6)

Entry point : fine_tune(model, train_loader, val_loader, config)

Key behaviours
--------------
• Full fine-tuning (backbone + head), not head-only linear probe.
• Gradient accumulation to simulate batch size 256 on memory-constrained GPUs.
• Linear LR warmup (5 epochs) → cosine annealing decay.
• Per-epoch checkpoint saving with SHA-256 hash recording.
• Mixed-precision (torch.amp) by default.
• Early stopping disabled — epoch count is fixed per dataset (§5.3).

Config dict keys (all optional — defaults match §5.1)
-----------------------------------------------------
  base_lr          float   1e-4
  batch_size       int     256
  weight_decay     float   0.05
  num_epochs       int     50
  warmup_epochs    int     5
  accum_steps      int     1          (set > 1 if GPU OOM)
  amp              bool    True
  save_dir         str     "checkpoints"
  log_every        int     10         (print every N steps)
  dataset          str     "cub200"
  num_classes      int     (required)
  model_name       str     (required — for checkpoint naming)
  seed             int     None       (fine-tuning seed; added to the checkpoint
                                       name as ``_seed{N}`` so several seeds per
                                       (model, dataset) can coexist)

Seeds: call ``set_seed(seed)`` *before* building the model (head initialisation)
and the DataLoaders; ``fine_tune`` re-seeds at its start for data order,
augmentation, Mixup and drop-path.
"""

from __future__ import annotations

import os
import math
import time
import datetime
import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .optimizer import build_optimiser, build_scheduler
from .loss      import build_loss
from .mixup     import build_mixup_fn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute top-k accuracy for integer (hard) labels."""
    maxk = max(topk)
    B = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / B for k in topk]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_name(model_name: str, dataset: str, epoch: int,
                    seed: int | None = None, date: str | None = None) -> str:
    """``{model}_ft_{dataset}[_seed{N}]_{YYYYMMDD}_ep{NNN}.pth`` (read by run_phase3)."""
    date = date or datetime.datetime.now().strftime("%Y%m%d")
    seed_part = f"_seed{seed}" if seed is not None else ""
    return f"{model_name}_ft_{dataset}{seed_part}_{date}_ep{epoch:03d}.pth"


def fine_tune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the standardised full fine-tuning protocol (Task 1.2 §5.5).

    Batches are (images, int labels) for single-label tasks and
    (images, multi-hot labels, entry weights) for multi-label tasks
    (utils/train_datasets.py). Images are [0, 1] pixels; the model normalises.

    Model selection: the FINAL epoch is kept for every model (no early
    stopping), so no choice is made on validation data. For multi-label tasks
    per-class thresholds are then estimated on the validation split
    (training/thresholds.py), frozen in the checkpoint metadata and written to
    ``<checkpoint>.thresholds.json``.

    Extra config keys: ``task`` ("single" | "multi"; default from
    utils.tasks.TASK_TYPE[dataset]), ``pos_weight`` (Tensor, multi-label BCE),
    ``save_every`` (also save every N epochs; 0 = final only), ``max_steps``
    (cap steps per epoch — smoke tests only).

    Returns
    -------
    history : dict with per-epoch 'train_loss', 'val_loss' and the task's
              validation metrics, plus 'checkpoint' and (multi-label)
              'decision_thresholds'.
    """
    from utils.tasks import MULTI, TASK_TYPE

    base_lr      = config.get("base_lr",      1e-4)
    batch_size   = config.get("batch_size",    256)
    weight_decay = config.get("weight_decay",  0.05)
    num_epochs   = config.get("num_epochs",    50)
    warmup_epochs = config.get("warmup_epochs", 5)
    accum_steps  = config.get("accum_steps",   1)
    use_amp      = config.get("amp",           True)
    save_dir     = config.get("save_dir",      "checkpoints")
    log_every    = config.get("log_every",     10)
    dataset      = config.get("dataset",       "cub200")
    task         = config.get("task", TASK_TYPE.get(dataset, "single"))
    save_every   = config.get("save_every",    0)
    max_steps    = config.get("max_steps")
    num_classes  = config["num_classes"]
    model_name   = config["model_name"]
    seed         = config.get("seed")
    if seed is not None:
        set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"
    model  = model.to(device)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    hash_log = os.path.join(save_dir, "finetuned_hashes.txt")

    optimiser, effective_lr = build_optimiser(
        model, base_lr=base_lr, batch_size=batch_size, weight_decay=weight_decay
    )
    scheduler = build_scheduler(optimiser, num_epochs=num_epochs, warmup_epochs=warmup_epochs)
    pos_weight = config.get("pos_weight")
    criterion_train, criterion_val = build_loss(
        task, pos_weight=None if pos_weight is None else pos_weight.to(device))
    mixup_fn = build_mixup_fn(num_classes=num_classes, task=task)
    scaler   = GradScaler(enabled=use_amp)

    log.info(
        f"Fine-tuning {model_name} | dataset={dataset} ({task}-label) | "
        f"epochs={num_epochs} | effective_lr={effective_lr:.6f} | "
        f"device={device} | amp={use_amp} | accum_steps={accum_steps}"
    )

    history: dict[str, Any] = {"train_loss": [], "val_loss": [], "val_metrics": []}
    val_out: dict[str, Any] = {}

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n_steps = 0.0, 0
        num_batches = len(train_loader) if max_steps is None else min(max_steps, len(train_loader))
        optimiser.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, total=num_batches,
                                          desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False)):
            if step >= num_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            if task == MULTI:
                labels, weights = batch[1].to(device), batch[2].to(device)
                images, labels, weights = mixup_fn(images, labels, weights)
            else:
                labels = batch[1].to(device, non_blocking=True)
                images, labels = mixup_fn(images, labels)        # soft targets

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = (criterion_train(outputs, labels, weights) if task == MULTI
                        else criterion_train(outputs, labels)) / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps
            n_steps += 1

            if (step + 1) % accum_steps == 0 or (step + 1) == num_batches:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()

        avg_train_loss = epoch_loss / max(n_steps, 1)
        history["train_loss"].append(avg_train_loss)
        scheduler.step()

        val_out = _evaluate(model, val_loader, criterion_val, device, use_amp, task)
        history["val_loss"].append(val_out["loss"])
        history["val_metrics"].append(val_out["metrics"])
        log.info(f"Epoch {epoch+1:>3}/{num_epochs} | train_loss={avg_train_loss:.4f} | "
                 f"val_loss={val_out['loss']:.4f} | {val_out['metrics']} | "
                 f"lr={optimiser.param_groups[0]['lr']:.2e}")

        if save_every and (epoch + 1) % save_every == 0 and (epoch + 1) < num_epochs:
            _save(model, save_dir, hash_log, model_name, dataset, seed, epoch + 1, num_epochs,
                  val_out, config, task, thresholds=None)

    # ---- final epoch is the selected model; thresholds from validation ----
    thresholds, thr_info = None, None
    if task == MULTI:
        from training.thresholds import f1_optimal_thresholds
        thresholds, thr_info = f1_optimal_thresholds(
            val_out["scores"], val_out["labels"], val_out["weights"])
        history["decision_thresholds"] = thresholds.tolist()
        history["threshold_info"] = thr_info
    history["checkpoint"] = _save(model, save_dir, hash_log, model_name, dataset, seed,
                                  num_epochs, num_epochs, val_out, config, task,
                                  thresholds=thresholds, thr_info=thr_info)
    log.info(f"Training complete. Final-epoch validation: {val_out['metrics']}")
    return history


def _save(model, save_dir, hash_log, model_name, dataset, seed, epoch, num_epochs,
          val_out, config, task, thresholds=None, thr_info=None) -> str:
    """Save checkpoint (+ thresholds sidecar) and append its SHA-256 to the hash log."""
    import json

    ckpt_path = os.path.join(save_dir, checkpoint_name(model_name, dataset, epoch, seed=seed))
    metadata = {
        "model_name": model_name, "dataset": dataset, "task": task, "seed": seed,
        "epoch": epoch, "num_epochs": num_epochs, "selection": "final_epoch",
        "val_loss": float(val_out["loss"]), "val_metrics": val_out["metrics"],
        "base_lr": config.get("base_lr", 1e-4), "batch_size": config.get("batch_size", 256),
        "model_kwargs": config.get("model_kwargs", {}),
        "mixup": "timm Mixup a=0.8, ls=0.1" if task == "single"
                 else "MultiLabelMixup a=0.8 (label mixing, weight product), no label smoothing",
    }
    if thresholds is not None:
        metadata["decision_thresholds"] = [float(t) for t in thresholds]
        metadata["threshold_info"] = thr_info
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, ckpt_path)
    if thresholds is not None:
        with open(ckpt_path + ".thresholds.json", "w") as f:
            json.dump({"decision_thresholds": metadata["decision_thresholds"], **thr_info}, f, indent=2)
    sha = _sha256(ckpt_path)
    with open(hash_log, "a") as f:
        f.write(f"Model:      {model_name}\nDataset:    {dataset}\nSeed:       {seed}\n"
                f"Epoch:      {epoch}\nValidation: {val_out['metrics']}\nFile:       {ckpt_path}\n"
                f"SHA-256:    {sha}\nDate:       {datetime.datetime.now().isoformat()}\n{'='*60}\n\n")
    return ckpt_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device,
              use_amp: bool = True, task: str = "single") -> dict[str, Any]:
    """
    One validation pass. single: loss, top-1/top-5 (%).
    multi: loss, macro AUROC and mAP over classes with both labels present,
    plus the raw scores/labels/weights used for threshold estimation.
    """
    import numpy as np

    model.eval()
    total_loss, n_batches = 0.0, 0
    top1_sum = top5_sum = 0.0
    scores, labels_all, weights_all = [], [], []
    for batch in tqdm(loader, desc="Validation", leave=False):
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            outputs = model(images).float()
        if task == "multi":
            weights = batch[2].to(device)
            total_loss += criterion(outputs, labels, weights).item()
            scores.append(torch.sigmoid(outputs).cpu())
            labels_all.append(labels.cpu())
            weights_all.append(weights.cpu())
        else:
            total_loss += criterion(outputs, labels).item()
            k5 = min(5, outputs.shape[1])
            t1, t5 = _accuracy(outputs, labels, topk=(1, k5))
            top1_sum += t1.item()
            top5_sum += t5.item()
        n_batches += 1

    out: dict[str, Any] = {"loss": total_loss / max(n_batches, 1)}
    if task != "multi":
        out["metrics"] = {"top1": top1_sum / max(n_batches, 1), "top5": top5_sum / max(n_batches, 1)}
        return out

    from sklearn.metrics import average_precision_score, roc_auc_score
    S = torch.cat(scores).numpy(); Y = torch.cat(labels_all).numpy(); W = torch.cat(weights_all).numpy()
    aucs, aps = [], []
    for c in range(S.shape[1]):
        v = W[:, c] > 0
        y = Y[v, c] >= 0.5
        if y.any() and (~y).any():
            aucs.append(roc_auc_score(y, S[v, c]))
            aps.append(average_precision_score(y, S[v, c]))
    out["metrics"] = {"macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
                      "mAP": float(np.mean(aps)) if aps else float("nan"),
                      "n_classes_scored": len(aucs)}
    out.update({"scores": S, "labels": Y, "weights": W})
    return out
