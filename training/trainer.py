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

def fine_tune(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    config: dict[str, Any],
) -> dict[str, list]:
    """
    Run the standardised full fine-tuning protocol (Task 1.2 §5.5).

    Parameters
    ----------
    model        : nn.Module  (backbone + head already attached)
    train_loader : DataLoader (yields (images, integer_labels))
    val_loader   : DataLoader (yields (images, integer_labels))
    config       : dict       (see module docstring for keys)

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'val_top1', 'val_top5'
    """
    # ---------------------------------------------------------------
    # Resolve config with defaults
    # ---------------------------------------------------------------
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
    num_classes  = config["num_classes"]
    model_name   = config["model_name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    hash_log = os.path.join(save_dir, "finetuned_hashes.txt")

    # ---------------------------------------------------------------
    # Build training components
    # ---------------------------------------------------------------
    optimiser, effective_lr = build_optimiser(
        model, base_lr=base_lr, batch_size=batch_size, weight_decay=weight_decay
    )
    scheduler = build_scheduler(optimiser, num_epochs=num_epochs, warmup_epochs=warmup_epochs)
    criterion_train, criterion_val = build_loss(dataset=dataset)
    mixup_fn = build_mixup_fn(num_classes=num_classes)
    scaler   = GradScaler(enabled=use_amp)

    log.info(
        f"Fine-tuning {model_name} | dataset={dataset} | "
        f"epochs={num_epochs} | effective_lr={effective_lr:.6f} | "
        f"device={device} | amp={use_amp} | accum_steps={accum_steps}"
    )

    history = {"train_loss": [], "val_loss": [], "val_top1": [], "val_top5": []}

    # ---------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        optimiser.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=num_batches,
            desc=f"Epoch {epoch+1}/{num_epochs} [train]",
            leave=False,
        )

        for step, (images, labels) in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Apply Mixup (returns soft labels)
            images, labels = mixup_fn(images, labels)

            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion_train(outputs, labels) / accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * accum_steps

            # Gradient accumulation step
            if (step + 1) % accum_steps == 0 or (step + 1) == num_batches:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()

            if step % log_every == 0:
                pbar.set_postfix(
                    loss=f"{loss.item() * accum_steps:.4f}",
                    lr=f"{optimiser.param_groups[0]['lr']:.2e}",
                )

        avg_train_loss = epoch_loss / num_batches
        history["train_loss"].append(avg_train_loss)

        # Step LR scheduler (once per epoch)
        scheduler.step()

        # ---------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------
        val_loss, top1, top5 = _evaluate(
            model, val_loader, criterion_val, device, use_amp
        )
        history["val_loss"].append(val_loss)
        history["val_top1"].append(top1.item())
        history["val_top5"].append(top5.item())

        log.info(
            f"Epoch {epoch+1:>3}/{num_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"top1={top1:.2f}% | top5={top5:.2f}% | "
            f"lr={optimiser.param_groups[0]['lr']:.2e}"
        )

        # ---------------------------------------------------------------
        # Save checkpoint + record hash
        # ---------------------------------------------------------------
        timestamp   = datetime.datetime.now().strftime("%Y%m%d")
        ckpt_name   = f"{model_name}_ft_{dataset}_{timestamp}_ep{epoch+1:03d}.pth"
        ckpt_path   = os.path.join(save_dir, ckpt_name)
        metadata    = {
            "model_name": model_name,
            "dataset":    dataset,
            "epoch":      epoch + 1,
            "num_epochs": num_epochs,
            "top1_acc":   f"{top1.item():.4f}",
            "val_loss":   f"{val_loss:.6f}",
            "base_lr":    base_lr,
            "batch_size": batch_size,
        }
        torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, ckpt_path)
        sha = _sha256(ckpt_path)

        with open(hash_log, "a") as f:
            f.write(
                f"Model:      {model_name}\n"
                f"Dataset:    {dataset}\n"
                f"Epoch:      {epoch+1}\n"
                f"Top-1 Acc:  {top1.item():.4f}\n"
                f"File:       {ckpt_path}\n"
                f"SHA-256:    {sha}\n"
                f"Date:       {datetime.datetime.now().isoformat()}\n"
                f"{'='*60}\n\n"
            )

    log.info(f"Training complete.  Best val top-1: {max(history['val_top1']):.2f}%")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Run one validation epoch and return (loss, top1_acc, top5_acc)."""
    model.eval()
    total_loss  = 0.0
    top1_total  = torch.tensor(0.0, device=device)
    top5_total  = torch.tensor(0.0, device=device)
    n_batches   = len(loader)

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        top1, top5  = _accuracy(outputs, labels, topk=(1, 5))
        total_loss += loss.item()
        top1_total += top1
        top5_total += top5

    return total_loss / n_batches, top1_total / n_batches, top5_total / n_batches
