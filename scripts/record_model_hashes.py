#!/usr/bin/env python3
"""
record_model_hashes.py  —  Task 1.2 §6.4
=========================================
Run ONCE immediately after downloading all six pre-trained checkpoints
and BEFORE any fine-tuning begins.

Writes two files:
  model_hashes.txt  — SHA-256 + HuggingFace commit SHA for all checkpoints
  (committed verbatim into the paper's implementation-details appendix)

Usage
-----
    python scripts/record_model_hashes.py [--out model_hashes.txt]

Requirements
------------
    pip install huggingface_hub torch

Notes
-----
• For HF checkpoints the script first tries model.safetensors, then
  pytorch_model.bin (some repos have migrated to safetensors).
• DINO-ViT-B/8 is loaded via torch.hub, its state_dict is saved to a
  temporary file, hashed, and the temp file is deleted immediately.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys
import tempfile

import torch
from huggingface_hub import hf_hub_download, model_info


# ---------------------------------------------------------------------------
# Checkpoint manifest
# ---------------------------------------------------------------------------
# Each entry: (model_display_name, repo_id, preferred_filename, fallback_filename)
HF_CHECKPOINTS = [
    (
        "ViT-B/16 (augreg IN-21K)",
        "google/vit-base-patch16-224-in21k",
        "model.safetensors",
        "pytorch_model.bin",
    ),
    (
        "DeiT-B/16 distilled (IN-1K)",
        "facebook/deit-base-distilled-patch16-224",
        "model.safetensors",
        "pytorch_model.bin",
    ),
    (
        "Swin-B (IN-22K→IN-1K)",
        "microsoft/swin-base-patch4-window7-224",
        "model.safetensors",
        "pytorch_model.bin",
    ),
    (
        "BEiT-B/16 pre-train (IN-22K)",
        "microsoft/beit-base-patch16-224-pt22k",
        "model.safetensors",
        "pytorch_model.bin",
    ),
    (
        "DINOv2-ViT-B/14 (LVD-142M)",
        "facebook/dinov2-base",
        "model.safetensors",
        "pytorch_model.bin",
    ),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    """Stream a file and return its SHA-256 hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _try_hf_download(repo_id: str, preferred: str, fallback: str) -> tuple[str, str]:
    """
    Try to download preferred filename; fall back to fallback on error.
    Returns (local_path, filename_used).
    """
    for fname in (preferred, fallback):
        try:
            local = hf_hub_download(repo_id=repo_id, filename=fname)
            return local, fname
        except Exception:
            continue
    raise RuntimeError(
        f"Could not download '{preferred}' or '{fallback}' from {repo_id}"
    )


def _hf_commit_sha(repo_id: str) -> str:
    try:
        info = model_info(repo_id)
        return info.sha or "unknown"
    except Exception:
        return "unknown"


def _write(log_path: str, text: str) -> None:
    with open(log_path, "a") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_log: str = "model_hashes.txt") -> None:
    sep = "=" * 60

    # Initialise log
    with open(output_log, "w") as f:
        f.write("ViT Explainability Benchmark — Pre-trained Model Hash Log\n")
        f.write(f"Generated : {datetime.datetime.now().isoformat()}\n")
        f.write(f"Purpose   : Task 1.2 §6.4 — Reproducibility record\n")
        f.write(f"{sep}\n\n")

    print(f"[record_model_hashes] Writing to: {output_log}\n")

    # ---------------------------------------------------------------
    # 1. HuggingFace checkpoints
    # ---------------------------------------------------------------
    for display_name, repo_id, preferred, fallback in HF_CHECKPOINTS:
        print(f"  →  {display_name}  ({repo_id})")
        try:
            local_path, fname_used = _try_hf_download(repo_id, preferred, fallback)
            sha   = sha256_file(local_path)
            commit = _hf_commit_sha(repo_id)

            entry = (
                f"Model      : {display_name}\n"
                f"Repo       : {repo_id}\n"
                f"File       : {fname_used}\n"
                f"SHA-256    : {sha}\n"
                f"HF Commit  : {commit}\n"
                f"Local path : {local_path}\n"
                f"Date       : {datetime.datetime.now().isoformat()}\n"
                f"{sep}\n\n"
            )
            _write(output_log, entry)
            print(f"     ✓  SHA-256: {sha[:24]}...\n")

        except Exception as exc:
            error_entry = (
                f"Model      : {display_name}\n"
                f"Repo       : {repo_id}\n"
                f"ERROR      : {exc}\n"
                f"{sep}\n\n"
            )
            _write(output_log, error_entry)
            print(f"     ✗  ERROR: {exc}\n", file=sys.stderr)

    # ---------------------------------------------------------------
    # 2. DINO-ViT-B/8 via PyTorch Hub
    # ---------------------------------------------------------------
    print("  →  DINO-ViT-B/8  (torch.hub 'facebookresearch/dino:main')")
    try:
        dino = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb8",
            pretrained=True,
            verbose=False,
        )
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            torch.save(dino.state_dict(), tmp.name)
            tmp_path = tmp.name

        sha = sha256_file(tmp_path)
        os.unlink(tmp_path)

        entry = (
            f"Model      : DINO-ViT-B/8\n"
            f"Source     : torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')\n"
            f"GitHub     : https://github.com/facebookresearch/dino\n"
            f"Direct URL : https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth\n"
            f"SHA-256    : {sha}\n"
            f"Note       : Hash is of the state_dict saved to a temp file; \n"
            f"             re-run to verify consistency.\n"
            f"Date       : {datetime.datetime.now().isoformat()}\n"
            f"{sep}\n\n"
        )
        _write(output_log, entry)
        print(f"     ✓  SHA-256: {sha[:24]}...\n")

    except Exception as exc:
        error_entry = (
            f"Model      : DINO-ViT-B/8\n"
            f"ERROR      : {exc}\n"
            f"{sep}\n\n"
        )
        _write(output_log, error_entry)
        print(f"     ✗  ERROR: {exc}\n", file=sys.stderr)

    print(f"[record_model_hashes] Done.  All hashes saved to: {output_log}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record SHA-256 hashes of all pre-trained checkpoints (Task 1.2 §6.4)"
    )
    parser.add_argument(
        "--out",
        default="model_hashes.txt",
        help="Output log file path (default: model_hashes.txt)",
    )
    args = parser.parse_args()
    main(output_log=args.out)
