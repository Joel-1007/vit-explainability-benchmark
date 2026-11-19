#!/usr/bin/env python3
"""
verify_model_zoo.py — load every model with its REAL pre-trained weights and
verify identity, architecture, token layout and attention extraction.

Per model (from model_zoo/specs.py):
  • weights   : the cached weight file's SHA-256 equals the hash published on
                the HF Hub (and recorded in model_hashes.txt); load is strict
  • config    : timm pretrained_cfg (mean/std, interpolation, crop) recorded
  • pooling   : how the classifier reads the representation (CLS token / mean)
  • tokens    : forward_features shape == prefix tokens + patch grid² (ViT-style)
                or (h, w, C) feature map (Swin); num_prefix_tokens matches spec
  • attention : post-softmax weights captured through the explainer hook for
                the last block: shape (1, heads, T, T), finite, rows sum to 1;
                Swin must raise UnsupportedArchitectureError
  • head      : a fresh n-class head gives finite (1, n) logits that depend on
                the input
Optionally (--compare-dino-official) the timm DINO weights are compared tensor
by tensor with the official dino_vitbase8_pretrain.pth.

Downloads ~2.4 GB on first run (cached afterwards). Writes a JSON report and
exits 1 if any check fails.

Usage:
    python scripts/verify_model_zoo.py [--models ...] [--compare-dino-official]
                                       [--out results/verify_model_zoo.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainers.base import (  # noqa: E402
    UnsupportedArchitectureError, _capture_attn_weights, _num_prefix_tokens,
)
from model_zoo import load_model  # noqa: E402
from model_zoo.specs import MODEL_SPECS, expected_tokens  # noqa: E402


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cached_weight_check(spec) -> dict:
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from record_model_hashes import published_weights

    pub = published_weights(spec.hf_repo)
    path = hf_hub_download(spec.hf_repo, pub["file"])          # cached by timm's load
    local = _sha256(path)
    return {"file": pub["file"], "commit": pub["commit"], "published_sha256": pub["sha256"],
            "local_sha256": local, "match": local == pub["sha256"]}


def _pooling(bb) -> str:
    gp = getattr(bb, "global_pool", None)
    if isinstance(gp, str):
        return gp or "none"
    return type(gp).__name__ if gp is not None else "unknown"


def verify(name: str, seed: int = 0) -> dict:
    spec = MODEL_SPECS[name]
    rep = {"model": name, "timm_id": spec.timm_id, "checks": {}, "info": {}, "errors": []}
    chk = rep["checks"]

    def ok(key, cond, detail=""):
        chk[key] = {"pass": bool(cond), "detail": detail}

    try:
        model = load_model(name, num_classes=5, pretrained=True).eval()
    except Exception as e:
        rep["errors"].append(f"load failed: {e}")
        ok("load", False, str(e))
        return rep
    ok("load", True, "timm strict load")
    bb = model.backbone
    cfg = getattr(bb, "pretrained_cfg", {}) or {}
    rep["info"].update({
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "mean": cfg.get("mean"), "std": cfg.get("std"),
        "interpolation": cfg.get("interpolation"), "crop_pct": cfg.get("crop_pct"),
        "pooling": _pooling(bb),
    })

    try:
        w = _cached_weight_check(spec)
        rep["info"]["weights"] = w
        ok("weights_sha256", w["match"], w["local_sha256"])
    except Exception as e:
        ok("weights_sha256", False, f"could not verify: {e}")

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(2, 3, 224, 224, generator=g)
    with torch.no_grad():
        feats = bb.forward_features(x)
        logits = model(x)
    rep["info"]["features_shape"] = list(feats.shape)
    ok("head_logits", logits.shape == (2, 5) and torch.isfinite(logits).all()
       and not torch.allclose(logits[0], logits[1]), f"logits {list(logits.shape)}")

    prefix, side = expected_tokens(name)
    if name == "swin_b":
        ok("tokens", feats.dim() == 4 and feats.shape[1:3] == (7, 7), f"feature map {list(feats.shape)}")
        try:
            _capture_attn_weights(model, x[:1], [-1])
            ok("attention", False, "Swin should raise UnsupportedArchitectureError")
        except UnsupportedArchitectureError:
            ok("attention", True, "unsupported (no CLS token), as expected")
        return rep

    T = prefix + side * side
    ok("tokens", feats.shape[1] == T, f"T={feats.shape[1]}, expected {prefix}+{side}²={T}")
    n_pre = _num_prefix_tokens(model, feats.shape[1])
    ok("prefix_tokens", n_pre == prefix, f"num_prefix_tokens={n_pre}, spec={prefix}")
    A = _capture_attn_weights(model, x[:1], [-1])[0]
    rows = A.sum(-1)
    ok("attention", A is not None and A.shape[-2:] == (T, T) and torch.isfinite(A).all()
       and torch.allclose(rows, torch.ones_like(rows), atol=1e-4),
       f"shape {list(A.shape)}, row-sum range [{rows.min():.5f}, {rows.max():.5f}]")
    return rep


def compare_dino_official() -> dict:
    """Tensor-by-tensor comparison of timm DINO weights with the official file."""
    from model_zoo.dino_vitb8 import DIRECT_URL, official_state_dict

    path = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(DIRECT_URL))
    if not os.path.exists(path):
        torch.hub.download_url_to_file(DIRECT_URL, path, progress=False)
    off = official_state_dict(path)
    timm_sd = load_model("dino_vitb8", num_classes=0, pretrained=True).backbone.state_dict()
    common = sorted(set(off) & set(timm_sd))
    diff = [k for k in common if off[k].shape != timm_sd[k].shape
            or not torch.equal(off[k].float(), timm_sd[k].float())]
    return {"official_sha256": _sha256(path), "n_official": len(off), "n_timm": len(timm_sd),
            "n_compared": len(common), "only_official": sorted(set(off) - set(timm_sd)),
            "only_timm": sorted(set(timm_sd) - set(off)), "n_different": len(diff),
            "different": diff[:10],
            "pass": len(diff) == 0 and len(common) == len(off) == len(timm_sd)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify real pre-trained model loading")
    p.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    p.add_argument("--compare-dino-official", action="store_true")
    p.add_argument("--out", default="results/verify_model_zoo.json")
    args = p.parse_args(argv)

    reports = []
    for name in args.models:
        try:
            r = verify(name)
        except Exception:
            r = {"model": name, "checks": {"run": {"pass": False, "detail": traceback.format_exc()}}}
        reports.append(r)
        status = "PASS" if all(c["pass"] for c in r["checks"].values()) else "FAIL"
        print(f"[{status}] {name}  {r.get('info', {}).get('pooling', '')}")
        for k, c in r["checks"].items():
            print(f"    {'ok ' if c['pass'] else 'BAD'} {k:<15} {str(c['detail'])[:110]}")
    out = {"models": reports}
    if args.compare_dino_official:
        d = compare_dino_official()
        out["dino_official"] = d
        print(f"[{'PASS' if d['pass'] else 'FAIL'}] DINO timm == official: "
              f"{d['n_compared']} tensors compared, {d['n_different']} different")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    failed = any(not c["pass"] for r in reports for c in r["checks"].values())
    failed |= args.compare_dino_official and not out["dino_official"]["pass"]
    print(f"Wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
