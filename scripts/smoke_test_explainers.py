#!/usr/bin/env python3
"""
smoke_test_explainers.py — real-model check of every model × explainer pair.

Runs each supported explainer on each model (real pre-trained weights; a
fine-tuned checkpoint with --weights-dir/--dataset) for a few images and checks:

  compatibility   Swin-B must raise UnsupportedArchitectureError for the
                  CLS-attention explainers (raw attention + variants, rollout,
                  Chefer LRP); every other pair must run
  shape           (224/P, 224/P) for the model's attribution grid
  finite          no NaN / inf
  non_degenerate  std > 0 and < 99% of entries equal
  deterministic   same input twice → identical map
  input_sensitive different images → different maps
  target          class-specific methods (GradCAM, Chefer LRP, RISE, LIME, IG):
                  maps for two different target classes differ (Spearman ρ < 0.999);
                  for RISE/LIME this is advisory unless fine-tuned heads are loaded
                  (--weights-dir/--dataset): an untrained head's scores barely move
                  under masking;
                  class-agnostic methods (raw attention + variants, rollout):
                  maps are identical across targets
  gradients       input gradient of the target logit is finite and non-zero
                  (models with gradient-based explainers)
  ig_completeness |Σφ − (f(x) − f(0))| ≤ 5% · max(|f(x) − f(0)|, 1 logit)
                  (Integrated Gradients; the floor avoids ill-conditioning at f(x) ≈ f(0))
  attention       every block's captured post-softmax attention has shape
                  (1, H, T, T) with T = prefix tokens + P², rows summing to 1
                  (CLS-token models)

Outputs (in --out): smoke_report.csv (one row per check), smoke_summary.json,
and <model>_maps.png — rows = explainers, columns = images, attribution
overlaid on the input — for visual inspection. Exit code 1 if any check fails.

Images: --images FILE... or --data-root (one evaluation image per dataset
found); otherwise a fixed synthetic image set (object shapes on texture).

Usage:
    python scripts/smoke_test_explainers.py --data-root /data --out results/smoke
    python scripts/smoke_test_explainers.py --models vit_b16 swin_b --rise-masks 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainers.base import UnsupportedArchitectureError, _capture_attn_weights, _get_timm_blocks  # noqa: E402
from model_zoo import MODEL_SPECS, load_model  # noqa: E402
from utils.eval_datasets import PairTransform  # noqa: E402

CLASS_AGNOSTIC = {"raw_attention", "rollout", "raw_attention_patch",
                  "raw_attention_last3", "raw_attention_all"}
NEEDS_CLS = CLASS_AGNOSTIC | {"chefer_lrp"}
GRADIENT_BASED = {"gradcam", "chefer_lrp", "integrated_gradients"}
PERTURBATION_BASED = {"rise", "lime"}


def _registries():
    import run_phase3 as rp
    return rp.ALL_EXPLAINERS, rp.MODEL_GRID_PATCH


def synthetic_images(n: int, seed: int = 0) -> List[torch.Tensor]:
    """Deterministic [0,1] images: textured background + a bright object at varying places."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        img = rng.uniform(0.2, 0.5, (224, 224, 3))
        cy, cx, r = 60 + 50 * k % 120, 70 + 40 * k % 100, 35
        yy, xx = np.mgrid[:224, :224]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        img[disk] = np.array([0.9, 0.3 + 0.2 * k % 0.6, 0.1])
        out.append(torch.tensor(img, dtype=torch.float32).permute(2, 0, 1))
    return out


def load_images(paths: List[str], data_root: Optional[str], n: int) -> List[torch.Tensor]:
    from PIL import Image
    tf = PairTransform()
    if paths:
        return [tf.image(Image.open(p).convert("RGB")) for p in paths[:max(n, 1)]]
    if data_root:
        from utils.eval_datasets import DATASET_CLASSES
        imgs = []
        for name, (sub, cls) in DATASET_CLASSES.items():
            if (Path(data_root) / sub).exists() and len(imgs) < n:
                try:
                    imgs.append(cls(Path(data_root) / sub)[0][0])
                except Exception as e:                    # report, keep going
                    print(f"  (could not read {name}: {e})")
        if imgs:
            return imgs
    return synthetic_images(n)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    import scipy.stats as ss
    a, b = a.flatten().numpy(), b.flatten().numpy()
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(ss.spearmanr(a, b).statistic)


def _build(cls, model, ps, name, rise_masks, lime_samples, ig_steps):
    kw = {}
    if name == "rise":
        kw = {"n_masks": rise_masks, "chunk_size": min(100, rise_masks)}
    elif name == "lime":
        kw = {"n_samples": lime_samples}
    elif name == "integrated_gradients":
        kw = {"n_steps": ig_steps}
    return cls(model, patch_size=ps, **kw)


def smoke_model(model_name: str, model: torch.nn.Module, explainers: Dict[str, type], ps: int,
                images: List[torch.Tensor], device: torch.device, args) -> tuple[list, dict]:
    rows, maps = [], {}
    P = 224 // ps
    model = model.to(device).eval()
    xs = [x.to(device) for x in images]
    with torch.no_grad():
        logits0 = model(xs[0][None])
    top = int(logits0.argmax())
    other = int(logits0.argsort(descending=True)[0, min(5, logits0.shape[1] - 1)])
    is_swin = MODEL_SPECS[model_name].prefix_tokens == 0

    def rec(exp, check, ok, detail=""):
        rows.append({"model": model_name, "explainer": exp, "check": check,
                     "pass": bool(ok), "detail": str(detail)[:200]})

    # model-level: input gradients
    if any(e in GRADIENT_BASED for e in explainers):
        x = xs[0][None].clone().requires_grad_(True)
        model(x)[0, top].backward()
        g = x.grad
        rec("(model)", "gradients", g is not None and torch.isfinite(g).all() and g.abs().sum() > 0,
            f"|grad|_1={float(g.abs().sum()) if g is not None else 'None'}")
        model.zero_grad()

    # model-level: attention extraction for every block
    if not is_swin:
        prefix = MODEL_SPECS[model_name].prefix_tokens
        T = prefix + P * P if ps == MODEL_SPECS[model_name].patch_size else None
        n_blocks = len(_get_timm_blocks(model))
        A = _capture_attn_weights(model, xs[0][None], list(range(n_blocks)))
        bad = [i for i, a in enumerate(A) if a is None or not torch.isfinite(a).all()
               or not torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-3)
               or (T is not None and a.shape[-1] != T)]
        rec("(model)", "attention", not bad,
            f"{n_blocks} blocks, shape {list(A[0].shape)}" + (f", bad blocks {bad}" if bad else ""))

    for name, cls in explainers.items():
        expect_unsupported = is_swin and name in NEEDS_CLS
        try:
            e = _build(cls, model, ps, name, args.rise_masks, args.lime_samples, args.ig_steps)
            t0 = time.time()
            m0 = e.explain(xs[0], top).detach().float().cpu()
            dt = time.time() - t0
        except UnsupportedArchitectureError as ex:
            rec(name, "compatibility", expect_unsupported, f"unsupported: {ex}")
            continue
        except Exception as ex:
            rec(name, "compatibility", False, f"{type(ex).__name__}: {ex}")
            continue
        rec(name, "compatibility", not expect_unsupported,
            "ran" if not expect_unsupported else "ran but should be unsupported")
        rec(name, "shape", tuple(m0.shape) == (P, P), f"{tuple(m0.shape)} vs {(P, P)}; {dt:.2f}s")
        rec(name, "finite", torch.isfinite(m0).all(), "")
        vals, counts = torch.unique(m0, return_counts=True)
        rec(name, "non_degenerate", m0.std() > 1e-8 and counts.max() < 0.99 * m0.numel(),
            f"std={float(m0.std()):.3g}, max share of one value={float(counts.max()) / m0.numel():.2f}")
        m0b = e.explain(xs[0], top).detach().float().cpu()
        rec(name, "deterministic", torch.allclose(m0, m0b, atol=1e-5, rtol=1e-4),
            f"max |diff|={float((m0 - m0b).abs().max()):.2e}")
        per_image = [m0]
        for x in xs[1:]:
            per_image.append(e.explain(x, top).detach().float().cpu())
        if len(per_image) > 1:
            rec(name, "input_sensitive", not torch.allclose(per_image[0], per_image[1]),
                f"ρ(img0, img1)={_spearman(per_image[0], per_image[1]):.3f}")
        m_other = e.explain(xs[0], other).detach().float().cpu()
        rho = _spearman(m0, m_other)
        if name in CLASS_AGNOSTIC:
            rec(name, "target", torch.allclose(m0, m_other), f"class-agnostic; ρ={rho:.3f}")
        elif name in PERTURBATION_BASED and not args.trained_head:
            # With an untrained head the class scores barely change under masking,
            # so RISE/LIME may legitimately be target-insensitive: advisory only.
            rec(name, "target", True,
                f"class-specific; ρ(top, other)={rho:.3f} — ADVISORY (untrained head; strict with --weights-dir)")
        else:
            rec(name, "target", rho < 0.999, f"class-specific; ρ(top, other)={rho:.3f}")
        if name == "integrated_gradients":
            e.explain(xs[0], top)              # gap must refer to THIS (image, target)
            with torch.no_grad():
                f = model(torch.stack([xs[0], torch.zeros_like(xs[0])]))[:, top]
            gap = abs(e.last_completeness_gap)
            df = abs(float(f[0] - f[1]))
            # 5% of |Δf|, with a 0.05-logit floor: a purely relative criterion is
            # ill-conditioned when f(x) ≈ f(0) (e.g. untrained heads).
            tol = 0.05 * max(df, 1.0)
            rec(name, "ig_completeness", gap <= tol,
                f"|gap|={gap:.3g} logits, |Δf|={df:.3g}, relative={gap / max(df, 1e-8):.2%}, tol={tol:.3g}")
        maps[name] = per_image
    return rows, maps


def save_grid(model_name: str, images: List[torch.Tensor], maps: Dict[str, list], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    names = list(maps)
    if not names:
        return
    fig, axes = plt.subplots(len(names) + 1, len(images),
                             figsize=(2.2 * len(images), 2.2 * (len(names) + 1)), squeeze=False)
    for j, x in enumerate(images):
        axes[0, j].imshow(x.permute(1, 2, 0).cpu().numpy())
        axes[0, j].set_title(f"image {j}", fontsize=8)
    for i, n in enumerate(names, start=1):
        for j, x in enumerate(images):
            m = maps[n][j]
            up = F.interpolate(m[None, None], size=(224, 224), mode="nearest")[0, 0]
            up = (up - up.min()) / (up.max() - up.min() + 1e-8)
            axes[i, j].imshow(x.permute(1, 2, 0).cpu().numpy())
            axes[i, j].imshow(up.numpy(), cmap="jet", alpha=0.5)
            if j == 0:
                axes[i, j].set_ylabel(n, fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(model_name, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None) -> int:
    all_explainers, grid_patch = _registries()
    p = argparse.ArgumentParser(description="Real-model explainer smoke test")
    p.add_argument("--models", nargs="+", default=list(MODEL_SPECS), choices=list(MODEL_SPECS))
    p.add_argument("--explainers", nargs="+", default=list(all_explainers), choices=list(all_explainers))
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--data-root", default=None)
    p.add_argument("--n-images", type=int, default=3)
    p.add_argument("--rise-masks", type=int, default=1000)
    p.add_argument("--lime-samples", type=int, default=300)
    p.add_argument("--ig-steps", type=int, default=400)
    p.add_argument("--num-classes", type=int, default=10,
                   help="Head size when no fine-tuned checkpoint is given (random head)")
    p.add_argument("--weights-dir", default=None)
    from utils.eval_datasets import DATASET_CLASSES, build_eval_loader
    p.add_argument("--dataset", choices=DATASET_CLASSES,
                   help="Dataset for smoke images and fine-tuned checkpoints")
    p.add_argument("--finetune-seed", type=int, help="Seed of the fine-tuned checkpoints")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/smoke")
    p.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)   # tests: small random models
    args = p.parse_args(argv)
    if args.weights_dir and not args.dataset:
        p.error("--weights-dir requires --dataset")
    if args.finetune_seed is not None and not args.weights_dir:
        p.error("--finetune-seed requires --weights-dir")
    if args.n_images < 1:
        p.error("--n-images must be at least 1")
    args.trained_head = bool(args.weights_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dataset and args.data_root and not args.images:
        loader = build_eval_loader(args.dataset, args.data_root, batch_size=args.n_images,
                                   max_samples=args.n_images, num_workers=0)
        if not len(loader.dataset):
            p.error(f"{args.dataset}: evaluation split is empty")
        images = list(next(iter(loader))[0])
    else:
        if args.dataset and not args.images:
            p.error("--dataset requires --data-root or explicit --images")
        images = load_images(args.images, args.data_root, args.n_images)
    os.makedirs(args.out, exist_ok=True)
    explainers = {n: all_explainers[n] for n in args.explainers}
    all_rows, summary = [], {}
    for name in args.models:
        torch.manual_seed(0)
        try:
            if args.weights_dir and args.dataset:
                import run_phase3 as rp
                model = rp.load_finetuned_model(name, args.dataset, args.weights_dir,
                                               seed=args.finetune_seed)
            else:
                model = load_model(name, num_classes=args.num_classes, pretrained=not args.tiny)
        except Exception as ex:
            all_rows.append({"model": name, "explainer": "(model)", "check": "load",
                             "pass": False, "detail": f"{type(ex).__name__}: {ex}"[:200]})
            continue
        rows, maps = smoke_model(name, model, explainers, grid_patch[name], images, device, args)
        all_rows += rows
        save_grid(name, images, maps, os.path.join(args.out, f"{name}_maps.png"))
        n_fail = sum(not r["pass"] for r in rows)
        summary[name] = {"checks": len(rows), "failed": n_fail}
        print(f"[{'PASS' if n_fail == 0 else 'FAIL'}] {name}: {len(rows)} checks, {n_fail} failed")
        for r in rows:
            if not r["pass"]:
                print(f"    BAD {r['explainer']:<22} {r['check']:<16} {r['detail']}")
        del model
    with open(os.path.join(args.out, "smoke_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "explainer", "check", "pass", "detail"])
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(args.out, "smoke_summary.json"), "w") as f:
        json.dump({"images": len(images), "device": str(device), "models": summary,
                   "settings": {"rise_masks": args.rise_masks, "lime_samples": args.lime_samples,
                                "ig_steps": args.ig_steps}}, f, indent=2)
    failed = any(not r["pass"] for r in all_rows)
    print(f"Wrote {args.out}/smoke_report.csv, smoke_summary.json and *_maps.png")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
