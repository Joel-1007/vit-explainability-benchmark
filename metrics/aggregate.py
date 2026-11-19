"""
aggregate.py — turn Phase 3 checkpoints into Phase 4 inputs.

Reads every ``<dataset>-<model>-<explainer>.pkl`` in a checkpoint directory
(written by Phase3Runner with a SampleMetricComputer) and produces:

* ``aggregated_results.csv`` — one row per sample:
  dataset, model, explainer, sample_id, label, pred, correct, <metric columns>
* ``combination_summary.csv`` — one row per combination: n_samples, accuracy,
  the mean of every metric, and L4 ``calibration_gap``
  = mean EGT (correct predictions) − mean EGT (incorrect predictions);
  95% bootstrap CIs and class-balanced (macro) means of the primary metrics.
* ``class_summary.csv`` — per class: n, accuracy and primary-metric means
  (class imbalance is reported, not hidden in a pooled mean).
* ``coverage.csv`` — every expected (dataset, model[, seed], explainer) with
  status ok / unsupported (+ reason) / missing. Unsupported or missing
  combinations never appear as scores.
* ``seed_summary.csv`` — when several fine-tuning seeds were evaluated:
  mean and std *across seeds* of each combination's primary-metric means.

Seeds: checkpoint model names "vit_b16@seed1" become model="vit_b16", seed=1;
unseeded runs have an empty seed.

Usage::

    python -m metrics.aggregate --checkpoint-dir results/phase3
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ID_COLUMNS = ["dataset", "model", "seed", "explainer", "sample_id"]
COMBO_KEYS = ["dataset", "model", "seed", "explainer"]
SEED_TAG = "@seed"
# Per-sample columns that are bookkeeping, not quality metrics.
NON_METRIC_COLUMNS = ID_COLUMNS + ["label", "pred", "correct"]


# Rows/columns that exist only for Phase 4 ablations; the main analyses
# (correlation, factor analysis, rankings) exclude them.
ABLATION_EXPLAINERS = {"raw_attention_patch", "raw_attention_last3", "raw_attention_all"}
ABLATION_MODELS = {"mae_b16"}


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_METRIC_COLUMNS]


_PRIMARY = ["miou", "pointing_game", "egt", "gini", "entropy_norm", "emr_90",
            "max_sensitivity", "model_randomisation", "head_permutation",
            "pgd_robustness"]


def primary_fidelity_k(df: pd.DataFrame) -> str | None:
    """Middle fidelity fraction present, e.g. '0.20' for k ∈ {0.10, 0.20, 0.50}."""
    ks = sorted({c.split("@", 1)[1] for c in df.columns
                 if c.startswith("comprehensiveness@") and "[" not in c}, key=float)
    return ks[len(ks) // 2] if ks else None


def primary_metric_columns(df: pd.DataFrame) -> list[str]:
    """
    One column per metric concept: miou (not each iou@τ), emr_90 (not 50/95),
    and fidelity at the middle k. Near-duplicate columns would otherwise
    dominate correlation-based analyses such as factor analysis.
    """
    cols = [c for c in _PRIMARY if c in df.columns]
    k = primary_fidelity_k(df)
    if k is not None:
        cols[6:6] = [f"sufficiency@{k}", f"comprehensiveness@{k}", f"log_odds_drop@{k}"]
    return [c for c in cols if c in df.columns]


def main_results(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ablation-only explainers, models and ``metric[mode]`` columns."""
    from .sample_metrics import is_ablation_column

    keep = ~df["explainer"].isin(ABLATION_EXPLAINERS) & ~df["model"].isin(ABLATION_MODELS)
    cols = [c for c in df.columns if not is_ablation_column(c)]
    return df.loc[keep, cols].dropna(axis=1, how="all").reset_index(drop=True)


def load_per_sample(checkpoint_dir: str | Path) -> pd.DataFrame:
    """One row per (combination, sample). Raises if a checkpoint has no metrics."""
    rows = []
    pkls = sorted(Path(checkpoint_dir).glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl checkpoints in {checkpoint_dir}")
    for pkl in pkls:
        with open(pkl, "rb") as f:
            res = pickle.load(f)
        if "metrics" not in res:
            raise ValueError(
                f"{pkl.name} has no per-sample metrics (written without a "
                "SampleMetricComputer). Delete it and re-run run_phase3.py."
            )
        ids = {k: res.get(k) for k in ("dataset", "model", "explainer")}
        if None in ids.values():
            ids = dict(zip(("dataset", "model", "explainer"), res["key"].split("-", 2)))
        ids["model"], ids["seed"] = split_seed(ids["model"])
        for i, (label, m) in enumerate(zip(res["labels"], res["metrics"])):
            rows.append({**ids, "sample_id": i, "label": label, **m})
    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype("Int64")
    return df


def split_seed(model: str):
    """'vit_b16@seed1' → ('vit_b16', 1); 'vit_b16' → ('vit_b16', None)."""
    if SEED_TAG in model:
        base, seed = model.split(SEED_TAG, 1)
        return base, int(seed)
    return model, None


def calibration_gap(group: pd.DataFrame) -> float:
    """L4: mean EGT on correct minus incorrect predictions (NaN if either is empty)."""
    if "egt" not in group:
        return math.nan
    ok, bad = group.loc[group["correct"] == 1, "egt"], group.loc[group["correct"] == 0, "egt"]
    if ok.dropna().empty or bad.dropna().empty:
        return math.nan
    return float(ok.mean() - bad.mean())


def bootstrap_ci(values, n_boot: int = 1000, level: float = 0.95, seed: int = 0):
    """
    Percentile bootstrap CI of the mean over samples (NaNs dropped).
    Returns (low, high); (nan, nan) for fewer than 2 values.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def summarise(per_sample: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """
    One row per (dataset, model, explainer): metric means, L4, and 95%
    percentile-bootstrap CIs (``<metric>_ci_low/_ci_high``) for the primary
    metrics. The CIs capture sample variation only; seed-to-seed variation of
    fine-tuning needs several checkpoints per (model, dataset).
    """
    keys = [k for k in COMBO_KEYS if k in per_sample]
    cols = metric_columns(per_sample)
    g = per_sample.groupby(keys, sort=True, dropna=False)
    summary = g[cols].mean()
    summary.insert(0, "accuracy", g["correct"].mean())
    summary.insert(0, "n_samples", g.size())
    summary["calibration_gap"] = g.apply(calibration_gap, include_groups=False)
    primary = primary_metric_columns(per_sample)
    for c in primary:
        cis = g[c].apply(lambda v: bootstrap_ci(v, n_boot=n_boot))
        summary[f"{c}_ci_low"] = cis.map(lambda t: t[0])
        summary[f"{c}_ci_high"] = cis.map(lambda t: t[1])
    # Class-balanced (macro) means: every class weighs the same, whatever its size
    per_class = per_sample.groupby(keys + ["label"], dropna=False)[["correct"] + primary].mean()
    macro = per_class.groupby(level=list(range(len(keys))), dropna=False).mean()
    summary["accuracy_macro"] = macro["correct"]
    summary["n_classes"] = per_class.groupby(level=list(range(len(keys))), dropna=False).size()
    for c in primary:
        summary[f"{c}_macro"] = macro[c]
    for c in ("max_sensitivity", "model_randomisation", "head_permutation"):
        if c in per_sample:
            summary[f"n_{c}"] = g[c].count()      # robustness runs on a subset
    return summary.reset_index()


def class_summary(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Per class: n, accuracy and primary-metric means (imbalance made visible)."""
    keys = [k for k in COMBO_KEYS if k in per_sample] + ["label"]
    g = per_sample.groupby(keys, sort=True, dropna=False)
    out = g[primary_metric_columns(per_sample)].mean()
    out.insert(0, "accuracy", g["correct"].mean())
    out.insert(0, "n", g.size())
    out = out.reset_index()
    total = out.groupby([k for k in keys if k != "label"], dropna=False)["n"].transform("sum")
    out.insert(out.columns.get_loc("n") + 1, "class_share", out["n"] / total)
    return out


def seed_summary(per_sample: pd.DataFrame) -> pd.DataFrame | None:
    """Across-seed mean/std of each combination's primary-metric means (None if <2 seeds)."""
    if "seed" not in per_sample or per_sample["seed"].nunique(dropna=True) < 2:
        return None
    cols = ["correct"] + primary_metric_columns(per_sample)
    per_seed = (per_sample.dropna(subset=["seed"])
                .groupby(["dataset", "model", "explainer", "seed"])[cols].mean())
    g = per_seed.groupby(level=["dataset", "model", "explainer"])
    out = g.mean().add_suffix("_mean")
    out = out.join(g.std(ddof=1).add_suffix("_seed_std"))
    out.insert(0, "n_seeds", g.size())
    return out.rename(columns=lambda c: c.replace("correct_", "accuracy_")).reset_index()


def coverage(checkpoint_dir: str | Path, per_sample: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Status of every expected combination: 'ok', 'unsupported' (with reason)
    or 'missing'. Expected = datasets × models × seeds × explainers from
    run_metadata.json, plus anything found on disk.
    """
    import json

    d = Path(checkpoint_dir)
    rows = {}
    meta_path = d / "run_metadata.json"
    if meta_path.exists():
        a = json.loads(meta_path.read_text()).get("args", {})
        seeds = a.get("finetune_seeds") or [None]
        for ds in a.get("datasets", []):
            for m in a.get("models", []):
                for sd in seeds:
                    for e in a.get("explainers", []):
                        rows[(ds, m, sd, e)] = {"status": "missing", "reason": ""}
    for pkl in d.glob("*.pkl"):
        ds, m, e = pkl.stem.split("-", 2)
        m, sd = split_seed(m)
        rows[(ds, m, sd, e)] = {"status": "ok", "reason": ""}
    for marker in d.glob("*.unsupported.json"):
        rec = json.loads(marker.read_text())
        m, sd = split_seed(rec["model"])
        rows[(rec["dataset"], m, sd, rec["explainer"])] = {
            "status": "unsupported", "reason": rec.get("reason", "")}
    df = pd.DataFrame([{"dataset": k[0], "model": k[1], "seed": k[2], "explainer": k[3], **v}
                       for k, v in sorted(rows.items(), key=lambda kv: tuple(str(x) for x in kv[0]))])
    if not df.empty:
        df["seed"] = df["seed"].astype("Int64")
    return df


def aggregate(checkpoint_dir: str | Path) -> Tuple[Path, Path]:
    """
    Write aggregated_results.csv, combination_summary.csv, class_summary.csv,
    coverage.csv and (with ≥2 seeds) seed_summary.csv; return the first two paths.
    """
    d = Path(checkpoint_dir)
    per_sample = load_per_sample(d)
    p1, p2 = d / "aggregated_results.csv", d / "combination_summary.csv"
    per_sample.to_csv(p1, index=False)
    summarise(per_sample).to_csv(p2, index=False)
    class_summary(per_sample).to_csv(d / "class_summary.csv", index=False)
    coverage(d).to_csv(d / "coverage.csv", index=False)
    seeds = seed_summary(per_sample)
    if seeds is not None:
        seeds.to_csv(d / "seed_summary.csv", index=False)
    return p1, p2


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-dir", default="results/phase3")
    args = p.parse_args(argv)
    p1, p2 = aggregate(args.checkpoint_dir)
    print(f"Wrote {p1}\nWrote {p2}")


if __name__ == "__main__":
    main()
