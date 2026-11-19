"""
Phase 4, Task 4.3: Ablation Studies A1–A4.

Each ablation compares variants that differ in exactly one variable, on the
SAME samples (paired by dataset, model, explainer and sample_id), and
reports per (dataset, model[, explainer]) and metric:
mean and std of each variant, Cohen's d, Wilcoxon signed-rank p-value and a
Bonferroni-corrected p-value (over all tests within the ablation).

A1 Token resolution  : raw_attention (CLS → patch) vs raw_attention_patch
                       (patch → patch), localization metrics.
A2 Layer depth       : raw_attention (last layer) vs raw_attention_last3,
                       raw_attention_all and rollout; fidelity + localization.
A3 Masking strategy  : fidelity with mean (default) vs zero vs blur baselines,
                       plus a masking sensitivity score = mean over samples of
                       the std of the metric across the three strategies.
A4 Pre-training      : vit_b16 (supervised) vs mae_b16 (MAE); all primary metrics.

Inputs come from a Phase 3 run with ``run_phase3.py --ablations`` (A1–A3)
and ``--models vit_b16 mae_b16`` (A4). Ablations whose inputs are missing
are skipped with a message saying what to run.

Usage:
    python scripts/phase4_ablations.py --results_csv results/phase3/aggregated_results.csv \\
        --output_dir results/phase4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as ss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.aggregate import primary_fidelity_k, primary_metric_columns  # noqa: E402

LOCALIZATION = ["miou", "pointing_game", "egt"]


def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Cohen's d effect size between two groups (implementation guide §4.3):
    d = (mu_A - mu_B) / sqrt((var_A + var_B) / 2)
    """
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt((var_a + var_b) / 2)
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return 0.0 if mean_a == mean_b else float(np.sign(mean_a - mean_b) * np.inf)
    return float((mean_a - mean_b) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    abs_d = abs(d)
    if not np.isfinite(abs_d):
        return "Large" if np.isinf(abs_d) else "n/a"
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    return "Large"


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if len(diff) < 2:
        return np.nan
    if np.allclose(diff, 0):
        return 1.0
    return float(ss.wilcoxon(a, b, zero_method="wilcox").pvalue)


def paired_compare(
    df: pd.DataFrame, ablation: str, variable: str,
    a_label: str, a_mask: pd.Series, b_label: str, b_mask: pd.Series,
    metrics: Iterable[str], pair_on: List[str], group_by: List[str],
    b_metric_suffix: str = "",
) -> pd.DataFrame:
    """Compare variant A vs B on samples present in both (inner join on ``pair_on``)."""
    a = df.loc[a_mask]
    b = df.loc[b_mask]
    merged = a.merge(b, on=pair_on, suffixes=("_A", "_B"))
    rows = []
    for metric in metrics:
        col_a, col_b = metric, metric + b_metric_suffix
        ca = f"{col_a}_A" if f"{col_a}_A" in merged else col_a
        cb = f"{col_b}_B" if f"{col_b}_B" in merged else col_b
        if ca not in merged or cb not in merged:
            continue
        for key, g in merged.groupby(group_by, sort=True):
            g = g[[ca, cb]].dropna()
            if len(g) < 2:
                continue
            xa, xb = g[ca].to_numpy(float), g[cb].to_numpy(float)
            d = compute_cohens_d(xa, xb)
            rows.append({
                "ablation": ablation, "variable": variable,
                **dict(zip(group_by, key if isinstance(key, tuple) else (key,))),
                "metric": metric, "variant_A": a_label, "variant_B": b_label,
                "n": len(g), "mean_A": xa.mean(), "std_A": xa.std(ddof=1),
                "mean_B": xb.mean(), "std_B": xb.std(ddof=1),
                "cohens_d": d, "effect_size": interpret_cohens_d(d),
                "wilcoxon_p": _wilcoxon(xa, xb),
            })
    return pd.DataFrame(rows)


def _bonferroni(table: pd.DataFrame) -> pd.DataFrame:
    if not table.empty:
        table["p_bonferroni"] = (table["wilcoxon_p"] * len(table)).clip(upper=1.0)
    return table


def _skip(name: str, why: str) -> None:
    print(f"SKIPPED {name}: {why}")


def _fidelity_metrics(df: pd.DataFrame) -> List[str]:
    k = primary_fidelity_k(df)
    return [f"{m}@{k}" for m in ("sufficiency", "comprehensiveness", "log_odds_drop")] if k else []


def run_ablation_A1(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """A1 Token Resolution: CLS-token vs patch-token attention (last layer)."""
    need = {"raw_attention", "raw_attention_patch"}
    if not need <= set(df.explainer):
        return _skip("A1", f"needs explainers {sorted(need)} — run run_phase3.py --ablations")
    return paired_compare(
        df, "A1", "token resolution",
        "CLS token", df.explainer == "raw_attention",
        "patch tokens", df.explainer == "raw_attention_patch",
        LOCALIZATION, pair_on=["dataset", "model", "sample_id"],
        group_by=["dataset", "model"],
    )


def run_ablation_A2(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """A2 Layer Depth: last layer vs last 3 / all layers / rollout."""
    variants = [("raw_attention_last3", "last 3 layers"),
                ("raw_attention_all", "all layers (mean)"),
                ("rollout", "all layers (rollout)")]
    present = [(e, lbl) for e, lbl in variants if e in set(df.explainer)]
    if "raw_attention" not in set(df.explainer) or not present:
        return _skip("A2", "needs raw_attention and at least one of "
                     f"{[e for e, _ in variants]} — run run_phase3.py --ablations")
    metrics = _fidelity_metrics(df) + LOCALIZATION
    return pd.concat([
        paired_compare(
            df, "A2", "layer depth",
            label, df.explainer == exp,
            "last layer", df.explainer == "raw_attention",
            metrics, pair_on=["dataset", "model", "sample_id"],
            group_by=["dataset", "model"],
        ) for exp, label in present
    ], ignore_index=True)


def run_ablation_A3(df: pd.DataFrame, output_dir: str) -> Optional[pd.DataFrame]:
    """A3 Masking Strategy: mean (default) vs zero vs blur fidelity baselines."""
    modes = [m for m in ("zero", "blur") if any(c.endswith(f"[{m}]") for c in df.columns)]
    if not modes:
        return _skip("A3", "no fidelity columns with [zero]/[blur] — run run_phase3.py "
                     "--ablations (or --fidelity-ablation-modes zero blur)")
    metrics = sorted(c for c in df.columns
                     if c.startswith(("sufficiency@", "comprehensiveness@", "log_odds_drop@"))
                     and "[" not in c)
    # Same samples, different column: no merge needed
    table = pd.concat([_compare_columns(df, metrics, mode) for mode in modes],
                      ignore_index=True)

    # Masking sensitivity: per-sample std across strategies, averaged
    sens_rows = []
    for metric in metrics:
        cols = [metric] + [f"{metric}[{m}]" for m in modes]
        if not all(c in df for c in cols):
            continue
        per_sample = df[cols].std(axis=1, ddof=0)
        for key, g in per_sample.groupby([df.dataset, df.model, df.explainer]):
            sens_rows.append({
                "dataset": key[0], "model": key[1], "explainer": key[2], "metric": metric,
                "strategies": ",".join(["mean", *modes]), "n": int(g.notna().sum()),
                "masking_sensitivity": float(g.mean()),
            })
    sens = pd.DataFrame(sens_rows)
    path = os.path.join(output_dir, "ablation_A3_masking_sensitivity.csv")
    sens.to_csv(path, index=False)
    print(f"Saved: {path}")
    return table


def _compare_columns(df: pd.DataFrame, metrics: List[str], mode: str) -> pd.DataFrame:
    """A3 helper: same rows, metric[mode] (A) vs metric under mean masking (B)."""
    rows = []
    for metric in metrics:
        alt = f"{metric}[{mode}]"
        if alt not in df:
            continue
        for key, g in df.groupby(["dataset", "model", "explainer"], sort=True):
            g = g[[alt, metric]].dropna()
            if len(g) < 2:
                continue
            xa, xb = g[alt].to_numpy(float), g[metric].to_numpy(float)
            d = compute_cohens_d(xa, xb)
            rows.append({
                "ablation": "A3", "variable": "masking strategy",
                "dataset": key[0], "model": key[1], "explainer": key[2],
                "metric": metric, "variant_A": f"{mode} mask", "variant_B": "mean mask",
                "n": len(g), "mean_A": xa.mean(), "std_A": xa.std(ddof=1),
                "mean_B": xb.mean(), "std_B": xb.std(ddof=1),
                "cohens_d": d, "effect_size": interpret_cohens_d(d),
                "wilcoxon_p": _wilcoxon(xa, xb),
            })
    return pd.DataFrame(rows)


def run_ablation_A4(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """A4 Pre-training Objective: supervised ViT-B/16 vs MAE ViT-B/16."""
    if not {"vit_b16", "mae_b16"} <= set(df.model):
        return _skip("A4", "needs models vit_b16 and mae_b16 — run run_phase3.py "
                     "--models vit_b16 mae_b16 (MAE needs fine-tuned weights)")
    return paired_compare(
        df, "A4", "pre-training objective",
        "MAE", df.model == "mae_b16",
        "supervised", df.model == "vit_b16",
        primary_metric_columns(df), pair_on=["dataset", "explainer", "sample_id"],
        group_by=["dataset", "explainer"],
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run Task 4.3: Ablation Studies A1–A4")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="aggregated_results.csv from Phase 3.")
    parser.add_argument("--output_dir", type=str, default="../results/phase4",
                        help="Directory to save phase 4 outputs.")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.results_csv):
        raise SystemExit(f"ERROR: results CSV not found: {args.results_csv}")
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.results_csv)

    tables = []
    for name, fn in [("A1", lambda: run_ablation_A1(df)), ("A2", lambda: run_ablation_A2(df)),
                     ("A3", lambda: run_ablation_A3(df, args.output_dir)),
                     ("A4", lambda: run_ablation_A4(df))]:
        t = fn()
        if t is None:
            continue
        if t.empty:
            _skip(name, "inputs present but fewer than 2 paired samples per group")
            continue
        t = _bonferroni(t)
        path = os.path.join(args.output_dir, f"ablation_{name}.csv")
        t.to_csv(path, index=False)
        print(f"Saved: {path} ({len(t)} comparisons)")
        tables.append(t)

    if not tables:
        raise SystemExit("ERROR: no ablation could be computed from this results file.")
    summary = pd.concat(tables, ignore_index=True)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Overall summary saved to {summary_path}")


if __name__ == "__main__":
    main()
