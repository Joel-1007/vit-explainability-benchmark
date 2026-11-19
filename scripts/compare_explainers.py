"""
Pairwise explainer comparisons with paired statistics.

For every (dataset, model[, seed]) and primary metric, each pair of explainers
is compared on the SAME images (inner join on sample_id). Reported per pair:
n, means, mean difference with a 95% bootstrap CI, Cohen's d (pooled, as in
the implementation guide) and paired d_z, Wilcoxon signed-rank p, and
Holm-adjusted p-values — within each (dataset, model, seed, metric) family
(``p_holm_family``) and over the whole table (``p_holm_global``).
``better`` names the explainer favoured by the metric's direction when the
family-adjusted p < alpha.

Main-matrix explainers/models only (ablation variants excluded).

Usage:
    python scripts/compare_explainers.py \\
        --results_csv results/phase3/aggregated_results.csv --output_dir results/phase4
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.aggregate import main_results, primary_metric_columns  # noqa: E402
from metrics.sample_metrics import metric_direction                 # noqa: E402
from metrics.stats import (cohens_d, holm, paired_diff_ci,          # noqa: E402
                           paired_dz, wilcoxon_p)


def compare_explainers(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    df = main_results(df)
    group_keys = [k for k in ("dataset", "model", "seed") if k in df and df[k].notna().any()]
    metrics = [m for m in primary_metric_columns(df) if metric_direction(m) is not None]
    rows = []
    for gkey, g in df.groupby(group_keys, sort=True, dropna=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        for metric in metrics:
            wide = g.pivot_table(index="sample_id", columns="explainer", values=metric)
            family = []
            for a, b in itertools.combinations(sorted(wide.columns), 2):
                pair = wide[[a, b]].dropna()
                if len(pair) < 2:
                    continue
                xa, xb = pair[a].to_numpy(), pair[b].to_numpy()
                lo, hi = paired_diff_ci(xa, xb)
                family.append({
                    **dict(zip(group_keys, gkey)), "metric": metric,
                    "explainer_A": a, "explainer_B": b, "n": len(pair),
                    "mean_A": xa.mean(), "mean_B": xb.mean(),
                    "mean_diff": (xa - xb).mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                    "cohens_d": cohens_d(xa, xb), "paired_dz": paired_dz(xa, xb),
                    "wilcoxon_p": wilcoxon_p(xa, xb),
                })
            if family:
                adj = holm([r["wilcoxon_p"] for r in family])
                for r, p in zip(family, adj):
                    r["p_holm_family"] = p
                rows.extend(family)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm_global"] = holm(out["wilcoxon_p"].to_numpy())

    def better(r):
        if not (r.p_holm_family < alpha) or r.mean_diff == 0:
            return ""
        higher = metric_direction(r.metric) > 0
        return r.explainer_A if (r.mean_diff > 0) == higher else r.explainer_B

    out["better"] = out.apply(better, axis=1)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Pairwise explainer comparisons (paired tests, Holm)")
    p.add_argument("--results_csv", required=True)
    p.add_argument("--output_dir", default="results/phase4")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)
    if not os.path.isfile(args.results_csv):
        raise SystemExit(f"ERROR: results CSV not found: {args.results_csv}")
    os.makedirs(args.output_dir, exist_ok=True)
    out = compare_explainers(pd.read_csv(args.results_csv), alpha=args.alpha)
    path = os.path.join(args.output_dir, "explainer_pairwise.csv")
    out.to_csv(path, index=False)
    n_sig = int((out["better"] != "").sum()) if not out.empty else 0
    print(f"Saved {path}: {len(out)} comparisons, {n_sig} significant after Holm (family-wise).")


if __name__ == "__main__":
    main()
