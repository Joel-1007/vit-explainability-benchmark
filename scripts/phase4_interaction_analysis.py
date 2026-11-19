"""
Phase 4, Task 4.2: Task-Metric Interaction Analysis
Answers: Does the ranking of explanation methods change across tasks (datasets)?
Computes Kendall tau concordance between rankings across datasets.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.aggregate import metric_columns  # noqa: E402
from metrics.sample_metrics import metric_direction  # noqa: E402
import pandas as pd
import numpy as np
import scipy.stats as ss

def compute_rankings(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each dataset d and metric m, ranks explainers by their mean score.
    Returns: DataFrame with columns [dataset, metric, explainer, rank].
    """
    # Group by dataset, explainer to get mean metric scores
    metric_cols = [c for c in metric_columns(results_df) if metric_direction(c) is not None]
    skipped = sorted(set(metric_columns(results_df)) - set(metric_cols))
    if skipped:
        print(f"Skipping metrics with unknown direction: {skipped}")
    mean_scores = results_df.groupby(["dataset", "explainer"])[metric_cols].mean().reset_index()
    
    rankings = []
    for dataset in mean_scores["dataset"].unique():
        ds_data = mean_scores[mean_scores["dataset"] == dataset]
        for metric in metric_cols:
            # Rank 1 = best: descending for higher-is-better, ascending otherwise
            higher_better = metric_direction(metric) > 0
            ds_data_sorted = ds_data.sort_values(by=metric, ascending=not higher_better).reset_index(drop=True)
            for idx, row in ds_data_sorted.iterrows():
                rankings.append({
                    "dataset": dataset,
                    "metric": metric,
                    "explainer": row["explainer"],
                    "rank": idx + 1
                })
    return pd.DataFrame(rankings)

def compute_concordance(rankings_df: pd.DataFrame, output_path: str):
    """
    Computes Kendall tau for explainer rankings between pairs of datasets for each metric.
    """
    results = []
    datasets = rankings_df["dataset"].unique()
    metrics = rankings_df["metric"].unique()
    
    for metric in metrics:
        subset = rankings_df[rankings_df["metric"] == metric]
        pivot = subset.pivot(index="explainer", columns="dataset", values="rank")
        
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                ds1, ds2 = datasets[i], datasets[j]
                if ds1 in pivot.columns and ds2 in pivot.columns:
                    # Drop NaNs just in case
                    valid = pivot[[ds1, ds2]].dropna()
                    if len(valid) >= 2:
                        tau, p_val = ss.kendalltau(valid[ds1], valid[ds2])
                        results.append({
                            "metric": metric,
                            "dataset_1": ds1,
                            "dataset_2": ds2,
                            "kendall_tau": tau,
                            "p_value": p_val
                        })
    
    scores = pd.DataFrame(results)
    scores.to_csv(output_path, index=False)
    print(f"Concordance analysis saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Task 4.2: Task-Metric Interaction Analysis")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to aggregated results CSV from Phase 3.")
    parser.add_argument("--output_dir", type=str, default="../results/phase4", help="Directory to save phase 4 outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.results_csv):
        raise SystemExit(
            f"ERROR: results CSV not found: {args.results_csv}\n"
            "Phase 4 needs real Phase 3 results; no dummy data is generated."
        )
    print(f"Loading results from {args.results_csv}...")
    df = pd.read_csv(args.results_csv)

    print("Computing rankings...")
    rankings_df = compute_rankings(df)
    rankings_out = os.path.join(args.output_dir, "explainer_rankings.csv")
    rankings_df.to_csv(rankings_out, index=False)
    
    print("Computing concordance...")
    concordance_out = os.path.join(args.output_dir, "dataset_concordance.csv")
    compute_concordance(rankings_df, concordance_out)

if __name__ == "__main__":
    main()
