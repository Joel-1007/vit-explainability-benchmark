"""
Phase 4, Task 4.1: Inter-Metric Correlation Analysis
Computes Spearman rank correlation between every pair of metrics across all method x model x dataset combinations.
Visualises an inter-metric correlation heatmap. Also includes factor analysis scaffolding.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.aggregate import metric_columns  # noqa: E402
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_metric_correlations(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    results_df: DataFrame with columns = metric names,
                rows = (explainer, model, dataset, sample) combinations.
    Returns a correlation matrix.
    """
    # Excludes ids and bookkeeping (label, pred, correct)
    metric_cols = metric_columns(results_df)
    corr_matrix = pd.DataFrame(index=metric_cols, columns=metric_cols, dtype=float)
    for m1 in metric_cols:
        for m2 in metric_cols:
            # Drop NaNs just in case some combinations weren't computable
            valid_idx = results_df[m1].notna() & results_df[m2].notna()
            if valid_idx.sum() > 0:
                rho, _ = ss.spearmanr(results_df.loc[valid_idx, m1], results_df.loc[valid_idx, m2])
                corr_matrix.loc[m1, m2] = rho
            else:
                corr_matrix.loc[m1, m2] = np.nan
    return corr_matrix

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """
    Plots and saves a heatmap of the correlation matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1)
    plt.title("Inter-Metric Spearman Rank Correlation")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Heatmap saved to {output_path}")

def run_factor_analysis(results_df: pd.DataFrame, output_path: str):
    """
    Factor analysis (PCA) of the latent structure of explanation quality.

    Not implemented yet. This used to write hard-coded eigenvalues
    (3.1 / 1.8 / 1.1); that was removed so placeholder numbers cannot reach
    the paper.
    """
    raise NotImplementedError(
        "Factor analysis is not implemented. Implement run_factor_analysis() "
        "in scripts/phase4_correlation_analysis.py before reporting it."
    )

def main():
    parser = argparse.ArgumentParser(description="Run Task 4.1: Inter-Metric Correlation Analysis")
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

    print("Computing correlations...")
    corr_matrix = compute_metric_correlations(df)
    
    csv_out = os.path.join(args.output_dir, "inter_metric_correlations.csv")
    corr_matrix.to_csv(csv_out)
    print(f"Correlation matrix saved to {csv_out}")

    heatmap_out = os.path.join(args.output_dir, "inter_metric_heatmap.pdf")
    plot_correlation_heatmap(corr_matrix, heatmap_out)

    fa_out = os.path.join(args.output_dir, "factor_analysis_results.csv")
    try:
        run_factor_analysis(df, fa_out)
    except NotImplementedError as e:
        print(f"WARNING: factor analysis SKIPPED — {e}")

if __name__ == "__main__":
    main()
