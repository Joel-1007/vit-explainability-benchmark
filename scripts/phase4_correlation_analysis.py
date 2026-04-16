"""
Phase 4, Task 4.1: Inter-Metric Correlation Analysis
Computes Spearman rank correlation between every pair of metrics across all method x model x dataset combinations.
Visualises an inter-metric correlation heatmap. Also includes factor analysis scaffolding.
"""

import os
import argparse
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
    metric_cols = [c for c in results_df.columns
                   if c not in ["explainer", "model", "dataset", "sample_id"]]
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
    Stub for running factor analysis (PCA) to find the latent structure of explanation quality.
    Generates dummy factor analysis output for scaffolding.
    """
    print(f"Generating dummy Factor Analysis variance breakdown to {output_path}.")
    
    latent_data = {
        "Factor": ["Factor 1 (Fidelity/Robustness)", "Factor 2 (Localization)", "Factor 3 (Complexity)"],
        "Eigenvalue": [3.1, 1.8, 1.1],
        "Variance_Explained_Pct": [45.2, 26.5, 16.3],
        "Cumulative_Variance_Pct": [45.2, 71.7, 88.0]
    }
    pd.DataFrame(latent_data).to_csv(output_path, index=False)
    print(f"Factor analysis summary saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run Task 4.1: Inter-Metric Correlation Analysis")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to aggregated results CSV from Phase 3.")
    parser.add_argument("--output_dir", type=str, default="../results/phase4", help="Directory to save phase 4 outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results_csv}...")
    # df = pd.read_csv(args.results_csv)
    # Placeholder for actual data loading logic
    
    # Generate dummy data for illustration
    print("WARNING: Using dummy data for scaffolding.")
    dummy_cols = ["F1_Sufficiency", "F2_Comprehensiveness", "L1_mIoU", "L2_PG", "R1_MaxSens", "C1_Gini"]
    
    # Generate somewhat correlated dummy data
    N = 300
    base_signal = np.random.randn(N)
    
    df = pd.DataFrame()
    df["F1_Sufficiency"] = base_signal + np.random.randn(N) * 0.5
    df["F2_Comprehensiveness"] = base_signal + np.random.randn(N) * 0.6  # Correlated with F1
    df["L1_mIoU"] = np.random.randn(N) * 1.0  # Orthogonal
    df["L2_PG"] = df["L1_mIoU"] + np.random.randn(N) * 0.4  # Correlated with L1
    df["R1_MaxSens"] = np.random.randn(N) * 1.5 # Orthogonal
    df["C1_Gini"] = np.random.randn(N) * 0.8 # Orthogonal

    # Normalize roughly to metric ranges 0-1
    for col in dummy_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df["explainer"] = np.random.choice(["GradCAM", "Rollout", "Chefer LRP"], N)
    df["model"] = "ViT-B"
    df["dataset"] = np.random.choice(["ImageNet-S-50", "CUB-200", "PASCAL VOC", "NIH ChestX-ray14"], N)
    df["sample_id"] = range(N)

    print("Computing correlations...")
    corr_matrix = compute_metric_correlations(df)
    
    csv_out = os.path.join(args.output_dir, "inter_metric_correlations.csv")
    corr_matrix.to_csv(csv_out)
    print(f"Correlation matrix saved to {csv_out}")

    heatmap_out = os.path.join(args.output_dir, "inter_metric_heatmap.pdf")
    plot_correlation_heatmap(corr_matrix, heatmap_out)

    fa_out = os.path.join(args.output_dir, "factor_analysis_results.csv")
    run_factor_analysis(df, fa_out)

if __name__ == "__main__":
    main()
