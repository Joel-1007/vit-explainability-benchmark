"""
Phase 4, Task 4.1: Inter-Metric Correlation Analysis
Computes Spearman rank correlation between every pair of metrics across all method x model x dataset combinations.
Visualises an inter-metric correlation heatmap, and runs a factor analysis
(PCA on the Spearman correlation matrix + varimax rotation) of the metric space.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.aggregate import main_results, metric_columns, primary_metric_columns  # noqa: E402
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

def varimax(loadings: np.ndarray, normalize: bool = True,
            max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """
    Varimax rotation (Kaiser, 1958) of a (variables x factors) loading matrix.

    Pairwise algorithm: each pair of factors is rotated by the closed-form
    angle that maximises the varimax criterion, sweeping until no angle
    exceeds ``tol``. With ``normalize`` (Kaiser normalisation, as in R's
    ``varimax``) rows are scaled to unit communality during rotation.
    """
    L = np.array(loadings, dtype=float)
    p, k = L.shape
    if k < 2:
        return L
    h = np.sqrt((L ** 2).sum(axis=1, keepdims=True)) if normalize else np.ones((p, 1))
    h[h == 0] = 1.0
    L = L / h
    for _ in range(max_iter):
        max_angle = 0.0
        for i in range(k - 1):
            for j in range(i + 1, k):
                x, y = L[:, i], L[:, j]
                u, v = x ** 2 - y ** 2, 2 * x * y
                A, B = u.sum(), v.sum()
                C, D = (u ** 2 - v ** 2).sum(), 2 * (u * v).sum()
                phi = 0.25 * np.arctan2(D - 2 * A * B / p, C - (A ** 2 - B ** 2) / p)
                if abs(phi) > tol:
                    c, s_ = np.cos(phi), np.sin(phi)
                    L[:, i], L[:, j] = x * c + y * s_, -x * s_ + y * c
                max_angle = max(max_angle, abs(phi))
        if max_angle < tol:
            break
    return L * h


def run_factor_analysis(results_df: pd.DataFrame, output_path: str,
                        level: str = "sample", min_coverage: float = 0.9) -> dict:
    """
    PCA on the Spearman correlation matrix of the primary metrics, with
    varimax rotation of the retained factors (Kaiser criterion: eigenvalue > 1,
    at least 2).

    level="sample"      : one row per sample. Metrics present for fewer than
                          ``min_coverage`` of samples (robustness runs on a
                          subset) are dropped, then incomplete rows.
    level="combination" : one row per (dataset, model, explainer) mean, which
                          includes the robustness metrics.

    Writes ``output_path`` (variance per component) and
    ``factor_loadings_varimax.csv`` next to it. Returns a summary dict.
    """
    cols = primary_metric_columns(results_df)
    if level == "combination":
        data = results_df.groupby(["dataset", "model", "explainer"])[cols].mean()
    elif level == "sample":
        data = results_df[cols]
    else:
        raise ValueError(f"level must be 'sample' or 'combination', got {level!r}")

    coverage = data.notna().mean()
    dropped = sorted(coverage[coverage < min_coverage].index)
    data = data.drop(columns=dropped).dropna()
    constant = sorted(c for c in data.columns if data[c].nunique() <= 1)
    data = data.drop(columns=constant)
    if data.shape[1] < 2 or len(data) < data.shape[1] + 1:
        raise ValueError(
            f"Not enough data for factor analysis: {len(data)} rows x {data.shape[1]} "
            f"metrics (dropped sparse {dropped}, constant {constant})."
        )

    corr = data.rank().corr().to_numpy()                 # Spearman
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]
    var_pct = 100 * eigval / eigval.sum()

    n_keep = max(2, int((eigval > 1.0).sum()))
    loadings = varimax(eigvec[:, :n_keep] * np.sqrt(eigval[:n_keep]))
    # Order rotated factors by explained variance; make the dominant loading positive
    ss = (loadings ** 2).sum(axis=0)
    idx = np.argsort(ss)[::-1]
    loadings, ss = loadings[:, idx], ss[idx]
    loadings *= np.sign(loadings[np.abs(loadings).argmax(axis=0), range(n_keep)])

    pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(eigval))],
        "Eigenvalue": eigval,
        "Variance_Explained_Pct": var_pct,
        "Cumulative_Variance_Pct": np.cumsum(var_pct),
        "Retained": [i < n_keep for i in range(len(eigval))],
    }).to_csv(output_path, index=False)

    factors = [f"F{i + 1}" for i in range(n_keep)]
    load_df = pd.DataFrame(loadings, index=data.columns, columns=factors)
    load_df["communality"] = (loadings ** 2).sum(axis=1)
    load_df.loc["rotated_variance_pct"] = list(100 * ss / len(data.columns)) + [np.nan]
    loadings_path = os.path.join(os.path.dirname(output_path), "factor_loadings_varimax.csv")
    load_df.to_csv(loadings_path)

    top3 = float(np.cumsum(var_pct)[min(2, len(var_pct) - 1)])
    print(f"Factor analysis ({level} level, {len(data)} rows, {data.shape[1]} metrics): "
          f"{n_keep} factors retained; top-3 components explain {top3:.1f}% "
          f"({'>' if top3 > 80 else '<='} 80% guide threshold).")
    if dropped:
        print(f"  Dropped sparse metrics (<{min_coverage:.0%} coverage): {dropped}")
    if constant:
        print(f"  Dropped constant metrics: {constant}")
    print(f"  Saved {output_path} and {loadings_path}")
    return {"n_factors": n_keep, "top3_pct": top3, "metrics": list(data.columns),
            "dropped": dropped + constant, "loadings": load_df}

def main():
    parser = argparse.ArgumentParser(description="Run Task 4.1: Inter-Metric Correlation Analysis")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to aggregated results CSV from Phase 3.")
    parser.add_argument("--output_dir", type=str, default="../results/phase4", help="Directory to save phase 4 outputs.")
    parser.add_argument("--fa_level", choices=["sample", "combination"], default="sample",
                        help="Factor analysis rows: per sample (default) or per combination mean "
                             "(includes the subset-only robustness metrics).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.results_csv):
        raise SystemExit(
            f"ERROR: results CSV not found: {args.results_csv}\n"
            "Phase 4 needs real Phase 3 results; no dummy data is generated."
        )
    print(f"Loading results from {args.results_csv}...")
    df = main_results(pd.read_csv(args.results_csv))   # ablation rows/columns excluded

    print("Computing correlations...")
    corr_matrix = compute_metric_correlations(df)
    
    csv_out = os.path.join(args.output_dir, "inter_metric_correlations.csv")
    corr_matrix.to_csv(csv_out)
    print(f"Correlation matrix saved to {csv_out}")

    heatmap_out = os.path.join(args.output_dir, "inter_metric_heatmap.pdf")
    plot_correlation_heatmap(corr_matrix, heatmap_out)

    fa_out = os.path.join(args.output_dir, "factor_analysis_results.csv")
    run_factor_analysis(df, fa_out, level=args.fa_level)

if __name__ == "__main__":
    main()
