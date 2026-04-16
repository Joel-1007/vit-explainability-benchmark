"""
Phase 4, Task 4.3: Ablation Studies
This script provides scaffolding for the 4 key ablations (A1-A4).
It includes the Cohen's d effect size calculation.
"""

import os
import argparse
import numpy as np
import pandas as pd

def compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Computes Cohen's d effect size between two groups.
    d = (mu_A - mu_B) / sqrt((var_A + var_B) / 2)
    """
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    
    pooled_std = np.sqrt((var_a + var_b) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std

def interpret_cohens_d(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def _run_mock_ablation(name, variant_a, variant_b, mean_a, mean_b, std_a, std_b, output_dir):
    N = 200
    group_a = np.random.normal(mean_a, std_a, N)
    group_b = np.random.normal(mean_b, std_b, N)
    d = compute_cohens_d(group_a, group_b)
    
    df = pd.DataFrame([{
        "Ablation": name,
        "Variant_A": variant_a,
        "Variant_B": variant_b,
        "Mean_A": mean_a,
        "Mean_B": mean_b,
        "Cohens_d": d,
        "Effect_Size": interpret_cohens_d(d)
    }])
    out_file = os.path.join(output_dir, f"ablation_{name.split(':')[0].replace(' ', '')}.csv")
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    return df

def run_ablation_A1(output_dir: str) -> pd.DataFrame:
    """Ablation A1: Token Resolution"""
    print("Running Ablation A1: Token Resolution")
    return _run_mock_ablation("A1", "Raw Patch", "CLS Token", 0.65, 0.40, 0.1, 0.15, output_dir)

def run_ablation_A2(output_dir: str) -> pd.DataFrame:
    """Ablation A2: Layer Depth"""
    print("Running Ablation A2: Layer Depth")
    return _run_mock_ablation("A2", "Rollout All", "Last Layer", 0.70, 0.68, 0.1, 0.12, output_dir)

def run_ablation_A3(output_dir: str) -> pd.DataFrame:
    """Ablation A3: Masking Strategy"""
    print("Running Ablation A3: Masking Strategy")
    return _run_mock_ablation("A3", "Mean Mask", "Zero Mask", 0.85, 0.82, 0.05, 0.06, output_dir)

def run_ablation_A4(output_dir: str) -> pd.DataFrame:
    """Ablation A4: Pre-training Objective"""
    print("Running Ablation A4: Pre-training Objective")
    return _run_mock_ablation("A4", "DINO Self-distill", "Supervised", 0.78, 0.60, 0.15, 0.20, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Run Task 4.3: Ablation Studies")
    parser.add_argument("--output_dir", type=str, default="../results/phase4", help="Directory to save phase 4 outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("This script is currently scaffolded. Update data ingestion logic within individual ablation functions.")

    dfs = []
    dfs.append(run_ablation_A1(args.output_dir))
    dfs.append(run_ablation_A2(args.output_dir))
    dfs.append(run_ablation_A3(args.output_dir))
    dfs.append(run_ablation_A4(args.output_dir))
    
    summary_df = pd.concat(dfs, ignore_index=True)
    summary_path = os.path.join(args.output_dir, "ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Overall summary saved to {summary_path}")

if __name__ == "__main__":
    main()
