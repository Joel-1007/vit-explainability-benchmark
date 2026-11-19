"""
Phase 4, Task 4.3: Ablation Studies
Ablations A1-A4 are NOT implemented yet; each raises NotImplementedError.
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

def _not_implemented(name: str) -> pd.DataFrame:
    # The previous version drew both groups from np.random.normal with
    # hand-picked means. Removed so placeholder effect sizes cannot reach the paper.
    raise NotImplementedError(
        f"Ablation {name} is not implemented: it needs real per-sample scores "
        "for both variants from Phase 3 runs. compute_cohens_d() is ready to use."
    )

def run_ablation_A1(output_dir: str) -> pd.DataFrame:
    """Ablation A1: Token Resolution (Raw Patch vs CLS Token)"""
    return _not_implemented("A1")

def run_ablation_A2(output_dir: str) -> pd.DataFrame:
    """Ablation A2: Layer Depth (Rollout All vs Last Layer)"""
    return _not_implemented("A2")

def run_ablation_A3(output_dir: str) -> pd.DataFrame:
    """Ablation A3: Masking Strategy (Mean Mask vs Zero Mask)"""
    return _not_implemented("A3")

def run_ablation_A4(output_dir: str) -> pd.DataFrame:
    """Ablation A4: Pre-training Objective (DINO Self-distill vs Supervised)"""
    return _not_implemented("A4")

def main():
    parser = argparse.ArgumentParser(description="Run Task 4.3: Ablation Studies")
    parser.add_argument("--output_dir", type=str, default="../results/phase4", help="Directory to save phase 4 outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    

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
