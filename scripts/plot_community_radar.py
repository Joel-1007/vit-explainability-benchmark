# scripts/plot_community_radar.py
"""
Plots a radar chart (spider plot) of aggregated metrics per Louvain community.
Visualizes how different communities trade-off factors like Sufficiency, 
Necessity, Robustness, Uniqueness, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

def plot_radar(data: dict, categories: list, save_path: str = None):
    """
    data: dict mapping { "Community_Name": [val1, val2, val3, ...] }
    categories: list of metric names corresponding to the values.
    """
    num_vars = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories, color='grey', size=11, fontweight="bold")
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1.0)
    
    colors = ["#FF3366", "#33CC99", "#3399FF", "#FF9933"]
    
    for i, (name, values) in enumerate(data.items()):
        # Normalize/clamp values to [0, 1] for radar chart compatibility
        norm_vals = np.clip(values, 0.0, 1.0).tolist()
        norm_vals += norm_vals[:1]  # Complete the loop
        
        color = colors[i % len(colors)]
        
        ax.plot(angles, norm_vals, linewidth=2, linestyle='solid', label=name, color=color)
        ax.fill(angles, norm_vals, alpha=0.25, color=color)

    plt.title("Community-Level Extrinsic Evaluation", size=15, fontweight='bold', y=1.05)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=11, frameon=True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Radar] Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    import json
    import ast
    
    parser = argparse.ArgumentParser(description="Plot Community Radar Chart from real benchmark data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON file containing community data dict.")
    parser.add_argument("--categories", type=str, required=True, help="List of category names as string, e.g. \"['mIoU', 'Necessity', 'Sufficiency', 'PGD Robustness', 'Uniqueness']\"")
    parser.add_argument("--output", type=str, default="figures/community_radar_chart.pdf", help="Output path for the plot.")
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
        
    categories = ast.literal_eval(args.categories)
    plot_radar(data, categories, save_path=args.output)
