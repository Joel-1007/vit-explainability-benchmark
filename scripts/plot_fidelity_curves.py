# scripts/plot_fidelity_curves.py
"""
Plot Insertion and Deletion Causal Fidelity curves.
This script demonstrates how to visualize the area-under-the-curve (AUC) metric
for the newly implemented MAE-based generative in-painting CausalMaskingMetric.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_curves(results: dict, mode: str, save_path: str = None):
    """
    results: dict formatting -> { "ExplainerName": [(fraction, confidence), ...] }
    mode: "Insertion" or "Deletion"
    """
    plt.figure(figsize=(8, 6))
    
    # Modern, rich aesthetic colors
    colors = ["#FF3366", "#33CC99", "#3399FF", "#FF9933", "#9966FF"]

    for i, (name, points) in enumerate(results.items()):
        fractions, confidences = zip(*points)
        auc = np.trapezoid(confidences, fractions)
        
        plt.plot(
            fractions, 
            confidences, 
            label=f"{name} (AUC: {auc:.3f})", 
            color=colors[i % len(colors)],
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.85
        )

    plt.xlabel(f"Fraction of Patches {'Revealed' if mode == 'Insertion' else 'Masked'}", fontsize=12, fontweight='bold')
    plt.ylabel("Classifier Confidence", fontsize=12, fontweight='bold')
    plt.title(f"Causal {mode} Curves", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=11, frameon=True, edgecolor="black")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[{mode}] Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    import pickle
    import json
    
    parser = argparse.ArgumentParser(description="Plot Causal Fidelity Curves from real benchmark data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON or PKL file containing true curve points.")
    parser.add_argument("--mode", type=str, required=True, choices=["Insertion", "Deletion"], help="Mode of the curve.")
    parser.add_argument("--output", type=str, default="figures/fidelity_curve.pdf", help="Output path for the plot.")
    args = parser.parse_args()
    
    path = Path(args.input)
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(path, 'r') as f:
            data = json.load(f)
            
    plot_curves(data, args.mode, save_path=args.output)
