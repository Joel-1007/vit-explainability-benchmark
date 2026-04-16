# scripts/plot_interaction_graph.py
"""
Explainer Interaction Graph Visualization.
Renders the graph built by ExplainerInteractionGraph with 
communities colored and sizes scaled by uniqueness.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from metrics.explainer_interaction import ExplainerInteractionGraph

def plot_eig(G: nx.Graph, save_path: str = None):
    plt.figure(figsize=(10, 8))
    
    # Extract attributes
    communities = nx.get_node_attributes(G, 'community')
    uniqueness = nx.get_node_attributes(G, 'uniqueness')
    
    # Setup layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Modern color palette for communities
    base_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    node_colors = [base_colors[communities.get(node, 0) % 8] for node in G.nodes()]
    
    # Node sizes scaled by uniqueness (so highly unique explainers pop out)
    # Give a base size of 500, plus up to 1500 depending on uniqueness
    node_sizes = [500 + 1500 * uniqueness.get(node, 0.5) for node in G.nodes()]
    
    # Draw edges with varying thickness/opacity based on NMI weight
    edges = G.edges(data=True)
    weights = [d.get("weight", 0.0) for u, v, d in edges]
    max_w = max(weights) if weights else 1.0
    normalized_weights = [3.0 * (w / max_w) for w in weights]
    
    nx.draw_networkx_edges(
        G, pos, 
        alpha=0.4, 
        width=normalized_weights,
        edge_color="gray"
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=2.0
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight="bold",
        font_family="sans-serif",
        font_color="black"
    )
    
    plt.title("Explainer Interaction Graph (EIG)", fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[EIG] Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    import pickle
    import json
    
    parser = argparse.ArgumentParser(description="Plot Explainer Interaction Graph from real benchmark data.")
    parser.add_argument("--input", type=str, required=True, help="Path to the JSON or PKL file containing dict of explainers to metrics arrays.")
    parser.add_argument("--output", type=str, default="figures/explainer_interaction_graph.pdf", help="Output path for the plot.")
    args = parser.parse_args()
    
    path = Path(args.input)
    if path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        with open(path, 'r') as f:
            data = json.load(f)
            
    eig = ExplainerInteractionGraph()
    G, comm, red = eig.run(data)
    
    plot_eig(G, save_path=args.output)
