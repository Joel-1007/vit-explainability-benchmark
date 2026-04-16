# metrics/explainer_interaction.py
"""
Explainer Interaction Graph (EIG)
--------------------------------
This module builds a weighted undirected graph whose nodes are explainers
and edge weights quantify multi‑metric agreement via **mutual information**.
The graph is then partitioned with the Louvain community detection algorithm
to reveal groups of redundant or complementary explainers.

Key steps:
1. Collect raw metric score vectors for each explainer across a set of
   evaluation samples.
2. Estimate pair‑wise mutual information (MI) for every metric family.
3. Aggregate MI across metrics using user‑provided weights (α).
4. Build a NetworkX graph, run Louvain, and compute redundancy/uniqueness
   scores (R_i, U_i).

The class is deliberately lightweight – it does not depend on the rest of
the benchmark code beyond the metric score dictionaries that the runner
produces.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple

# Optional imports – we lazily import to avoid hard dependencies if the user
# does not need this functionality.


class ExplainerInteractionGraph:
    """Construct and analyse an explainer interaction graph.

    Parameters
    ----------
    metric_weights: Dict[str, float] | None, default None
        Mapping from metric name (e.g., "causal_fidelity") to a non‑negative
        weight α_m. If ``None`` all metrics receive equal weight.
    mi_estimator: callable | None, default None
        Function ``mi(x, y) -> float`` that estimates mutual information
        between two 1‑D arrays. If ``None`` a simple k‑NN estimator based on
        ``sklearn.metrics.mutual_info_regression`` is used.
    random_state: int | None, default 42
        Seed for the MI estimator (reproducibility).
    """

    def __init__(
        self,
        metric_weights: Dict[str, float] | None = None,
        mi_estimator: callable | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.metric_weights = metric_weights
        self.random_state = random_state
        if mi_estimator is None:
            # Lazy import of sklearn's MI estimator.
            try:
                from sklearn.metrics import mutual_info_regression
            except Exception as e:
                raise RuntimeError(
                    "scikit‑learn is required for the default MI estimator. "
                    "Install it or provide a custom `mi_estimator`."
                ) from e

            def _default_mi(x: np.ndarray, y: np.ndarray) -> float:
                # mutual_info_regression expects 2‑D X and 1‑D y.
                x = x.reshape(-1, 1)
                y = y.ravel()
                mi = mutual_info_regression(x, y, random_state=self.random_state)
                return float(mi[0])

            self.mi_estimator = _default_mi
        else:
            self.mi_estimator = mi_estimator

    # ---------------------------------------------------------------------
    # Helper: compute pairwise MI matrix for a single metric
    # ---------------------------------------------------------------------
    @staticmethod
    def _pairwise_mi_matrix(
        scores: Dict[str, np.ndarray],
        mi_func: callable,
    ) -> np.ndarray:
        """Return an |V|×|V| symmetric matrix of MI values.

        ``scores`` maps explainer name → 1‑D score vector (same length for all).
        ``mi_func`` is a function ``mi(x, y)`` returning a float.
        """
        explainer_names = list(scores.keys())
        n = len(explainer_names)
        mi_mat = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                xi = scores[explainer_names[i]]
                xj = scores[explainer_names[j]]
                mi = mi_func(xi, xj)
                mi_mat[i, j] = mi_mat[j, i] = mi
        return mi_mat

    # ---------------------------------------------------------------------
    # Public API – build graph
    # ---------------------------------------------------------------------
    def build_graph(
        self,
        explainer_score_dict: Dict[str, Dict[str, np.ndarray]],
    ) -> Tuple[nx.Graph, Dict[str, int], Dict[str, float]]:
        """Construct the interaction graph.

        Parameters
        ----------
        explainer_score_dict: dict
            Mapping ``explainer_name -> {metric_name -> score_vector}``.
            All score vectors for a given metric must have the same length
            (number of evaluation samples).

        Returns
        -------
        G: networkx.Graph
            Weighted undirected graph with edge attribute ``weight``.
        communities: dict
            Mapping ``explainer_name -> community_id`` obtained from Louvain.
        redundancy: dict
            Mapping ``explainer_name -> R_i`` (total similarity to others).
        """
        # -----------------------------------------------------------------
        # 1. Determine metric weights (α)
        # -----------------------------------------------------------------
        metric_names = set()
        for scores in explainer_score_dict.values():
            metric_names.update(scores.keys())
        metric_names = sorted(metric_names)
        if self.metric_weights is None:
            # Uniform weighting
            alpha = {m: 1.0 / len(metric_names) for m in metric_names}
        else:
            # Use provided weights, normalise to sum to 1.
            total = sum(self.metric_weights.get(m, 0.0) for m in metric_names)
            if total == 0:
                raise ValueError("Sum of metric_weights is zero.")
            alpha = {m: self.metric_weights.get(m, 0.0) / total for m in metric_names}

        # -----------------------------------------------------------------
        # 2. Compute per‑metric MI matrices and aggregate
        # -----------------------------------------------------------------
        explainer_names = list(explainer_score_dict.keys())
        n = len(explainer_names)
        agg_mat = np.zeros((n, n), dtype=float)
        for m in metric_names:
            # Gather score vectors for metric m across all explainers
            scores_m = {
                name: explainer_score_dict[name][m]
                for name in explainer_names
                if m in explainer_score_dict[name]
            }
            if len(scores_m) != n:
                raise ValueError(
                    f"Metric '{m}' missing for some explainers. Expected {n}, got {len(scores_m)}."
                )
            mi_mat = self._pairwise_mi_matrix(scores_m, self.mi_estimator)
            agg_mat += alpha[m] * mi_mat

        # -----------------------------------------------------------------
        # 3. Build NetworkX graph
        # -----------------------------------------------------------------
        G = nx.Graph()
        for name in explainer_names:
            G.add_node(name)
        for i in range(n):
            for j in range(i + 1, n):
                w = agg_mat[i, j]
                if w > 0:
                    G.add_edge(explainer_names[i], explainer_names[j], weight=w)

        # -----------------------------------------------------------------
        # 4. Louvain community detection (python‑louvain)
        # -----------------------------------------------------------------
        try:
            import community as community_louvain
        except Exception as e:
            raise RuntimeError(
                "python‑louvain package is required for community detection. "
                "Install it or skip this step."
            ) from e
        partition = community_louvain.best_partition(G, weight="weight")  # dict node->community

        # -----------------------------------------------------------------
        # 5. Redundancy (R_i) and uniqueness (U_i)
        # -----------------------------------------------------------------
        redundancy = {}
        for node in G.nodes:
            # Sum of incident edge weights (excluding self‑loops which do not exist)
            redundancy[node] = sum(d["weight"] for _, d in G[node].items())
        max_r = max(redundancy.values()) if redundancy else 1.0
        uniqueness = {k: 1.0 - (v / max_r) for k, v in redundancy.items()}

        # Attach redundancy/uniqueness as node attributes for easy visualisation
        nx.set_node_attributes(G, redundancy, "redundancy")
        nx.set_node_attributes(G, uniqueness, "uniqueness")
        nx.set_node_attributes(G, partition, "community")

        return G, partition, redundancy

    # ---------------------------------------------------------------------
    # Convenience wrapper for end‑to‑end usage from the benchmark runner
    # ---------------------------------------------------------------------
    def run(
        self,
        explainer_score_dict: Dict[str, Dict[str, np.ndarray]],
    ) -> Tuple[nx.Graph, Dict[str, int], Dict[str, float]]:
        """One‑line entry point used by ``metrics/runner.py``.

        It simply forwards to :meth:`build_graph`.
        """
        return self.build_graph(explainer_score_dict)

# -------------------------------------------------------------------------
# Example usage (not executed during import)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data for quick sanity check
    np.random.seed(0)
    dummy_scores = {
        "RawAttention": {
            "causal_fidelity": np.random.rand(100),
            "adversarial_robustness": np.random.rand(100),
        },
        "GradCAM": {
            "causal_fidelity": np.random.rand(100),
            "adversarial_robustness": np.random.rand(100),
        },
        "Rollout": {
            "causal_fidelity": np.random.rand(100),
            "adversarial_robustness": np.random.rand(100),
        },
    }
    eig = ExplainerInteractionGraph()
    G, comm, red = eig.run(dummy_scores)
    print("Communities:", comm)
    print("Redundancy:", red)
    print("Graph size:", G.number_of_nodes(), G.number_of_edges())
"""
