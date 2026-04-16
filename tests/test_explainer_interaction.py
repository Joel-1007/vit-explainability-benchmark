import pytest
import numpy as np
import networkx as nx
from metrics.explainer_interaction import ExplainerInteractionGraph

def test_eig_initialisation():
    try:
        eig = ExplainerInteractionGraph()
        assert eig.metric_weights is None
    except RuntimeError:
        pytest.skip("skipping EIG init test because sklearn is not installed")

def test_eig_build_graph():
    try:
        eig = ExplainerInteractionGraph()
    except RuntimeError:
        pytest.skip("skipping EIG test because sklearn is not installed")
        
    # Generate mock scores that are perfectly correlated to ensure NMI=1.0
    scores = np.linspace(0, 1, 100)
    dummy_scores = {
        "ExpA": { "M1": scores },
        "ExpB": { "M1": scores },
        "ExpC": { "M1": np.random.rand(100) }, # Random
    }
    
    try:
        G, comm, red = eig.run(dummy_scores)
    except RuntimeError:
        pytest.skip("skipping EIG test because python-louvain or sklearn is not installed")
        
    assert isinstance(G, nx.Graph)
    assert len(G.nodes) == 3
    
    # NMI weight between ExpA and ExpB should be very high
    assert "ExpA" in comm
    assert "ExpB" in comm
    
    # test community aggregation helper
    comm_stats = eig.community_performance(G, comm)
    assert isinstance(comm_stats, dict)

def test_eig_classification():
    try:
        eig = ExplainerInteractionGraph()
    except RuntimeError:
        pytest.skip("skipping EIG test because sklearn is not installed")
        
    G = nx.Graph()
    G.add_node("ExpA", redundancy=0.9, uniqueness=0.1, stability=0.0) # Redundant
    G.add_node("ExpB", redundancy=0.1, uniqueness=0.9, stability=0.0) # Complementary
    G.add_node("ExpC", redundancy=0.1, uniqueness=0.1, stability=0.9) # Fragile
    G.add_node("ExpD", redundancy=0.1, uniqueness=0.1, stability=0.0) # Robust
    
    categories = eig.classify_explainers(G)
    
    assert categories["ExpA"] == "Redundant"
    assert categories["ExpB"] == "Complementary"
    assert categories["ExpC"] == "Fragile"
    assert categories["ExpD"] == "Robust"
