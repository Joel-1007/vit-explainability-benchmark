# tests/test_adversarial_robustness.py
import pytest
import torch
from metrics.adversarial_robustness import PGDRobustnessMetric

class DummyModel(torch.nn.Module):
    def forward(self, x):
        # We need gradients to flow back to x, so we make output depend on x
        val = x.sum() * 0.0001
        return torch.tensor([[0.5, 2.0, 0.1]], device=x.device) + val

def dummy_explainer(x):
    # Returns a fixed attribution map regardless of input
    return torch.ones(1, 14, 14)

class SensitiveExplainer:
    def __init__(self):
        self.call_count = 0
        
    def __call__(self, x):
        self.call_count += 1
        # First call (original) returns ones. Second call (adversarial) returns zeros.
        if self.call_count % 2 == 1:
            return torch.ones(1, 14, 14)
        else:
            return torch.zeros(1, 14, 14)

def test_pgd_robustness_computation():
    metric = PGDRobustnessMetric(steps=2, random_start=False)
    model = DummyModel()
    
    x = torch.rand(3, 224, 224)
    
    # A totally invariant explainer should have score 1.0
    score_invariant = metric.compute(model, x, explainer_fn=dummy_explainer)
    assert score_invariant == pytest.approx(1.0, 1e-4)

    # A fully sensitive explainer that outputs disjoint maps has score 0.0
    sensitive_exp = SensitiveExplainer()
    score_sensitive = metric.compute(model, x, explainer_fn=sensitive_exp)
    assert score_sensitive == pytest.approx(0.0, 1e-4)
