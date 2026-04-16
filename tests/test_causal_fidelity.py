import pytest
import torch
from metrics.causal_fidelity import CausalMaskingMetric

class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return probability distributions
        # Make the first class highly confident if input is unmasked (mostly > 0)
        # If masked (mostly < 0 or zeros), confidence drops.
        v = x.mean()
        # Returns shape (1, 3)
        return torch.tensor([[v, 1.0 - v, 0.0]], device=x.device)

def test_causal_masking_initialisation():
    metric = CausalMaskingMetric(mode="noise")
    assert metric.mode == "noise"

def test_causal_necessity_computation():
    metric = CausalMaskingMetric(mode="noise")
    model = DummyModel()
    
    # Fake image (positive mean) and fake saliency
    x = torch.ones(3, 224, 224) 
    saliency = torch.ones(224, 224) # Unanimous saliency, everything is masked
    
    # Target class 0 expects drop in confidence
    drop = metric.compute(model, x, saliency, target_class=0, tau=0.5)
    
    # Should be non-negative
    assert drop >= 0.0
    
def test_causal_sufficiency_computation():
    metric = CausalMaskingMetric(mode="noise")
    model = DummyModel()
    x = torch.ones(3, 224, 224) 
    saliency = torch.ones(224, 224) 
    
    cs = metric.compute_sufficiency(model, x, saliency, target_class=0, tau=0.5)
    
    # Normalized score between 0 and 1, roughly
    assert list(model(x.unsqueeze(0)).shape) == [1, 3]
    assert cs >= 0.0

def test_curve_method_returns_auc():
    metric = CausalMaskingMetric(mode="noise")
    model = DummyModel()
    x = torch.ones(3, 224, 224)
    saliency = torch.rand(224, 224)
    
    points, auc = metric.curve(model, x, saliency, target_class=0, mode="insertion", steps=5)
    
    assert len(points) == 6 # steps + 1
    assert all(0.0 <= frac <= 1.0 for frac, conf in points)
    assert 0.0 <= auc <= 1.0
