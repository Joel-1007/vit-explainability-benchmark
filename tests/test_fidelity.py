"""
Tests for Fidelity Metrics (F1-F3)
"""
import pytest
torch = pytest.importorskip("torch")

import torch.nn as nn
from metrics.fidelity import FidelityMetrics

class DummyModel(nn.Module):
    def forward(self, x):
        # Return a deterministic "logit" based on the input sum
        # Shape x: (B, C, H, W)
        B = x.shape[0]
        logits = torch.zeros(B, 2)
        # Class 1 logit is sum of x, Class 0 is constant minus sum
        val = x.reshape(B, -1).sum(dim=1) / 100.0
        logits[:, 1] = val
        logits[:, 0] = 1.0 - val
        return logits

def test_generate_mask():
    fm = FidelityMetrics(mask_mode="zero", k_fractions=(0.5,))
    # (B, H_a, W_a) = (2, 2, 2)
    att_map = torch.tensor([
        [[0.1, 0.4],
         [0.2, 0.3]],
        [[0.9, 0.8],
         [0.1, 0.2]]
    ])
    # For k=0.5, N=4 -> k=2 top elements
    # Batch 0: top 2 are 0.4 and 0.3 -> index (0,1) and (1,1)
    mask = fm._generate_mask(att_map, 0.5)
    assert mask.shape == (2, 1, 2, 2)
    assert torch.allclose(mask[0, 0], torch.tensor([[0., 1.], [0., 1.]]))
    assert torch.allclose(mask[1, 0], torch.tensor([[1., 1.], [0., 0.]]))

def test_apply_mask_zero():
    fm = FidelityMetrics(mask_mode="zero", k_fractions=(0.5,))
    x = torch.ones(1, 1, 4, 4)
    # att map 2x2
    mask_small = torch.tensor([[[[1., 0.], [0., 1.]]]])
    # Interpolated to 4x4 (nearest) -> 2x2 blocks
    x_keep = fm._apply_mask(x, mask_small, keep=True)
    assert x_keep[0, 0, 0, 0] == 1.0
    assert x_keep[0, 0, 0, 3] == 0.0
    
    x_drop = fm._apply_mask(x, mask_small, keep=False)
    assert x_drop[0, 0, 0, 0] == 0.0
    assert x_drop[0, 0, 0, 3] == 1.0

def test_apply_mask_mean():
    fm = FidelityMetrics(mask_mode="mean", k_fractions=(0.5,))
    x = torch.tensor([[[
        [1., 1., 3., 3.],
        [1., 1., 3., 3.],
        [3., 3., 1., 1.],
        [3., 3., 1., 1.]
    ]]]) # mean is 2.0
    mask_small = torch.tensor([[[[1., 0.], [0., 1.]]]])
    
    x_drop = fm._apply_mask(x, mask_small, keep=False)
    # drop where mask_small == 1
    # top left dropped -> replaced by mean (2.0)
    assert x_drop[0, 0, 0, 0] == 2.0
    # top right kept (mask=0 -> drop keeps it) -> 3.0
    assert x_drop[0, 0, 0, 3] == 3.0

def test_metrics_standalone():
    fm = FidelityMetrics(mask_mode="zero", k_fractions=(0.5,))
    model = DummyModel()
    
    # 2 samples, 1 channel, 4x4
    x = torch.ones(2, 1, 4, 4)
    targets = torch.tensor([1, 1])
    att_map = torch.tensor([
        [[1., 0.], [0., 1.]],
        [[0., 1.], [1., 0.]]
    ])
    
    suff = fm.sufficiency(model, x, targets, att_map, 0.5)
    comp = fm.comprehensiveness(model, x, targets, att_map, 0.5)
    lod = fm.log_odds_drop(model, x, targets, att_map, 0.5)
    
    assert suff.shape == (2,)
    assert comp.shape == (2,)
    assert lod.shape == (2,)

def test_compute_all():
    fm = FidelityMetrics(mask_mode="mean", k_fractions=(0.25, 0.5))
    model = DummyModel()
    x = torch.rand(2, 1, 4, 4)
    targets = torch.tensor([0, 1])
    att_map = torch.rand(2, 2, 2)
    
    results = fm.compute_all(model, x, targets, att_map)
    
    expected_keys = [
        "sufficiency@0.25", "comprehensiveness@0.25", "log_odds_drop@0.25",
        "sufficiency@0.50", "comprehensiveness@0.50", "log_odds_drop@0.50",
    ]
    for key in expected_keys:
        assert key in results
        assert results[key].shape == (2,)
