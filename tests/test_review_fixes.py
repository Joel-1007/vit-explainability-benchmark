"""
Regression tests for the theory/design review fixes:
C2 entropy normalisation (T6), R2 re-initialisation, the axiom verifier's
full-suite path, MetricSuite construction and bootstrap CIs.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
import torch


def test_c2_uses_mass_distribution_matching_t6():
    """T6 example: e_sym=(.5,.5,0,…) has entropy log2/logN, e_asym=(1,0,…) has 0."""
    from metrics.complexity import ComplexityMetrics, attribution_entropy, entropy_batch_torch
    N = 196
    sym, asym = np.zeros(N), np.zeros(N)
    sym[:2], asym[0] = 0.5, 1.0
    cm = ComplexityMetrics()
    assert cm.compute(sym).entropy_norm == pytest.approx(math.log(2) / math.log(N))
    assert cm.compute(asym).entropy_norm == pytest.approx(0.0, abs=1e-12)
    # class, standalone function and batched torch version agree
    rng = np.random.default_rng(0)
    maps = rng.uniform(0, 1, size=(5, N))
    torch_h = entropy_batch_torch(torch.tensor(maps, dtype=torch.float32)).numpy()
    for m, th in zip(maps, torch_h):
        a = cm.compute(m).entropy_norm
        assert a == pytest.approx(attribution_entropy(m)["entropy_norm"], abs=1e-9)
        assert a == pytest.approx(float(th), abs=1e-5)


def test_c2_not_flattened_on_unit_scale_maps():
    """A concentrated [0,1] map must be far from maximal entropy (softmax gave ~0.99)."""
    from metrics.complexity import ComplexityMetrics
    m = np.zeros(196)
    m[:10] = 1.0
    assert ComplexityMetrics().compute(m).entropy_norm < 0.5


def test_randomised_vit_stays_finite():
    import timm
    from metrics.robustness import randomise_model_weights
    torch.manual_seed(0)
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=10).eval()
    rand = randomise_model_weights(model, seed=0).eval()
    with torch.no_grad():
        out = rand(torch.randn(2, 3, 224, 224))
    assert torch.isfinite(out).all()
    assert torch.equal(rand.blocks[0].norm1.weight, torch.ones_like(rand.blocks[0].norm1.weight))
    assert not torch.equal(rand.head.weight, model.head.weight)
    assert not torch.equal(rand.pos_embed, model.pos_embed)


def test_metric_suite_and_axiom_verifier_full_path():
    from metrics.axiom_verifier import AxiomVerifier
    from metrics.suite import MetricSuite
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = AxiomVerifier(metric_suite=MetricSuite(), n_patches=16).run_all()
    assert not [x for x in w if "Test failed" in str(x.message)]
    names = {r.metric_name for r in res}
    assert names == {"L1-mIoU", "L2-PointingGame", "L3-EGT", "C1-Gini", "C2-Entropy", "C3-EMR90"}
    # Theorem T6: complexity metrics reward the A3-violating attribution
    for r in res:
        if r.metric_name.startswith("C") and r.axiom_name == "A3":
            assert not r.satisfies and r.delta < 0


def test_bootstrap_ci_in_summary():
    from metrics.aggregate import bootstrap_ci, summarise
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 0.1, 400)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo == pytest.approx(2 * 1.96 * v.std(ddof=1) / 20, rel=0.15)
    assert all(math.isnan(t) for t in bootstrap_ci([1.0]))
    df = pd.DataFrame({"dataset": "d", "model": "m", "explainer": "e", "sample_id": range(400),
                       "label": 0, "pred": 0, "correct": 1, "miou": v})
    s = summarise(df).iloc[0]
    assert s.miou_ci_low < s.miou < s.miou_ci_high
