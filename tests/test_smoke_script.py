"""
Stage 4 regression test: the real-model smoke-test script, run here on tiny
random timm models with the same token layouts (no downloads).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

TINY = {"vit_b16": "vit_tiny_patch16_224", "deit_b16": "deit_tiny_distilled_patch16_224",
        "swin_b": "swin_tiny_patch4_window7_224"}


class _Wrap(torch.nn.Module):
    """Mimics a model-zoo wrapper (backbone inside, normalisation in forward).
    (RISE is left out here: a random tiny model's output barely depends on the
    input, so RISE maps are legitimately target-insensitive on it; RISE is
    covered by the real-weight run.)"""
    def __init__(self, arch, n):
        super().__init__()
        import timm
        torch.manual_seed(0)
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=n)

    def forward(self, x):
        return self.backbone((x - 0.45) / 0.25)


def test_smoke_script_on_tiny_models(tmp_path, monkeypatch):
    import smoke_test_explainers as smoke
    monkeypatch.setattr(smoke, "load_model", lambda name, num_classes=10, pretrained=True: _Wrap(TINY[name], num_classes))
    out = tmp_path / "smoke"
    code = smoke.main(["--models", "vit_b16", "swin_b", "--explainers", "raw_attention", "rollout",
                       "gradcam", "chefer_lrp", "integrated_gradients", "raw_attention_last3",
                       "--n-images", "2", "--ig-steps", "64", "--device", "cpu", "--out", str(out)])
    rep = pd.read_csv(out / "smoke_report.csv")
    assert code == 0, rep[~rep["pass"]].to_string()
    # DeiT (CLS + distillation token): attention explainers and IG. On this RANDOM
    # tiny DeiT the pre-ReLU GradCAM map is entirely negative for the top class, so
    # GradCAM is legitimately all-zero; its DeiT path is covered exactly in
    # tests/test_explainers.py (token layout + target-layer gradient tests).
    out2 = tmp_path / "smoke_deit"
    code2 = smoke.main(["--models", "deit_b16", "--explainers", "raw_attention", "rollout",
                        "chefer_lrp", "integrated_gradients", "--n-images", "2", "--ig-steps", "64",
                        "--device", "cpu", "--out", str(out2)])
    rep2 = pd.read_csv(out2 / "smoke_report.csv")
    assert code2 == 0, rep2[~rep2["pass"]].to_string()
    rep = pd.concat([rep, rep2])
    # Swin: CLS-attention explainers recorded as (expected) unsupported, others run
    sw = rep[(rep.model == "swin_b") & (rep.check == "compatibility")].set_index("explainer")
    for e in ("raw_attention", "rollout", "chefer_lrp", "raw_attention_last3"):
        assert sw.loc[e, "pass"] and "unsupported" in sw.loc[e, "detail"]
    assert sw.loc["gradcam", "detail"] == "ran"
    # every check family was exercised
    assert {"shape", "finite", "non_degenerate", "deterministic", "input_sensitive", "target",
            "gradients", "attention", "ig_completeness"} <= set(rep.check)
    # class-agnostic vs class-specific target behaviour is checked with the right expectation
    t = rep[(rep.model == "vit_b16") & (rep.check == "target")].set_index("explainer")
    assert "class-agnostic" in t.loc["rollout", "detail"] and "class-specific" in t.loc["gradcam", "detail"]
    assert (out / "vit_b16_maps.png").exists() and (out / "swin_b_maps.png").exists()
    assert json.loads((out2 / "smoke_summary.json").read_text())["models"]["deit_b16"]["failed"] == 0


def test_smoke_script_flags_a_broken_explainer(tmp_path, monkeypatch):
    """A constant (degenerate, target-insensitive) explainer must fail the checks."""
    import smoke_test_explainers as smoke
    from explainers.base import BaseExplainer

    class Constant(BaseExplainer):
        def explain(self, x, target_class, **kw):
            return torch.ones(224 // self.patch_size, 224 // self.patch_size)

    monkeypatch.setattr(smoke, "load_model", lambda name, num_classes=10, pretrained=True: _Wrap(TINY[name], num_classes))
    monkeypatch.setattr(smoke, "_registries", lambda: ({"gradcam": Constant}, {"vit_b16": 16}))
    code = smoke.main(["--models", "vit_b16", "--explainers", "gradcam", "--n-images", "2",
                       "--device", "cpu", "--out", str(tmp_path / "s")])
    rep = pd.read_csv(tmp_path / "s" / "smoke_report.csv").set_index("check")
    assert code == 1
    assert not rep.loc["non_degenerate", "pass"] and not rep.loc["target", "pass"]
    assert not rep.loc["input_sensitive", "pass"]
