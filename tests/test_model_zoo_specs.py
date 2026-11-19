"""
Stage 1 regression tests (offline): model identity, provenance table, token
layout and attention extraction for every model architecture, built with
pretrained=False. Real-weight checks live in scripts/verify_model_zoo.py.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import timm
import torch

from explainers.base import (UnsupportedArchitectureError, _capture_attn_weights,
                             _num_prefix_tokens)
from model_zoo import ABLATION_MODEL_REGISTRY, MODEL_REGISTRY, MODEL_SPECS, load_model
from model_zoo.specs import expected_tokens

ROOT = Path(__file__).resolve().parents[1]
ALL = list(MODEL_SPECS)


def test_specs_cover_registry_and_tags_exist_in_timm():
    assert set(MODEL_SPECS) == set(MODEL_REGISTRY) | set(ABLATION_MODEL_REGISTRY)
    for spec in MODEL_SPECS.values():
        arch, tag = spec.timm_id.split(".", 1)
        # Fully pinned "arch.tag" that this timm version can resolve
        assert tag and spec.timm_id in timm.list_pretrained(arch), spec.timm_id
        assert spec.hf_repo == f"timm/{spec.timm_id}"
    assert MODEL_SPECS["mae_b16"].main_matrix is False


def test_beit_identifier_and_provenance_are_honest():
    spec = MODEL_SPECS["beit_b16"]
    assert spec.timm_id == "beit_base_patch16_224.in22k_ft_in22k"
    assert "supervised" in spec.supervision and "self-supervised" in spec.supervision
    assert "FOLLOWED BY supervised" in spec.pretraining
    # the old, non-existent pure-MIM tag must not come back
    assert "beit_base_patch16_224.in22k" not in timm.list_pretrained("beit_base_patch16_224")


def test_dino_uses_timm_not_torch_hub():
    import model_zoo.dino_vitb8 as dino
    code = inspect.getsource(dino)
    assert "torch.hub.load" not in code
    assert MODEL_SPECS["dino_vitb8"].timm_id == "vit_base_patch8_224.dino"
    m = load_model("dino_vitb8", num_classes=3, pretrained=False)
    assert isinstance(m.backbone, timm.models.vision_transformer.VisionTransformer)
    assert m.backbone.global_pool == "token"


def test_dino_official_state_dict_key_mapping(tmp_path):
    from model_zoo.dino_vitb8 import load_dino_vitb8_from_pth, official_state_dict
    ref = timm.create_model("vit_base_patch8_224", pretrained=False, num_classes=0)
    wrapped = {"teacher": {f"backbone.{k}": v for k, v in ref.state_dict().items()}}
    wrapped["teacher"]["head.mlp.0.weight"] = torch.zeros(2, 2)      # DINO projection head
    path = tmp_path / "dino.pth"
    torch.save(wrapped, path)
    sd = official_state_dict(str(path))
    assert set(sd) == set(ref.state_dict())
    m = load_dino_vitb8_from_pth(str(path), num_classes=0)
    assert torch.equal(m.backbone.pos_embed, ref.pos_embed)


@pytest.mark.parametrize("name", ALL)
def test_architecture_token_layout_and_attention(name):
    torch.manual_seed(0)
    model = load_model(name, num_classes=4, pretrained=False).eval()
    assert model.backbone.pretrained_cfg.get("architecture") == MODEL_SPECS[name].timm_id.split(".")[0]
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        feats = model.backbone.forward_features(x)
        logits = model(x)
    assert logits.shape == (1, 4) and torch.isfinite(logits).all()
    prefix, side = expected_tokens(name)
    if name == "swin_b":
        assert feats.shape[1:3] == (7, 7)
        with pytest.raises(UnsupportedArchitectureError):
            _capture_attn_weights(model, x, [-1])
        return
    T = prefix + side * side
    assert feats.shape[1] == T
    assert _num_prefix_tokens(model, T) == prefix
    A = _capture_attn_weights(model, x, [-1])[0]
    assert A.shape[-2:] == (T, T)
    assert torch.allclose(A.sum(-1), torch.ones(A.shape[:-1]), atol=1e-4)


def test_record_model_hashes_covers_every_spec(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import record_model_hashes as rmh
    monkeypatch.setattr(rmh, "published_weights", lambda repo: {
        "repo": repo, "commit": "c0ffee", "file": "model.safetensors", "size": 1, "sha256": "ab" * 32})
    out = tmp_path / "hashes.txt"
    assert rmh.main(["--out", str(out)]) == 0
    text = out.read_text()
    for spec in MODEL_SPECS.values():
        assert f"timm ID    : {spec.timm_id}" in text and spec.pretraining[:30] in text
