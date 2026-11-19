"""
Stage 2 regression tests: pixel-space input protocol.

* every model normalises internally with its PRE-TRAINING statistics
* evaluation and fine-tuning validation produce identical [0, 1] tensors
* GT masks stay aligned with the pixels through the frozen resize/crop,
  for every loader and several aspect ratios
* perturbation baselines are defined in pixel space (RISE black, IG black,
  LIME image-mean fill, R1 noise clamped, erasing in [0, 1])
"""

from __future__ import annotations

import numpy as np
import pytest
import timm
import torch
from PIL import Image, ImageDraw

from model_zoo import MODEL_SPECS, load_model
from utils.eval_datasets import (CUB200Eval, ImageNetS50Eval, NIHBBoxEval, PairTransform,
                                 VOCEval)

ASPECTS = [(300, 200), (200, 300), (500, 375), (224, 224), (640, 480)]


# ---------------------------------------------------------------------------
# Model-specific normalisation inside the wrapper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_spec_statistics_match_timm_pretraining_config(name):
    spec = MODEL_SPECS[name]
    cfg = timm.models.get_pretrained_cfg(spec.timm_id)
    assert tuple(cfg.mean) == pytest.approx(spec.mean)
    assert tuple(cfg.std) == pytest.approx(spec.std)


@pytest.mark.parametrize("name", ["vit_b16", "deit_b16"])   # 0.5/0.5 and ImageNet stats
def test_wrapper_normalises_internally(name):
    torch.manual_seed(0)
    m = load_model(name, num_classes=3, pretrained=False).eval()
    x = torch.rand(2, 3, 224, 224)
    mean = torch.tensor(MODEL_SPECS[name].mean).view(1, 3, 1, 1)
    std = torch.tensor(MODEL_SPECS[name].std).view(1, 3, 1, 1)
    with torch.no_grad():
        torch.testing.assert_close(m(x), m.backbone((x - mean) / std))
    # buffers are not part of the checkpoint → fine-tuned weights load strictly
    assert not any(k.startswith("normalize.") for k in m.state_dict())
    m2 = load_model(name, num_classes=3, pretrained=False)
    m2.load_state_dict(m.state_dict(), strict=True)


def test_gradients_flow_to_pixels_through_normalisation():
    m = load_model("vit_b16", num_classes=3, pretrained=False).eval()
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    m(x)[0, 1].backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# One geometry for evaluation and fine-tuning validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("size", ASPECTS)
def test_validation_transform_equals_evaluation_transform(size):
    from training.transforms import build_transforms
    _, val_tf = build_transforms()
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8))
    a, b = val_tf(img), PairTransform().image(img)
    assert torch.equal(a, b) and a.shape == (3, 224, 224)
    assert a.min() >= 0 and a.max() <= 1


def test_train_transform_stays_in_pixel_range():
    from training.transforms import PixelSpaceRandomErasing, build_transforms
    torch.manual_seed(0)
    train_tf, _ = build_transforms()
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (300, 400, 3), dtype=np.uint8))
    for _ in range(5):
        x = train_tf(img)
        assert x.shape == (3, 224, 224) and x.min() >= 0 and x.max() <= 1
    erase = PixelSpaceRandomErasing(probability=1.0)
    flat = torch.full((3, 224, 224), 0.5)
    out = erase(flat.clone())
    assert out.min() >= 0 and out.max() <= 1 and not torch.equal(out, flat)


# ---------------------------------------------------------------------------
# Mask alignment through the frozen resize/crop
# ---------------------------------------------------------------------------

def _scene(w, h, box):
    """Black image with a white rectangle, and the matching binary mask."""
    img = Image.new("RGB", (w, h), (0, 0, 0))
    ImageDraw.Draw(img).rectangle(box, fill=(255, 255, 255))
    mask = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1 + 1, x0:x1 + 1] = 1
    return img, mask


def _iou(x, m):
    bright = x.mean(0) > 0.5
    obj = m == 1
    return float((bright & obj).sum() / (bright | obj).sum())


@pytest.mark.parametrize("size", ASPECTS)
def test_pair_transform_keeps_mask_aligned(size):
    w, h = size
    box = (w // 4, h // 3, w // 4 + w // 3, h // 3 + h // 3)
    img, mask = _scene(w, h, box)
    x, m = PairTransform()(img, mask)
    assert _iou(x, m) > 0.95


def _check_loader(ds):
    assert len(ds) > 0
    for i in range(len(ds)):
        x, _, m = ds[i]
        if (m == 1).any():
            assert _iou(x, m) > 0.95, (type(ds).__name__, i)


def test_alignment_through_every_loader(tmp_path):
    rng = np.random.default_rng(0)
    # CUB: object drawn where the segmentation is
    cub = tmp_path / "cub"
    for k, (w, h) in enumerate([(300, 200), (200, 300)]):
        box = (40, 30, 40 + w // 2, 30 + h // 2)
        img, mask = _scene(w, h, box)
        (cub / "images" / "001.A").mkdir(parents=True, exist_ok=True)
        (cub / "segmentations" / "001.A").mkdir(parents=True, exist_ok=True)
        img.save(cub / "images" / "001.A" / f"i{k}.jpg", quality=100)
        Image.fromarray(mask * 255).save(cub / "segmentations" / "001.A" / f"i{k}.png")
    (cub / "images.txt").write_text("1 001.A/i0.jpg\n2 001.A/i1.jpg\n")
    (cub / "train_test_split.txt").write_text("1 0\n2 0\n")
    _check_loader(CUB200Eval(cub))

    # VOC: palette segmentation, class 3 (bird)
    voc = tmp_path / "voc"
    for d in ("JPEGImages", "SegmentationClass", "ImageSets/Segmentation"):
        (voc / d).mkdir(parents=True)
    img, mask = _scene(500, 333, (100, 60, 300, 250))
    img.save(voc / "JPEGImages" / "a.jpg", quality=100)
    seg = Image.fromarray(mask * 3, mode="P"); seg.putpalette([0, 0, 0] * 256)
    seg.save(voc / "SegmentationClass" / "a.png")
    (voc / "ImageSets/Segmentation/val.txt").write_text("a\n")
    _check_loader(VOCEval(voc))

    # ImageNet-S: RGB-encoded ids
    ins = tmp_path / "ins"
    (ins / "validation" / "n0").mkdir(parents=True)
    img, mask = _scene(400, 300, (50, 50, 250, 200))
    img.save(ins / "validation" / "n0" / "a.JPEG", quality=100)
    ids = mask.astype(np.int64) * 7
    Image.fromarray(np.stack([ids % 256, ids // 256, 0 * ids], -1).astype(np.uint8)).save(
        ins / "validation" / "n0" / "a.png")
    _check_loader(ImageNetS50Eval(ins))

    # NIH: grayscale image, box from CSV
    nih = tmp_path / "nih"
    (nih / "images").mkdir(parents=True)
    img, _ = _scene(1024, 1024, (300, 200, 699, 599))
    img.convert("L").save(nih / "images" / "00000001_000.png")
    (nih / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n00000001_000.png,Mass,300,200,400,400,,,\n")
    _check_loader(NIHBBoxEval(nih))


# ---------------------------------------------------------------------------
# Perturbation baselines are pixel-space and model-independent
# ---------------------------------------------------------------------------

class _Recorder(torch.nn.Module):
    """Tiny classifier that records every input batch it sees."""
    def __init__(self):
        super().__init__()
        self.seen = []
        self.lin = torch.nn.Linear(3, 4)

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.lin(x.mean(dim=(2, 3)))


def test_rise_masks_to_black():
    from explainers import RISEExplainer
    m = _Recorder()
    e = RISEExplainer(m, patch_size=16, n_masks=8, chunk_size=8)
    e.explain(torch.ones(3, 224, 224), 0)
    inp = m.seen[0]                                       # (8, 3, H, W) = 1 ⊙ masks
    masks = e._masks.float()
    torch.testing.assert_close(inp, masks.expand(-1, 3, -1, -1).to(inp.dtype))


def test_ig_baseline_is_black():
    """Every path point is α·x, i.e. the straight line from the black image."""
    from explainers import IntegratedGradientsExplainer
    m = _Recorder()
    e = IntegratedGradientsExplainer(m, n_steps=4, chunk_size=5)
    x = torch.rand(3, 224, 224)
    e.explain(x, 0)
    alphas, _ = e.quadrature(4, "gausslegendre")
    for a, p in zip(alphas, m.seen[0]):
        torch.testing.assert_close(p, a * x)
    m2 = _Recorder()
    IntegratedGradientsExplainer(m2, n_steps=4, chunk_size=5, method="trapezoid").explain(x, 0)
    assert torch.equal(m2.seen[0][0], torch.zeros(3, 224, 224))     # α = 0 node = baseline


def test_ig_quadrature_resolves_sharp_feature_near_baseline():
    """
    Regression for real ViT behaviour: the path gradient spikes near α = 0.
    On ∫₀¹ 50·e^{-50α} dα (= 1 − e^{-50}) Gauss–Legendre is accurate at 200
    nodes while the uniform trapezoid rule is not.
    """
    import math
    from explainers import IntegratedGradientsExplainer as IG
    truth = 1 - math.exp(-50)
    g = lambda a: 50 * torch.exp(-50 * a)                          # noqa: E731
    errs = {}
    for method in ("gausslegendre", "trapezoid"):
        a, w = IG.quadrature(200, method)
        assert w.sum().item() == pytest.approx(1.0, abs=1e-5)
        errs[method] = abs(float((g(a) * w).sum()) - truth)
    assert errs["gausslegendre"] < 1e-4 < errs["trapezoid"]


def test_lime_fills_with_image_mean():
    from explainers import LIMEExplainer
    m = _Recorder()
    x = torch.rand(3, 224, 224)
    e = LIMEExplainer(m, patch_size=16, n_samples=4)
    recon = e._reconstruct(x, torch.zeros(196), x.mean(dim=(-2, -1)), 14, 16)
    torch.testing.assert_close(recon, x.mean(dim=(-2, -1)).view(3, 1, 1).expand_as(x))


def test_r1_noise_is_clamped_to_valid_pixels():
    import random
    from metrics.robustness import _perturb
    x = torch.rand(1, 3, 32, 32)
    p = _perturb(x, 0.2, random.Random(0), clip_range=(0.0, 1.0))
    assert p.min() >= 0 and p.max() <= 1 and not torch.equal(p, x)
    assert (p - x).abs().max() <= 0.2 + 1e-6
