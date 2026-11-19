"""
Fast tests for utils/eval_datasets.py and the run_phase3 wiring.

Builds tiny fake versions of all four datasets on disk (same layout and file
formats as the real ones) and uses small randomly-initialised timm models,
so nothing is downloaded and the whole file runs in seconds on CPU.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from utils.eval_datasets import (
    IGNORE, NIH_CLASSES, VOC_CLASSES, CUB200Eval, ImageNetS50Eval,
    NIHBBoxEval, VOCEval, build_eval_loader, spread_indices,
)


# ---------------------------------------------------------------------------
# Fake dataset builders
# ---------------------------------------------------------------------------

def _rgb(path: Path, w: int, h: int, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (w, h), 128).save(path)


def make_cub(root: Path) -> None:
    classes = ["001.Albatross", "002.Auklet", "003.Blackbird"]
    lines, split = [], []
    i = 0
    for c in classes:
        for k in range(2):
            i += 1
            rel = f"{c}/img_{k}.jpg"
            _rgb(root / "images" / rel, 300, 200)
            seg = np.zeros((200, 300), np.uint8)
            seg[50:150, 100:200] = 255          # all annotators agree
            seg[0:10, 0:10] = 51                # weak agreement → background
            (root / "segmentations" / c).mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg).save(root / "segmentations" / c / f"img_{k}.png")
            lines.append(f"{i} {rel}")
            split.append(f"{i} {k}")            # img_0 = test, img_1 = train
    (root / "images.txt").write_text("\n".join(lines) + "\n")
    (root / "train_test_split.txt").write_text("\n".join(split) + "\n")


def make_voc(root: Path) -> None:
    ids = ["2007_000001", "2007_000002", "2007_000003"]
    palette = [0, 0, 0] * 256
    for img_id, (cls_big, cls_small) in zip(ids, [(15, 12), (1, 0), (0, 0)]):
        _rgb(root / "JPEGImages" / f"{img_id}.jpg", 250, 250)
        seg = np.zeros((250, 250), np.uint8)
        if cls_big:
            seg[20:200, 20:200] = cls_big        # dominant class
            seg[19, 20:200] = IGNORE             # boundary band
        if cls_small:
            seg[210:240, 210:240] = cls_small
        m = Image.fromarray(seg, mode="P")
        m.putpalette(palette)
        (root / "SegmentationClass").mkdir(parents=True, exist_ok=True)
        m.save(root / "SegmentationClass" / f"{img_id}.png")
    (root / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)
    (root / "ImageSets" / "Segmentation" / "val.txt").write_text("\n".join(ids) + "\n")


def make_imagenet_s(root: Path, masks_beside_images: bool = True) -> None:
    for wnid in ["n01440764", "n01443537"]:
        for k in range(2):
            img = root / "validation" / wnid / f"ILSVRC_{k}.JPEG"
            _rgb(img, 256, 256)
            ids = np.zeros((256, 256), np.int64)
            ids[64:192, 64:192] = 7             # category id 7
            ids[0:8, :] = 1000                  # ignore
            rgb = np.stack([ids % 256, ids // 256, np.zeros_like(ids)], -1).astype(np.uint8)
            mpath = (img.with_suffix(".png") if masks_beside_images
                     else root / "validation-segmentation" / wnid / f"ILSVRC_{k}.png")
            mpath.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(mpath)


def make_nih(root: Path) -> None:
    for name in ["00000001_000.png", "00000002_000.png"]:
        _rgb(root / "images" / name, 1024, 1024, mode="L")   # grayscale like NIH
    (root / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\n"
        "00000001_000.png,Infiltrate,100,100,200,200,,,\n"
        "00000001_000.png,Infiltrate,600,600,100,100,,,\n"
        "00000001_000.png,Mass,400,400,50,50,,,\n"
        "00000002_000.png,Cardiomegaly,300,500,400,300,,,\n"
    )


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    make_cub(tmp_path / "cub200")
    make_voc(tmp_path / "voc2012")
    make_imagenet_s(tmp_path / "imagenet_s50")
    make_nih(tmp_path / "nih_chestxray")
    return tmp_path


def _check_sample(ds, idx=0):
    x, y, m = ds[idx]
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32
    assert m.shape == (224, 224) and m.dtype == torch.int64
    assert set(m.unique().tolist()) <= {0, 1, IGNORE}
    assert isinstance(y, int)
    return x, y, m


# ---------------------------------------------------------------------------
# Per-dataset tests
# ---------------------------------------------------------------------------

def test_cub_uses_test_split_and_folder_order(data_root):
    ds = CUB200Eval(data_root / "cub200")
    assert len(ds) == 3                                   # only img_0 of each class
    assert [y for _, y in ds.samples] == [0, 1, 2]        # sorted folder order
    _, _, m = _check_sample(ds)
    assert m.sum() > 0
    assert m[0, 0] == 0                                   # weak (51) agreement → background


def test_voc_dominant_class_mask_and_ignore(data_root):
    ds = VOCEval(data_root / "voc2012")
    assert ds.skipped == ["2007_000003"]                  # no foreground → skipped
    assert [y for _, y in ds.samples] == [14, 0]          # seg idx 15 → "person"(14)
    assert VOC_CLASSES[14] == "person"
    _, _, m = _check_sample(ds, 0)
    assert (m == 1).any() and (m == IGNORE).any()
    # the smaller class-12 object must not be marked as target
    seg = ds._seg("2007_000001")
    raw = ds._load(0)[2]
    assert raw[seg == 12].max() == 0


@pytest.mark.parametrize("beside", [True, False])
def test_imagenet_s_mask_locations_and_rgb_decoding(tmp_path, beside):
    make_imagenet_s(tmp_path, masks_beside_images=beside)
    ds = ImageNetS50Eval(tmp_path)
    assert len(ds) == 4 and [s[2] for s in ds.samples] == [0, 0, 1, 1]
    raw = ds._load(0)[2]
    assert raw[128, 128] == 1 and raw[100 // 2, 0] == 0 and raw[0, 0] == IGNORE
    _check_sample(ds)


def test_imagenet_s_missing_mask_fails_loudly(tmp_path):
    make_imagenet_s(tmp_path)
    next((tmp_path / "validation").rglob("*.png")).unlink()
    with pytest.raises(FileNotFoundError, match="no mask"):
        ImageNetS50Eval(tmp_path)


def test_nih_bbox_union_alias_and_grayscale(data_root):
    ds = NIHBBoxEval(data_root / "nih_chestxray")
    labels = sorted(y for _, y, _ in ds.samples)
    assert len(ds) == 3                                    # (img1,Infiltration),(img1,Mass),(img2,Cardio)
    assert NIH_CLASSES.index("Infiltration") in labels     # "Infiltrate" alias resolved
    s = next(i for i, (n, y, _) in enumerate(ds.samples)
             if y == NIH_CLASSES.index("Infiltration"))
    raw = ds._load(s)[2]
    assert raw.sum() == 200 * 200 + 100 * 100              # union of both boxes
    x, _, _ = _check_sample(ds, s)                         # grayscale → 3 channels


def test_mask_geometry_matches_image(tmp_path):
    """A mask that marks the left half must stay the left half after resize+crop."""
    root = tmp_path / "nih"
    _rgb(root / "images" / "a.png", 400, 300, mode="L")
    (root / "BBox_List_2017.csv").write_text(
        "Image Index,Finding Label,Bbox [x,y,w,h],,,\na.png,Mass,0,0,200,300,,,\n")
    _, _, m = NIHBBoxEval(root)[0]
    # 400x300 → resize 341x256 → centre crop 224: left half ends near column 112
    cols = (m == 1).any(0).nonzero().flatten()
    assert cols.min() == 0 and 105 <= cols.max() <= 115


def test_loader_batches_and_spread_subset(data_root):
    loader = build_eval_loader("cub200", data_root, batch_size=2, max_samples=2, num_workers=0)
    x, y, m = next(iter(loader))
    assert x.shape == (2, 3, 224, 224) and m.shape == (2, 224, 224)
    assert y.tolist() == [0, 2]                            # spread, not first-N
    assert spread_indices(10, 3) == [0, 4, 9] or spread_indices(10, 3) == [0, 5, 9]


# ---------------------------------------------------------------------------
# run_phase3 end-to-end (tiny random models, no downloads)
# ---------------------------------------------------------------------------

def _tiny(arch):
    import timm

    def factory(name, num_classes=0, pretrained=True, **_):
        torch.manual_seed(0)
        return timm.create_model(arch, pretrained=False, num_classes=num_classes,
                                 img_size=224).eval()
    return factory


def test_run_phase3_end_to_end(data_root, tmp_path, monkeypatch):
    import run_phase3

    monkeypatch.setattr(run_phase3, "load_model", _tiny("vit_tiny_patch16_224"))
    weights = tmp_path / "weights"
    weights.mkdir()
    for ds, n in [("cub200", 200), ("pascal_voc", 20)]:
        m = run_phase3.load_model("vit_b16", num_classes=n)
        torch.save({"model_state_dict": m.state_dict(), "metadata": {}},
                   weights / f"vit_b16_ft_{ds}_20260101_ep001.pth")

    ckpt = tmp_path / "ckpt"
    run_phase3.main([
        "--data-root", str(data_root), "--datasets", "cub200", "pascal_voc",
        "--models", "vit_b16", "--explainers", "gradcam", "rollout",
        "--weights-dir", str(weights), "--checkpoint-dir", str(ckpt),
        "--batch-size", "2", "--num-workers", "0", "--device", "cpu",
    ])
    files = sorted(p.name for p in ckpt.glob("*.pkl"))
    assert files == [
        "cub200-vit_b16-gradcam.pkl", "cub200-vit_b16-rollout.pkl",
        "pascal_voc-vit_b16-gradcam.pkl", "pascal_voc-vit_b16-rollout.pkl",
    ]
    res = pickle.loads((ckpt / "pascal_voc-vit_b16-gradcam.pkl").read_bytes())
    assert res["n_samples"] == 2 and res["labels"] == [14, 0]
    assert all(a.shape == (14, 14) for a in res["attributions"])
    assert all(m is not None and m.shape == (224, 224) for m in res["gt_masks"])


def test_run_phase3_refuses_missing_weights(data_root, tmp_path, monkeypatch):
    import run_phase3

    monkeypatch.setattr(run_phase3, "load_model", _tiny("vit_tiny_patch16_224"))
    with pytest.raises(SystemExit):
        run_phase3.main([
            "--data-root", str(data_root), "--datasets", "cub200",
            "--models", "vit_b16", "--explainers", "gradcam",
            "--checkpoint-dir", str(tmp_path / "ckpt"),
            "--num-workers", "0", "--device", "cpu",
        ])
    assert not list((tmp_path / "ckpt").glob("*.pkl"))


def test_run_phase3_swin_skips_unsupported_explainers(data_root, tmp_path, monkeypatch):
    import run_phase3

    monkeypatch.setattr(run_phase3, "load_model", _tiny("swin_tiny_patch4_window7_224"))
    ckpt = tmp_path / "ckpt"
    run_phase3.main([
        "--data-root", str(data_root), "--datasets", "cub200", "--max-samples", "1",
        "--models", "swin_b", "--explainers", "rollout", "gradcam",
        "--allow-random-head", "--checkpoint-dir", str(ckpt),
        "--num-workers", "0", "--device", "cpu",
    ])
    assert sorted(p.name for p in ckpt.glob("*.pkl")) == ["cub200-swin_b-gradcam.pkl"]


def test_model_grid_patch_sizes():
    from run_phase3 import MODEL_GRID_PATCH
    from model_zoo import MODEL_REGISTRY
    assert set(MODEL_GRID_PATCH) == set(MODEL_REGISTRY)
    assert MODEL_GRID_PATCH["dino_vitb8"] == 8 and MODEL_GRID_PATCH["dinov2_vitb14"] == 14
