"""
eval_datasets.py — evaluation datasets with ground-truth localization masks.

Every dataset yields ``(image, label, gt_mask)``:

    image   : (3, 224, 224) float32, ImageNet-normalised
    label   : int, class index in the order the fine-tuned head uses
    gt_mask : (224, 224) int64 — 1 = object, 0 = background, 255 = ignore

Image and mask go through the same geometry (resize shorter side to 256,
centre-crop 224); masks use nearest-neighbour resampling so labels stay
discrete. ``metrics.localization._binary_gt`` treats 255 as "ignore".

Class orders (fine-tuning MUST use the same order, or the explained class
is wrong):

* CUB-200      : sorted class-folder names under ``images/`` — identical to
                 ``ImageFolder``, which ``scripts/pilot_finetune.py`` uses.
* ImageNet-S50 : sorted WordNet-ID folder names under ``validation/``.
* PASCAL VOC   : the 20 standard VOC classes (``VOC_CLASSES``); segmentation
                 index c ∈ 1..20 → label c-1.
* NIH CXR-14   : the 14 standard findings (``NIH_CLASSES``).

Expected layouts (under ``<data_root>``)::

    cub200/        images/<class>/*.jpg, segmentations/<class>/*.png,
                   images.txt, train_test_split.txt
    voc2012/       JPEGImages/, SegmentationClass/, ImageSets/Segmentation/val.txt
    imagenet_s50/  validation/<wnid>/*.JPEG, masks either next to the images
                   (<stem>.png) or in validation-segmentation/<wnid>/<stem>.png
    nih_chestxray/ images/*.png, BBox_List_2017.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

IGNORE = 255
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)

NIH_CLASSES = (
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
)
# BBox_List_2017.csv spells one finding differently from Data_Entry_2017.
_NIH_BBOX_ALIASES = {"Infiltrate": "Infiltration"}

NUM_CLASSES = {
    "cub200": 200,
    "pascal_voc": len(VOC_CLASSES),
    "imagenet_s50": 50,
    "nih_chestxray": len(NIH_CLASSES),
}


# ---------------------------------------------------------------------------
# Joint image / mask transform
# ---------------------------------------------------------------------------

class PairTransform:
    """Resize shorter side → ``resize``, centre-crop ``size``; same geometry for both."""

    def __init__(self, size: int = 224, resize: int = 256) -> None:
        self.size, self.resize = size, resize
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, img: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        m = Image.fromarray(mask.astype(np.uint8))
        img = TF.center_crop(TF.resize(img, self.resize, InterpolationMode.BILINEAR), self.size)
        m   = TF.center_crop(TF.resize(m,   self.resize, InterpolationMode.NEAREST),  self.size)
        x = self.normalize(TF.to_tensor(img))
        return x, torch.from_numpy(np.array(m, dtype=np.int64))


class _MaskedDataset(Dataset):
    """Base: subclasses fill ``self.samples`` and implement ``_load``."""

    samples: list

    def __init__(self, transform: Optional[Callable] = None) -> None:
        self.transform = transform or PairTransform()

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, idx: int) -> Tuple[Image.Image, int, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        img, label, mask = self._load(idx)
        if mask.shape != (img.height, img.width):
            raise ValueError(
                f"{type(self).__name__}[{idx}]: mask {mask.shape} does not match "
                f"image {(img.height, img.width)}"
            )
        x, m = self.transform(img, mask)
        return x, label, m


# ---------------------------------------------------------------------------
# CUB-200-2011 (official test split)
# ---------------------------------------------------------------------------

class CUB200Eval(_MaskedDataset):
    """CUB-200 test split. Masks are annotator-agreement maps; ≥ 128 → object."""

    def __init__(self, root: str | Path, transform: Optional[Callable] = None) -> None:
        super().__init__(transform)
        self.root = Path(root)
        classes = sorted(p.name for p in (self.root / "images").iterdir() if p.is_dir())
        cls_idx = {c: i for i, c in enumerate(classes)}

        id_to_path = {}
        for line in (self.root / "images.txt").read_text().splitlines():
            if line.strip():
                i, rel = line.split()
                id_to_path[i] = rel
        test_ids = [
            parts[0] for parts in
            (l.split() for l in (self.root / "train_test_split.txt").read_text().splitlines())
            if len(parts) == 2 and parts[1] == "0"
        ]
        self.samples = [
            (id_to_path[i], cls_idx[id_to_path[i].split("/")[0]])
            for i in sorted(test_ids, key=int)
        ]

    def _load(self, idx):
        rel, label = self.samples[idx]
        img  = Image.open(self.root / "images" / rel).convert("RGB")
        seg  = np.array(Image.open(self.root / "segmentations" / Path(rel).with_suffix(".png")).convert("L"))
        mask = (seg >= 128).astype(np.uint8)
        return img, label, mask


# ---------------------------------------------------------------------------
# PASCAL VOC 2012 segmentation val
# ---------------------------------------------------------------------------

class VOCEval(_MaskedDataset):
    """
    VOC 2012 segmentation val. VOC images are multi-object, so each image is
    assigned its *dominant* class (most labelled pixels); the mask marks that
    class's pixels as 1, other objects and background as 0, and VOC's
    object-boundary band (255) as ignore.
    """

    def __init__(self, root: str | Path, split: str = "val",
                 transform: Optional[Callable] = None) -> None:
        super().__init__(transform)
        self.root = Path(root)
        ids = [l.strip() for l in
               (self.root / "ImageSets" / "Segmentation" / f"{split}.txt").read_text().splitlines()
               if l.strip()]
        self.samples = []
        self.skipped: List[str] = []
        for img_id in ids:
            seg = self._seg(img_id)
            counts = np.bincount(seg[(seg > 0) & (seg != IGNORE)].ravel(), minlength=21)
            if counts.sum() == 0:
                self.skipped.append(img_id)
                continue
            c = int(counts.argmax())
            self.samples.append((img_id, c - 1))

    def _seg(self, img_id: str) -> np.ndarray:
        # Palette PNG: np.array yields class indices directly.
        return np.array(Image.open(self.root / "SegmentationClass" / f"{img_id}.png"))

    def _load(self, idx):
        img_id, label = self.samples[idx]
        img  = Image.open(self.root / "JPEGImages" / f"{img_id}.jpg").convert("RGB")
        seg  = self._seg(img_id)
        mask = np.where(seg == label + 1, 1, 0).astype(np.uint8)
        mask[seg == IGNORE] = IGNORE
        return img, label, mask


# ---------------------------------------------------------------------------
# ImageNet-S50 validation
# ---------------------------------------------------------------------------

class ImageNetS50Eval(_MaskedDataset):
    """
    ImageNet-S50 validation. Official masks encode the category as
    R + 256·G (0 = background, 1000 = ignore); single-channel masks are
    read as-is. Any non-zero, non-ignore id counts as object.
    """

    def __init__(self, root: str | Path, split: str = "validation",
                 transform: Optional[Callable] = None) -> None:
        super().__init__(transform)
        self.root = Path(root)
        split_dir = self.root / split
        classes = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        seg_dir = self.root / f"{split}-segmentation"

        self.samples, missing = [], []
        for label, wnid in enumerate(classes):
            for img_path in sorted((split_dir / wnid).glob("*.JPEG")):
                mask_path = img_path.with_suffix(".png")
                if not mask_path.exists():
                    mask_path = seg_dir / wnid / f"{img_path.stem}.png"
                if mask_path.exists():
                    self.samples.append((img_path, mask_path, label))
                else:
                    missing.append(img_path.name)
        if missing:
            raise FileNotFoundError(
                f"ImageNet-S50: {len(missing)} images have no mask "
                f"(e.g. {missing[:3]}). Looked next to each image and in {seg_dir}/."
            )

    def _load(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        raw = np.array(Image.open(mask_path)).astype(np.int64)
        ids = raw[..., 0] + 256 * raw[..., 1] if raw.ndim == 3 else raw
        mask = (ids > 0).astype(np.uint8)
        mask[ids == 1000] = IGNORE
        return img, label, mask


# ---------------------------------------------------------------------------
# NIH ChestX-ray14 bounding-box subset
# ---------------------------------------------------------------------------

class NIHBBoxEval(_MaskedDataset):
    """
    NIH ChestX-ray14 images that have expert bounding boxes
    (BBox_List_2017.csv, ~880 images / 984 boxes). One sample per
    (image, finding); the mask is the union of that finding's boxes.
    Only these images have localization ground truth.

    Patient-level leakage guard: NIH images of the same patient share the
    filename prefix ``<patient>_<follow-up>.png``. If ``train_val_list.txt``
    is present, box images whose *patient* appears in it are excluded (the
    fine-tuned model may have seen that patient). Exclusions are recorded in
    ``self.leakage`` and logged; ``require_split=True`` makes a missing
    train_val_list an error.
    """

    def __init__(self, root: str | Path, transform: Optional[Callable] = None,
                 exclude_train_patients: bool = True, require_split: bool = False) -> None:
        super().__init__(transform)
        import pandas as pd

        self.root = Path(root)
        df = pd.read_csv(self.root / "BBox_List_2017.csv")
        # Header is 'Image Index, Finding Label, Bbox [x, y, w, h]' (+ empty cols)
        cols = list(df.columns)
        x_col, y_col, w_col, h_col = cols[2], cols[3], cols[4], cols[5]

        boxes = {}
        for _, r in df.iterrows():
            finding = _NIH_BBOX_ALIASES.get(r[cols[1]], r[cols[1]])
            if finding not in NIH_CLASSES:
                raise ValueError(f"NIH: unknown finding label {r[cols[1]]!r}")
            key = (r[cols[0]], NIH_CLASSES.index(finding))
            boxes.setdefault(key, []).append(
                (float(r[x_col]), float(r[y_col]), float(r[w_col]), float(r[h_col]))
            )
        samples = [(img, label, b) for (img, label), b in sorted(boxes.items())]

        self.leakage = {"checked": False, "excluded_samples": 0, "excluded_patients": []}
        split_file = self.root / "train_val_list.txt"
        if exclude_train_patients and split_file.exists():
            train_patients = {nih_patient_id(l.strip())
                              for l in split_file.read_text().splitlines() if l.strip()}
            leaky = sorted({nih_patient_id(img) for img, _, _ in samples} & train_patients)
            samples = [s for s in samples if nih_patient_id(s[0]) not in train_patients]
            self.leakage = {"checked": True, "excluded_patients": leaky,
                            "excluded_samples": len(boxes) - len(samples)}
            if leaky:
                import warnings
                warnings.warn(
                    f"NIH: excluded {self.leakage['excluded_samples']} box samples from "
                    f"{len(leaky)} patients that also appear in train_val_list.txt",
                    RuntimeWarning, stacklevel=2,
                )
        elif require_split:
            raise FileNotFoundError(f"{split_file} not found; cannot check patient leakage")
        self.samples = samples

    def _load(self, idx):
        img_name, label, boxes = self.samples[idx]
        img  = Image.open(self.root / "images" / img_name).convert("RGB")
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        for x, y, w, h in boxes:
            x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
            x1, y1 = min(int(round(x + w)), img.width), min(int(round(y + h)), img.height)
            mask[y0:y1, x0:x1] = 1
        return img, label, mask


def nih_patient_id(image_name: str) -> str:
    """'00000013_005.png' → '00000013' (NIH filenames are <patient>_<follow-up>)."""
    return Path(image_name).name.split("_", 1)[0]


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

DATASET_CLASSES = {
    "cub200":        ("cub200",        CUB200Eval),
    "pascal_voc":    ("voc2012",       VOCEval),
    "imagenet_s50":  ("imagenet_s50",  ImageNetS50Eval),
    "nih_chestxray": ("nih_chestxray", NIHBBoxEval),
}


def spread_indices(n: int, k: int) -> List[int]:
    """k evenly spaced indices in [0, n) — a deterministic, class-spread subset."""
    if k >= n:
        return list(range(n))
    return [int(i) for i in np.linspace(0, n - 1, k).round().astype(int)]


def build_eval_loader(
    name: str,
    data_root: str | Path,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
) -> DataLoader:
    """DataLoader yielding ``(images, labels, gt_masks)`` for dataset ``name``."""
    subdir, cls = DATASET_CLASSES[name]
    dataset: Dataset = cls(Path(data_root) / subdir)
    if max_samples is not None and max_samples < len(dataset):
        dataset = Subset(dataset, spread_indices(len(dataset), max_samples))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
