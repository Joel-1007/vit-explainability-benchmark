"""
specs.py — single source of truth for model identity and provenance.

Every wrapper in ``model_zoo`` builds its backbone from ``MODEL_SPECS[...]
.timm_id``; ``scripts/record_model_hashes.py`` and
``scripts/verify_model_zoo.py`` read the same table, so the hash log always
describes the exact artifact the benchmark loads.

All weights come from the timm organisation on the HuggingFace Hub
(``hf_repo = "timm/<timm_id>"``). The ``pretraining`` field states what the
weights have been trained on *before* our fine-tuning, including any
supervised stages — this is what the paper must report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str              # benchmark key
    display: str
    timm_id: str           # architecture.pretrained_tag (pinned)
    pretraining: str       # full provenance, including supervised stages
    supervision: str       # "supervised" | "self-supervised" | "self-supervised + supervised"
    patch_size: int        # native patch size (Swin: 4, merged ×8 in stage 4)
    prefix_tokens: int     # CLS / distillation tokens before the patch tokens
    reference: str
    main_matrix: bool = True   # False → ablation-only (A4)
    # Input normalisation used in PRE-TRAINING (applied inside the wrapper;
    # the data pipeline delivers [0, 1] pixels to every model).
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    @property
    def hf_repo(self) -> str:
        return f"timm/{self.timm_id}"


MODEL_SPECS = {
    "vit_b16": ModelSpec(
        name="vit_b16", display="ViT-B/16",
        timm_id="vit_base_patch16_224.augreg_in21k",
        pretraining="Supervised on ImageNet-21k (21,843 classes) with the AugReg "
                    "recipe; no ImageNet-1k fine-tuning.",
        supervision="supervised", patch_size=16, prefix_tokens=1,
        reference="Steiner et al., 2022 (How to train your ViT?)",
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    "deit_b16": ModelSpec(
        name="deit_b16", display="DeiT-B/16 (distilled)",
        timm_id="deit_base_distilled_patch16_224.fb_in1k",
        pretraining="Supervised on ImageNet-1k with hard distillation from a "
                    "RegNetY-16GF teacher (CLS + distillation token).",
        supervision="supervised", patch_size=16, prefix_tokens=2,
        reference="Touvron et al., 2021 (DeiT)",
    ),
    "swin_b": ModelSpec(
        name="swin_b", display="Swin-B",
        timm_id="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        pretraining="Supervised on ImageNet-22k, then fine-tuned on ImageNet-1k.",
        supervision="supervised", patch_size=4, prefix_tokens=0,
        reference="Liu et al., 2021 (Swin Transformer)",
    ),
    "beit_b16": ModelSpec(
        name="beit_b16", display="BEiT-B/16",
        timm_id="beit_base_patch16_224.in22k_ft_in22k",
        pretraining="Self-supervised masked image modelling on ImageNet-22k "
                    "(DALL-E tokenizer targets), FOLLOWED BY supervised intermediate "
                    "fine-tuning on ImageNet-22k (21,841 classes) by the BEiT authors "
                    "(original file beit_base_patch16_224_pt22k_ft22k.pth). The "
                    "pure-MIM checkpoint is not distributed by timm; MIM vs supervised "
                    "pre-training is isolated by ablation A4 (MAE vs ViT-B/16) instead.",
        supervision="self-supervised + supervised", patch_size=16, prefix_tokens=1,
        reference="Bao et al., 2022 (BEiT)",
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    "dino_vitb8": ModelSpec(
        name="dino_vitb8", display="DINO ViT-B/8",
        timm_id="vit_base_patch8_224.dino",
        pretraining="Self-supervised self-distillation (no labels) on ImageNet-1k; "
                    "backbone weights of the official release "
                    "(dino_vitbase8_pretrain.pth), no classification head.",
        supervision="self-supervised", patch_size=8, prefix_tokens=1,
        reference="Caron et al., 2021 (DINO)",
    ),
    "dinov2_vitb14": ModelSpec(
        name="dinov2_vitb14", display="DINOv2 ViT-B/14",
        timm_id="vit_base_patch14_dinov2.lvd142m",
        pretraining="Self-supervised on LVD-142M (distilled from ViT-g/14); "
                    "variant without register tokens. Positional embeddings are "
                    "interpolated from 518 px to 224 px.",
        supervision="self-supervised", patch_size=14, prefix_tokens=1,
        reference="Oquab et al., 2024 (DINOv2)",
    ),
    "mae_b16": ModelSpec(
        name="mae_b16", display="MAE ViT-B/16",
        timm_id="vit_base_patch16_224.mae",
        pretraining="Self-supervised masked autoencoder (pixel reconstruction) on "
                    "ImageNet-1k; encoder only.",
        supervision="self-supervised", patch_size=16, prefix_tokens=1,
        reference="He et al., 2022 (MAE)", main_matrix=False,
    ),
}


def expected_tokens(name: str, img_size: int = 224) -> Tuple[int, int]:
    """(prefix tokens, patch-grid side) for a ViT-style model at ``img_size``."""
    s = MODEL_SPECS[name]
    return s.prefix_tokens, img_size // s.patch_size
