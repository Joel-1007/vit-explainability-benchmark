"""
Model 1 — ViT-B/16
Architecture : Standard Vision Transformer
Pre-training  : Supervised ImageNet-21K (augreg variant)
Patch size    : 16 × 16
Params        : ~86 M
IN-1K Top-1   : 84.2 %

timm identifier : vit_base_patch16_224.augreg_in21k
HuggingFace     : google/vit-base-patch16-224-in21k

Explainability note
-------------------
Standard CLS-token architecture.  All attention-based methods
(raw attention, rollout, GradCAM, etc.) apply directly.
Baseline model for the benchmark.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .normalize import PixelNormalize
from .specs import MODEL_SPECS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMM_ID = MODEL_SPECS["vit_b16"].timm_id   # pinned in model_zoo/specs.py
HF_REPO   = "google/vit-base-patch16-224-in21k"
PATCH_SIZE = 16
N_PATCHES  = (224 // PATCH_SIZE) ** 2   # 196


class ViTB16(nn.Module):
    """
    Thin wrapper around the timm ViT-B/16 backbone.

    Parameters
    ----------
    num_classes : int
        0  → return raw CLS-token features (dim=768).
        >0 → append a linear classification head.
    pretrained : bool
        Download ImageNet-21K weights when True.
    """

    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        # Pre-training normalisation, applied inside the model (pixel-space protocol)
        self.normalize = PixelNormalize(MODEL_SPECS["vit_b16"].mean, MODEL_SPECS["vit_b16"].std)
        self.backbone = timm.create_model(
            TIMM_ID,
            pretrained=pretrained,
            num_classes=num_classes,  # 0 strips the head
        )
        self.num_classes = num_classes
        self.patch_size  = PATCH_SIZE
        self.embed_dim   = self.backbone.embed_dim   # 768

    # ------------------------------------------------------------------
    # Head management
    # ------------------------------------------------------------------
    def reset_classifier(self, num_classes: int) -> None:
        """Replace (or strip) the classification head in-place."""
        self.backbone.reset_classifier(num_classes)
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))

    # ------------------------------------------------------------------
    # Attention extraction helper (for explainability methods)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def load_vit_b16(num_classes: int = 0, pretrained: bool = True) -> ViTB16:
    """
    Load ViT-B/16 with optional pre-trained IN-21K weights.

    Parameters
    ----------
    num_classes : int
        Downstream classes for classification head (0 = feature extractor).
    pretrained : bool
        Whether to load pre-trained weights.

    Returns
    -------
    ViTB16
    """
    return ViTB16(num_classes=num_classes, pretrained=pretrained)


# ---------------------------------------------------------------------------
# HuggingFace alternative (for reference / hash recording)
# ---------------------------------------------------------------------------
def load_vit_b16_hf(num_classes: int = 0):
    """
    Load ViT-B/16 via HuggingFace Transformers.
    Useful for hash recording of the HF checkpoint.
    """
    from transformers import ViTModel
    model = ViTModel.from_pretrained(HF_REPO)
    return model
