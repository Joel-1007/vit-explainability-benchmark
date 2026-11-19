"""
Model 5 — DINO ViT-B/8
Architecture : Standard Vision Transformer (1 CLS token)
Pre-training : DINO self-distillation, no labels, ImageNet-1k
Patch size   : 8 × 8  →  28 × 28 = 784 patch tokens at 224 × 224
Params       : ~85 M

timm identifier : vit_base_patch8_224.dino
                  (HF Hub repo timm/vit_base_patch8_224.dino — the backbone of the
                  official release dino_vitbase8_pretrain.pth; no head)
Official file   : https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth

Loading
-------
Loaded through timm like every other model, so explainer hooks (``blocks``,
``attn.attn_drop``, ``num_prefix_tokens``) behave identically across the zoo.
The previous PyTorch-Hub loader (facebookresearch/dino repository) executed
remote code at load time and returned a non-timm module; it was removed.
``load_dino_vitb8_from_pth`` loads the official file into the same timm
architecture; ``scripts/verify_model_zoo.py --compare-dino-official`` checks
that both sources give identical tensors.

Explainability note
-------------------
The 8-pixel patch yields a 28×28 attribution grid — 4× the spatial resolution
of the 16-pixel models.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .normalize import PixelNormalize
from .specs import MODEL_SPECS

TIMM_ID    = MODEL_SPECS["dino_vitb8"].timm_id
ARCH       = TIMM_ID.split(".")[0]          # vit_base_patch8_224
DIRECT_URL = (
    "https://dl.fbaipublicfiles.com/dino/"
    "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
)
PATCH_SIZE = 8
N_PATCHES  = (224 // PATCH_SIZE) ** 2   # 784


class DINOViTB8(nn.Module):
    """
    Wrapper for DINO ViT-B/8 (timm).

    Parameters
    ----------
    num_classes : int
        0  → CLS-token features (dim 768).
        >0 → linear classification head on the CLS token.
    pretrained : bool
        Load the DINO weights from the timm HF Hub repo.
    """

    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        # Pre-training normalisation, applied inside the model (pixel-space protocol)
        self.normalize = PixelNormalize(MODEL_SPECS["dino_vitb8"].mean, MODEL_SPECS["dino_vitb8"].std)
        self.backbone = timm.create_model(
            TIMM_ID if pretrained else ARCH,
            pretrained=pretrained,
            num_classes=num_classes,
            global_pool="token",     # classify from the CLS token
        )
        self.num_classes = num_classes
        self.patch_size  = PATCH_SIZE
        self.embed_dim   = self.backbone.embed_dim   # 768

    def reset_classifier(self, num_classes: int) -> None:
        self.backbone.reset_classifier(num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x))


def load_dino_vitb8(num_classes: int = 0, pretrained: bool = True) -> DINOViTB8:
    """Load DINO ViT-B/8 with the official self-supervised weights (via timm)."""
    return DINOViTB8(num_classes=num_classes, pretrained=pretrained)


def official_state_dict(pth_path: str) -> dict:
    """
    Backbone tensors from the official ``dino_vitbase8_pretrain.pth``, with
    keys in timm naming (the official ViT uses the same names; any
    ``backbone.`` / ``module.`` prefixes and head tensors are dropped).
    """
    sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    for key in ("teacher", "state_dict", "model"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        if k.startswith("head."):
            continue
        out[k] = v
    return out


def load_dino_vitb8_from_pth(pth_path: str, num_classes: int = 0) -> DINOViTB8:
    """Load the official DINO file into the timm architecture (strict on the backbone)."""
    model = DINOViTB8(num_classes=num_classes, pretrained=False)
    missing, unexpected = model.backbone.load_state_dict(official_state_dict(pth_path), strict=False)
    head_only = all(k.startswith("head.") for k in missing)
    if unexpected or not head_only:
        raise RuntimeError(f"DINO official weights mismatch: missing={missing} unexpected={unexpected}")
    return model
