"""
Ablation model (A4) — MAE ViT-B/16
Architecture : Standard Vision Transformer (identical to ViT-B/16)
Pre-training : Masked Autoencoder, self-supervised on ImageNet-1K (He et al., 2022)
Patch size   : 16 × 16
Params       : ~86 M

timm identifier : vit_base_patch16_224.mae   (encoder only, no head)

Purpose
-------
Phase 4 ablation A4 (pre-training objective): compared against ViT-B/16
(supervised) with the same architecture and the same fine-tuning protocol,
so pre-training is the only variable. Not part of the main 6-model matrix.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

TIMM_ID    = "vit_base_patch16_224.mae"
PATCH_SIZE = 16


class MAEB16(nn.Module):
    """Thin wrapper around the timm MAE ViT-B/16 encoder (CLS-token head)."""

    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            TIMM_ID,
            pretrained=pretrained,
            num_classes=num_classes,  # 0 = feature extractor
            global_pool="token",      # CLS-token head, same as ViT-B/16
        )
        self.num_classes = num_classes
        self.patch_size  = PATCH_SIZE
        self.embed_dim   = self.backbone.embed_dim

    def reset_classifier(self, num_classes: int) -> None:
        self.backbone.reset_classifier(num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_mae_b16(num_classes: int = 0, pretrained: bool = True) -> MAEB16:
    return MAEB16(num_classes=num_classes, pretrained=pretrained)
