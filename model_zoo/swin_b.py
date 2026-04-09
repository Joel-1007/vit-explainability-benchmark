"""
Model 3 — Swin-B
Architecture : Swin Transformer (shifted-window local self-attention)
Pre-training  : Supervised IN-22K → fine-tuned IN-1K
Patch size    : 4 × 4 (merged into 7 × 7 windows)
Params        : ~88 M
IN-1K Top-1   : 85.2 %

timm identifier : swin_base_patch4_window7_224.ms_in22k_ft_in1k
HuggingFace     : microsoft/swin-base-patch4-window7-224

Explainability note
-------------------
CRITICAL LIMITATION — Swin has NO CLS token and uses LOCAL shifted-window
attention.  Standard attention rollout does NOT apply.

Supported explanation methods:
  ✓  GradCAM (on any stage's output feature map)
  ✓  Gradient × Input
  ✓  Integrated Gradients
  ✗  Raw attention rollout (cannot be directly applied)
  ✗  Attention-based methods assuming a global CLS token

This limitation must be flagged explicitly in the paper.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
TIMM_ID    = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
HF_REPO    = "microsoft/swin-base-patch4-window7-224"
PATCH_SIZE = 4   # base patch; merged in stages


class SwinB(nn.Module):
    """
    Wrapper for Swin-B.

    Parameters
    ----------
    num_classes : int
        0  → return final pooled feature vector (dim=1024).
        >0 → attach classification head.
    pretrained : bool
        Use IN-22K→IN-1K fine-tuned weights.
    """

    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            TIMM_ID,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.patch_size  = PATCH_SIZE
        # Swin-B final feature dim
        self.embed_dim   = self.backbone.num_features   # 1024

    # ------------------------------------------------------------------
    def reset_classifier(self, num_classes: int) -> None:
        self.backbone.reset_classifier(num_classes)
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    def get_stage_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Return the spatial feature maps after each Swin stage.
        Useful for GradCAM — hook onto stage 3 (final stage) output.

        Returns
        -------
        List of Tensors, one per stage, shape (B, H_i, W_i, C_i).
        """
        feature_maps = []

        x = self.backbone.patch_embed(x)          # (B, H/4, W/4, C)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        for layer in self.backbone.layers:
            x = layer(x)
            feature_maps.append(x)

        return feature_maps   # 4 stages for Swin-B

    # ------------------------------------------------------------------
    @property
    def grad_cam_target_layer(self) -> nn.Module:
        """
        Return the recommended target layer for GradCAM —
        the last Swin stage (stage index 3).
        """
        return self.backbone.layers[-1]


# ---------------------------------------------------------------------------
def load_swin_b(num_classes: int = 0, pretrained: bool = True) -> SwinB:
    """
    Load Swin-B with IN-22K→IN-1K weights via timm.

    Parameters
    ----------
    num_classes : int
        Downstream classification classes (0 = feature extractor).
    pretrained : bool
        Load pre-trained weights.
    """
    return SwinB(num_classes=num_classes, pretrained=pretrained)


def load_swin_b_hf(num_classes: int = 0):
    """Load Swin-B via HuggingFace Transformers."""
    from transformers import SwinModel
    return SwinModel.from_pretrained(HF_REPO)
