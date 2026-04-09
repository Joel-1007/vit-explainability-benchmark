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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMM_ID   = "vit_base_patch16_224.augreg_in21k"
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
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Attention extraction helper (for explainability methods)
    # ------------------------------------------------------------------
    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the attention weight matrix from the final transformer block.

        Returns
        -------
        attn : Tensor, shape (B, num_heads, N+1, N+1)
            N = number of patch tokens;  position 0 is the CLS token.
        """
        # timm ViT exposes forward_features and individual blocks
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        attn = None
        for i, blk in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
                x = blk(x)
            else:
                # Capture attention weights from the last block
                x_norm = blk.norm1(x)
                B, N, C = x_norm.shape
                qkv = blk.attn.qkv(x_norm).reshape(
                    B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                scale = blk.attn.scale
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
        return attn   # (B, H, N+1, N+1)


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
