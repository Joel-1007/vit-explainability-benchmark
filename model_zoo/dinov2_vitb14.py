"""
Model 6 — DINOv2-ViT-B/14
Architecture : Standard Vision Transformer (self-distillation v2)
Pre-training  : DINOv2 on LVD-142M (142 M curated images)
Patch size    : 14 × 14  →  16 × 16 = 256 patch tokens at 224 × 224
Params        : ~86 M
IN-1K Top-1   : 86.5 %

HuggingFace (official Meta) : facebook/dinov2-base
timm identifier             : vit_base_patch14_dinov2.lvd142m
With register tokens        : facebook/dinov2-with-registers-base

IMPORTANT — use the standard variant (no register tokens).
            Register tokens alter attention distribution patterns and
            would complicate comparison with other models.

Explainability note
-------------------
Standard CLS-token architecture; all methods apply.
256 patch tokens (16 × 16 grid) at 224 × 224 — midpoint between
DINO-B/8 (784 patches) and standard ViT-B/16 (196 patches).
LVD-142M pre-training is orders of magnitude larger than IN-1K/21K.
State of the art in self-supervised visual features as of 2023-2024.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
HF_REPO        = "facebook/dinov2-base"
HF_REPO_REG    = "facebook/dinov2-with-registers-base"   # avoid — see note
TIMM_ID        = "vit_base_patch14_dinov2.lvd142m"
PATCH_SIZE     = 14
N_PATCHES      = (224 // PATCH_SIZE) ** 2   # 256


class DINOv2ViTB14(nn.Module):
    """
    Wrapper for DINOv2-ViT-B/14.

    Parameters
    ----------
    num_classes : int
        0  → CLS-token feature extractor (dim=768).
        >0 → attach linear classification head.
    pretrained : bool
        Load LVD-142M self-supervised weights via timm.
    use_hf : bool
        If True, load via HuggingFace Transformers instead of timm.
        (timm is recommended for benchmark consistency.)
    """

    def __init__(
        self,
        num_classes: int = 0,
        pretrained: bool = True,
        use_hf: bool = False,
    ):
        super().__init__()
        self.patch_size  = PATCH_SIZE
        self.num_classes = num_classes
        self._use_hf     = use_hf

        if use_hf:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(HF_REPO)
            self.embed_dim = self.backbone.config.hidden_size   # 768
            self.head = (
                nn.Linear(self.embed_dim, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        else:
            self.backbone = timm.create_model(
                TIMM_ID,
                pretrained=pretrained,
                num_classes=num_classes,
                img_size=224,
            )
            self.embed_dim = self.backbone.embed_dim   # 768
            self.head = nn.Identity()

    # ------------------------------------------------------------------
    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        if self._use_hf:
            self.head = (
                nn.Linear(self.embed_dim, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        else:
            self.backbone.reset_classifier(num_classes)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_hf:
            out = self.backbone(pixel_values=x)
            cls = out.last_hidden_state[:, 0]   # CLS token
            return self.head(cls)
        return self.backbone(x)

    # ------------------------------------------------------------------
    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return attention weights from the final transformer block.

        Returns
        -------
        attn : Tensor, shape (B, num_heads, N+1, N+1)  [N=256]
        """
        if self._use_hf:
            out = self.backbone(
                pixel_values=x,
                output_attentions=True,
            )
            # last layer attentions
            return out.attentions[-1]   # (B, H, N+1, N+1)

        # timm path
        b = self.backbone
        x = b.patch_embed(x)
        cls_token = b.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = b.pos_drop(x + b.pos_embed)

        attn = None
        for i, blk in enumerate(b.blocks):
            if i < len(b.blocks) - 1:
                x = blk(x)
            else:
                x_norm = blk.norm1(x)
                B, N, C = x_norm.shape
                qkv = blk.attn.qkv(x_norm).reshape(
                    B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * blk.attn.scale
                attn = attn.softmax(dim=-1)
        return attn   # (B, H, 257, 257)


# ---------------------------------------------------------------------------
def load_dinov2_vitb14(
    num_classes: int = 0,
    pretrained: bool = True,
    use_hf: bool = False,
) -> DINOv2ViTB14:
    """
    Load DINOv2-ViT-B/14 with LVD-142M self-supervised weights.

    Parameters
    ----------
    num_classes : int
        Downstream classification classes (0 = feature extractor).
    pretrained : bool
        Load pre-trained weights (via timm by default).
    use_hf : bool
        Use HuggingFace Transformers if True, else timm (recommended).
    """
    return DINOv2ViTB14(
        num_classes=num_classes,
        pretrained=pretrained,
        use_hf=use_hf,
    )


def load_dinov2_vitb14_hf(num_classes: int = 0):
    """Load DINOv2-ViT-B/14 via HuggingFace Transformers (reference)."""
    from transformers import AutoModel, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained(HF_REPO)
    model = AutoModel.from_pretrained(HF_REPO)
    return model, processor
