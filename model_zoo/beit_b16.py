"""
Model 4 — BEiT-B/16
Architecture : BERT-style Masked Image Modelling ViT
Pre-training  : Masked Image Modelling on IN-22K
                (DALL-E discrete VAE token targets)
Patch size    : 16 × 16
Params        : ~86 M
IN-1K Top-1   : 85.2 % (after downstream fine-tune)

timm identifier (pre-train only) : beit_base_patch16_224.in22k
HuggingFace  (pre-train only)    : microsoft/beit-base-patch16-224-pt22k

IMPORTANT — always load the pre-train-only checkpoint.
            The already-fine-tuned checkpoint (beit_..._in22k_in1k)
            must NOT be used as the starting point; your standardised
            fine-tuning recipe is applied on top of pre-trained weights.

Explainability note
-------------------
Standard CLS-token architecture identical to ViT-B/16.
All attention-based methods apply directly.
The discrete-token MIM objective may produce more semantically
structured representations than pixel-reconstruction MAE.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
TIMM_ID_PRETRAIN = "beit_base_patch16_224.in22k"           # pre-train only ✓
TIMM_ID_FINETUNED = "beit_base_patch16_224.in22k_ft_in22k_in1k"  # reference only
HF_REPO_PRETRAIN  = "microsoft/beit-base-patch16-224-pt22k"
PATCH_SIZE = 16
N_PATCHES  = (224 // PATCH_SIZE) ** 2   # 196


class BEiTB16(nn.Module):
    """
    Wrapper for BEiT-B/16 (pre-train checkpoint only).

    Parameters
    ----------
    num_classes : int
        0  → return CLS-token features (dim=768).
        >0 → attach linear classification head.
    pretrained : bool
        Load IN-22K pre-trained weights (NOT the fine-tuned version).
    """

    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            TIMM_ID_PRETRAIN,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_classes = num_classes
        self.patch_size  = PATCH_SIZE
        self.embed_dim   = self.backbone.embed_dim   # 768

    # ------------------------------------------------------------------
    def reset_classifier(self, num_classes: int) -> None:
        self.backbone.reset_classifier(num_classes)
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    def get_last_selfattention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return attention weights from the final transformer block.

        Returns
        -------
        attn : Tensor, shape (B, num_heads, N+1, N+1)
            Index 0 = CLS token.
        """
        # BEiT uses relative position bias; timm handles this internally.
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # BEiT does NOT use a learned pos embed on input (uses rel-pos bias)
        x = self.backbone.pos_drop(x)

        attn = None
        for i, blk in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
                x = blk(x)
            else:
                x_norm = blk.norm1(x)
                B, N, C = x_norm.shape
                head_dim = C // blk.attn.num_heads
                qkv = blk.attn.qkv(x_norm).reshape(
                    B, N, 3, blk.attn.num_heads, head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv.unbind(0)
                scale = head_dim ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                # BEiT adds relative position bias
                if hasattr(blk.attn, "get_rel_pos_bias"):
                    attn = attn + blk.attn.get_rel_pos_bias()
                attn = attn.softmax(dim=-1)
        return attn   # (B, H, N+1, N+1)


# ---------------------------------------------------------------------------
def load_beit_b16(num_classes: int = 0, pretrained: bool = True) -> BEiTB16:
    """
    Load BEiT-B/16 pre-train-only checkpoint.

    Parameters
    ----------
    num_classes : int
        Downstream classification classes (0 = feature extractor).
    pretrained : bool
        Load IN-22K pre-trained (MIM) weights.
    """
    return BEiTB16(num_classes=num_classes, pretrained=pretrained)


def load_beit_b16_hf():
    """Load BEiT-B/16 pre-train weights via HuggingFace Transformers."""
    from transformers import BeitModel
    return BeitModel.from_pretrained(HF_REPO_PRETRAIN)
