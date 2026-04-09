"""
Model 2 — DeiT-B/16 (distilled)
Architecture : Data-efficient Image Transformer with distillation token
Pre-training  : Knowledge distillation (IN-1K, RegNetY-160 teacher)
Patch size    : 16 × 16
Params        : ~87 M
IN-1K Top-1   : 83.4 %

timm identifier : deit_base_distilled_patch16_224
HuggingFace     : facebook/deit-base-distilled-patch16-224

Explainability note
-------------------
Has BOTH a CLS token and a distillation token.
When extracting attention maps, document whether you attend to
the CLS token, the distillation token, or their average.
The distillation token carries CNN-teacher knowledge and may
produce different spatial distributions than the CLS token.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMM_ID    = "deit_base_distilled_patch16_224"
HF_REPO    = "facebook/deit-base-distilled-patch16-224"
PATCH_SIZE = 16
N_PATCHES  = (224 // PATCH_SIZE) ** 2   # 196


class DeiTB16(nn.Module):
    """
    Wrapper for DeiT-B/16 distilled.

    Parameters
    ----------
    num_classes : int
        0  → return concatenated (CLS + distillation) feature vector.
        >0 → full classification head (DeiT averages both tokens).
    pretrained : bool
        Download IN-1K distilled weights when True.
    attention_mode : str
        'cls'   — use CLS  token attention for explanation methods.
        'dist'  — use distillation token attention.
        'mean'  — average both.  (Default: 'cls')
    """

    def __init__(
        self,
        num_classes: int = 0,
        pretrained: bool = True,
        attention_mode: str = "cls",
    ):
        super().__init__()
        self.backbone = timm.create_model(
            TIMM_ID,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.num_classes    = num_classes
        self.patch_size     = PATCH_SIZE
        self.embed_dim      = self.backbone.embed_dim   # 768
        self.attention_mode = attention_mode

    # ------------------------------------------------------------------
    def reset_classifier(self, num_classes: int) -> None:
        self.backbone.reset_classifier(num_classes)
        self.num_classes = num_classes

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    # ------------------------------------------------------------------
    def get_last_selfattention(
        self,
        x: torch.Tensor,
        token: str | None = None,
    ) -> torch.Tensor:
        """
        Return attention weights from the last block.

        Parameters
        ----------
        token : str | None
            'cls'  → CLS token row  (index 0)
            'dist' → distillation token row  (index 1)
            'mean' → mean of both rows
            None   → use self.attention_mode

        Returns
        -------
        attn : Tensor, shape (B, H, N+2, N+2)
            Indices: 0=CLS, 1=distillation, 2..N+1=patch tokens
        """
        mode = token or self.attention_mode

        # Forward through patch embedding + positional encoding
        x = self.backbone.patch_embed(x)
        cls_tok  = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        dist_tok = self.backbone.dist_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tok, dist_tok, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        attn = None
        for i, blk in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
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

        if mode == "cls":
            return attn[:, :, 0:1, :]    # (B, H, 1, N+2)
        elif mode == "dist":
            return attn[:, :, 1:2, :]    # (B, H, 1, N+2)
        elif mode == "mean":
            return attn[:, :, :2, :].mean(dim=2, keepdim=True)
        else:
            return attn                  # Full (B, H, N+2, N+2)


# ---------------------------------------------------------------------------
def load_deit_b16(
    num_classes: int = 0,
    pretrained: bool = True,
    attention_mode: str = "cls",
) -> DeiTB16:
    """
    Load DeiT-B/16 distilled with optional IN-1K distilled weights.

    Parameters
    ----------
    num_classes : int
        Downstream classes (0 = feature extractor).
    pretrained : bool
        Load pre-trained weights.
    attention_mode : str
        Default token for attention extraction: 'cls', 'dist', or 'mean'.
    """
    return DeiTB16(
        num_classes=num_classes,
        pretrained=pretrained,
        attention_mode=attention_mode,
    )


def load_deit_b16_hf(num_classes: int = 0):
    """Load DeiT-B/16 distilled via HuggingFace Transformers."""
    from transformers import DeiTModel
    return DeiTModel.from_pretrained(HF_REPO)
