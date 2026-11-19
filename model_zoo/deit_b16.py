"""
Model 2 — DeiT-B/16 (distilled)
Architecture : Data-efficient Image Transformer with distillation token
Pre-training  : Knowledge distillation (IN-1K, RegNetY-160 teacher)
Patch size    : 16 × 16
Params        : ~87 M
IN-1K Top-1   : 83.4 %

timm identifier : deit_base_distilled_patch16_224.fb_in1k
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

from .normalize import PixelNormalize
from .specs import MODEL_SPECS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMM_ID = MODEL_SPECS["deit_b16"].timm_id   # pinned in model_zoo/specs.py
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
        # Pre-training normalisation, applied inside the model (pixel-space protocol)
        self.normalize = PixelNormalize(MODEL_SPECS["deit_b16"].mean, MODEL_SPECS["deit_b16"].std)
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
        return self.backbone(self.normalize(x))



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
