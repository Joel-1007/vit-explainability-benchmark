"""
Model 4 — BEiT-B/16
Architecture : ViT-B/16 with relative position bias and layer scale
Pre-training : Self-supervised masked image modelling on ImageNet-22k
               (DALL-E dVAE token targets), FOLLOWED BY supervised intermediate
               fine-tuning on ImageNet-22k (21,841 classes) by the BEiT authors.
Patch size   : 16 × 16
Params       : ~86 M

timm identifier : beit_base_patch16_224.in22k_ft_in22k
                  (HF Hub repo timm/beit_base_patch16_224.in22k_ft_in22k; ported
                  from the official beit_base_patch16_224_pt22k_ft22k.pth)

Provenance note (paper)
-----------------------
The pure-MIM checkpoint (microsoft/beit-base-patch16-224-pt22k) is not
distributed by timm. The only timm BEiT-B/16 weights include the supervised
IN-22k stage, so BEiT must be described as "MIM + supervised IN-22k", NOT as a
purely self-supervised model. The clean MIM-vs-supervised comparison is ablation
A4 (MAE-B/16 vs ViT-B/16, same architecture). The 21,841-class IN-22k head is
discarded; our fine-tuning attaches a new head.

Explainability note
-------------------
CLS-token architecture (1 prefix token, 14×14 patch grid); attention includes
a per-block relative position bias. Attention-based explainers capture the
post-softmax weights through timm's ``attn_drop`` hook (verified in
scripts/verify_model_zoo.py).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from .normalize import PixelNormalize
from .specs import MODEL_SPECS

# ---------------------------------------------------------------------------
TIMM_ID    = MODEL_SPECS["beit_b16"].timm_id
PATCH_SIZE = 16
N_PATCHES  = (224 // PATCH_SIZE) ** 2   # 196


class BEiTB16(nn.Module):
    """
    Wrapper for BEiT-B/16 (timm ``beit_base_patch16_224.in22k_ft_in22k``).

    Parameters
    ----------
    num_classes : int
        0  → return CLS-token features (dim=768).
        >0 → attach linear classification head.
    pretrained : bool
        Load the MIM + supervised IN-22k weights (see module provenance note).
    """

    def __init__(self, num_classes: int = 0, pretrained: bool = True):
        super().__init__()
        # Pre-training normalisation, applied inside the model (pixel-space protocol)
        self.normalize = PixelNormalize(MODEL_SPECS["beit_b16"].mean, MODEL_SPECS["beit_b16"].std)
        self.backbone = timm.create_model(
            TIMM_ID,
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
        return self.backbone(self.normalize(x))



# ---------------------------------------------------------------------------
def load_beit_b16(num_classes: int = 0, pretrained: bool = True) -> BEiTB16:
    """
    Load BEiT-B/16 (MIM + supervised IN-22k weights, see module docstring).

    Parameters
    ----------
    num_classes : int
        Downstream classification classes (0 = feature extractor).
    pretrained : bool
        Load pre-trained weights.
    """
    return BEiTB16(num_classes=num_classes, pretrained=pretrained)
