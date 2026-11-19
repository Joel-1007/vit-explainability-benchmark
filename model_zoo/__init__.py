"""
model_zoo — Task 1.2
Six pre-trained backbones for the ViT Explainability Benchmark.

Exports
-------
load_vit_b16, load_deit_b16, load_swin_b,
load_beit_b16, load_dino_vitb8, load_dinov2_vitb14,
load_model   (dispatcher)
"""

from .vit_b16      import load_vit_b16
from .deit_b16     import load_deit_b16
from .swin_b       import load_swin_b
from .beit_b16     import load_beit_b16
from .dino_vitb8   import load_dino_vitb8
from .dinov2_vitb14 import load_dinov2_vitb14
from .mae_b16      import load_mae_b16      # ablation A4 only

MODEL_REGISTRY = {
    "vit_b16":       load_vit_b16,
    "deit_b16":      load_deit_b16,
    "swin_b":        load_swin_b,
    "beit_b16":      load_beit_b16,
    "dino_vitb8":    load_dino_vitb8,
    "dinov2_vitb14": load_dinov2_vitb14,
}

# Models used only by Phase 4 ablations (not part of the main 6-model matrix)
ABLATION_MODEL_REGISTRY = {
    "mae_b16":       load_mae_b16,       # A4: MAE vs supervised ViT-B/16
}


def load_model(name: str, num_classes: int = 0, **kwargs):
    """
    Unified model loader.

    Parameters
    ----------
    name : str
        One of the keys in MODEL_REGISTRY.
    num_classes : int
        Number of downstream classes.  0 = strip classifier head
        (returns CLS-token features only).
    """
    registry = {**MODEL_REGISTRY, **ABLATION_MODEL_REGISTRY}
    if name not in registry:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(registry.keys())}"
        )
    return registry[name](num_classes=num_classes, **kwargs)


__all__ = list(MODEL_REGISTRY.keys()) + ["load_mae_b16", "load_model", "MODEL_REGISTRY", "ABLATION_MODEL_REGISTRY"]
