"""
explainers/__init__.py  —  Phase 3 Task 3.1 Explainer Package
=============================================================
Exports all 7 explainer classes and the BaseExplainer ABC.

Explainer index
---------------
  E1  RawAttentionExplainer      raw_attention.py
  E2  AttentionRolloutExplainer  rollout.py
  E3  GradCAMExplainer           gradcam.py
  E4  CheferLRPExplainer         chefer_lrp.py     (pure-PyTorch)
  E5  RISEExplainer              rise.py           (vectorised, M=4000)
  E6  LIMEExplainer              lime.py           (patch-grid superpixels)
  E7  DIMEExplainer              dime.py           ⚠ PENDING RESOLUTION
                                  (guide requires DIME but DIME is a
                                   multimodal image+text method; raises
                                   NotImplementedError until resolved.
                                   See BENCHMARK.md §3.1 and dime.py.)

Swin-B compatibility
--------------------
E1, E2, E4 raise UnsupportedArchitectureError for Swin-B (no CLS token).
E3 (GradCAM) and E5, E6 are fully Swin-B compatible.
"""

from .base         import BaseExplainer, UnsupportedArchitectureError
from .raw_attention import RawAttentionExplainer
from .rollout      import AttentionRolloutExplainer
from .gradcam      import GradCAMExplainer
from .chefer_lrp   import CheferLRPExplainer
from .rise         import RISEExplainer
from .lime         import LIMEExplainer
from .dime         import DIMEExplainer

__all__ = [
    # ABC + exceptions
    "BaseExplainer",
    "UnsupportedArchitectureError",
    # Explainers E1–E7
    "RawAttentionExplainer",          # E1
    "AttentionRolloutExplainer",       # E2
    "GradCAMExplainer",               # E3
    "CheferLRPExplainer",             # E4
    "RISEExplainer",                  # E5
    "LIMEExplainer",                  # E6
    "DIMEExplainer",                  # E7  ⚠ pending
]
