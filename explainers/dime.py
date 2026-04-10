"""
dime.py  —  E7  DIMEExplainer  (PENDING RESOLUTION)
====================================================

⚠ GUIDE COMPLIANCE NOTICE ⚠
----------------------------
The implementation guide (Task 3.1) explicitly lists ``DIMEExplainer``
as a required explainer class.

However, DIME — "Fine-grained Interpretations of Multimodal Models via
Disentangled Local Explanations" (Lyu & Apidianaki, AIES 2022) — is
designed exclusively for **multimodal models** that jointly process
image AND text inputs (e.g. LXMERT, MDETR).  It cannot be applied
to a single-input Vision Transformer in a meaningful way.

This inconsistency is flagged in the guide PDF as an open issue.  Rather
than silently substituting another method, this placeholder class:

1. Implements the full ``BaseExplainer`` interface so that any code that
   *imports* DIMEExplainer will not break at import time.
2. Raises ``NotImplementedError`` at call time with a human-readable
   explanation of the issue.
3. Is tracked in BENCHMARK.md §3.1 as "pending project-level signoff".

Resolution options (to be confirmed by the project lead before final runs):
  A. Confirm that the guide intended Gradient Attention Rollout
     (Chefer et al., ICCV 2021) — we implement that under the same name.
  B. Confirm a different "DIME" reference — we implement accordingly.
  C. Remove DIMEExplainer from the benchmark and adjust all result tables.

DO NOT remove this file — it documents an open issue in the methodology.
"""

from __future__ import annotations

import torch

from .base import BaseExplainer


class DIMEExplainer(BaseExplainer):
    """
    E7 — DIMEExplainer  [PENDING RESOLUTION — see module docstring]

    Parameters
    ----------
    model      : fine-tuned nn.Module.
    patch_size : ViT patch size (pixels).  Default 16.

    Notes
    -----
    This class is a **documented placeholder**.  ``explain()`` raises
    ``NotImplementedError`` with a full explanation of the guide
    inconsistency.  ``is_resolved`` is a class attribute that will be
    set to ``True`` once the issue is formally resolved.
    """

    is_resolved: bool = False  # Set to True once E7 is formally resolved.

    def explain(
        self,
        x:            torch.Tensor,
        target_class: int,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DIMEExplainer — PENDING RESOLUTION.\n\n"
            "The implementation guide lists DIMEExplainer as a required "
            "method, but DIME (Lyu & Apidianaki, AIES 2022) is a multimodal "
            "image+text framework that cannot be applied to single-input "
            "Vision Transformers.\n\n"
            "This cannot be silently substituted without project sign-off.\n"
            "See BENCHMARK.md §3.1 for the full discussion and resolution "
            "options (A/B/C).\n\n"
            "To resolve: update DIMEExplainer.is_resolved = True and "
            "implement explain() with the confirmed algorithm."
        )
