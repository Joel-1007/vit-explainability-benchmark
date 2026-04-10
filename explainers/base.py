"""
base.py  —  Phase 3 / Task 3.1  BaseExplainer
==============================================
Abstract base class that ALL explanation methods must subclass.
Also contains shared utilities used across explainer modules.

Public API
----------
BaseExplainer           ABC for all 7 explainers
UnsupportedArchitectureError
                        Raised when a method cannot operate on the model
                        (e.g. attention-based methods on Swin-B which has
                        no CLS token and uses local window attention).

Internal helpers (module-level, used by submodules)
----------------------------------------------------
_get_timm_blocks(model)     → list[nn.Module]
_has_cls_token(model)       → bool
_to_patch_grid(v, P)        → Tensor(P, P)
_capture_attn_weights(model, x, block_indices)
                            → list[Tensor(B,H,N,N)]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Public exceptions
# ---------------------------------------------------------------------------

class UnsupportedArchitectureError(RuntimeError):
    """
    Raised by attention-based explainers (E1, E2, E4, E7) when called on a
    model that has no global CLS token (e.g. Swin-B, which uses shifted-window
    local attention and hierarchical feature maps).

    For Swin-B, use GradCAMExplainer instead — see BENCHMARK.md §1.1.
    """


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class BaseExplainer(ABC):
    """
    All explanation methods must subclass this.

    Parameters
    ----------
    model      : fine-tuned nn.Module.  Set to eval() on construction.
    patch_size : ViT patch size (pixels).  Default 16.

    Contract
    --------
    ``explain(x, target_class)`` must return a 2-D attribution map
    of shape ``(H_img // patch_size, W_img // patch_size)``
    with raw (un-normalised) float values.

    ``explain_batch`` has a default loop implementation; override it
    for methods where batching is critical (RISE, LIME).
    """

    def __init__(self, model: nn.Module, patch_size: int = 16) -> None:
        self.model      = model
        self.patch_size = patch_size
        self.model.eval()

    @abstractmethod
    def explain(
        self,
        x:            torch.Tensor,
        target_class: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute a 2-D attribution map for one image.

        Parameters
        ----------
        x            : (3, H, W) float32 in [0, 1].
        target_class : integer class index.

        Returns
        -------
        torch.Tensor of shape (H // patch_size, W // patch_size),
        dtype float32, values un-normalised (any range).
        """

    def explain_batch(
        self,
        xs:             torch.Tensor,
        target_classes: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Vectorised batch explanation — defaults to a loop over ``explain``.

        Override this in any method where amortising setup cost across the
        batch yields significant speed-up (e.g. RISE mask generation, LIME
        sample generation).

        Parameters
        ----------
        xs             : (B, 3, H, W) float32 in [0, 1].
        target_classes : (B,) int64.

        Returns
        -------
        (B, H // patch_size, W // patch_size) float32 tensor.
        """
        return torch.stack([
            self.explain(xs[i], int(target_classes[i].item()), **kwargs)
            for i in range(len(xs))
        ])


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _get_timm_blocks(model: nn.Module) -> list:
    """
    Return the ordered list of transformer blocks from a timm-style ViT.

    Tries in order: ``model.blocks``, ``model.layers``,
    ``model.encoder.layers``.  Works for ViT-B/16, DeiT, BEiT, DINO,
    DINOv2, and the MockViT used in unit tests.

    Raises
    ------
    AttributeError  if no block container is found.
    """
    _ATTRS = ("blocks", "layers", "encoder.layers")
    for attr in _ATTRS:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        except AttributeError:
            continue
    raise AttributeError(
        f"Cannot locate transformer blocks on {type(model).__name__}. "
        f"Tried: {_ATTRS}.  Pass a timm ViT / DeiT / BEiT / DINO model."
    )


def _has_cls_token(model: nn.Module) -> bool:
    """Return True if the model has a CLS token parameter."""
    return hasattr(model, "cls_token")


def _to_patch_grid(
    v:    torch.Tensor,
    P:    int,
) -> torch.Tensor:
    """
    Reshape a flat attribution vector of length P² to a (P, P) grid.

    Parameters
    ----------
    v : 1-D tensor with numel == P * P.
    P : number of patches per spatial dimension.
    """
    assert v.numel() == P * P, (
        f"Expected {P*P} elements for a {P}×{P} patch grid, got {v.numel()}."
    )
    return v.reshape(P, P)


def _capture_attn_weights(
    model:         nn.Module,
    x:             torch.Tensor,
    block_indices: Sequence[int],
) -> List[torch.Tensor]:
    """
    Run a forward pass and capture raw attention weights (B, H, N, N)
    from the requested block indices.

    Strategy
    --------
    Hooks ``block.attn.attn_drop`` for each target block.  This works for:
    • timm ViT (with ``fused_attn=False``): ``attn_drop`` receives the
      post-softmax attention matrix as its first input argument.
    • MockViT (unit tests): same ``attn_drop`` interface.

    For models that use fused SDPA (``fused_attn=True``), temporarily
    disables it to obtain explicit attention tensors.

    Parameters
    ----------
    model         : the ViT model (eval mode assumed).
    x             : (B, 3, H, W) input batch.
    block_indices : which blocks to hook (negative indices allowed).

    Returns
    -------
    List of (B, H, N, N) attention tensors, one per requested block,
    in the same order as ``block_indices``.
    """
    if not _has_cls_token(model):
        raise UnsupportedArchitectureError(
            f"{type(model).__name__} has no CLS token.  Attention-based "
            "explainers (E1 RawAttention, E2 Rollout, E4 CheferLRP) require "
            "a global CLS token.  Use GradCAMExplainer for this model."
        )

    blocks   = _get_timm_blocks(model)
    n_blocks = len(blocks)
    # Normalise negative indices
    indices  = [i % n_blocks for i in block_indices]

    captured  = {}     # block_idx → Tensor(B, H, N, N)
    handles   = []
    fused_states: dict = {}  # block_idx → original fused_attn value

    for idx in indices:
        attn_mod = blocks[idx].attn

        # Disable fused SDPA so weights are materialised
        if getattr(attn_mod, "fused_attn", False):
            fused_states[idx] = True
            attn_mod.fused_attn = False

        def _make_hook(i):
            def _hook(module, inp, out):
                # inp[0] is the attention weight tensor fed into Dropout/Identity
                captured[i] = inp[0].detach()
            return _hook

        if hasattr(attn_mod, "attn_drop"):
            h = attn_mod.attn_drop.register_forward_hook(_make_hook(idx))
        else:
            # Fallback: hook the whole attn module and read _attn_weights attr
            def _make_fallback(i):
                def _hook(module, inp, out):
                    if hasattr(module, "_attn_weights"):
                        captured[i] = module._attn_weights.detach()
                return _hook
            h = attn_mod.register_forward_hook(_make_fallback(idx))

        handles.append(h)

    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
        for idx, orig in fused_states.items():
            blocks[idx].attn.fused_attn = orig

    return [captured.get(i) for i in indices]
