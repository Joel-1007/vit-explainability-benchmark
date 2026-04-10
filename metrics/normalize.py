"""
normalize.py  —  Phase 3 / Task 3.2  Attribution Normalisation
===============================================================
Guide Listing 3 — canonical normalisation pipeline that sits between
explainers (raw attribution maps) and metrics (expect [0,1] inputs).

Public API
----------
normalize_attribution(att_map, mode='minmax')
    Normalise a single (Hp, Wp) or batched (B, Hp, Wp) attribution map.

normalize_batch(att_maps, mode='minmax')
    Convenience wrapper for (B, Hp, Wp) tensors.

NormMode                  String enum for mode validation.
AttributionNormError      Raised on unrecognised modes or shape errors.

Modes
-----
'minmax'     (default)
    att_norm = (att - min) / (max - min)
    Degenerate case (max == min): returns all-zeros map.
    Preserves spatial structure; scale-invariant.

'percentile'
    Clamps att at 99th percentile then applies minmax.
    Removes outlier hot-spots that can saturate metrics.
    Percentile is computed per-sample (not across the batch).

'softmax'
    att_norm = softmax(att.flatten()).reshape(Hp, Wp)
    Produces a proper probability distribution over patches.
    Used by EGT (L3) and Effective Mass Ratio (C3).
    Numerically stable: subtracted max before exp.

Notes
-----
- All operations are pure-PyTorch (no external deps beyond torch).
- Input may be any float dtype; output is always float32.
- Negative values are accepted; only 'minmax' / 'percentile' clamp implicitly
  via the normalisation formula (result = [0,1]).
- 'softmax' output sums to 1.0 over all patches (by construction).

Reference
---------
Implementation guide §3.2, Listing 3.
"""

from __future__ import annotations

from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants and exceptions
# ---------------------------------------------------------------------------

class NormMode(str, Enum):
    MINMAX     = "minmax"
    PERCENTILE = "percentile"
    SOFTMAX    = "softmax"


_VALID_MODES = {m.value for m in NormMode}

# 99th percentile clamp threshold (guide-mandated)
_PERCENTILE_Q = 99.0


class AttributionNormError(ValueError):
    """Raised for invalid mode strings or shape violations."""


# ---------------------------------------------------------------------------
# Core implementation
# ---------------------------------------------------------------------------

def _minmax_1d(flat: torch.Tensor) -> torch.Tensor:
    """
    Minmax-normalise a 1-D float tensor to [0, 1].
    Degenerate (constant) maps return all-zeros.
    """
    lo = flat.min()
    hi = flat.max()
    if (hi - lo).abs() < 1e-8:
        return torch.zeros_like(flat)
    return (flat - lo) / (hi - lo)


def _softmax_1d(flat: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable spatial softmax over a 1-D tensor.
    Subtracts max before exp to prevent overflow.
    Output sums to 1.0.
    """
    shifted = flat - flat.max()
    exp     = shifted.exp()
    return exp / exp.sum()


def _normalise_single(
    att:  torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    Normalise a single (Hp, Wp) or 1-D attribution map.

    Parameters
    ----------
    att  : (Hp, Wp) or (N,) float tensor.
    mode : one of 'minmax', 'percentile', 'softmax'.

    Returns
    -------
    Same shape as input, float32, values in [0, 1].
    """
    att  = att.float()
    flat = att.flatten()

    if mode == NormMode.MINMAX:
        norm_flat = _minmax_1d(flat)

    elif mode == NormMode.PERCENTILE:
        # Clamp at 99th percentile, then minmax
        p99      = torch.quantile(flat, _PERCENTILE_Q / 100.0)
        clamped  = flat.clamp(max=p99.item())
        norm_flat = _minmax_1d(clamped)

    elif mode == NormMode.SOFTMAX:
        norm_flat = _softmax_1d(flat)

    else:
        raise AttributionNormError(
            f"Unknown normalisation mode '{mode}'. "
            f"Valid modes: {sorted(_VALID_MODES)}"
        )

    return norm_flat.reshape(att.shape)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_attribution(
    att_map: torch.Tensor,
    mode:    str = "minmax",
) -> torch.Tensor:
    """
    Guide Listing 3 — normalise a raw attribution map to [0, 1].

    Parameters
    ----------
    att_map : (Hp, Wp) **or** (B, Hp, Wp) float tensor.
        Raw (un-normalised) attribution values from any explainer.
        May contain negative values; any float dtype accepted.
    mode    : str, one of:
        ``'minmax'``     — (att - min) / (max - min); degenerate → zeros.
        ``'percentile'`` — clamp at 99th percentile then minmax.
        ``'softmax'``    — spatial softmax; output sums to 1.0.

    Returns
    -------
    torch.Tensor — same shape as ``att_map``, dtype float32, values in [0, 1].

    Raises
    ------
    AttributionNormError  if ``mode`` is not one of the three valid strings.
    AttributionNormError  if ``att_map`` has fewer than 2 dimensions.

    Examples
    --------
    >>> import torch
    >>> att = torch.rand(14, 14)
    >>> n   = normalize_attribution(att, mode='minmax')
    >>> float(n.min()), float(n.max())
    (0.0, 1.0)

    >>> batch = torch.rand(4, 14, 14)
    >>> nb    = normalize_attribution(batch, mode='softmax')
    >>> nb.shape
    torch.Size([4, 14, 14])
    >>> nb[0].sum().item()  # ≈ 1.0
    1.0
    """
    if mode not in _VALID_MODES:
        raise AttributionNormError(
            f"Unknown normalisation mode '{mode}'. "
            f"Valid modes: {sorted(_VALID_MODES)}"
        )

    if att_map.ndim < 2:
        raise AttributionNormError(
            f"att_map must be at least 2-D (Hp, Wp) or (B, Hp, Wp); "
            f"got shape {tuple(att_map.shape)}."
        )

    if att_map.ndim == 2:
        # Single map (Hp, Wp)
        return _normalise_single(att_map, mode)

    if att_map.ndim == 3:
        # Batched (B, Hp, Wp) — normalise each sample independently
        return torch.stack([
            _normalise_single(att_map[i], mode)
            for i in range(att_map.shape[0])
        ])

    raise AttributionNormError(
        f"att_map must be 2-D or 3-D; got {att_map.ndim}-D shape "
        f"{tuple(att_map.shape)}."
    )


def normalize_batch(
    att_maps: torch.Tensor,
    mode:     str = "minmax",
) -> torch.Tensor:
    """
    Convenience wrapper for (B, Hp, Wp) batch input.

    Equivalent to ``normalize_attribution(att_maps, mode)`` but asserts
    the input is exactly 3-D, which can catch shape bugs earlier.

    Parameters
    ----------
    att_maps : (B, Hp, Wp) float tensor.
    mode     : same as :func:`normalize_attribution`.

    Returns
    -------
    (B, Hp, Wp) float32 tensor.
    """
    if att_maps.ndim != 3:
        raise AttributionNormError(
            f"normalize_batch expects (B, Hp, Wp) tensor; "
            f"got {att_maps.ndim}-D shape {tuple(att_maps.shape)}."
        )
    return normalize_attribution(att_maps, mode)
