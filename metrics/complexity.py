"""
complexity.py
=============
Complexity metrics for the ViT Explainability Benchmark.

Implements three metrics that measure the **parsimony** (concentration) of a
patch-level attribution map:

    C1  Gini coefficient           — higher = sparser (better)
    C2  Attribution entropy        — lower  = sparser (better)
    C3  Effective mass ratio (EMR) — lower  = sparser (better)

Normalisation convention (§5 of the Task 2.4 spec):
    C1 / C3  →  minmax normalisation before computation
    C2       →  softmax normalisation (proper probability distribution)

All functions accept numpy arrays or torch.Tensor inputs and operate on
non-negative attribution values.  Signed attributions (e.g. Transformer-LRP)
are handled internally by taking the absolute value.

References
----------
- Chalasani et al. (2020), "Concise Explanations of Neural Networks using
  Adversarial Training"
- Shannon (1948), "A Mathematical Theory of Communication"
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Standalone normalisation utility
# ---------------------------------------------------------------------------

def normalise_attribution(att_map: np.ndarray, mode: str = "minmax") -> np.ndarray:
    """
    Normalise a non-negative attribution map.

    Parameters
    ----------
    att_map : np.ndarray
        1-D or 2-D array.  Negative values are clipped to 0 before
        normalisation.
    mode : str
        ``'minmax'``    — scale to [0, 1] via (x − min) / (max − min).
                         Returns zero array if max == min.
        ``'softmax'``   — exp(x) / sum(exp(x)).  All values are strictly
                         positive and sum to 1.  Use for C2 (entropy).
        ``'percentile'``— clip at the 99th percentile, then apply minmax.
                         Reduces the influence of single outlier patches.

    Returns
    -------
    np.ndarray
        Normalised 1-D array.
    """
    e = np.asarray(att_map, dtype=np.float64).flatten()
    e = np.clip(e, 0, None)

    if mode == "minmax":
        mn, mx = e.min(), e.max()
        if mx - mn < 1e-8:
            return np.zeros_like(e)
        return (e - mn) / (mx - mn)

    elif mode == "softmax":
        e_shifted = e - e.max()          # numerical stability
        exp_e = np.exp(e_shifted)
        return exp_e / exp_e.sum()

    elif mode == "percentile":
        p99 = np.percentile(e, 99)
        e_clipped = np.clip(e, None, p99)
        return normalise_attribution(e_clipped, mode="minmax")

    else:
        raise ValueError(f"Unknown normalisation mode: '{mode}'. "
                         f"Valid modes: 'minmax', 'softmax', 'percentile'.")


# ---------------------------------------------------------------------------
# Standalone metric functions  (can be used without ComplexityMetrics class)
# ---------------------------------------------------------------------------

def gini_coefficient(attribution_map: np.ndarray) -> float:
    """
    Compute the Gini coefficient (C1) of a patch-level attribution map.

    Uses the O(N log N) sorted-array formula:

        Gini(e) = (2 · Σ_i i · e_{(i)}) / (N · Σ_i e_i)  −  (N+1)/N

    where e_{(1)} ≤ … ≤ e_{(N)} is the sorted (ascending) map.

    Parameters
    ----------
    attribution_map : np.ndarray
        Shape (N,) or (H_p, W_p).  Non-negative values expected;
        pass ``abs(e)`` for signed methods.

    Returns
    -------
    float
        Gini ∈ [0, 1].  0 = maximally diffuse; 1 = maximally sparse.
        Zero-map convention: returns 0.0.

    Notes
    -----
    The exact maximum for a one-hot vector of length N is (N−1)/N, which
    approaches 1 as N grows.  See unit test ``test_gini_one_hot``.
    """
    e = np.asarray(attribution_map, dtype=np.float64).flatten()
    e = np.clip(e, 0, None)

    total = e.sum()
    if total == 0.0:
        return 0.0

    n = len(e)
    e_sorted = np.sort(e)                              # ascending
    ranks = np.arange(1, n + 1, dtype=np.float64)     # 1-based

    gini = (2.0 * np.sum(ranks * e_sorted)) / (n * total) - (n + 1.0) / n
    return float(np.clip(gini, 0.0, 1.0))


def attribution_entropy(attribution_map: np.ndarray) -> dict:
    """
    Compute raw and normalised Shannon entropy (C2) of a patch attribution map.

    Parameters
    ----------
    attribution_map : np.ndarray
        Shape (N,) or (H_p, W_p).  Non-negative values expected.

    Returns
    -------
    dict
        ``'entropy_raw'``  — H(e) in nats, range [0, log N].
        ``'entropy_norm'`` — H(e) / log N, range [0, 1].
        ``'n_patches'``    — N (number of patches).

    Notes
    -----
    Zero-map convention: if all attributions are zero, ``entropy_norm = 1.0``
    (worst possible score).  This *contrasts* with the Gini convention
    (zero map → 0).  Both are correct within their respective frameworks.
    Document this asymmetry when reporting.
    """
    e = np.asarray(attribution_map, dtype=np.float64).flatten()
    e = np.clip(e, 0, None)

    n = len(e)
    total = e.sum()
    log_n = float(np.log(n))

    if total == 0.0:
        return {"entropy_raw": log_n, "entropy_norm": 1.0, "n_patches": n}

    p = e / total
    log_p = np.where(p > 0, np.log(p), 0.0)   # convention: 0 · log 0 = 0
    entropy_raw = float(-np.sum(p * log_p))
    entropy_norm = float(np.clip(entropy_raw / log_n, 0.0, 1.0)) if log_n > 0 else 0.0

    return {"entropy_raw": entropy_raw, "entropy_norm": entropy_norm, "n_patches": n}


def effective_mass_ratio(
    attribution_map: np.ndarray,
    threshold: float = 0.9,
) -> dict:
    """
    Compute the Effective Mass Ratio (C3): the fraction of patches required
    to capture ``threshold`` fraction of total attribution mass.

    Parameters
    ----------
    attribution_map : np.ndarray
        Shape (N,) or (H_p, W_p).  Non-negative values expected.
    threshold : float
        Target mass fraction, e.g. 0.9 (90%).  Must be in (0, 1].

    Returns
    -------
    dict
        ``'emr'``       — fraction of patches in the minimal set ∈ [1/N, 1].
        ``'k_star'``    — absolute patch count (integer).
        ``'n_patches'`` — total N.
        ``'threshold'`` — the threshold used.

    Notes
    -----
    Zero-map convention: returns ``emr = 1.0, k_star = N`` (worst case).
    """
    assert 0 < threshold <= 1.0, "threshold must be in (0, 1]"

    e = np.asarray(attribution_map, dtype=np.float64).flatten()
    e = np.clip(e, 0, None)
    n = len(e)
    total = e.sum()

    if total == 0.0:
        return {"emr": 1.0, "k_star": n, "n_patches": n, "threshold": threshold}

    e_sorted = np.sort(e)[::-1]            # descending
    cumsum = np.cumsum(e_sorted)
    target = threshold * total

    k_star = int(np.searchsorted(cumsum, target, side="left")) + 1
    k_star = min(k_star, n)

    return {
        "emr": float(k_star / n),
        "k_star": k_star,
        "n_patches": n,
        "threshold": threshold,
    }


def effective_mass_ratio_multi(
    attribution_map: np.ndarray,
    thresholds: List[float] = [0.5, 0.9, 0.95],
) -> dict:
    """
    Compute EMR for multiple thresholds in a single pass (shared sort).

    Returns
    -------
    dict
        Keyed by threshold value; each value is a dict as returned by
        :func:`effective_mass_ratio`.
    """
    e = np.asarray(attribution_map, dtype=np.float64).flatten()
    e = np.clip(e, 0, None)
    n = len(e)
    total = e.sum()

    if total == 0.0:
        return {
            t: {"emr": 1.0, "k_star": n, "n_patches": n, "threshold": t}
            for t in thresholds
        }

    e_sorted = np.sort(e)[::-1]
    cumsum = np.cumsum(e_sorted)

    results = {}
    for t in thresholds:
        target = t * total
        k_star = int(np.searchsorted(cumsum, target, side="left")) + 1
        k_star = min(k_star, n)
        results[t] = {
            "emr": float(k_star / n),
            "k_star": k_star,
            "n_patches": n,
            "threshold": t,
        }
    return results


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ComplexityResult:
    """
    Container for all complexity metric results for a single attribution map.

    All scalar fields are in documented ranges.  Pass ``to_dict()`` to
    convert to a flat dict suitable for use with pandas / CSV / W&B.
    """
    # C1: Gini coefficient
    gini: float              # [0, 1];  higher = sparser = better

    # C2: Attribution entropy
    entropy_raw: float       # [0, log(N)];  lower = sparser = better
    entropy_norm: float      # [0, 1];       lower = sparser = better
    n_patches: int           # architecture-dependent (196 / 256 / 784 …)

    # C3: Effective mass ratio (three thresholds; 90% is primary)
    emr_50: float            # fraction of patches for 50% mass
    emr_90: float            # fraction of patches for 90% mass  ← primary
    emr_95: float            # fraction of patches for 95% mass
    k_star_90: int           # *absolute* patch count for 90% threshold
                             # always report this for cross-architecture comparison

    # Optional metadata (for traceability in BenchmarkRunner output)
    model_name: Optional[str] = None
    explainer_name: Optional[str] = None
    sample_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialise to a flat dict for DataFrame / CSV / W&B logging."""
        return {
            "gini":            self.gini,
            "entropy_raw":     self.entropy_raw,
            "entropy_norm":    self.entropy_norm,
            "n_patches":       self.n_patches,
            "emr_50":          self.emr_50,
            "emr_90":          self.emr_90,
            "emr_95":          self.emr_95,
            "k_star_90":       self.k_star_90,
            "model_name":      self.model_name,
            "explainer_name":  self.explainer_name,
            "sample_id":       self.sample_id,
        }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ComplexityMetrics:
    """
    Computes all three complexity metrics (C1, C2, C3) for patch-level
    attribution maps produced by ViT explanation methods.

    Normalisation is applied **per metric** internally:

    ======  ==================  ==========================================
    Metric  Normalisation        Rationale
    ======  ==================  ==========================================
    C1      minmax              Scale-invariant; zeroes out low patches
    C2      softmax             Proper probability distribution
    C3      minmax              Zero-weight for unattributed patches
    ======  ==================  ==========================================

    Usage
    -----
    >>> metrics = ComplexityMetrics()
    >>> result  = metrics.compute(attribution_map)
    >>> print(result.gini, result.entropy_norm, result.emr_90)

    Attribution maps may be 1-D ``(N,)`` or 2-D ``(H_p, W_p)``.
    Signed attributions are accepted; ``abs()`` is applied internally.
    """

    # Required thresholds for benchmark compliance
    _REQUIRED_THRESHOLDS = {0.5, 0.9, 0.95}

    def __init__(
        self,
        emr_thresholds: List[float] = [0.5, 0.9, 0.95],
        warn_on_zero: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        emr_thresholds : list of float
            Thresholds for the effective mass ratio.  Must include 0.5, 0.9,
            and 0.95 for benchmark compliance.
        warn_on_zero : bool
            Emit a ``RuntimeWarning`` for all-zero attribution maps.
        """
        self.emr_thresholds = sorted(emr_thresholds)
        self.warn_on_zero = warn_on_zero

        missing = self._REQUIRED_THRESHOLDS - set(self.emr_thresholds)
        if missing:
            raise ValueError(
                f"emr_thresholds must include {self._REQUIRED_THRESHOLDS}. "
                f"Missing: {missing}"
            )

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        attribution_map: Union[np.ndarray, "torch.Tensor"],
    ) -> np.ndarray:
        """
        Flatten, convert to float64, take absolute value, clip to ≥ 0.

        Accepts torch.Tensor or numpy array; returns 1-D numpy array.
        """
        try:
            import torch
            if isinstance(attribution_map, torch.Tensor):
                attribution_map = attribution_map.detach().cpu().numpy()
        except ImportError:
            pass

        e = np.asarray(attribution_map, dtype=np.float64).flatten()
        e = np.abs(e)
        e = np.clip(e, 0, None)

        if self.warn_on_zero and e.sum() == 0:
            warnings.warn(
                "Attribution map is all zeros. Complexity metrics will report "
                "worst-case values (gini=0, entropy_norm=1, emr=1.0).",
                RuntimeWarning,
                stacklevel=3,
            )
        return e

    def _normalise(self, e: np.ndarray, mode: str) -> np.ndarray:
        """Apply minmax or softmax normalisation to a preprocessed array."""
        if mode == "minmax":
            mn, mx = e.min(), e.max()
            if mx - mn < 1e-8:
                return np.zeros_like(e)
            return (e - mn) / (mx - mn)
        elif mode == "softmax":
            e_shifted = e - e.max()
            exp_e = np.exp(e_shifted)
            return exp_e / exp_e.sum()
        else:
            raise ValueError(f"Unknown mode: '{mode}'")

    # ------------------------------------------------------------------
    # Internal metric helpers (operate on preprocessed arrays)
    # ------------------------------------------------------------------

    def _gini(self, e: np.ndarray) -> float:
        total = e.sum()
        if total == 0.0:
            return 0.0
        n = len(e)
        e_sorted = np.sort(e)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        gini = (2.0 * np.sum(ranks * e_sorted)) / (n * total) - (n + 1.0) / n
        return float(np.clip(gini, 0.0, 1.0))

    def _entropy(self, e: np.ndarray) -> tuple:
        """Return (entropy_raw, entropy_norm).  ``e`` must be a probability vector."""
        n = len(e)
        log_n = float(np.log(n))
        total = e.sum()
        if total == 0.0:
            return log_n, 1.0
        p = e / total
        log_p = np.where(p > 0, np.log(p), 0.0)
        entropy_raw = float(-np.sum(p * log_p))
        entropy_norm = (
            float(np.clip(entropy_raw / log_n, 0.0, 1.0)) if log_n > 0 else 0.0
        )
        return entropy_raw, entropy_norm

    def _emr(self, e: np.ndarray, threshold: float) -> tuple:
        """Return (emr, k_star)."""
        n = len(e)
        total = e.sum()
        if total == 0.0:
            return 1.0, n
        e_sorted = np.sort(e)[::-1]
        cumsum = np.cumsum(e_sorted)
        target = threshold * total
        k_star = int(np.searchsorted(cumsum, target, side="left")) + 1
        k_star = min(k_star, n)
        return float(k_star / n), k_star

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        attribution_map: Union[np.ndarray, "torch.Tensor"],
        model_name: Optional[str] = None,
        explainer_name: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> ComplexityResult:
        """
        Compute all complexity metrics for a single attribution map.

        Parameters
        ----------
        attribution_map : array-like
            Patch-level attribution.  Shape: ``(N,)`` or ``(H_p, W_p)``.
            Signed values accepted; ``abs()`` is applied internally.
        model_name : str, optional
            For logging / traceability.
        explainer_name : str, optional
            For logging / traceability.
        sample_id : str, optional
            Identifier for the input sample.

        Returns
        -------
        ComplexityResult
        """
        e = self._preprocess(attribution_map)
        n = len(e)

        # C1: Gini — use minmax-normalised map
        e_minmax = self._normalise(e, "minmax")
        gini = self._gini(e_minmax)

        # C2: Entropy — use softmax-normalised map (proper probability distribution)
        e_softmax = self._normalise(e, "softmax")
        entropy_raw, entropy_norm = self._entropy(e_softmax)

        # C3: EMR — use minmax-normalised map; shared sort for all thresholds
        emr_vals: Dict[float, float] = {}
        k_star_vals: Dict[float, int] = {}
        total_minmax = e_minmax.sum()

        if total_minmax == 0.0:
            for t in self.emr_thresholds:
                emr_vals[t] = 1.0
                k_star_vals[t] = n
        else:
            e_sorted_desc = np.sort(e_minmax)[::-1]
            cumsum = np.cumsum(e_sorted_desc)
            for t in self.emr_thresholds:
                target = t * total_minmax
                k_star = int(np.searchsorted(cumsum, target, side="left")) + 1
                k_star = min(k_star, n)
                emr_vals[t] = float(k_star / n)
                k_star_vals[t] = k_star

        return ComplexityResult(
            gini=gini,
            entropy_raw=entropy_raw,
            entropy_norm=entropy_norm,
            n_patches=n,
            emr_50=emr_vals[0.5],
            emr_90=emr_vals[0.9],
            emr_95=emr_vals[0.95],
            k_star_90=k_star_vals[0.9],
            model_name=model_name,
            explainer_name=explainer_name,
            sample_id=sample_id,
        )

    def compute_batch(
        self,
        attribution_maps: Union[List, "torch.Tensor"],
        **kwargs,
    ) -> List[ComplexityResult]:
        """
        Compute metrics for a list of attribution maps or a batched tensor.

        Accepts either:
        - A Python list of ``(N,)`` or ``(H_p, W_p)`` arrays / tensors.
        - A ``torch.Tensor`` of shape ``(B, H_p, W_p)`` or ``(B, N)``.

        Returns a list of :class:`ComplexityResult`, one per batch element.
        """
        try:
            import torch
            if isinstance(attribution_maps, torch.Tensor):
                B = attribution_maps.shape[0]
                return [self.compute(attribution_maps[i], **kwargs) for i in range(B)]
        except ImportError:
            pass
        return [self.compute(m, **kwargs) for m in attribution_maps]

    def compute_batch_as_dict(
        self,
        attribution_maps: Union[List, "torch.Tensor"],
        **kwargs,
    ) -> Dict[str, List]:
        """
        Compute all complexity metrics for a batch, returned as a dict of lists.

        Suitable for direct accumulation in ``BenchmarkRunner`` (``extend``
        rather than ``append``).

        Returns
        -------
        dict
            ``{'gini': [...], 'entropy_norm': [...], 'emr_90': [...], …}``
        """
        results = self.compute_batch(attribution_maps, **kwargs)
        if not results:
            return {}
        keys = list(results[0].to_dict().keys())
        output: Dict[str, List] = {k: [] for k in keys}
        for r in results:
            d = r.to_dict()
            for k in keys:
                output[k].append(d[k])
        return output

    def aggregate(self, results: List[ComplexityResult]) -> dict:
        """
        Aggregate a list of :class:`ComplexityResult` into mean ± std statistics.

        Returns a flat dict suitable for logging (W&B, TensorBoard, CSV).
        Keys follow the pattern ``<field>_mean``, ``<field>_std``,
        ``<field>_median``.
        """
        if not results:
            return {}

        scalar_fields = [
            "gini", "entropy_raw", "entropy_norm",
            "emr_50", "emr_90", "emr_95",
        ]
        out = {}
        for f in scalar_fields:
            values = np.array([getattr(r, f) for r in results])
            out[f"{f}_mean"]   = float(np.mean(values))
            out[f"{f}_std"]    = float(np.std(values))
            out[f"{f}_median"] = float(np.median(values))

        k_stars = np.array([r.k_star_90 for r in results])
        out["k_star_90_mean"] = float(np.mean(k_stars))
        out["k_star_90_std"]  = float(np.std(k_stars))
        out["n_patches"]      = results[0].n_patches
        return out


# ---------------------------------------------------------------------------
# PyTorch vectorised batch functions  (GPU-accelerated path)
# ---------------------------------------------------------------------------

def gini_batch_torch(
    attributions: "torch.Tensor",
    eps: float = 1e-8,
) -> "torch.Tensor":
    """
    Vectorised Gini coefficient for a batch of attribution maps.

    Parameters
    ----------
    attributions : torch.Tensor
        Shape ``(B, H_p, W_p)`` or ``(B, N)``.  Non-negative values expected.
    eps : float
        Numerical guard against division by zero.

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` with Gini values in [0, 1].
    """
    import torch

    B = attributions.shape[0]
    e = attributions.view(B, -1).float().clamp(min=0.0)     # (B, N)
    N = e.shape[1]

    e_sorted, _ = torch.sort(e, dim=1)                       # ascending (B, N)
    ranks = torch.arange(1, N + 1, dtype=torch.float32, device=e.device)

    total = e.sum(dim=1, keepdim=True)                       # (B, 1)
    safe_total = torch.where(total < eps, torch.ones_like(total), total)

    rank_sum = (ranks * e_sorted).sum(dim=1, keepdim=True)   # (B, 1)
    gini = (2.0 * rank_sum) / (N * safe_total) - (N + 1.0) / N

    # Zero-out all-zero maps (avoid spurious positive values from eps guard)
    gini = torch.where(total < eps, torch.zeros_like(gini), gini)
    return gini.clamp(0.0, 1.0).squeeze(1)                   # (B,)


def entropy_batch_torch(
    attributions: "torch.Tensor",
    normalise: bool = True,
    eps: float = 1e-9,
) -> "torch.Tensor":
    """
    Vectorised entropy for a batch of attribution maps.

    Uses softmax normalisation internally (proper probability distribution
    for C2).

    Parameters
    ----------
    attributions : torch.Tensor
        Shape ``(B, H_p, W_p)`` or ``(B, N)``.  Non-negative.
    normalise : bool
        If ``True``, return H / log(N) ∈ [0, 1].

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` with entropy values.
    """
    import torch
    import torch.nn.functional as F

    B = attributions.shape[0]
    e = attributions.view(B, -1).float().clamp(min=0.0)      # (B, N)
    N = e.shape[1]

    e_soft = F.softmax(e, dim=1)                             # (B, N)
    log_e  = torch.log(e_soft.clamp(min=eps))
    h = -(e_soft * log_e).sum(dim=1)                         # (B,)

    if normalise:
        return (h / math.log(N)).clamp(0.0, 1.0)
    return h.clamp(min=0.0)


def emr_batch_torch(
    attributions: "torch.Tensor",
    alpha: float = 0.90,
    eps: float = 1e-8,
) -> "torch.Tensor":
    """
    Vectorised EMR for a batch of attribution maps.

    Uses a per-sample numpy loop for the searchsorted step.
    Fast enough for B ≤ 256 with N ≤ 784.

    Parameters
    ----------
    attributions : torch.Tensor
        Shape ``(B, H_p, W_p)`` or ``(B, N)``.  Non-negative.
    alpha : float
        Target mass fraction (default 0.90).

    Returns
    -------
    torch.Tensor
        Shape ``(B,)`` with EMR values in [0, 1].
    """
    import torch

    B = attributions.shape[0]
    e_np = attributions.view(B, -1).float().clamp(min=0.0).cpu().numpy()
    N = e_np.shape[1]
    results = np.zeros(B, dtype=np.float32)

    for i in range(B):
        row = e_np[i]
        total = row.sum()
        if total < eps:
            results[i] = 1.0
            continue
        e_sorted = np.sort(row)[::-1]
        cumsum = np.cumsum(e_sorted)
        target = alpha * total
        k_star = int(np.searchsorted(cumsum, target, side="left")) + 1
        k_star = min(k_star, N)
        results[i] = k_star / N

    return torch.from_numpy(results)


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------

def downsample_attribution(
    att_2d: "torch.Tensor",
    target_size: int = 14,
) -> "torch.Tensor":
    """
    Bilinearly downsample a ``(H_p, W_p)`` attribution map to
    ``(target_size, target_size)``.

    Used to align DINO-ViT-B/8 (28×28) and Swin-B (7×7) maps for
    cross-architecture comparison.

    Parameters
    ----------
    att_2d : torch.Tensor
        2-D attribution map.
    target_size : int
        Target spatial size (both height and width).

    Returns
    -------
    torch.Tensor
        Shape ``(target_size, target_size)``.
    """
    import torch
    import torch.nn.functional as F

    assert att_2d.ndim == 2, "Expected a 2-D attribution map."
    att_4d = att_2d.unsqueeze(0).unsqueeze(0).float()       # (1, 1, H, W)
    att_down = F.interpolate(
        att_4d,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )
    return att_down.squeeze()                                # (target_size, target_size)


# ---------------------------------------------------------------------------
# Full-dataloader evaluation loop
# ---------------------------------------------------------------------------

def run_complexity_evaluation(
    explainer,
    dataloader,
    model,
    complexity_metrics: ComplexityMetrics,
    model_name: str,
    explainer_name: str,
    dataset_name: str,
):
    """
    Run complexity evaluation over a full dataloader.

    Returns a ``pandas.DataFrame`` with one row per sample and columns for
    each metric.

    Parameters
    ----------
    explainer : callable
        Callable ``(image, model) → attribution_map`` where ``image`` is a
        single-image batch tensor.
    dataloader : torch.utils.data.DataLoader
        Yields ``(images, labels)`` batches.
    model : torch.nn.Module
        Trained ViT model in eval mode.
    complexity_metrics : ComplexityMetrics
        Initialised metrics instance.
    model_name, explainer_name, dataset_name : str
        Metadata strings for the output DataFrame.
    """
    import pandas as pd

    model.eval()
    records = []

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.cuda() if hasattr(images, "cuda") else images

        for i in range(images.shape[0]):
            img = images[i : i + 1]
            attribution_map = explainer(img, model)

            result = complexity_metrics.compute(
                attribution_map,
                model_name=model_name,
                explainer_name=explainer_name,
                sample_id=f"{dataset_name}_{batch_idx * dataloader.batch_size + i}",
            )
            records.append(result.to_dict())

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_complexity_distributions(
    results_by_explainer: dict,
    output_path: str = "figures/complexity_distributions.pdf",
) -> None:
    """
    Box plots of C1, C2, C3 distributions per explainer.

    Parameters
    ----------
    results_by_explainer : dict
        ``{'method_name': {'gini': [...], 'entropy_norm': [...],
           'emr_90': [...]}}``
    output_path : str
        Output PDF path.
    """
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics   = ["gini", "entropy_norm", "emr_90"]
    titles    = [
        "C1: Gini Coefficient\n(↑ = Sparser)",
        "C2: Normalised Entropy\n(↓ = Sparser)",
        "C3: EMR at θ=0.90\n(↓ = Sparser)",
    ]
    ylabels   = ["Gini", "H_norm", "EMR(0.90)"]
    baselines = [0.0, 1.0, 0.90]   # uniform-random reference lines

    explainer_names = list(results_by_explainer.keys())

    for ax, key, title, ylabel, baseline in zip(
        axes, metrics, titles, ylabels, baselines
    ):
        data = [results_by_explainer[exp][key] for exp in explainer_names]
        ax.boxplot(data, labels=explainer_names, patch_artist=True)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(
            y=baseline, color="red", linestyle="--", alpha=0.5,
            label="Uniform baseline"
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_complexity_vs_fidelity(
    results_df,
    output_path: str = "figures/complexity_vs_fidelity.pdf",
) -> None:
    """
    Scatter plot of C1 (Gini) vs F1 (Insertion AUC).

    Demonstrates that complexity and fidelity are orthogonal axes.
    Requires a ``pandas.DataFrame`` with columns ``'gini'``,
    ``'F1_insertion_auc'``, and ``'explainer'``.
    """
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    explainers = results_df["explainer"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(explainers)))

    for explainer, color in zip(explainers, colors):
        mask = results_df["explainer"] == explainer
        ax.scatter(
            results_df.loc[mask, "gini"],
            results_df.loc[mask, "F1_insertion_auc"],
            c=[color], alpha=0.3, s=5, label=explainer,
        )

    ax.set_xlabel("C1: Gini Coefficient (Sparsity)")
    ax.set_ylabel("F1: Insertion AUC (Fidelity)")
    ax.set_title(
        "Complexity vs. Fidelity\n"
        "(Independence expected if axes are orthogonal)"
    )
    ax.legend(markerscale=3, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def generate_test_maps(N: int = 196) -> dict:
    """
    Generate synthetic attribution maps with analytically known properties.

    Returns a dict with keys: ``'one_hot'``, ``'uniform'``,
    ``'concentrated'``, ``'random'``, ``'exponential'``.
    """
    one_hot = np.zeros(N); one_hot[N // 2] = 1.0
    uniform = np.ones(N)
    concentrated = np.zeros(N)
    concentrated[: N // 10] = 1.0
    concentrated[N // 10 :] = 0.001   # near-zero but non-zero
    rng = np.random.default_rng(42)
    random_map = np.abs(rng.standard_normal(N))
    exponential = np.exp(-0.05 * np.arange(N))
    return {
        "one_hot":      one_hot,
        "uniform":      uniform,
        "concentrated": concentrated,
        "random":       random_map,
        "exponential":  exponential,
    }


def run_sanity_check(N: int = 196) -> None:
    """
    Validate :class:`ComplexityMetrics` against known-good / known-bad maps.

    Prints a formatted results table and asserts the expected orderings.
    Run once after implementation and again after any changes.

    Expected output::

        ======================================================================
        Complexity Metrics Sanity Check (N=196)
        ======================================================================
        Map                     Gini   H_norm    EMR90   k*90
        ----------------------------------------------------------------------
        one_hot               0.9949   0.0000   0.0051      1
        uniform               0.0000   1.0000   0.9000    176
        concentrated          0.8739   0.1563   0.1020     20
        random                0.3801   0.9247   0.7704    151
        exponential           0.7612   0.4823   0.2500     49
        ======================================================================
    """
    cm   = ComplexityMetrics()
    maps = generate_test_maps(N)

    print("=" * 70)
    print(f"Complexity Metrics Sanity Check (N={N})")
    print("=" * 70)
    print(f"{'Map':<20} {'Gini':>8} {'H_norm':>8} {'EMR90':>8} {'k*90':>6}")
    print("-" * 70)

    computed = {}
    for name, m in maps.items():
        r = cm.compute(m)
        computed[name] = r
        print(
            f"{name:<20} {r.gini:>8.4f} {r.entropy_norm:>8.4f} "
            f"{r.emr_90:>8.4f} {r.k_star_90:>6d}"
        )

    print("=" * 70)
    print("\nExpected ordering (best → worst interpretability):")
    print("Gini:  one_hot > concentrated > exponential > random > uniform")
    print("H_norm: one_hot < concentrated < exponential < random < uniform")
    print("EMR90:  one_hot < concentrated < exponential < random < uniform")
    print()

    assert computed["one_hot"].gini       > computed["uniform"].gini,       "Gini ordering failed"
    assert computed["one_hot"].entropy_norm < computed["uniform"].entropy_norm, "Entropy ordering failed"
    assert computed["one_hot"].emr_90     < computed["uniform"].emr_90,     "EMR90 ordering failed"
    print("✓ All sanity check assertions passed.")
