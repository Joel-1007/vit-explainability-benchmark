"""
stats.py — paired comparisons, effect sizes and multiple-comparison correction.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import scipy.stats as ss


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """d = (mean_A − mean_B) / sqrt((var_A + var_B) / 2)  (implementation guide §4.3)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    diff = a.mean() - b.mean()
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return float(diff / pooled)


def paired_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Paired effect size d_z = mean(A − B) / sd(A − B)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1) if len(d) > 1 else math.nan
    if not np.isfinite(sd) or sd == 0:
        return 0.0 if d.mean() == 0 else math.copysign(math.inf, d.mean())
    return float(d.mean() / sd)


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 if all differences are zero."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return math.nan
    if np.allclose(d, 0):
        return 1.0
    return float(ss.wilcoxon(a, b, zero_method="wilcox").pvalue)


def paired_diff_ci(a, b, n_boot: int = 1000, level: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of mean(A − B) over paired samples."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    q = (1 - level) / 2
    return float(np.quantile(means, q)), float(np.quantile(means, 1 - q))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """
    Holm–Bonferroni adjusted p-values (controls the family-wise error rate,
    uniformly more powerful than Bonferroni). NaNs are kept and not counted.
    """
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    m = len(idx)
    if m == 0:
        return out
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out
