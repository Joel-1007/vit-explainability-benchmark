"""
thresholds.py — per-class decision thresholds for multi-label models.

Estimated ONLY on the validation split, after the final epoch, then frozen
in the checkpoint (``metadata["decision_thresholds"]``) and a JSON sidecar.

For class c, t_c maximises F1 of the rule "predict c iff s_c ≥ t" over the
valid validation entries (weight w = 1). Candidates are the observed scores;
ties in F1 are broken towards the larger threshold (fewer false positives).
These are detection operating points; they say nothing about probability
calibration. A class without validation positives gets no estimate
(t_c = 0.5, recorded in ``fallback_classes``).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def f1_optimal_thresholds(scores: np.ndarray, labels: np.ndarray,
                          weights: np.ndarray | None = None,
                          fallback: float = 0.5) -> Tuple[np.ndarray, Dict]:
    """
    scores, labels, weights : (N, C). Returns (thresholds (C,), info).
    info: per-class best F1, number of validation positives, fallback classes.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    weights = np.ones_like(labels) if weights is None else np.asarray(weights, float)
    C = scores.shape[1]
    t = np.full(C, fallback, float)
    best_f1 = np.full(C, np.nan)
    n_pos = np.zeros(C, int)
    fallback_classes: List[int] = []
    for c in range(C):
        valid = weights[:, c] > 0
        s, y = scores[valid, c], labels[valid, c] >= 0.5
        n_pos[c] = int(y.sum())
        if n_pos[c] == 0:
            fallback_classes.append(c)
            continue
        order = np.argsort(-s, kind="stable")
        s_sorted, y_sorted = s[order], y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(~y_sorted)
        # evaluate only at the last index of each run of equal scores
        last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
        tp, fp, cand = tp[last], fp[last], s_sorted[last]
        f1 = 2 * tp / (2 * tp + fp + (n_pos[c] - tp))
        k = int(np.flatnonzero(f1 == f1.max())[0])          # largest threshold among ties
        t[c], best_f1[c] = cand[k], f1[k]
    return t, {"best_f1": best_f1.tolist(), "n_val_positives": n_pos.tolist(),
               "fallback_classes": fallback_classes, "rule": "s_c >= t_c, F1-optimal on validation"}
