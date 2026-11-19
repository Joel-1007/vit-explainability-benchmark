"""
aggregate.py — turn Phase 3 checkpoints into Phase 4 inputs.

Reads every ``<dataset>-<model>-<explainer>.pkl`` in a checkpoint directory
(written by Phase3Runner with a SampleMetricComputer) and produces:

* ``aggregated_results.csv`` — one row per sample:
  dataset, model, explainer, sample_id, label, pred, correct, <metric columns>
* ``combination_summary.csv`` — one row per combination: n_samples, accuracy,
  the mean of every metric, and L4 ``calibration_gap``
  = mean EGT (correct predictions) − mean EGT (incorrect predictions).

Usage::

    python -m metrics.aggregate --checkpoint-dir results/phase3
"""

from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd

ID_COLUMNS = ["dataset", "model", "explainer", "sample_id"]
# Per-sample columns that are bookkeeping, not quality metrics.
NON_METRIC_COLUMNS = ID_COLUMNS + ["label", "pred", "correct"]


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_METRIC_COLUMNS]


def load_per_sample(checkpoint_dir: str | Path) -> pd.DataFrame:
    """One row per (combination, sample). Raises if a checkpoint has no metrics."""
    rows = []
    pkls = sorted(Path(checkpoint_dir).glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No .pkl checkpoints in {checkpoint_dir}")
    for pkl in pkls:
        with open(pkl, "rb") as f:
            res = pickle.load(f)
        if "metrics" not in res:
            raise ValueError(
                f"{pkl.name} has no per-sample metrics (written without a "
                "SampleMetricComputer). Delete it and re-run run_phase3.py."
            )
        ids = {k: res.get(k) for k in ("dataset", "model", "explainer")}
        if None in ids.values():
            ids = dict(zip(("dataset", "model", "explainer"), res["key"].split("-", 2)))
        for i, (label, m) in enumerate(zip(res["labels"], res["metrics"])):
            rows.append({**ids, "sample_id": i, "label": label, **m})
    return pd.DataFrame(rows)


def calibration_gap(group: pd.DataFrame) -> float:
    """L4: mean EGT on correct minus incorrect predictions (NaN if either is empty)."""
    if "egt" not in group:
        return math.nan
    ok, bad = group.loc[group["correct"] == 1, "egt"], group.loc[group["correct"] == 0, "egt"]
    if ok.dropna().empty or bad.dropna().empty:
        return math.nan
    return float(ok.mean() - bad.mean())


def summarise(per_sample: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model, explainer) with metric means and L4."""
    keys = ["dataset", "model", "explainer"]
    cols = metric_columns(per_sample)
    g = per_sample.groupby(keys, sort=True)
    summary = g[cols].mean()
    summary.insert(0, "accuracy", g["correct"].mean())
    summary.insert(0, "n_samples", g.size())
    summary["calibration_gap"] = g.apply(calibration_gap, include_groups=False)
    for c in ("max_sensitivity", "model_randomisation", "label_randomisation"):
        if c in per_sample:
            summary[f"n_{c}"] = g[c].count()      # robustness runs on a subset
    return summary.reset_index()


def aggregate(checkpoint_dir: str | Path) -> Tuple[Path, Path]:
    """Write aggregated_results.csv and combination_summary.csv; return their paths."""
    d = Path(checkpoint_dir)
    per_sample = load_per_sample(d)
    p1, p2 = d / "aggregated_results.csv", d / "combination_summary.csv"
    per_sample.to_csv(p1, index=False)
    summarise(per_sample).to_csv(p2, index=False)
    return p1, p2


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint-dir", default="results/phase3")
    args = p.parse_args(argv)
    p1, p2 = aggregate(args.checkpoint_dir)
    print(f"Wrote {p1}\nWrote {p2}")


if __name__ == "__main__":
    main()
