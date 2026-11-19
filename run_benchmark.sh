#!/usr/bin/env bash
# run_benchmark.sh
# E2E runner for the ViT Explainability Benchmark (Phase 5.4.4)
# Note: For TPAMI submission, reproducibility is paramount. This script isolates
# execution by enforcing strict hardware/seed parity.

set -euo pipefail

SEED=42
CHECKPOINT_DIR="results/phase3"
DATA_ROOT=${DATA_ROOT:-"/data"}
OUTPUT_DIR="results/phase4"

echo "================================================="
echo "  ViT Explainability Benchmark - Full Pipeline   "
echo "================================================="

echo "[1/4] Environment Integrity Check"
uv run pytest tests/ || { echo "Tests failed! Aborting."; exit 1; }

echo "[2/4] Verifying Datasets"
# Assumes structure matches config specs
# Paths match the layout run_phase3.py loads from ({DATA_ROOT}/cub200, voc2012, ...)
uv run python scripts/verify_datasets.py \
  --cub        "$DATA_ROOT/cub200" \
  --voc        "$DATA_ROOT/voc2012" \
  --imagenet_s "$DATA_ROOT/imagenet_s50" \
  --nih        "$DATA_ROOT/nih_chestxray"

echo "[3/4] Running Baseline Evaluation Suite (Phase 3)"
mkdir -p "$CHECKPOINT_DIR"
# The Phase3Runner caches results incrementally per (dataset, model, explainer).
# It can safely be interrupted and resumed.
uv run python run_phase3.py \
  --data-root "$DATA_ROOT" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seed $SEED \
  --norm-mode percentile \
  --patch-size 16

echo "[4/4] Generating Analytical Artifacts (Phase 4)"
mkdir -p "$OUTPUT_DIR"

# Phase 4 reads an aggregated per-sample CSV. No script writes it yet: the
# Phase 3 .pkl checkpoints still need an aggregation step. The Phase 4 scripts
# refuse to run without real results (they no longer generate dummy data).
RESULTS_CSV="$CHECKPOINT_DIR/aggregated_results.csv"
if [ -f "$RESULTS_CSV" ]; then
  echo "Running Analytical Modules..."
  uv run python scripts/phase4_correlation_analysis.py --results_csv "$RESULTS_CSV" --output_dir "$OUTPUT_DIR"
  uv run python scripts/phase4_interaction_analysis.py --results_csv "$RESULTS_CSV" --output_dir "$OUTPUT_DIR"
  # phase4_ablations.py (A1–A4) is not implemented yet and is not run here.
else
  echo "WARNING: $RESULTS_CSV not found — Phase 4 analytics SKIPPED."
  echo "         Phase 3 checkpoints are in $CHECKPOINT_DIR; aggregate them first."
fi

echo "================================================="
echo " Pipeline Complete."
echo " Results available in -> $OUTPUT_DIR"
echo "================================================="
