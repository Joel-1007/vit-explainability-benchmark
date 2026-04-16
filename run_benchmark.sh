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
uv run python scripts/verify_datasets.py --data_root "$DATA_ROOT" --dataset cub200
uv run python scripts/verify_datasets.py --data_root "$DATA_ROOT" --dataset voc
uv run python scripts/verify_datasets.py --data_root "$DATA_ROOT" --dataset imagenet_s50
uv run python scripts/verify_datasets.py --data_root "$DATA_ROOT" --dataset nih_chestxray

echo "[3/4] Running Baseline Evaluation Suite (Phase 3)"
mkdir -p "$CHECKPOINT_DIR"
# The Phase3Runner caches results incrementally per (dataset, model, explainer).
# It can safely be interrupted and resumed.
uv run python -m metrics.runner \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seed $SEED \
  --norm-mode percentile \
  --patch-size 16

echo "[4/4] Generating Analytical Artifacts (Phase 4)"
mkdir -p "$OUTPUT_DIR"

# Concatenate all pickle outputs into a single CSV for Phase 4 ingestion if a script exists,
# or assume the Phase 4 analytics scripts operate directly via aggregated CSV structure.
echo "Running Analytical Modules..."
uv run python scripts/phase4_correlation_analysis.py --results_csv "$CHECKPOINT_DIR/aggregated_results.csv" --output_dir "$OUTPUT_DIR"
uv run python scripts/phase4_interaction_analysis.py --results_csv "$CHECKPOINT_DIR/aggregated_results.csv" --output_dir "$OUTPUT_DIR"
uv run python scripts/phase4_ablations.py --output_dir "$OUTPUT_DIR"

echo "================================================="
echo " Pipeline Complete."
echo " Results available in -> $OUTPUT_DIR"
echo "================================================="
