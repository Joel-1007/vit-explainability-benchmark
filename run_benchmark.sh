#!/usr/bin/env bash
# run_benchmark.sh
# E2E runner for the ViT Explainability Benchmark (Phase 5.4.4)
# Note: For TPAMI submission, reproducibility is paramount. This script isolates
# execution by enforcing strict hardware/seed parity.

set -euo pipefail

SEED=42
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"results/phase3"}
DATA_ROOT=${DATA_ROOT:-"/data"}
# Fine-tuned checkpoints: {model}_{dataset}.pth or {model}_ft_{dataset}_*.pth
WEIGHTS_DIR=${WEIGHTS_DIR:-"checkpoints"}
OUTPUT_DIR=${OUTPUT_DIR:-"results/phase4"}

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
# Evaluation protocol audit: splits, labels, annotation coverage, NIH patient leakage
uv run python scripts/audit_datasets.py --data-root "$DATA_ROOT" --output_dir "$OUTPUT_DIR/audit"

echo "[3/4] Running Baseline Evaluation Suite (Phase 3)"
mkdir -p "$CHECKPOINT_DIR"
# --ablations adds the A1/A2 raw-attention variants and A3 zero/blur fidelity.
# A4 (MAE vs supervised) runs only if fine-tuned MAE weights are present.
MODELS=(vit_b16 deit_b16 swin_b beit_b16 dino_vitb8 dinov2_vitb14)
if ls "$WEIGHTS_DIR"/mae_b16_* >/dev/null 2>&1; then
  MODELS+=(mae_b16)
else
  echo "  Note: no mae_b16 weights in $WEIGHTS_DIR — ablation A4 will be skipped."
fi
# The Phase3Runner caches results incrementally per (dataset, model, explainer).
# It can safely be interrupted and resumed.
# Several fine-tuning seeds: export FINETUNE_SEEDS="0 1 2" (weights {model}_ft_{dataset}_seed{N}_*.pth)
SEED_ARGS=()
if [ -n "${FINETUNE_SEEDS:-}" ]; then
  read -r -a _seeds <<< "$FINETUNE_SEEDS"
  SEED_ARGS=(--finetune-seeds "${_seeds[@]}")
fi
uv run python run_phase3.py \
  --data-root "$DATA_ROOT" \
  ${SEED_ARGS[@]+"${SEED_ARGS[@]}"} \
  --models "${MODELS[@]}" \
  --ablations \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --seed $SEED \
  --weights-dir "$WEIGHTS_DIR" \
  --norm-mode percentile

echo "[4/4] Generating Analytical Artifacts (Phase 4)"
mkdir -p "$OUTPUT_DIR"

# run_phase3.py writes aggregated_results.csv (one row per sample) and
# combination_summary.csv into the checkpoint dir at the end of a run.
# Re-create them any time with: uv run python -m metrics.aggregate --checkpoint-dir DIR
RESULTS_CSV="$CHECKPOINT_DIR/aggregated_results.csv"
if [ -f "$RESULTS_CSV" ]; then
  echo "Running Analytical Modules..."
  uv run python scripts/phase4_correlation_analysis.py --results_csv "$RESULTS_CSV" --output_dir "$OUTPUT_DIR"
  uv run python scripts/phase4_interaction_analysis.py --results_csv "$RESULTS_CSV" --output_dir "$OUTPUT_DIR"
  uv run python scripts/phase4_ablations.py --results_csv "$RESULTS_CSV" --output_dir "$OUTPUT_DIR"
  uv run python scripts/compare_explainers.py --results_csv "$RESULTS_CSV" --output_dir "$OUTPUT_DIR"
else
  echo "WARNING: $RESULTS_CSV not found — Phase 4 analytics SKIPPED."
  echo "         Did Phase 3 finish? Re-run: uv run python -m metrics.aggregate --checkpoint-dir $CHECKPOINT_DIR"
fi

echo "================================================="
echo " Pipeline Complete."
echo " Results available in -> $OUTPUT_DIR"
echo "================================================="
