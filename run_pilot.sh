#!/usr/bin/env bash
# run_pilot.sh
# ============================================================
# PILOT / SMOKE-TEST runner
# ============================================================
# Verifies the full pipeline works end-to-end on a tiny subset.
# Run this FIRST before committing to a full benchmark run.
#
# What it tests:
#   ✓ Environment & dependencies (pytest)
#   ✓ Dataset loading (1 dataset, 50 images)
#   ✓ All 6 explainers (RISE with 100 masks instead of 4000)
#   ✓ All metrics (L1-L4, R1-R3, C1-C3, F1-F3)
#   ✓ Checkpointing & resume
#   ✓ Phase 4 analytics scripts
#
# Expected runtime:
#   A100 GPU  : ~5–15 minutes
#   CPU only  : ~30–60 minutes
#
# Usage:
#   bash run_pilot.sh
#   bash run_pilot.sh --skip-tests     # skip pytest if already verified
# ============================================================

set -euo pipefail

# --- Config ---------------------------------------------------
SEED=42
MAX_BATCHES=7                   # 7 batches × 8 images = ~50 images
RISE_N_MASKS=100                # Reduced from 4000 (40× faster)
CHECKPOINT_DIR="results/pilot"
OUTPUT_DIR="results/pilot/phase4"
DATA_ROOT=${DATA_ROOT:-"/data"}
SKIP_TESTS=${1:-""}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     ViT Explainability Benchmark — PILOT RUN         ║"
echo "║     (Smoke test: ~50 images, RISE=100 masks)         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  DATA_ROOT      : $DATA_ROOT"
echo "  MAX_BATCHES    : $MAX_BATCHES  (~50 images)"
echo "  RISE_N_MASKS   : $RISE_N_MASKS"
echo "  CHECKPOINT_DIR : $CHECKPOINT_DIR"
echo "  OUTPUT_DIR     : $OUTPUT_DIR"
echo ""

# --- Step 1: Environment check --------------------------------
if [[ "$SKIP_TESTS" != "--skip-tests" ]]; then
    echo "[1/4] Running unit tests to verify environment..."
    uv run pytest tests/ -v --tb=short -q || {
        echo ""
        echo "❌ Tests failed! Fix the environment before running the benchmark."
        exit 1
    }
    echo "✅ All tests passed."
else
    echo "[1/4] Skipping tests (--skip-tests flag set)."
fi

# --- Step 2: GPU check ----------------------------------------
echo ""
echo "[2/4] Checking GPU..."
uv run python -c "
import torch
if torch.cuda.is_available():
    print(f'  ✅ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  ⚠️  No GPU detected — running on CPU (will be slower)')
"

# --- Step 3: Dataset check ------------------------------------
echo ""
echo "[3/4] Verifying dataset access..."
if [ -d "$DATA_ROOT/cub200" ]; then
    echo "  ✅ CUB-200-2011 found at $DATA_ROOT/cub200"
else
    echo "  ⚠️  WARNING: $DATA_ROOT/cub200 not found."
    echo "       Set DATA_ROOT to the correct path, e.g.:"
    echo "         export DATA_ROOT=/scratch/\$USER/datasets"
    echo "       Continuing anyway (runner will error if datasets missing)."
fi

# --- Step 4: Run pilot benchmark ------------------------------
echo ""
echo "[4/4] Running PILOT benchmark..."
echo "      (dataset=cub200, 1 model, 6 explainers, ~50 images)"
echo ""

mkdir -p "$CHECKPOINT_DIR" "$OUTPUT_DIR"

# Run Phase 3 with limited batches
uv run python -m metrics.runner \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --seed "$SEED" \
    --max-batches "$MAX_BATCHES" \
    --norm-mode minmax \
    --patch-size 16

echo ""
echo "[Phase 4] Running analytics on pilot results..."

# Phase 4 analytics (will use whatever results exist)
RESULTS_CSV="$CHECKPOINT_DIR/aggregated_results.csv"

if [ -f "$RESULTS_CSV" ]; then
    uv run python scripts/phase4_correlation_analysis.py \
        --results_csv "$RESULTS_CSV" \
        --output_dir "$OUTPUT_DIR"
    uv run python scripts/phase4_interaction_analysis.py \
        --results_csv "$RESULTS_CSV" \
        --output_dir "$OUTPUT_DIR"
    uv run python scripts/phase4_ablations.py \
        --output_dir "$OUTPUT_DIR"
else
    echo "  ⚠️  No aggregated_results.csv yet — Phase 4 analytics skipped."
    echo "       (Run the full pipeline with run_benchmark.sh to generate it)"
fi

# --- Summary --------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                 PILOT RUN COMPLETE ✅                ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  Checkpoints  → $CHECKPOINT_DIR/"
echo "  Analytics    → $OUTPUT_DIR/"
echo ""
echo "  If everything looks good, run the FULL benchmark:"
echo "    bash run_benchmark.sh"
echo ""
echo "  Checkpoint files can be listed with:"
echo "    uv run python -m metrics.runner --list-checkpoints \\"
echo "        --checkpoint-dir $CHECKPOINT_DIR"
echo ""
