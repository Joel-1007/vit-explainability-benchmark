# ViT Explainability Benchmark

**A Comprehensive, Axiomatically-Grounded Explainability Benchmark for Vision Transformers**

> **Target Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)  
> **Status:** Phases 1–3 code complete · Phase 3 computational run pending · Phase 4 analysis scripts ready

---

## Overview

Existing evaluation frameworks for Vision Transformer (ViT) explanations are often inconsistent, narrowly scoped, and lack standardised methodology. This benchmark provides a rigorous, unified, and mathematically grounded evaluation suite that generates consistent empirical evidence across **6 ViT architectures**, **7 explanation methods**, and **4 datasets**, measuring **13 metrics** spanning Fidelity, Localization, Robustness, and Complexity.

> 📖 **Full technical reference:** [BENCHMARK.md](BENCHMARK.md) — contains all metric definitions, mathematical proofs, experimental protocols, and API specifications.

---

## Key Capabilities

| Dimension | Metrics | What it measures |
|-----------|---------|------------------|
| **Fidelity (F1–F3)** | Sufficiency, Comprehensiveness, Log-odds | Does removing highlighted patches destroy model confidence? |
| **Localization (L1–L4)** | mIoU, Pointing Game, EGT, CalibGap | Does the explanation highlight the object of interest? |
| **Robustness (R1–R3)** | MaxSens, ModelRand, LabelRand | Is the explanation stable under perturbations? |
| **Complexity (C1–C3)** | Gini, Entropy, Effective Mass Ratio | Is the explanation parsimonious and interpretable? |

Additionally, the **Axiomatic Analysis Tool** (`AxiomVerifier`) maps every metric against the four Shapley axioms (Dummy, Completeness, Symmetry, Linearity), surfacing representational biases — including Theorem T6: the anti-alignment of complexity and symmetry.

### Model Zoo

Six ViT variants (~86M parameters each), fine-tuned under an identical standardised protocol:

| Model | Pre-training | Patch Size | Architecture |
|-------|-------------|------------|--------------|
| ViT-B/16 | Supervised (augreg) | 16×16 | Standard CLS |
| DeiT-B/16 | Distilled | 16×16 | CLS + distillation |
| Swin-B | Shifted-window | 7×7 | Hierarchical |
| BEiT-B/16 | Masked Image Modelling | 16×16 | Standard CLS |
| DINO-ViT-B/8 | Self-supervised | 8×8 | Standard CLS |
| DINOv2-ViT-B/14 | Self-supervised v2 | 14×14 | Standard CLS |

### Explainers

| # | Method | Class | Swin-B |
|---|--------|-------|--------|
| E1 | Raw CLS Attention | `RawAttentionExplainer` | ✗ |
| E2 | Attention Rollout | `AttentionRolloutExplainer` | ✗ |
| E3 | GradCAM | `GradCAMExplainer` | ✓ |
| E4 | Chefer LRP | `CheferLRPExplainer` | ✗ |
| E5 | RISE | `RISEExplainer` | ✓ |
| E6 | LIME | `LIMEExplainer` | ✓ |
| E7 | DIME | `DIMEExplainer` | — *(placeholder)* |

---

## Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/Joel-1007/vit-explainability-benchmark.git
cd vit-explainability-benchmark

# Install dependencies via pip
pip install -r requirements.txt

# Or use uv (recommended for fast, deterministic, reproducible environment management)
pip install uv --user
uv sync
```

### Run Tests

The metric suite is heavily tested (200+ unit tests):

```bash
pytest tests/ -v
# Or with uv:
uv run pytest tests/ -v
```

### Run the Phase 3 Benchmark

```bash
# Full evaluation (6 models × 6 explainers × 4 datasets)
python run_phase3.py --data-root /path/to/datasets

# Dry run (1 batch per combination, for validation)
python run_phase3.py --data-root /path/to/datasets --dry-run

# Subset: specific models and explainers
python run_phase3.py --data-root /path/to/datasets \
    --models vit_b16 deit_b16 \
    --explainers gradcam rollout \
    --datasets imagenet_s50 cub200
```

The benchmark automatically checkpoints each `(dataset, model, explainer)` combination as a `.pkl` file, so interrupted runs can be resumed seamlessly.

### Usage Example: Single Evaluation Loop

```python
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics
from metrics.robustness import RobustnessMetrics, randomise_model_weights

runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_custom_explainer_function,
    robustness=RobustnessMetrics(epsilon=0.05, n_samples=50),
    randomised_model=randomise_model_weights(model, seed=42)
)

results = runner.evaluate(model, val_loader, dataset_name="cub200")
print(f"Mean IoU: {results['macro']['miou']:.4f}")
print(f"Max Sensitivity: {results['macro']['max_sensitivity']:.4f}")
```

---

## 🎯 Recommended Running Strategy

The benchmark is designed to be run in three progressive stages.  
**Always start with the Pilot to verify everything works before committing GPU hours.**

| Stage | Images | RISE masks | Models | Runtime (A100) | Command |
|-------|--------|-----------|--------|----------------|---------|
| **1. Pilot** | 50 | 100 | 1 | ~10 min | `bash run_pilot.sh` |
| **2. Subsample** | 500 | 500 | 6 | ~4–6 hours | `uv run python run_phase3.py --data-root /data --max-samples 500` |
| **3. Full** | 5000+ | 4000 | 6 | ~3–5 days | `bash run_benchmark.sh` |

### Stage 1 — Pilot Run (Start Here)

Verifies the complete pipeline end-to-end in ~10 minutes:

```bash
# Set your dataset path first
export DATA_ROOT="/data"    # adjust to actual path

# Run the pilot
bash run_pilot.sh
```

What the pilot tests:
- ✅ All unit tests pass
- ✅ GPU is detected and working  
- ✅ Dataset loads correctly
- ✅ All 6 explainers run without errors  
- ✅ All metrics compute correctly
- ✅ Checkpointing works (safe to interrupt/resume)
- ✅ Phase 4 analytics scripts execute

If the pilot completes without errors, you're ready for the full run.

### Stage 2 — Subsample Run (Recommended for Validation)

500 images per dataset gives statistically meaningful results in a manageable time:

```bash
# Uses max-samples and max-batches to limit dataset size
uv run python run_phase3.py \
    --data-root $DATA_ROOT \
    --checkpoint-dir results/subsample \
    --max-samples 500 \
    --seed 42
```

> RISE with 500 masks on 500 images is the approach recommended in the RISE paper for GPU-constrained evaluation. Results are reported with a table footnote noting the subsample.

### Stage 3 — Full Run (Publication Results)

```bash
bash run_benchmark.sh
```

---

## 📂 Repository Structure

```text
vit-explainability-benchmark/
│
├── README.md                   # This file
├── BENCHMARK.md                # Authoritative technical reference (metrics, proofs, protocols)
├── requirements.txt            # Python dependencies (pip install -r requirements.txt)
├── pyproject.toml              # uv/pip project configuration
├── Dockerfile                  # Reproducible container build
├── run_phase3.py               # ★ Main Phase 3 benchmark runner script
├── run_benchmark.sh            # End-to-end pipeline (bash)
├── model_hashes.txt            # SHA-256 hashes of all pre-trained checkpoints
│
├── model_zoo/                  # 6 standardised ViT model wrappers
│   ├── __init__.py             #   load_model() dispatcher
│   ├── vit_b16.py              #   ViT-B/16 (timm augreg)
│   ├── deit_b16.py             #   DeiT-B/16 (distilled)
│   ├── swin_b.py               #   Swin-B (shifted-window)
│   ├── beit_b16.py             #   BEiT-B/16 (MIM)
│   ├── dino_vitb8.py           #   DINO-ViT-B/8
│   └── dinov2_vitb14.py        #   DINOv2-ViT-B/14
│
├── explainers/                 # 7 explanation method implementations
│   ├── base.py                 #   BaseExplainer ABC
│   ├── raw_attention.py        #   E1 — Raw CLS attention
│   ├── rollout.py              #   E2 — Attention Rollout
│   ├── gradcam.py              #   E3 — GradCAM
│   ├── chefer_lrp.py           #   E4 — Chefer et al. LRP
│   ├── rise.py                 #   E5 — RISE (4000 masks)
│   ├── lime.py                 #   E6 — LIME (patch-grid)
│   └── dime.py                 #   E7 — DIME (placeholder)
│
├── metrics/                    # Evaluation suite
│   ├── fidelity.py             #   F1–F3 (Sufficiency, Comprehensiveness, Log-odds)
│   ├── localization.py         #   L1–L4 (mIoU, PG, EGT, CalibGap)
│   ├── robustness.py           #   R1–R3 (MaxSens, ModelRand, LabelRand)
│   ├── complexity.py           #   C1–C3 (Gini, Entropy, EMR)
│   ├── axiom_verifier.py       #   Axiomatic analysis (A1–A4) + Theorems T1–T6
│   ├── normalize.py            #   Attribution normalisation pipeline
│   ├── sanity.py               #   Sanity checks S1–S3
│   ├── suite.py                #   MetricSuite unified class
│   ├── causal_fidelity.py      #   Causal masking metric
│   ├── adversarial_robustness.py #  PGD adversarial robustness
│   ├── explainer_interaction.py #  Explainer Interaction Graph (EIG)
│   └── runner.py               #   BenchmarkRunner + Phase3Runner
│
├── training/                   # Standardised fine-tuning protocol
│   ├── transforms.py           #   RandAugment + random erasing
│   ├── mixup.py                #   Batch Mixup (α=0.8)
│   ├── optimizer.py            #   AdamW + warmup + cosine decay
│   ├── loss.py                 #   SoftTargetCE / BCE
│   └── trainer.py              #   Fine-tune loop with AMP
│
├── configs/                    # Per-dataset YAML configurations
│   ├── cub200.yaml             #   CUB-200-2011 (200 classes)
│   ├── pascal_voc.yaml         #   PASCAL VOC 2012 (20 classes)
│   ├── imagenet.yaml           #   ImageNet-1K (linear probe)
│   ├── imagenet_s50.yaml       #   ImageNet-S-50 (50-class subset)
│   └── nih_chestxray.yaml      #   NIH ChestX-ray14 (14 classes)
│
├── scripts/                    # Utilities, data prep, Phase 4 analytics
│   ├── verify_datasets.py      #   Dataset integrity verification
│   ├── pilot_finetune.py       #   5-epoch sanity check
│   ├── record_model_hashes.py  #   SHA-256 hash logging
│   ├── create_cub_val_split.py #   Stratified CUB val split
│   ├── phase4_correlation_analysis.py  # Task 4.1 — Spearman + PCA
│   ├── phase4_interaction_analysis.py  # Task 4.2 — Kendall τ
│   ├── phase4_ablations.py     #   Task 4.3 — Ablations + Cohen's d
│   ├── plot_interaction_graph.py #  EIG visualisation
│   ├── plot_community_radar.py #   Community radar chart
│   └── plot_fidelity_curves.py #   Fidelity curve plots
│
├── tests/                      # 200+ unit tests
├── utils/                      # Shared utilities (PGD attack)
├── paper/                      # LaTeX manuscript drafts
├── results/                    # Output directory for checkpoints
│
└── docs/                       # Project documentation
    ├── implementation_guide.md #   Detailed 5-phase implementation guide
    ├── phase4_instructions.md  #   Phase 4 operational instructions
    ├── project_status_checklist.md # Full task audit
    ├── compute_budget_breakdown.md # GPU hour estimates
    ├── anaconda_compatibility_and_runtime_analysis.md
    └── literature_review.md    #   Literature review & gap analysis
```

---

## 🖥️ Running on an NVIDIA A100 HPC / SLURM Cluster

This section covers running the benchmark on an **NVIDIA A100 GPU node** in a high-performance computing (HPC) cluster environment.

### Prerequisites

| Tool | Version |
|------|---------|
| CUDA | 12.1 (matches Dockerfile) |
| Python | 3.13+ |
| PyTorch | 2.2.0 |
| `uv` | Latest |

### 1. Log In & Verify the GPU

```bash
nvidia-smi
# Should show: NVIDIA A100 with CUDA 12.x
```

### 2. Load Environment Modules

Most HPC clusters use `module` to manage CUDA/Python versions:

```bash
module purge
module load cuda/12.1
module load python/3.13
module list            # verify what is loaded
```

> ⚠️ Module names vary by cluster — run `module avail cuda` and `module avail python` to find the correct names.

### 3. Clone & Install

```bash
# Use scratch/work storage (not home — usually quota-limited)
cd /scratch/$USER

git clone https://github.com/Joel-1007/vit-explainability-benchmark.git
cd vit-explainability-benchmark

pip install uv --user
uv sync
```

Verify PyTorch sees the A100:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True
#           NVIDIA A100 ...
```

### 4. Set the Dataset Path

Point `DATA_ROOT` to where the datasets live on the cluster's shared storage:

```bash
export DATA_ROOT="/data"    # adjust to the correct path
```

The benchmark expects the following structure under `$DATA_ROOT`:

| Dataset | Expected Path |
|---------|---------------|
| CUB-200-2011 | `$DATA_ROOT/cub200/` |
| PASCAL VOC | `$DATA_ROOT/voc/` |
| ImageNet-S50 | `$DATA_ROOT/imagenet_s50/` |
| NIH ChestX-ray | `$DATA_ROOT/nih_chestxray/` |

### 5. Run the Full Pipeline

```bash
bash run_benchmark.sh
```

The script runs 4 automated stages:
1. ✅ Environment integrity check (all unit tests)
2. 📋 Dataset verification (all 4 datasets)
3. 🔬 Phase 3 metric evaluation — checkpointed per `(dataset, model, explainer)`, safe to interrupt and resume
4. 📊 Phase 4 analytics → `results/phase4/`

### 6. SLURM Job Submission

If the cluster uses SLURM, use the following job script:

```bash
cat > run_vit_benchmark.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=vit-bench
#SBATCH --partition=gpu          # adjust: check available partitions with `sinfo`
#SBATCH --gres=gpu:a100:1        # 1 × A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

module purge
module load cuda/12.1
module load python/3.13

cd /scratch/$USER/vit-explainability-benchmark
export DATA_ROOT="/data"

bash run_benchmark.sh
EOF

mkdir -p logs
sbatch run_vit_benchmark.slurm

# Monitor job
squeue -u $USER
tail -f logs/bench_<JOB_ID>.out
```

### 7. A100-Specific Optimizations

The A100 supports **bfloat16** and **TF32** natively for significantly faster training and inference. Add to `main.py`:

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Expected Runtime on A100

| Stage | Estimated Time |
|-------|----------------|
| Test suite | ~2–5 min |
| Full Phase 3 benchmark | ~4–8 hours |
| Phase 4 analytics | ~15–30 min |

> The `Phase3Runner` checkpoints results per `(dataset, model, explainer)` — it can be safely interrupted and resumed at any point.

### Docker / Singularity (Alternative)

The repo includes a `Dockerfile` pre-configured for **CUDA 12.1 + PyTorch 2.2**:

```bash
# Docker
docker build -t vit-bench .
docker run --gpus all -v /data:/data -e DATA_ROOT=/data vit-bench bash run_benchmark.sh

# Singularity (common in HPC environments)
singularity build vit_bench.sif docker://pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
```

---

## Reproducibility

- **Deterministic seeds:** All pseudo-random sampling accepts explicit integer seeds, injected before every combination.
- **Model hashes:** SHA-256 of all pre-trained checkpoints locked in `model_hashes.txt`.
- **Checkpointing:** `Phase3Runner` saves per-combination `.pkl` files with atomic writes for crash recovery.
- **Data splits:** Deterministic and documented in `configs/*.yaml`.
- **Run metadata:** Each benchmark run saves a `run_metadata.json` with environment details.

---

## Extending the Benchmark

### Adding a New Explainer

1. Create `explainers/my_method.py` and subclass `BaseExplainer`.
2. Implement `explain(x, target_class) → (H_patches, W_patches)` tensor.
3. Optionally override `explain_batch()` for amortised computation.
4. Register in `explainers/__init__.py`.
5. Add unit tests following `tests/test_explainers.py` patterns.

```python
from explainers.base import BaseExplainer
import torch

class MyMethodExplainer(BaseExplainer):
    def explain(self, x: torch.Tensor, target_class: int, **kwargs) -> torch.Tensor:
        # Your attribution logic here
        # Return shape: (H // patch_size, W // patch_size)
        ...
```

### Adding a New Metric

1. Create or extend a file in `metrics/`.
2. Follow patterns in `LocalizationMetrics`, `ComplexityMetrics`.
3. Ensure the metric accepts normalised attribution maps (use `metrics/normalize.py`).
4. Integrate into `BenchmarkRunner` or `Phase3Runner` in `metrics/runner.py`.
5. Write unit tests with known-good and known-bad inputs.
6. Document the formal definition, range, and direction in `BENCHMARK.md`.

### Reproducing Figures

```bash
# Axiom satisfaction heatmap
python -c "from metrics.axiom_verifier import generate_axiom_satisfaction_heatmap; generate_axiom_satisfaction_heatmap('figures/axiom_satisfaction.pdf')"

# Complexity distributions
python -c "from metrics.complexity import run_sanity_check; run_sanity_check()"

# Phase 4 analytics (requires Phase 3 results)
python scripts/phase4_correlation_analysis.py \
    --results_csv results/phase3/aggregated_results.csv \
    --output_dir results/phase4
```

---

## Phase 4: Analysis & Findings

> *Pending Phase 3 benchmark results. See `docs/phase4_instructions.md` for the execution workflow.*

| Analysis | Script | Output |
|----------|--------|--------|
| Inter-Metric Correlation | `scripts/phase4_correlation_analysis.py` | Spearman heatmap + PCA |
| Task-Metric Concordance | `scripts/phase4_interaction_analysis.py` | Kendall τ + decision tree |
| Ablation Studies (A1–A4) | `scripts/phase4_ablations.py` | Cohen's d effect sizes |

---

## License

MIT
