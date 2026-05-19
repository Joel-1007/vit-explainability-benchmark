# ViT Explainability Benchmark

**A Comprehensive, Axiomatically-Grounded Explainability Benchmark for Vision Transformers**

> **Target Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)  
> **Status:** Active Development (Phases 1–3 Complete; Phase 4 analysis pending benchmark results)

Existing evaluation frameworks for Vision Transformer (ViT) explanations are often inconsistent, narrowly scoped, and lack standardised methodology. This repository provides a rigorous, unified, and mathematically grounded evaluation suite designed to generate consistent empirical evidence across multiple ViT architectures and diverse explanation methods.

---

## 📖 Authoritative Documentation

All formal metric definitions, mathematical proofs, experimental protocols, API layouts, and unit test specifications are documented in the single authoritative reference file:

👉 **[BENCHMARK.md](BENCHMARK.md)**

Please read `BENCHMARK.md` before diving into the code. It supersedes all individual task documents.

---

## 🚀 Key Capabilities

The framework systematically evaluates spatial interpretation methods across four critical dimensions:

1. **Localization (L1–L4):** Does the explanation highlight the object of interest? (mIoU, Pointing Game, EGT, CalibGap).
2. **Robustness (R1–R3):** How sensitive is the explanation to irrelevant input perturbations, model randomisation, and label randomisation? (MaxSens, ModelRand, LabelRand).
3. **Complexity (C1–C3):** How parsimonious is the explanation? Are the decisive patches compact and easily understandable? (Gini, Entropy, Effective Mass Ratio).
4. **Fidelity (F1–F3):** Does removing the highlighted patches actually destroy the model's confidence? (Sufficiency, Comprehensiveness, Log-odds Drop).

Furthermore, the suite includes an **Axiomatic Analysis Tool** (`AxiomVerifier`) to map every empirical metric against the four fundamental Shapley axioms (Dummy, Completeness, Symmetry, Linearity) to surface representational biases (e.g., Theorem T6: the anti-alignment of complexity and symmetry).

### Controlled Model Zoo
To decouple explanation variation from training noise, the benchmark includes a controlled zoo of six ViT variants (~86M parameters each) fine-tuned under an identical, standardised protocol:
- Standard ViT-B/16 (augreg)
- DeiT-B/16 (distilled)
- Swin-B (shifted-window)
- BEiT-B/16 (MIM)
- DINO-ViT-B/8 & DINOv2-ViT-B/14 (self-supervised)

---

## 🛠️ Quickstart

We use [`uv`](https://docs.astral.sh/uv/) for fast and deterministic Python environment management. 

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/Joel-1007/vit-explainability-benchmark.git
cd vit-explainability-benchmark

# Install uv (fast, deterministic package manager)
pip install uv --user

# Install all dependencies from the lock file (fully reproducible)
uv sync
```

### 2. Run the Benchmark Test Suite

The metric suite is heavily tested (200+ unit tests). Run them to verify your environment. Some tests require PyTorch; if it is not installed in the environment, those tests will be gracefully skipped.

```bash
# Run all metrics and explainer tests
uv run pytest tests/ -v
```

### 3. Usage Example: Unified Evaluation Loop

The `BenchmarkRunner` provides a single loop to evaluate an explainer against a dataset.

```python
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics
from metrics.robustness import RobustnessMetrics, randomise_model_weights

# Initialize the runner with your desired metrics
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_custom_explainer_function,
    robustness=RobustnessMetrics(epsilon=0.05, n_samples=50),
    randomised_model=randomise_model_weights(model, seed=42)
)

# Run over a standard PyTorch DataLoader yielding (images, masks, labels)
results = runner.evaluate(model, val_loader, dataset_name="cub200")

print(f"Mean IoU: {results['macro']['miou']:.4f}")
print(f"Max Sensitivity (Robustness): {results['macro']['max_sensitivity']:.4f}")
```

---

## 📂 Repository Architecture

```text
vit-explainability-benchmark/
├── BENCHMARK.md                # The core reference document
├── model_zoo/                  # Standardised ViT wrappers
│   ├── __init__.py             # load_model() dispatcher
│   ├── vit_b16.py              # Model 1 — ViT-B/16
│   ├── deit_b16.py             # Model 2 — DeiT-B/16
│   ├── swin_b.py               # Model 3 — Swin-B
│   ├── beit_b16.py             # Model 4 — BEiT-B/16
│   ├── dino_vitb8.py           # Model 5 — DINO-ViT-B/8
│   └── dinov2_vitb14.py        # Model 6 — DINOv2-ViT-B/14
├── training/                   # Shared fine-tuning protocol (AdamW + Cosine + Mixup)
│   ├── transforms.py           # RandAugment + random erasing
│   ├── mixup.py                # Batch Mixup α=0.8
│   ├── optimizer.py            # AdamW + warmup + cosine
│   ├── loss.py                 # SoftTargetCE / BCE
│   └── trainer.py              # Full fine-tune loop
├── explainers/                 # 7 explanation method implementations
│   ├── base.py                 # BaseExplainer ABC
│   ├── raw_attention.py        # E1 — Raw CLS attention
│   ├── rollout.py              # E2 — Attention Rollout
│   ├── gradcam.py              # E3 — GradCAM
│   ├── chefer_lrp.py           # E4 — Chefer et al. LRP
│   ├── rise.py                 # E5 — RISE (4000 masks)
│   ├── lime.py                 # E6 — LIME (patch-grid)
│   └── dime.py                 # E7 — DIME (placeholder)
├── metrics/                    # The evaluation suite
│   ├── fidelity.py             # F1–F3 (Sufficiency, Comprehensiveness, Log-odds)
│   ├── localization.py         # L1–L4 (mIoU, PG, EGT, CalibGap)
│   ├── robustness.py           # R1–R3 (MaxSens, ModelRand, LabelRand)
│   ├── complexity.py           # C1–C3 (Gini, Entropy, EMR)
│   ├── axiom_verifier.py       # Empirical axiom testing (A1–A4)
│   ├── normalize.py            # Attribution normalisation pipeline
│   ├── sanity.py               # Sanity checks S1–S3
│   └── runner.py               # BenchmarkRunner + Phase3Runner
├── tests/                      # 200+ unit tests
├── configs/                    # YAML configs for dataset-specific runs
├── scripts/                    # Reproducibility, data prep, Phase 4 analytics
└── pyproject.toml              # uv dependency management
```

## Reproducibility Guarantees
- SHA-256 hashes of all pre-trained models are locked in `model_hashes.txt`.
- Data splits are deterministic.
- All pseudo-random sampling in metrics (e.g., L2 tie-breaking, R1 noise generation) accepts explicit integer seeds.
- The `Phase3Runner` checkpoints results per `(dataset, model, explainer)` combination for crash recovery.

---

## 🖥️ Running on an A100 Lab (HPC / SLURM Cluster)

This section covers running the benchmark on an **NVIDIA A100 GPU node** in an HPC lab environment.

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

## 🔧 Extending the Benchmark

### Adding a New Explanation Method

1. Create a new file in `explainers/`, e.g. `explainers/my_method.py`.
2. Subclass `BaseExplainer` from `explainers/base.py`.
3. Implement the `explain(x, target_class, **kwargs) → torch.Tensor` method:
   - Input: `x` is a `(3, H, W)` float32 tensor in `[0, 1]`.
   - Output: a `(H // patch_size, W // patch_size)` float32 tensor (un-normalised).
4. Optionally override `explain_batch()` for amortised computation.
5. Register the explainer in `explainers/__init__.py`.
6. Add unit tests in `tests/` following the pattern in `test_explainers.py` (shape, finite, batch, variance checks).

```python
# explainers/my_method.py
from explainers.base import BaseExplainer
import torch

class MyMethodExplainer(BaseExplainer):
    def explain(self, x: torch.Tensor, target_class: int, **kwargs) -> torch.Tensor:
        # Your attribution logic here
        # Return shape: (H_patches, W_patches)
        ...
```

### Adding a New Metric

1. Create or extend a file in `metrics/`.
2. Follow the pattern of existing metric classes (e.g., `LocalizationMetrics`, `ComplexityMetrics`).
3. Ensure the metric accepts normalised attribution maps (use `metrics/normalize.py` upstream).
4. Add the metric to the `BenchmarkRunner` or `Phase3Runner` integration in `metrics/runner.py`.
5. Write unit tests with known-good inputs (perfect attribution → expected score) and known-bad inputs (random/misaligned → chance level).
6. Document the formal definition, range, and direction (higher-is-better vs lower-is-better) in `BENCHMARK.md`.

### Reproducing Figures

```bash
# Axiom satisfaction heatmap (Figure F1)
uv run python -c "from metrics.axiom_verifier import generate_axiom_satisfaction_heatmap; generate_axiom_satisfaction_heatmap('figures/axiom_satisfaction.pdf')"

# Complexity distributions
uv run python -c "from metrics.complexity import run_sanity_check; run_sanity_check()"

# Phase 4 analytics (requires Phase 3 results CSV)
uv run python scripts/phase4_correlation_analysis.py --results_csv results/aggregated_results.csv --output_dir results/phase4
uv run python scripts/phase4_interaction_analysis.py --results_csv results/aggregated_results.csv --output_dir results/phase4
uv run python scripts/phase4_ablations.py --output_dir results/phase4
```

---

## 📊 Phase 4: Analysis & Findings (Experimental)
*(Note: These are placeholders for Phase 4 empirical results.)*

- **Inter-Metric Correlation:** [PLACEHOLDER: Summary of metric orthogonality and heatmap results]
- **Task-Metric Concordance:** [PLACEHOLDER: Discordance insights and Kendall tau findings]
- **Ablation Studies:** 
  - Token Resolution: `[PLACEHOLDER: Cohen's d effect size]`
  - Layer Depth: `[PLACEHOLDER: Cohen's d effect size]`
  - Masking Strategy: `[PLACEHOLDER: Cohen's d effect size]`
  - Pre-training Objective: `[PLACEHOLDER: Cohen's d effect size]`
