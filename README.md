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

# Install uv (if not already installed)
pip install uv --user

# Install all dependencies from the lock file (fully reproducible)
uv sync
```

### 2. Run the Benchmark Test Suite

The metric suite is heavily tested (200+ unit tests). Run them to verify your environment. Some tests require PyTorch; if it is not installed in the environment, those tests will be gracefully skipped.

```bash
# Run all metrics tests (Localization, Robustness, Complexity, Axiomatic)
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
├── BENCHMARK.md             # The core reference document
├── model_zoo/               # Standardised ViT wrappers
├── training/                # Shared fine-tuning protocol (AdamW + Cosine + Mixup)
├── metrics/                 # The evaluation suite
│   ├── fidelity.py          # F1–F3
│   ├── localization.py      # L1–L4
│   ├── robustness.py        # R1–R3
│   ├── complexity.py        # C1–C3
│   ├── axiom_verifier.py    # Empirical axiom testing (A1–A4)
│   └── runner.py            # BenchmarkRunner loop
├── explainers/              # 7 XAI methods (Raw Attn, Rollout, GradCAM, LRP, RISE, LIME, DIME)
├── tests/                   # 200+ unit tests
├── configs/                 # YAML configs for dataset-specific runs
├── scripts/                 # Phase 4 analytics, data prep, reproducibility
├── paper/                   # LaTeX / manuscript assets
└── pyproject.toml           # uv dependency management
```

---

## 🖥️ Running on an A100 Lab (HPC / SLURM Cluster)

This section covers running the benchmark on an A100 GPU node in a university HPC environment.

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
# Expected: True \n NVIDIA A100 ...
```

### 4. Set the Dataset Path

Point `DATA_ROOT` to where the datasets live on the cluster:

```bash
export DATA_ROOT="/data"    # adjust to the correct shared storage path
```

The benchmark expects these sub-directories under `$DATA_ROOT`:

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

The script runs 4 stages automatically:
1. ✅ Environment integrity check (unit tests)
2. 📋 Dataset verification
3. 🔬 Phase 3 metric evaluation (checkpointed — safe to interrupt and resume)
4. 📊 Phase 4 analytics → `results/phase4/`

### 6. SLURM Job Submission

If the cluster uses SLURM, use the following job script:

```bash
cat > run_vit_benchmark.slurm << 'EOF'
#!/bin/bash
#SBATCH --job-name=vit-bench
#SBATCH --partition=gpu          # adjust partition name: run `sinfo`
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

# Monitor
squeue -u $USER
tail -f logs/bench_<JOB_ID>.out
```

### 7. A100-Specific Optimizations

The A100 supports **bfloat16** and **TF32** natively. Add these lines to `main.py` for maximum throughput:

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

> The `Phase3Runner` checkpoints results per `(dataset, model, explainer)` — it can be safely interrupted and resumed at any point without re-running completed combinations.

### Docker / Singularity (Alternative)

The repo ships with a `Dockerfile` pre-configured for **CUDA 12.1 + PyTorch 2.2**:

```bash
# Docker
docker build -t vit-bench .
docker run --gpus all -v /data:/data -e DATA_ROOT=/data vit-bench bash run_benchmark.sh

# Singularity (common in HPC environments)
singularity build vit_bench.sif docker://pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
```

---

## Reproducibility Guarantees
- SHA-256 hashes of all pre-trained models are locked in `model_hashes.txt`.
- Data splits are deterministic.
- All pseudo-random sampling in metrics (e.g., L2 tie-breaking, R1 noise generation) accepts explicit integer seeds.

## 📊 Phase 4: Analysis & Findings (Experimental)
*(Note: These are placeholders for Phase 4 empirical results.)*

- **Inter-Metric Correlation:** [PLACEHOLDER: Summary of metric orthogonality and heatmap results]
- **Task-Metric Concordance:** [PLACEHOLDER: Discordance insights and Kendall tau findings]
- **Ablation Studies:** 
  - Token Resolution: `[PLACEHOLDER: Cohen's d effect size]`
  - Layer Depth: `[PLACEHOLDER: Cohen's d effect size]`
  - Masking Strategy: `[PLACEHOLDER: Cohen's d effect size]`
  - Pre-training Objective: `[PLACEHOLDER: Cohen's d effect size]`
