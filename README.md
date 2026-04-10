# ViT Explainability Benchmark

**A Comprehensive, Axiomatically-Grounded Explainability Benchmark for Vision Transformers**

> **Target Venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)  
> **Status:** Active Development (Phases 1 & 2 Complete)

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
4. **Fidelity (F1–F3):** (Phase 3) Does removing the highlighted patches actually destroy the model's confidence? (Insertion/Deletion AUC, Comprehensiveness).

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
git clone https://github.com/[YOUR-ORG]/vit-explainability-benchmark.git
cd vit-explainability-benchmark

# The repository uses uv for dependency management (Python 3.13)
# Dependencies are specified in pyproject.toml
```

### 2. Run the Benchmark Test Suite

The metric suite is heavily tested (100+ unit tests). Run them to verify your environment. Some tests require PyTorch; if it is not installed in the environment, those tests will be gracefully skipped.

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
│   ├── localization.py      # L1–L4
│   ├── robustness.py        # R1–R3
│   ├── complexity.py        # C1–C3
│   ├── axiom_verifier.py    # Empirical axiom testing (A1–A4)
│   └── runner.py            # BenchmarkRunner loop
├── tests/                   # 100+ Unit tests
├── configs/                 # YAML configs for dataset-specific runs
├── scripts/                 # Reproducibility data prep & logging
└── pyproject.toml           # uv dependency management
```

## Reproducibility Guarantees
- SHA-256 hashes of all pre-trained models are locked in `model_hashes.txt`.
- Data splits are deterministic.
- All pseudo-random sampling in metrics (e.g., L2 tie-breaking, R1 noise generation) accepts explicit integer seeds.
