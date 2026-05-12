# Anaconda Compatibility & Single A100 Runtime Analysis

## 1. Anaconda Compatibility Assessment

### Verdict: ✅ Fully Compatible — with minor adjustments

The project uses only standard PyPI packages (PyTorch, timm, transformers, scipy, etc.) that are all available via `conda` or `conda-forge`. There are no exotic C-extension dependencies or build requirements that would be incompatible with Anaconda. Below are the specific items to address.

---

### 1.1 Python Version Constraint

> [!WARNING]
> The project's `.python-version` file specifies **Python 3.13**, and `pyproject.toml` requires `>=3.13`. Anaconda's default Python channel typically lags behind. As of mid-2026, `conda-forge` should have 3.13, but if not, you'll need to relax this to `>=3.11`.

**Action required:**
- Check if your Anaconda distribution ships Python 3.13: `conda search python=3.13`
- If not available, change `requires-python` in [pyproject.toml](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/pyproject.toml) to `>=3.11` and `.python-version` to `3.11`. All code uses only standard Python 3.10+ features (type union syntax `X | Y` requires 3.10+).

---

### 1.2 Package Manager: `uv` → `conda`/`pip`

The project currently uses **`uv`** (Astral's fast pip replacement) as the package manager, as seen in:
- [Dockerfile](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/Dockerfile) — `RUN pip install uv`, `uv pip install -r requirements.txt`
- [run_benchmark.sh](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/run_benchmark.sh) — all commands use `uv run python ...`
- A `uv.lock` file exists for deterministic resolution

**Changes for Anaconda:**

| File | Current | Anaconda Equivalent |
|------|---------|-------------------|
| `Dockerfile` | `pip install uv` + `uv pip install` + `uv venv` | Use `conda` base image or `pip install` directly |
| `run_benchmark.sh` | `uv run python ...` | `python ...` (inside activated conda env) |
| `run_benchmark.sh` | `uv run pytest tests/` | `pytest tests/` |
| `uv.lock` | `uv`'s lock format | Replace with `environment.yml` for conda |

---

### 1.3 Dependencies — All Conda-Compatible

Every dependency in [requirements.txt](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/requirements.txt) is available via conda or conda-forge:

| Package | `requirements.txt` Version | Conda Channel | Notes |
|---------|---------------------------|---------------|-------|
| `torch>=2.2.0` | ✅ | `pytorch` channel | Install with `conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia` |
| `torchvision>=0.17.0` | ✅ | `pytorch` channel | Bundled with PyTorch |
| `timm>=0.9.16` | ✅ | `conda-forge` or `pip` | `pip install timm` inside conda env is fine |
| `transformers>=4.40.0` | ✅ | `conda-forge` | Or `pip install` |
| `huggingface_hub>=0.22.0` | ✅ | `conda-forge` | |
| `datasets>=2.19.0` | ✅ | `conda-forge` | |
| `numpy>=1.26.0` | ✅ | `defaults` | |
| `scipy>=1.13.0` | ✅ | `defaults` | |
| `scikit-learn>=1.4.0` | ✅ | `defaults` | |
| `pandas>=2.2.0` | ✅ | `defaults` | |
| `matplotlib>=3.8.0` | ✅ | `defaults` | |
| `seaborn>=0.13.0` | ✅ | `conda-forge` | |
| `Pillow>=10.3.0` | ✅ | `defaults` | |
| `tqdm>=4.66.0` | ✅ | `defaults` | |
| `pyyaml>=6.0.1` | ✅ | `defaults` | |
| `omegaconf>=2.3.0` | ✅ | `conda-forge` | |

> [!TIP]
> For PyTorch with CUDA on Anaconda, always install from the `pytorch` channel, not `defaults`:
> ```bash
> conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
> ```

---

### 1.4 `environment.yml` — Recommended for Anaconda

Create this file to replace the `uv`-based workflow:

```yaml
# environment.yml
name: vit-bench
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python>=3.11
  - pytorch>=2.2.0
  - torchvision>=0.17.0
  - pytorch-cuda=12.1
  - numpy>=1.26.0
  - scipy>=1.13.0
  - scikit-learn>=1.4.0
  - pandas>=2.2.0
  - matplotlib>=3.8.0
  - seaborn>=0.13.0
  - pillow>=10.3.0
  - tqdm>=4.66.0
  - pyyaml>=6.0.1
  - pytest>=9.0.0
  - pip
  - pip:
    - timm>=0.9.16
    - transformers>=4.40.0
    - huggingface_hub>=0.22.0
    - datasets>=2.19.0
    - omegaconf>=2.3.0
```

Usage: `conda env create -f environment.yml && conda activate vit-bench`

---

### 1.5 Code Compatibility Issues

> [!IMPORTANT]
> **Zero code-level incompatibilities found.** The Python code itself is entirely standard and does not use any `uv`-specific APIs, Rust extensions, or system-level bindings that would differ between Anaconda and a regular venv/pip setup.

Specific items verified:
- **Phase 4 scripts** ([phase4_correlation_analysis.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_correlation_analysis.py), [phase4_interaction_analysis.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_interaction_analysis.py), [phase4_ablations.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_ablations.py)) — use only `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`. All pure Python/Cython packages.
- **Training code** ([trainer.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/training/trainer.py)) — standard PyTorch with `torch.cuda.amp`. Fully compatible.
- **Model zoo** — loads via `timm`, which works identically under conda.
- **Metrics engine** ([runner.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/metrics/runner.py)) — pure PyTorch. No environment-specific code.
- **`torch.cuda.amp.GradScaler` / `autocast`** — Anaconda's PyTorch includes these by default when installed with CUDA.

---

### 1.6 Summary of Changes for Anaconda

| # | Change | Effort | Priority |
|---|--------|--------|----------|
| 1 | Create `environment.yml` (above) | 5 min | **High** |
| 2 | Relax `requires-python` to `>=3.11` if 3.13 unavailable | 1 min | **High** |
| 3 | Replace `uv run python` → `python` in `run_benchmark.sh` | 2 min | **Medium** |
| 4 | Replace `uv run pytest` → `pytest` in `run_benchmark.sh` | 1 min | **Medium** |
| 5 | Update `Dockerfile` to use conda base image (optional) | 15 min | **Low** |

**Total effort: ~25 minutes.** The project is essentially Anaconda-ready today.

---

---

## 2. Single A100 (80GB) Runtime Estimation

### Compute Budget Breakdown

The original budget was estimated for parallelised multi-GPU execution. Below is a **revised estimate for a single A100 80GB GPU**, accounting for sequential execution and the project's actual workload.

---

### 2.1 Fine-tuning: 6 Models × 4 Datasets (24 combinations)

**Per-combination estimate:**

| Model | Params | Approx. time per dataset (50 epochs, A100) |
|-------|--------|---------------------------------------------|
| ViT-B/16 | 86M | ~3–4 hrs |
| DeiT-B/16 | 86M | ~3–4 hrs |
| Swin-B | 88M | ~4–5 hrs |
| BEiT-B/16 | 86M | ~3–4 hrs |
| DINO-ViT-B/8 | 86M | ~5–6 hrs (patch size 8 → 4× more tokens) |
| DINOv2-ViT-B/14 | 86M | ~4–5 hrs |

**Reasoning:**
- Batch size 256, 50 epochs, mixed precision (AMP enabled by default)
- CUB-200: ~6K train images → ~24 batches/epoch → ~1,200 steps total → ~1.5 hrs
- ImageNet-S-50: ~65K images → ~254 batches/epoch → ~12,700 steps → ~4 hrs
- PASCAL VOC: ~5K train → ~20 batches/epoch → ~1 hr
- NIH ChestX-ray14: ~87K images → ~340 batches/epoch → ~5 hrs
- A100 throughput at batch=256, 224×224: ~1,200 img/s (ViT-B), ~900 img/s (DINO-ViT-B/8)

| Dataset | Images | Time per model (est.) |
|---------|--------|-----------------------|
| CUB-200 | ~6K | 1.5–2 hrs |
| ImageNet-S-50 | ~65K | 3.5–5 hrs |
| PASCAL VOC | ~5K | 1–1.5 hrs |
| NIH ChestX-ray14 | ~87K | 4–6 hrs |
| **Total per model** | | **10–14.5 hrs** |

**Total for 6 models: 60–87 hrs** (sequential, single A100)

> [!NOTE]
> The original estimate of 120–180 hrs assumed 4 GPUs with parallelisation overhead. On a single A100, the raw compute is actually *less* because there's no multi-GPU communication overhead, but you lose parallelism. The A100 80GB can handle batch_size=256 for all models without gradient accumulation (except possibly DINO-ViT-B/8), making it efficient.

---

### 2.2 Pilot Benchmark (100 samples)

| Step | Estimate |
|------|----------|
| 100 samples × 7 explainers × 6 models × 4 datasets × all metrics | ~2–3 hrs |

No change from original — this is a quick sanity check.

---

### 2.3 Full Benchmark (Phase 3)

This is the largest task. The workload is:
- **4 datasets × 6 models × 7 explainers = 168 combinations**
- For each: run explainer on N samples, compute all metrics (L1–L4, F1–F3, R1–R3, C1–C3, sanity checks)

**Per-combination cost drivers:**
- RISE (E5): M=4,000 masks per sample → ~2s/sample on A100
- LIME (E6): ~1.5s/sample (superpixel + regression)
- Chefer LRP (E4): ~0.3s/sample (backward pass)
- GradCAM (E3): ~0.2s/sample
- Attention-based (E1, E2): ~0.1s/sample
- DIME (E7): placeholder/pending — excluded for now

**Effective samples per dataset:**

| Dataset | Val/Test Samples | Notes |
|---------|-----------------|-------|
| CUB-200 | ~5,794 | Official test split |
| ImageNet-S-50 | ~2,500 | Subset with masks |
| PASCAL VOC | ~1,449 | Val 2012 |
| NIH ChestX-ray14 | ~25,596 | Official test set |
| **Total** | **~35,339** | |

**Per-combination time (averaged across explainers):**

| Explainer | Time/sample | × ~35K samples | Total |
|-----------|-------------|-----------------|-------|
| E1 Raw Attention | 0.1s | 0.97 hrs | × 6 models = ~6 hrs |
| E2 Rollout | 0.1s | 0.97 hrs | × 6 models = ~6 hrs |
| E3 GradCAM | 0.2s | 1.94 hrs | × 6 models = ~12 hrs |
| E4 Chefer LRP | 0.3s | 2.92 hrs | × 6 models = ~18 hrs |
| E5 RISE | 2.0s | 19.4 hrs | × 6 models = ~117 hrs |
| E6 LIME | 1.5s | 14.6 hrs | × 6 models = ~88 hrs |
| **Subtotal** | | | **~247 hrs** |

**Adding metric computation overhead** (~20% on top of explainer time for all metric families): **~296 hrs**

> [!IMPORTANT]
> **RISE and LIME dominate the runtime.** These two perturbation-based methods account for ~70% of the full benchmark time. Consider reducing RISE masks from 4,000 to 2,000 (diminishing returns above ~2,000 per the original paper) to save ~50 hrs.

---

### 2.4 Randomisation Tests (R3, R4)

These require re-running explainers on weight-randomised and label-randomised model variants.

| Test | What it does | Time estimate |
|------|-------------|---------------|
| R2: Model randomisation | Run all explainers on weight-randomised model copies | ~100 hrs (same as full benchmark but metric computation is lighter — just Spearman ρ) |
| R3: Label randomisation | Run all explainers on label-shuffled models | ~80 hrs |

The code already supports these via `RobustnessMetrics` in [runner.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/metrics/runner.py#L99-L136). If run simultaneously with the main benchmark (which the `BenchmarkRunner` supports), the overhead is ~50% additional.

**Estimated: 80–130 hrs** (depends on whether run inline or separately)

---

### 2.5 Ablation Runs (A1–A4)

| Ablation | What varies | Combinations | Time |
|----------|-------------|--------------|------|
| A1: Token Resolution | CLS vs patch attention (E1, E2) | 2 variants × 6 models × 4 datasets | ~8 hrs |
| A2: Layer Depth | Last layer vs last-3 vs all layers | 3 variants × 6 models × 4 datasets | ~12 hrs |
| A3: Masking Strategy | Zero vs mean vs blur for F1/F2 metrics | 3 variants × 6 models × 4 datasets | ~25 hrs |
| A4: Pre-training Objective | Supervised vs MAE (ViT-B/16 only) | 2 variants × 1 model × 4 datasets | ~10 hrs |
| **Total** | | | **~55–75 hrs** |

> [!NOTE]
> The current ablation scripts ([phase4_ablations.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_ablations.py)) are **scaffolded with mock data** — they generate random numbers and compute Cohen's d on them. The actual GPU-intensive ablation runs will require connecting these to the real `Phase3Runner` with modified configurations.

---

### 2.6 Phase 4 Analysis Scripts (Pure CPU)

The three Phase 4 analysis scripts are **pure CPU analytics** — no GPU required:

| Script | What it does | Runtime |
|--------|-------------|---------|
| `phase4_correlation_analysis.py` | Spearman correlations + heatmap + factor analysis | < 1 min |
| `phase4_interaction_analysis.py` | Kendall tau concordance analysis | < 1 min |
| `phase4_ablations.py` | Cohen's d effect size computation + tables | < 1 min (once real data is plugged in) |

**These scripts require zero GPU time.** They consume the CSV outputs from Phase 3 and produce statistical summaries and plots.

---

### 2.7 Revised Total Estimate — Single A100 (80GB)

| Task | Single A100 Estimate | Original Estimate (multi-GPU) |
|------|---------------------|-------------------------------|
| Fine-tuning (6 models × 4 datasets) | **60–87 hrs** | 120–180 hrs |
| Pilot benchmark (100 samples) | **2–3 hrs** | 2–4 hrs |
| Full benchmark (all combos) | **250–300 hrs** | 250–400 hrs |
| Randomisation tests (R2, R3) | **80–130 hrs** | 80–120 hrs |
| Ablation runs (A1–A4) | **55–75 hrs** | 60–100 hrs |
| Phase 4 analysis (CPU only) | **< 0.1 hrs** | N/A |
| **Total** | **447–595 hrs** | **512–804 hrs** |

---

### 2.8 Wall-Clock Time Translation

| Scenario | GPU-hours | Calendar Time |
|----------|-----------|---------------|
| 24/7 continuous | 447–595 hrs | **18.6–24.8 days** |
| 18 hrs/day (maintenance windows) | 447–595 hrs | **24.8–33.1 days** |
| 12 hrs/day (shared cluster) | 447–595 hrs | **37.3–49.6 days** |

> [!TIP]
> **Optimisation opportunities to reduce total by ~100 hrs:**
> 1. Reduce RISE masks from 4,000 → 2,000 (saves ~50 hrs)
> 2. Run randomisation tests inline with main benchmark via `BenchmarkRunner` (saves ~30–50 hrs of model reload time)
> 3. Use `max_batches` for ablations — A1/A2 only need attention-based explainers, not all 7

---

### 2.9 Memory Considerations (A100 80GB)

| Operation | Peak VRAM | Fits in 80GB? |
|-----------|-----------|---------------|
| Fine-tuning ViT-B/16 at batch=256, AMP | ~28 GB | ✅ Yes |
| Fine-tuning DINO-ViT-B/8 at batch=256, AMP | ~55 GB | ✅ Yes |
| Fine-tuning Swin-B at batch=256, AMP | ~32 GB | ✅ Yes |
| RISE explainer (4,000 masks, batch processing) | ~12 GB | ✅ Yes |
| Full benchmark inference | ~8–15 GB | ✅ Yes |

The A100 80GB has ample memory for all operations. You can likely increase batch sizes beyond 256 during inference-only phases (benchmark/ablations) to speed things up further.

---

## 3. Summary

| Question | Answer |
|----------|--------|
| Can this run on Anaconda? | **Yes** — with ~25 min of config changes |
| Code changes needed? | **None** — only tooling/config files |
| Blocking issues? | Python 3.13 availability in conda (easy to relax to 3.11) |
| Total GPU time (single A100)? | **~447–595 hrs (~19–25 days continuous)** |
| Biggest time sinks? | RISE & LIME explainers (~70% of benchmark time) |
| Phase 4 scripts GPU cost? | **Zero** — pure CPU analytics |
