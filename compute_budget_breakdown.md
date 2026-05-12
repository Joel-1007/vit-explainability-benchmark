# Compute Budget Breakdown — What Each Part Does & Rerun Policy

Every GPU-hour in the 447–595 hr budget falls into one of five stages. This document explains each one: what it computes, why it exists, how long it takes, and whether you lose work if something crashes.

---

## Stage 1: Fine-Tuning (60–87 hrs)

### What it does

Takes 6 pre-trained ViT architectures (ViT-B/16, DeiT-B/16, Swin-B, BEiT-B/16, DINO-ViT-B/8, DINOv2-ViT-B/14) and fine-tunes each one on 4 downstream datasets (CUB-200, ImageNet-S-50, PASCAL VOC, NIH ChestX-ray14). That's **24 independent training runs**, each running 50 epochs with identical hyperparameters (lr=1e-4, batch=256, AdamW, cosine schedule, Mixup augmentation).

### Why it exists

The benchmark needs all models to be trained under the *exact same conditions* so that any differences in explanation quality are attributable to the architecture, not the training recipe. A TPAMI reviewer will reject the paper if models were trained with different hyperparameters. The [trainer.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/training/trainer.py) enforces this standardised protocol.

### Time breakdown

| Model | CUB-200 (~6K imgs) | ImageNet-S-50 (~65K imgs) | PASCAL VOC (~5K imgs) | NIH ChestX-ray (~87K imgs) | **Per-model total** |
|-------|---------------------|---------------------------|----------------------|---------------------------|---------------------|
| ViT-B/16 | 1.5 hrs | 4 hrs | 1 hr | 5 hrs | ~11.5 hrs |
| DeiT-B/16 | 1.5 hrs | 4 hrs | 1 hr | 5 hrs | ~11.5 hrs |
| Swin-B | 2 hrs | 5 hrs | 1.5 hrs | 6 hrs | ~14.5 hrs |
| BEiT-B/16 | 1.5 hrs | 4 hrs | 1 hr | 5 hrs | ~11.5 hrs |
| DINO-ViT-B/8 | 2.5 hrs | 5.5 hrs | 2 hrs | 6 hrs | ~16 hrs |
| DINOv2-ViT-B/14 | 2 hrs | 4.5 hrs | 1.5 hrs | 5.5 hrs | ~13.5 hrs |

The time differences come from architecture — DINO-ViT-B/8 uses patch size 8 (4× more tokens than patch-16 models), making each forward/backward pass slower. Swin-B's windowed attention is slightly heavier than vanilla ViT attention.

### Rerun policy: **Per-epoch checkpointing — very safe**

The trainer saves a `.pth` checkpoint after every single epoch and records its SHA-256 hash. If the GPU crashes at epoch 37/50, you **do not** lose epochs 1–36. However, the current code doesn't have a `--resume` flag built into `pilot_finetune.py` — you'd need to manually load the last checkpoint and set `start_epoch`. This is a standard 5-line change.

Each of the 24 training runs is fully independent, so a failure in one doesn't affect any other. You can restart just the failed run.

> **Risk of needing reruns:** Low. Fine-tuning ViTs is a well-understood procedure and unlikely to fail silently. The pilot check (5-epoch sanity test) catches data pipeline bugs before you commit to the full 50 epochs. Budget an extra ~5% for restarts.

---

## Stage 2: Pilot Benchmark (2–3 hrs)

### What it does

Runs the full metric evaluation pipeline on a **tiny subset** (100 samples) before committing to the expensive full run. It tests every combination: 6 models × 7 explainers × 4 datasets × all metric families (fidelity F1–F3, localization L1–L4, robustness R1–R3, complexity C1–C3). This is essentially a "does everything actually work end-to-end?" check.

### Why it exists

If there's a shape mismatch between a particular explainer and a particular model (e.g., `RawAttentionExplainer` crashing on Swin-B because Swin has no CLS token), you want to discover that in 2 hours, not 300 hours into the full run. The code in [explainers/__init__.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/explainers/__init__.py) already documents known incompatibilities (E1, E2, E4 don't work with Swin-B), but the pilot catches anything unexpected.

### Time breakdown

100 samples is tiny — the time is dominated by the slowest explainers (RISE, LIME) and model loading overhead. Each of the 168 combinations (6×7×4) processes just 100 samples:
- Fast explainers (attention-based): ~1 sec per combination
- Slow explainers (RISE, LIME): ~3-4 min per combination
- Total: ~2–3 hrs

### Rerun policy: **Designed to be rerun**

This is a diagnostic step. You're *expected* to run it, find problems, fix code, and run it again. It has the same per-combination checkpointing as the full benchmark (via `Phase3Runner`), so even within the pilot you can resume from where you left off. Budget 2–3 iterations.

> **Risk of needing reruns:** High — that's the whole point. But each run is only 2–3 hrs so it's cheap. This is where you catch bugs before they cost you days.

---

## Stage 3: Full Benchmark (250–300 hrs)

### What it does

This is the main experiment. For every sample in every dataset, for every model, for every explainer:

1. **Generate an attribution map** — the explainer produces a heatmap showing which parts of the image the model "looked at" to make its prediction
2. **Normalise the map** — clip outliers, scale to [0,1] using the method specified in [normalize.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/metrics/normalize.py)
3. **Compute all metrics on that map** — this is the actual measurement:
   - **Fidelity (F1–F3):** Does masking the "important" regions actually change the model's prediction? (Insertion/deletion curves, sufficiency, comprehensiveness)
   - **Localization (L1–L4):** Does the heatmap overlap with the ground-truth object location? (mIoU, Pointing Game, Energy-based GT coverage, calibration gap)
   - **Robustness (R1–R3):** Is the explanation stable under small perturbations? (Max-sensitivity, model weight randomisation, label randomisation)
   - **Complexity (C1–C3):** Is the explanation concise or noisy? (Gini coefficient, sparsity, effective resolution)
4. **Save results** — checkpoint each (dataset, model, explainer) combination to a `.pkl` file

### Why it exists

This is the entire empirical contribution of the paper. The results matrix (7 explainers × 6 models × 4 datasets × ~12 metrics) is what every table, figure, and conclusion in the TPAMI paper is built from. Without this, there's no paper.

### Time breakdown — by explainer (the dominant cost axis)

The explainer is what determines how long each sample takes. Metrics are computed on the attribution map *after* the explainer runs — that's cheap (~20% overhead on top of explainer time).

| Explainer | What it does per sample | Time/sample | Why it's fast or slow |
|-----------|------------------------|-------------|----------------------|
| **E1: Raw Attention** | Extracts the attention weights from the last transformer layer — no extra computation | ~0.1s | Just reads existing tensors from the forward pass |
| **E2: Attention Rollout** | Multiplies attention matrices across all layers to trace information flow from input to output | ~0.1s | Matrix multiplications, but the matrices are small (num_tokens × num_tokens) |
| **E3: GradCAM** | Computes gradients of the predicted class w.r.t. intermediate feature maps, then weights the feature maps by those gradients | ~0.2s | One backward pass — about the same cost as one training step |
| **E4: Chefer LRP** | Propagates relevance scores backward through every layer using Layer-wise Relevance Propagation rules specific to transformers | ~0.3s | More complex backward pass than GradCAM — custom rules per layer type |
| **E5: RISE** | Generates 4,000 random binary masks, applies each to the input image, runs a forward pass on each masked image, and averages the masks weighted by prediction confidence | **~2.0s** | **4,000 forward passes per sample.** This is the single most expensive operation in the entire benchmark |
| **E6: LIME** | Segments the image into superpixels, randomly perturbs subsets of superpixels (~1,000 perturbations), runs a forward pass on each, fits a linear regression to explain the predictions | **~1.5s** | ~1,000 forward passes + regression fitting |
| **E7: DIME** | Currently a `NotImplementedError` placeholder — excluded | 0s | Not implemented |

**Where the 250–300 hours comes from:**

| Component | Calculation | Hours |
|-----------|-------------|-------|
| E1+E2 (attention-based) on ~35K samples × 6 models | 35K × 0.1s × 2 × 6 = 42K sec | ~12 hrs |
| E3 (GradCAM) on ~35K samples × 6 models | 35K × 0.2s × 6 = 42K sec | ~12 hrs |
| E4 (Chefer LRP) on ~35K samples × 6 models | 35K × 0.3s × 6 = 63K sec | ~18 hrs |
| E5 (RISE) on ~35K samples × 6 models | 35K × 2.0s × 6 = 420K sec | **~117 hrs** |
| E6 (LIME) on ~35K samples × 6 models | 35K × 1.5s × 6 = 315K sec | **~88 hrs** |
| Metric computation overhead (~20%) | | ~49 hrs |
| **Total** | | **~296 hrs** |

The range (250–300) accounts for variance in dataset image sizes, model loading time, and the fact that Swin-B and DINO-ViT-B/8 are slower per forward pass than vanilla ViT-B/16.

### Rerun policy: **Per-combination checkpointing — highly resumable**

The [Phase3Runner](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/metrics/runner.py#L360-L509) saves a `.pkl` file after each of the 168 combinations completes. If the GPU crashes while processing combination #87, combinations #1–86 are fully saved. When you restart, it prints `↩ Skip: cub200-vit_b16-rollout` for every completed combination and resumes from #87.

However, **within** a single combination, there's no mid-combination checkpointing. If a combination has 25,000 samples (e.g., NIH ChestX-ray × RISE) and crashes at sample 20,000, you lose those 20,000 samples and must restart that combination from scratch. The worst-case loss is one combination of RISE on NIH ChestX-ray (~14 hrs).

> **Risk of needing reruns:** Medium. The most likely failures are:
> - OOM on a specific model+explainer combo (fix: reduce batch size for that combo)
> - CUDA errors from long-running processes (fix: just restart — checkpoints are safe)
> - An explainer crashing on a specific model architecture (caught in pilot, but edge cases exist)
>
> Budget an extra ~10–15% for restarts (~30 hrs).

---

## Stage 4: Randomisation Tests (80–130 hrs)

### What it does

Runs two "sanity check" experiments that test whether the explainers are actually explaining the model's behaviour or just producing arbitrary patterns.

**R2 — Model Weight Randomisation:**
1. Take the trained model
2. Progressively randomise its weights (re-initialise layers from the top down)
3. Run each explainer on the *randomised* model
4. Compare the explanation to the explanation from the *real* model using Spearman ρ
5. A good explainer should produce *very different* explanations when the model is randomised. If the explanations look the same, the explainer is not actually capturing learned behaviour — it's just a function of the input image's pixel structure

**R3 — Label Randomisation:**
1. Train a model on shuffled labels (the model memorises random label assignments)
2. Run each explainer on this "junk" model
3. A good explainer should produce different (likely worse) explanations compared to the properly trained model

### Why it exists

This is a **mandatory** sanity check for TPAMI. Adebayo et al. (2018) showed that some popular explanation methods (like vanilla gradient visualisations) pass through randomised model weights essentially unchanged — meaning they're not explaining the model at all, just edge-detecting the input image. Your benchmark *must* include this test, or reviewers will immediately reject.

### Time breakdown

The time cost comes from re-running the explainers on the randomised models. You need the same forward passes as the full benchmark, but:
- You typically only need a subset of samples (~5K–10K, not the full 35K) — statistical significance is reached quickly
- Metric computation is lighter (just computing Spearman ρ between two attribution maps, not the full metric suite)
- You still need to run all 7 explainers × 6 models × 4 datasets for both R2 and R3

| Sub-test | Samples | Explainer cost | Overhead | Total |
|----------|---------|----------------|----------|-------|
| R2 (model randomisation) | ~10K per dataset | Same as full benchmark but on fewer samples | ~40–50% of full benchmark time | ~50–70 hrs |
| R3 (label randomisation) | ~10K per dataset | Same | Requires training on shuffled labels first (~5 hrs extra) | ~35–60 hrs |

### Rerun policy: **Same checkpointing as full benchmark**

The `BenchmarkRunner` supports passing `randomised_model` and `label_randomised_model` as constructor arguments. Each combination is checkpointed identically to Stage 3. If run as a separate pass (recommended), you get the same resume behaviour.

> **Risk of needing reruns:** Low-to-medium. The randomised models are intentionally broken, so occasional numerical instabilities (NaN attention weights, etc.) can occur. These are caught per-combination and don't invalidate other results. Budget ~10% extra.

---

## Stage 5: Ablation Runs (55–75 hrs)

### What it does

Four controlled experiments, each changing **exactly one variable** while keeping everything else fixed. Each ablation answers a specific research question:

**A1 — Token Resolution (~8 hrs):**
- **Question:** Does using CLS-token attention vs. raw patch-token attention change explanation quality?
- **What runs:** E1 and E2 (attention-based explainers) in two configurations × 6 models × 4 datasets
- **What's measured:** Localization metrics (L1–L4). Hypothesis: CLS attention is more class-discriminative but less spatially precise.
- **Why it matters:** Practitioners need to know *which tokens to look at* when extracting attention-based explanations.

**A2 — Layer Depth (~12 hrs):**
- **Question:** Should you use attention from the last layer, the last 3 layers averaged, or all layers (rollout)?
- **What runs:** E2 (rollout) in 3 depth configurations × 6 models × 4 datasets
- **What's measured:** Fidelity (F1–F3) and localization (L1–L4). Does deeper aggregation help?
- **Why it matters:** Every paper that uses attention rollout uses a different number of layers. Your benchmark settles this empirically.

**A3 — Masking Strategy (~25 hrs):**
- **Question:** Do fidelity metrics (insertion/deletion) give different answers depending on how you mask pixels — zeroing them out, replacing with mean pixel value, or Gaussian blurring?
- **What runs:** All explainers × 6 models × 4 datasets × 3 masking strategies
- **What's measured:** Fidelity scores under each masking strategy. If the ranking of explainers changes with the masking strategy, the metric is fragile.
- **Why it matters:** This is a known problem (Hooker et al., 2019) — masked images are out-of-distribution, so the model's response to them may be unreliable. Your ablation quantifies how bad this problem is.
- **This is the most expensive ablation** because it requires recomputing fidelity metrics (which involve forward passes on masked images) under 3 different strategies.

**A4 — Pre-training Objective (~10 hrs):**
- **Question:** Do self-supervised models (MAE) produce better explanations than supervised models?
- **What runs:** ViT-B/16 (supervised) vs. MAE-ViT-B/16 (self-supervised) with identical fine-tuning, all explainers, all datasets
- **What's measured:** All metrics. Hypothesis: MAE pre-training learns richer spatial features → better localization in explanations.
- **Why it matters:** If true, this is a concrete recommendation for practitioners: "Use self-supervised pre-training if you care about explainability."
- **Only 1 model pair** (not 6), so it's cheaper than the others.

### Time breakdown

| Ablation | # of runs | Samples per run | Dominant cost | Total |
|----------|-----------|-----------------|---------------|-------|
| A1 | 2 variants × 2 explainers × 6 models × 4 datasets = 96 | ~35K total | Attention extraction (fast) | ~8 hrs |
| A2 | 3 variants × 1 explainer × 6 models × 4 datasets = 72 | ~35K total | Attention extraction (fast) | ~12 hrs |
| A3 | 3 masking × 7 explainers × 6 models × 4 datasets = 504 | ~5K subset per combo | Fidelity metric forward passes | ~25 hrs |
| A4 | 2 models × 7 explainers × 4 datasets = 56 | ~35K total | RISE/LIME on two models | ~10 hrs |

### Rerun policy: **Not currently checkpointed — but should be**

> [!WARNING]
> The current [phase4_ablations.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_ablations.py) is **scaffolded with mock data** — it generates random numbers, not real results. When you implement the real ablations, you'll need to either:
> 1. Wire them through `Phase3Runner` (recommended — inherits checkpointing for free), or
> 2. Add your own per-combination save logic

If implemented through `Phase3Runner`, you get the same resume behaviour as Stage 3. If implemented as standalone scripts, a crash loses whatever wasn't saved.

Each ablation is independent of the others. A failure in A3 doesn't affect A1, A2, or A4.

> **Risk of needing reruns:** Medium. Ablations are where you discover unexpected behaviour — an explainer might crash under a specific masking strategy, or a depth configuration might produce degenerate attention maps. Budget ~15% extra.

---

## Stage 6: Phase 4 Analysis (≈ 0 GPU-hours)

### What it does

Three Python scripts that consume the CSV/pickle output from Stages 3–5 and produce statistical analyses + paper-ready figures:

1. **[phase4_correlation_analysis.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_correlation_analysis.py):** Computes Spearman ρ between every pair of metrics → heatmap figure + factor analysis (PCA) to find latent structure
2. **[phase4_interaction_analysis.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_interaction_analysis.py):** Computes Kendall τ concordance of explainer rankings across datasets → "are rankings stable across tasks?"
3. **[phase4_ablations.py](file:///c:/Users/Jayan/College/Research/vit-explainability-benchmark/scripts/phase4_ablations.py):** Computes Cohen's d effect sizes for each ablation → "are the differences statistically meaningful?"

### Why it exists

Raw numbers are not findings. These scripts convert the results matrix into the paper's arguments: "Fidelity and localization are orthogonal (ρ = 0.12)", "RISE ranks first on fine-grained datasets but third on medical imaging (τ = 0.34)", "Masking strategy has a large effect on fidelity scores (d = 0.91)".

### Time: < 5 minutes total (CPU only)

These scripts process DataFrames and compute statistics. Zero GPU involved. You can run them on a laptop.

### Rerun policy: **Rerun freely and often**

These are pure analysis scripts — no state, no checkpoints, no side effects. You'll run them dozens of times as you iterate on the analysis, tweak visualisations, and respond to reviewer comments. Each run takes seconds.

---

## Summary: Rerun Risk & Checkpointing

| Stage | Hours | Checkpointed? | Worst-case data loss on crash | Likely reruns | Effective total |
|-------|-------|---------------|-------------------------------|---------------|-----------------|
| Fine-tuning | 60–87 | ✅ Per-epoch | Last partial epoch (~1–6 hrs) | 1–2 failed runs | 65–95 hrs |
| Pilot benchmark | 2–3 | ✅ Per-combination | One combination (~5 min) | 2–3 full reruns (intentional) | 4–9 hrs |
| Full benchmark | 250–300 | ✅ Per-combination | One combination (up to ~14 hrs for RISE×NIH) | ~10% restarts | 275–330 hrs |
| Randomisation tests | 80–130 | ✅ Per-combination | One combination (~4 hrs) | ~10% restarts | 88–143 hrs |
| Ablations | 55–75 | ⚠️ Needs implementation | Entire ablation run if no saves | ~15% restarts | 63–86 hrs |
| Phase 4 analysis | ~0 | N/A (stateless) | Nothing | Run freely | ~0 |
| **Total (with reruns)** | | | | | **495–663 hrs** |

> [!IMPORTANT]
> The raw estimate is 447–595 hrs assuming zero failures. With realistic rerun overhead (~10–15%), budget **~500–660 hrs**, or roughly **21–28 days** of continuous single-A100 time.

### What you should absolutely not rerun unnecessarily

- **RISE on NIH ChestX-ray14** (~14 hrs per model). If this combination completes, protect that `.pkl` file. Back it up.
- **LIME on NIH ChestX-ray14** (~10 hrs per model). Same logic.

### What's cheap to rerun

- Attention-based explainers (E1, E2) — minutes per combination
- Phase 4 analysis scripts — seconds
- Pilot benchmark — 2–3 hrs and you're expected to iterate
