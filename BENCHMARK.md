# ViT Explainability Benchmark — Full Technical Reference

> **Venue target: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**
>
> This document is the single authoritative reference for all phases, formal
> definitions, implementation details, axiomatic analyses, unit tests, and integration
> specifications of the ViT Explainability Benchmark project.

---

## Table of Contents

| §                                                              | Title                             | Status         |
| -------------------------------------------------------------- | --------------------------------- | -------------- |
| [Phase 1](#phase-1--model-zoo--training-protocol)              | Model Zoo & Training Protocol     | ✅ Complete    |
| [1.1](#11--model-zoo-selection)                                | Model Zoo Selection               | ✅             |
| [1.2](#12--standardised-fine-tuning-protocol)                  | Standardised Fine-Tuning Protocol | ✅             |
| [1.3](#13--dataset-registry)                                   | Dataset Registry                  | ✅             |
| [1.4](#14--reproducibility-infrastructure)                     | Reproducibility Infrastructure    | ✅             |
| [Phase 2](#phase-2--metrics-suite)                             | Metrics Suite                     | ✅ Complete    |
| [2.1](#21--fidelity-metrics-f1f3)                              | Fidelity Metrics (F1–F3)          | ✅             |
| [2.2](#22--localization-metrics-l1l4)                          | Localization Metrics (L1–L4)      | ✅             |
| [2.3](#23--robustness-metrics-r1r3)                            | Robustness Metrics (R1–R3)        | ✅             |
| [2.4](#24--complexity-metrics-c1c3)                            | Complexity Metrics (C1–C3)        | ✅             |
| [2.5](#25--axiomatic-analysis)                                 | Axiomatic Analysis                | ✅             |
| [Phase 3](#phase-3--baseline-evaluation-pipeline)              | Baseline Evaluation Pipeline      | ✅ Complete    |
| [3.1](#31--standardised-explainer-interface)                   | Standardised Explainer Interface  | ✅             |
| [3.2](#32--attribution-normalisation)                          | Attribution Normalisation         | ✅             |
| [3.3](#33--benchmarkrunner-phase3runner)                       | BenchmarkRunner / Phase3Runner    | ✅             |
| [3.4](#34--sanity-checks-s1s3)                                 | Sanity Checks (S1–S3)             | ✅             |
| [Phase 4](#phase-4--analysis-ablations--theoretical-grounding) | Analysis & Ablations              | 🔄 In Progress |
| [4.1](#41-inter-metric-correlation)                            | Inter-Metric Correlation          | 🔄             |
| [4.2](#42-task-metric-interaction)                             | Task-Metric Interaction           | 🔄             |
| [4.3](#43-ablation-studies)                                    | Ablation Studies                  | 🔄             |
| [Appendix A](#appendix-a--project-layout)                      | Project Layout                    | —              |
| [Appendix B](#appendix-b--complete-metric-index)               | Complete Metric Index             | —              |
| [Appendix C](#appendix-c--master-checklist)                    | Master Checklist                  | —              |

---

---

# Phase 1 — Model Zoo & Training Protocol

## 1.1 Model Zoo Selection

Six Vision Transformers are selected with controlled variation: architecture family,
pre-training objective, and patch size are varied while **parameter count (~86 M),
input resolution (224×224), and fine-tuning protocol are held constant**.
This isolation ensures that any differences in explanation quality are attributable
to architecture, not training conditions.

| #   | Model               | Architecture     | Pre-training                 | Patch | Params | IN-1K Top-1   | timm ID                                         |
| --- | ------------------- | ---------------- | ---------------------------- | ----- | ------ | ------------- | ----------------------------------------------- |
| 1   | **ViT-B/16**        | Standard ViT     | Supervised IN-21K (AugReg)   | 16    | 86 M   | 84.2 %        | `vit_base_patch16_224.augreg_in21k`             |
| 2   | **DeiT-B/16**       | DeiT (distilled) | Knowledge distillation IN-1K | 16    | 87 M   | 83.4 %        | `deit_base_distilled_patch16_224`               |
| 3   | **Swin-B**          | Shifted-window   | Supervised IN-22K → IN-1K    | 4     | 88 M   | 85.2 %        | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| 4   | **BEiT-B/16**       | BERT-style MIM   | MIM IN-22K (DALL-E dVAE)     | 16    | 86 M   | 85.2 %        | `beit_base_patch16_224.in22k` (pre-train only)  |
| 5   | **DINO-ViT-B/8**    | Standard ViT     | Self-distillation IN-1K      | 8     | 85 M   | 80.1 % (k-NN) | torch.hub `facebookresearch/dino`               |
| 6   | **DINOv2-ViT-B/14** | Standard ViT     | Self-distillation LVD-142M   | 14    | 86 M   | 86.5 %        | `vit_base_patch14_dinov2.lvd142m`               |
| A4  | MAE-ViT-B/16        | Standard ViT     | Masked autoencoder IN-1K     | 16    | 86 M   | —             | `vit_base_patch16_224.mae` (ablation A4 only)   |

IN-1K Top-1 is the published accuracy of each architecture/pre-training, for reference;
the benchmark loads the IDs above (matching `model_zoo/`) and fine-tunes them per dataset.
MAE-ViT-B/16 is not part of the main matrix; it is the supervised-vs-MAE comparison of
ablation A4 (same architecture as ViT-B/16, so pre-training is the only variable).

**Architectural variation axes:**

| Axis            | Values represented                                                    |
| --------------- | --------------------------------------------------------------------- |
| Supervision     | Supervised (ViT, DeiT, Swin), Self-supervised (BEiT MIM, DINO, DINOv2; MAE in A4) |
| Attention scope | Global (ViT, DeiT, BEiT, DINO, DINOv2), Local-window (Swin)           |
| Patch size      | 8 (DINO), 14 (DINOv2), 16 (ViT/DeiT/BEiT), 32 (window-merged in Swin) |
| Special tokens  | CLS only (ViT/BEiT/DINO/DINOv2), CLS+Distil (DeiT), None (Swin)       |

> [!IMPORTANT]
> **Swin-B structural limitation.** Standard attention rollout and CLS-token attribution
> methods do not apply to Swin-B (no CLS token; attention is local-window-based).
> For Swin-B only the gradient- and perturbation-based explainers apply: GradCAM (E3,
> final-stage feature map), RISE (E5), LIME (E6) and Integrated Gradients (E8, optional).
> Raw attention, rollout and Chefer LRP (E1, E2, E4) are skipped and reported as n/a.

---

## 1.2 Standardised Fine-Tuning Protocol

All six models are fine-tuned under an identical protocol. No per-model hyperparameter
tuning is performed — the protocol is fixed once and applied uniformly.

### Hyperparameters

| Component                   | Value                                           | Source                     |
| --------------------------- | ----------------------------------------------- | -------------------------- |
| **Optimiser**               | AdamW                                           | Loshchilov & Hutter (2019) |
| β₁, β₂, ε                   | 0.9, 0.999, 1e-8                                | —                          |
| **Weight decay**            | 0.05 (biases and LayerNorm excluded)            | —                          |
| **Base LR**                 | 1e-4 @ batch 256; scaled linearly               | —                          |
| **LR schedule**             | Cosine annealing + 5-epoch linear warmup        | —                          |
| **Epochs**                  | 50 (CUB-200-2011); 30 (VOC, NIH ChestX-ray14, ImageNet-S-50) | —             |
| **Input size**              | 224 × 224, bicubic interpolation                | —                          |
| **Augmentation**            | RandAugment M=9 N=2, random erasing p=0.25      | Cubuk et al. (2020)        |
| **Mixup**                   | α=0.8; CutMix disabled                          | Zhang et al. (2018)        |
| **Label smoothing**         | ε=0.1                                           | Szegedy et al. (2016)      |
| **Stochastic depth**        | drop_path_rate=0.1                              | Huang et al. (2016)        |
| **Dropout**                 | 0.0 (disabled)                                  | —                          |
| **Loss (CUB/VOC/ImageNet)** | SoftTargetCrossEntropy (Mixup-compatible)       | —                          |
| **Loss (NIH ChestX-ray14)** | Binary CE, class-weighted (multi-label)         | —                          |

### Implementation files

| File                     | Responsibility                             |
| ------------------------ | ------------------------------------------ |
| `training/transforms.py` | RandAugment + random erasing pipeline      |
| `training/mixup.py`      | Batch-level Mixup (α=0.8)                  |
| `training/optimizer.py`  | AdamW + LR warmup + cosine decay           |
| `training/loss.py`       | SoftTargetCrossEntropy / BCE               |
| `training/trainer.py`    | Full fine-tune loop with grad accumulation |

### Pilot sanity check (§7)

```bash
python scripts/pilot_finetune.py \
    --data_root /path/to/CUB_200_2011 \
    --batch_size 64 \
    --accum_steps 4
```

**Required result:** top-1 ≥ 65 % on CUB-200-2011 val after 5 epochs.

---

## 1.3 Dataset Registry

Four benchmark datasets are used. All are resized (shorter side 256) and centre-cropped
to 224×224 for inference; GT masks go through the same geometry with nearest-neighbour
resampling (`utils/eval_datasets.py`).

| Dataset              | Task                                         | Classes | Fine-tune train | **Evaluation set (with GT)**                     | GT type used                    | Config                       |
| -------------------- | -------------------------------------------- | ------- | --------------- | ------------------------------------------------ | ------------------------------- | ---------------------------- |
| **CUB-200-2011**     | Fine-grained bird classification             | 200     | 5,994           | 5,794 (official test split)                      | Pixel segmentation masks        | `configs/cub200.yaml`        |
| **PASCAL VOC 2012**  | Object classification (dominant class)       | 20      | 10,582 (SBD aug.) | 1,449 (segmentation `val`)                     | Pixel segmentation masks        | `configs/pascal_voc.yaml`    |
| **ImageNet-S-50**    | 50-class subset of ImageNet-1K               | 50      | ImageNet-1K images of the 50 classes | 752 (`validation`, as checked by `verify_datasets.py`) | Pixel segmentation masks | `configs/imagenet_s50.yaml`  |
| **NIH ChestX-ray14** | Multi-label chest pathology (14 findings)    | 14      | 86,524 (train/val list) | **984 boxes on ~880 images** (`BBox_List_2017.csv`) | Bounding boxes, 8 findings | `configs/nih_chestxray.yaml` |

> [!IMPORTANT]
> **Localization on NIH runs on a small subset.** Only ~880 of the 112,120 NIH images
> have expert boxes, covering 8 of the 14 findings. All NIH localization metrics are
> computed on that subset (one sample per image–finding pair), and per-finding counts
> are small, so report per-finding results with their confidence intervals.
> The fourth dataset is NIH ChestX-ray14 throughout; CheXpert/CheXlocalize is only
> discussed as an alternative in `docs/dataset_sources.md`.

> **VOC labels.** VOC images often contain several objects; each image is labelled with
> its dominant class (most segmented pixels) and the GT mask covers that class only.

### Dataset verification protocol

```bash
python scripts/verify_datasets.py \
    --cub /data/cub200 --voc /data/voc2012 \
    --imagenet_s /data/imagenet_s50 --nih /data/nih_chestxray
```

All four datasets must pass: file-count checks, annotation bounds checks, class-balance
checks, and stratified validation split creation.

---

## 1.4 Reproducibility Infrastructure

| Asset                                | Description                              | Location     |
| ------------------------------------ | ---------------------------------------- | ------------ |
| `model_hashes.txt`                   | SHA-256 of all 6 pre-trained checkpoints | project root |
| `checkpoints/*/finetuned_hashes.txt` | Per-epoch SHA-256 of fine-tuned weights  | per dataset  |
| `requirements.txt`                   | Pinned dependency versions               | project root |
| `configs/*.yaml`                     | Full hyperparameter specs                | `configs/`   |

```bash
# Record pre-trained hashes once after download
python scripts/record_model_hashes.py --out model_hashes.txt
```

**Checkpoints**: large binaries are `.gitignore`d; only hash logs are committed.

**Environment.** Python ≥ 3.13, locked by `uv.lock` (PyTorch 2.15.0, timm 1.0.31 at the
time of writing); `requirements.txt` gives minimum versions only.
Current test run (2026-11-19, CPU, Python 3.13.8 / PyTorch 2.15.0):
**300 passed, 2 skipped (CUDA), 0 failed** (`uv run pytest`). Result blocks below dated
2026-04-10 were recorded under Python 3.11.8 / PyTorch 2.10.0 and are kept as history.

**Seeds and uncertainty.** Each (model, dataset) pair currently has a single fine-tuned
checkpoint, so architecture effects are confounded with fine-tuning-seed variance.
`combination_summary.csv` reports 95% bootstrap CIs over samples; seed variance needs
several fine-tuning seeds per pair (planned).

---

---

# Phase 2 — Metrics Suite

The metrics suite evaluates attribution maps $\phi(f, x) \in \mathbb{R}^{H_a \times W_a}$
produced by an explanation method $\phi$ for model $f$ at input $x$.
Thirteen metrics are implemented across four complementary families:

| Family                  | Metrics                                           | Question answered                                          |
| ----------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| **Fidelity** (§2.1)     | F1 Sufficiency, F2 Comprehensiveness, F3 Log-odds | Does the attribution reflect what the model actually uses? |
| **Localization** (§2.2) | L1 mIoU, L2 PG, L3 EGT, L4 CalibGap               | Does the attribution point at the right region?            |
| **Robustness** (§2.3)   | R1 MaxSens, R2 ModelRand, R3 HeadPerm             | Is the attribution stable and model-sensitive?             |
| **Complexity** (§2.4)   | C1 Gini, C2 Entropy, C3 EMR                       | Is the attribution concise?                                |

All metrics reside in `metrics/`. `Phase3Runner` (with `metrics/sample_metrics.py`)
computes all four families per sample; L4 is computed per combination from
per-sample EGT and prediction correctness (`metrics/aggregate.py`).

Direction summary (used for rankings): higher is better for F2, F3, L1–L4, R2, R3, C1;
lower is better for F1, R1, C2, C3.

---

## 2.1 Fidelity Metrics (F1–F3)

> [!NOTE]
> Fidelity metrics quantify whether the attribution reflects what inputs the model
> **actually relies on** for its prediction. They operate by masking or removing
> high-attribution regions and measuring the resulting change in model output.

### F1 — Sufficiency

$$F1 = \text{Suff}(\phi, f, x) = f(x)_{y} - f(\tilde{x}_{\text{keep}})_{y}$$

where $\tilde{x}_{\text{keep}}$ keeps the top-$k$ attributed patches and replaces the rest
with the baseline, and $y$ is the explained class (the ground-truth label in the
benchmark). The attributed region alone should be _sufficient_ to reproduce the
model's confidence. **Lower is better** (ERASER convention, DeYoung et al., 2020):
a small drop means the kept region suffices; negative values mean the kept region
alone is *more* convincing than the full image.

### F2 — Comprehensiveness

$$F2 = \text{Comp}(\phi, f, x) = f(x)_{y} - f(\tilde{x}_{\text{drop}})_{y}$$

where $\tilde{x}_{\text{drop}}$ replaces the top-$k$ attributed patches with the baseline.
Removing the attributed region should _reduce_ confidence. **Higher is better.**

### F3 — Log-odds Drop

$$F3 = \text{LogOddsDrop}(\phi, f, x) = \log \frac{f(x)_{y^*}}{1 - f(x)_{y^*}} - \log \frac{f(x')_{y^*}}{1 - f(x')_{y^*}}$$

where $x' = \tilde{x}_{\text{drop}}$. Measures the log-odds change of the explained class
after removing the top-$k$ attributed patches. **Higher is better.**

**Masking baseline.** The default replaces removed patches with the per-channel image
mean (`mask_mode="mean"`); zero and Gaussian-blur baselines are compared in ablation
A3. Pixel-space masking produces inputs a ViT never saw during training (out of
distribution); token removal would avoid this and is noted as a limitation.
All three metrics are evaluated at $k \in \{0.1, 0.2, 0.5\}$ (fraction of patches).

---

### 2.1.1 Implementation — `FidelityMetrics`

```python
from metrics.fidelity import FidelityMetrics

fm = FidelityMetrics(mask_mode="mean", k_fractions=(0.1, 0.2, 0.5))   # or "zero" / "blur"

# All three metrics in one call over fractions
scores = fm.compute_all(model, images, targets, att_maps)
# -> {'sufficiency@0.10': tensor([...]), 'comprehensiveness@0.10': tensor([...]), ...}

# Or individually
f1 = fm.sufficiency(model, images, targets, att_maps, k_frac=0.2)
f2 = fm.comprehensiveness(model, images, targets, att_maps, k_frac=0.2)
f3 = fm.log_odds_drop(model, images, targets, att_maps, k_frac=0.2)
```

### 2.1.2 Unit Tests (F1–F3)

File: `tests/test_fidelity.py` — **5 tests**, all passing.

| ID  | Test                      | Assertion                                                  |
| --- | ------------------------- | ---------------------------------------------------------- |
| F01 | `test_generate_mask`      | Masks top `k_frac` elements correctly.                     |
| F02 | `test_apply_mask_zero`    | Selected regions retain content, others zeroed.            |
| F03 | `test_apply_mask_mean`    | Drops selected regions, interpolates with baseline mean.   |
| F04 | `test_metrics_standalone` | Individual metrics execution shapes are correct.           |
| F05 | `test_compute_all`        | Computes F1, F2, F3 recursively over configured fractions. |

```
Results: 5/5 passed, 0 failed
```

---

## 2.2 Localization Metrics (L1–L4)

All four metrics operate on a pair $(e, M^{GT})$ where $e \in \mathbb{R}^{H \times W}$
is the raw attribution map and $M^{GT}$ is the binary ground-truth mask
(bounding box or pixel segmentation).

> [!NOTE]
> Attribution maps at patch resolution (e.g., 14×14 for ViT-B/16) are upsampled to
> the GT mask resolution before metric computation: bilinearly for L1/L2, and by
> nearest neighbour for L3 (each patch's attribution covers exactly its own pixels).
> GT pixels marked ignore (255, e.g. VOC object boundaries) are excluded.
> See the ViT patch grid reference table in §2.2.4.

### L1 — Multi-Threshold IoU (mIoU)

**Notation.** Let $\tilde{e}_p = (e_p - \min_q e_q) / (\max_q e_q - \min_q e_q + \varepsilon)$
be the min-max normalised attribution. For threshold $\tau$, define the predicted binary mask:

$$\hat{M}_\tau = \{p : \tilde{e}_p \geq \tau\}$$

The Intersection-over-Union at threshold $\tau$:

$$\text{IoU}(\hat{M}_\tau, M^{GT}) = \frac{|\hat{M}_\tau \cap M^{GT}|}{|\hat{M}_\tau \cup M^{GT}|}$$

The multi-threshold mIoU averaged over $\tau \in \{0.25, 0.50, 0.75\}$:

$$L1 = \text{mIoU} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{IoU}(\hat{M}_\tau, M^{GT})$$

**Range:** $[0, 1]$. Lower thresholds ($\tau=0.25$) are permissive; higher ($\tau=0.75$)
are conservative. Reporting all three prevents cherry-picking.

---

### L2 — Pointing Game (PG)

Let $p^* = \arg\max_p e_p$ be the pixel with maximum raw attribution.

$$L2 = \text{PG} = \mathbf{1}\!\left[p^* \in M^{GT}\right]$$

**Range:** $\{0, 1\}$ per sample; macro-average is the dataset-level hit rate.

**Tie-breaking.** Multiple pixels at the maximum → draw uniformly at random (seed=42).
Prevents degenerate implementations from always picking $(0,0)$.

**Dataset-level:**

$$\overline{\text{PG}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[p^*_i \in M^{GT}_i\right]$$

---

### L3 — Energy-Ground-Truth (EGT)

Energy-based pointing game (Wang et al., 2020). With $e^+_p = \max(e_p, 0)$:

$$L3 = \text{EGT} = \frac{\sum_{p \in M^{GT}} e^+_p}{\sum_{p} e^+_p}$$

the fraction of the map's positive attribution mass that lies inside the GT region.

**Range:** $[0, 1]$. EGT = 1 iff no positive mass lies outside the GT; EGT = 0 iff none
lies inside. **Scale-invariant:** $\text{EGT}(c\,e) = \text{EGT}(e)$ for $c > 0$, so
explainers with different raw scales are comparable. Negative attribution (evidence
*against* the class) carries no energy. If $\sum_p e^+_p = 0$, EGT is defined as the GT
area fraction $|M^{GT}|/|M|$, the value of an uninformative uniform map.

> **Changed from softmax.** An earlier version used $\tilde{e} = \mathrm{softmax}(e)$.
> Softmax is strictly positive, so EGT could never reach 1; on $[0,1]$-normalised maps
> it is nearly flat (a perfect 14×14 map covering half the image scored 0.72, a uniform
> map 0.50), and on raw maps it depends on each method's scale.

---

### L4 — Calibration Gap (CalibGap)

Let $\mathcal{C}$ and $\mathcal{I}$ be correctly and incorrectly classified samples:

$$L4 = \text{CalibGap} = \mathbb{E}_{i \in \mathcal{C}}\!\left[\text{EGT}_i\right] - \mathbb{E}_{i \in \mathcal{I}}\!\left[\text{EGT}_i\right]$$

**Range:** $(-1, 1)$. Positive = explanation aligns better with GT on correct predictions
(necessary condition for faithfulness). Near-zero or negative = sanity-check failure.

**Undefined** when all predictions correct ($\mathcal{I} = \emptyset$) or all incorrect
($\mathcal{C} = \emptyset$) — raises `ValueError`.

---

### 2.2.1 Implementation — `LocalizationMetrics`

```python
from metrics.localization import LocalizationMetrics

lm = LocalizationMetrics(
    thresholds=[0.25, 0.50, 0.75],   # L1 binarisation thresholds
    seed=42,                          # tie-breaking seed for L2
)

# L1 — mIoU
result = lm.multi_threshold_iou(att_map, gt_mask)
# → {'iou@0.25': 0.74, 'iou@0.50': 0.61, 'iou@0.75': 0.42, 'miou': 0.59}

# L2 — Pointing Game
pg = lm.pointing_game(att_map, gt_mask)      # → 1.0 or 0.0

# L3 — EGT
e = lm.egt(att_map, gt_mask)                 # → float in [0, 1]

# L4 — CalibGap (dataset-level, requires partitioned lists)
gap = lm.calibration_gap(atts_correct, atts_incorrect, gt_masks=all_masks)

# All three per-sample metrics in one call (L4 excluded)
scores = lm.compute_all(att_map, gt_mask)
```

---

### 2.2.2 Axiomatic Analysis (L1–L4)

#### Proposition 2.2.A — IoU is argument-symmetric but penalises over-segmentation

**Argument symmetry.** $\text{IoU}(\hat{M}, M^{GT}) = \text{IoU}(M^{GT}, \hat{M})$ always,
since set intersection and union are commutative. (This is symmetry *in the metric's
arguments*; it is unrelated to the Shapley Symmetry axiom A3 of §2.5.)

**Area-penalty corollary.** Despite argument symmetry, IoU penalises over-segmentation:
a large predicted mask that contains the GT region gives low IoU because the union is large.

$$\text{IoU}(\hat{M}_\text{large}, M^{GT}_\text{small}) = \frac{|M^{GT}_\text{small}|}{|\hat{M}_\text{large}|} \ll 1$$

**Implication.** An explanation that highlights the entire image is correctly penalised,
even if it technically contains the GT region. IoU must be reported at all three thresholds.

---

#### Theorem 2.2.B — EGT satisfies Monotone Coverage but NOT Boundary Sensitivity

**Monotone Coverage.** If $M^{GT}_1 \subseteq M^{GT}_2$ with $e$ fixed:

$$\text{EGT}(e, M^{GT}_1) \leq \text{EGT}(e, M^{GT}_2)$$

_Proof._ The denominator $\sum_p e^+_p$ does not depend on the mask, and every
$e^+_p \geq 0$, so enlarging $M^{GT}$ can only increase the numerator. $\square$

**Boundary Sensitivity violation.** EGT is a sum over $M^{GT}$ — it cannot distinguish mass
concentrated at the GT boundary vs. the GT semantic centre. Pointing Game (L2) is
complementary: it penalises boundary-concentrated peaks.

**Implication.** L2 and L3 are not redundant — both are needed.

---

#### Proposition 2.2.C — CalibGap is gameable

CalibGap is a difference of conditional means, not a scoring rule, so it carries no
incentive-compatibility guarantee.

**Claim.** CalibGap can be gamed by an adversarial explainer with access to the
prediction correctness flag.

_Proof by construction._ If $\phi$ outputs high-EGT maps for correct predictions
and low-EGT maps for incorrect ones, CalibGap → 1 without faithfulness. $\square$

**Mitigation.** The benchmark:

1. Passes only `(model, image)` to explainers — never correctness flags.
2. Reports L1–L3 alongside L4 so inflated CalibGap is detectable.
3. Requires consistent CalibGap across all four datasets.

**Implication.** CalibGap is a _diagnostic_ metric, never a primary ranking criterion.

---

### 2.2.3 Unit Tests (L1–L4)

File: `tests/test_localization.py` — **15 tests**, all passing (collected by pytest
through `test_registered`; earlier the T-functions were not collected at all).

| ID  | Test                         | Assertion                                    |
| --- | ---------------------------- | -------------------------------------------- |
| T01 | `random_iou_not_one`         | mIoU < 1.0 for 20 random trials              |
| T02 | `random_pg_within_bounds`    | PG ∈ {0,1}; not always 1.0 over 100 trials   |
| T03 | `random_egt_within_bounds`   | EGT ∈ [0,1] for 50 random pairs              |
| T04 | `perfect_iou_is_one`         | att=GT → mIoU = 1.0 at all τ                 |
| T05 | `perfect_pg_is_one`          | peak at GT pixel → PG = 1.0                  |
| T06 | `perfect_egt_is_one`         | all mass inside GT → EGT = 1.0               |
| T07 | `perfect_calibgap_positive`  | perfect-correct vs wrong-incorrect → gap > 0 |
| T08 | `misaligned_iou_is_zero`     | att outside GT → IoU@0.75 = 0.0              |
| T09 | `misaligned_pg_is_zero`      | peak outside GT → PG = 0.0                   |
| T10 | `misaligned_egt_near_zero`   | all mass outside GT → EGT < 0.01             |
| T11 | `constant_att_map_iou`       | constant map → no NaN; result in [0,1]       |
| T12 | `calibgap_empty_list_raises` | empty correct/incorrect → ValueError         |
| T13 | `egt_reaches_one_on_unit_scale_maps` | [0,1] map fully inside GT → EGT = 1 |
| T14 | `egt_scale_invariant`        | EGT(c·e) = EGT(e) for c > 0                  |
| T15 | `egt_degenerate_and_ignore`  | no positive mass → GT area; ignore excluded  |

```
Results: 15/15 passed, 0 failed
```

---

### 2.2.4 ViT Patch Grid Reference

| Model                          | Patch size | Grid          | Patches | Upsample to 224×224 |
| ------------------------------ | ---------- | ------------- | ------- | ------------------- |
| ViT-B/16, DeiT-B/16, BEiT-B/16, MAE-B/16 | 16 | 14×14       | 196     | ×16                 |
| DINOv2-ViT-B/14                | 14         | 16×16         | 256     | ×14                 |
| DINO-ViT-B/8                   | 8          | 28×28         | 784     | ×8                  |
| Swin-B                         | 16 (grid)  | 14×14         | 196     | ×16                 |

Swin-B's GradCAM map is native 7×7 (stage 4) and is bilinearly resampled to the
14×14 grid; RISE, LIME and IG are computed directly on the 14×14 grid.

---

## 2.3 Robustness Metrics (R1–R3)

Robustness metrics test whether an explanation is **stable and faithful** under
systematic perturbations to the input or the model. A faithful attribution must:

- Change **when the input moves** toward a worst-case perturbation (R1).
- Change **when the model weights are randomised** — sanity check (R2).
- Change **when the classifier head is permuted** — dependence on the class mapping (R3).

> [!NOTE]
> R1–R3 complement L1–L4, which test spatial faithfulness to GT.
> Robustness metrics test faithfulness to the _model's internal computation_.

---

### R1 — Max-Sensitivity (MaxSens)

**Reference**: Yeh et al. (2019), _On the (In)fidelity and Sensitivity of Explanations_.

Draw $K$ random perturbations $\delta_k \sim \text{Uniform}(-\epsilon, +\epsilon)^d$
from the $\ell_\infty$ ball of radius $\epsilon$ around $x$:

$$R1 = \text{MaxSens}(\phi, x) = \max_{k=1,\ldots,K} \frac{\|\phi(f, x+\delta_k) - \phi(f, x)\|_2}{\|\phi(f, x)\|_2 + \varepsilon_\text{num}}$$

where $\varepsilon_\text{num} = 10^{-8}$.

**Default:** $K=20$ (unit tests/fast evaluation). Use $K=50$ for production benchmark
runs per the Yeh et al. (2019) specification.

**Range:** $[0, \infty)$. Lower = more stable attribution. Zero for a fully deterministic
constant explainer.

---

### R2 — Model Randomisation (ModelRand)

**Reference**: Adebayo et al. (2018), _Sanity Checks for Saliency Maps_.

Let $f_\text{rand}$ be a deep copy of $f$ with every layer re-initialised to an untrained
state: each module's standard initialisation (PyTorch `reset_parameters()`; LayerNorm
back to scale 1, shift 0), and $\mathcal{N}(0, 0.02^2)$ for parameters outside such
modules (`cls_token`, `pos_embed`, LayerScale). An earlier version drew *every*
parameter, LayerNorm included, from $\mathcal{N}(0,1)$, which can overflow through 12
transformer blocks and yield NaN maps instead of a random-model baseline.
Let $\tilde{\phi}(\cdot) = \text{MinMax}(\phi(\cdot))$ be attribution normalised to $[0,1]$.

SSIM (Wang et al., 2004) between the two normalised maps:

$$\text{SSIM}(A, B) = \frac{(2\mu_A\mu_B + C_1)(2\sigma_{AB} + C_2)}{(\mu_A^2 + \mu_B^2 + C_1)(\sigma_A^2 + \sigma_B^2 + C_2)}$$

$$R2 = \text{ModelRand}(\phi, x) = 1 - \text{SSIM}\!\left(\tilde{\phi}(f_\text{orig}, x),\; \tilde{\phi}(f_\text{rand}, x)\right)$$

**Range:** $[0, 1]$. Higher = explanation is model-sensitive (desired). Near 0 = sanity-check
failure (attribution ignores the model weights entirely).

**SSIM implementation:** Pure-PyTorch, 3×3 Gaussian window ($\sigma=1.0$), GPU-native.
No `scikit-image` dependency.

---

### R3 — Head Permutation (HeadPerm)

Let $f_\text{shuf}$ be a deep copy of $f$ with the final classifier head weights
**column-permuted** (feature-to-class mapping shuffled, backbone unchanged, no retraining).

> **Not Adebayo et al.'s label randomisation.** Their data-randomisation test *retrains*
> the model on shuffled labels. HeadPerm only permutes the trained head, so it tests
> whether the explanation depends on the head's class mapping, at no training cost.
> It was previously called "Label Randomisation"; `label_randomisation` and
> `randomise_classifier_labels` remain as aliases in the code.

Let $\rho(\cdot, \cdot)$ denote the Spearman rank correlation of two flattened maps:

$$R3 = \text{HeadPerm}(\phi, x) = 1 - \frac{|\rho(\phi_\text{orig}, \phi_\text{shuf})| + 1}{2}$$

**Derivation:**

- $\rho \in [-1, 1]$; $(|\rho|+1)/2 \in [0.5, 1]$ measures structural preservation.
- $1 - (|\rho|+1)/2 \in [0, 0.5]$ measures structural divergence.
- $|\rho|$ (not $\rho$) is used because sign-flipped attributions still preserve spatial structure.

**Range:** $[0, 0.5]$. Higher = explanation is sensitive to the head. Near 0 = failure
(explanation unchanged despite the permutation). Class-agnostic explainers (raw
attention, rollout) are expected to score ≈ 0 by construction.

**Spearman implementation:** Pure-PyTorch rank-via-argsort, stable, handles ties,
no SciPy dependency.

---

### 2.3.1 Implementation — `RobustnessMetrics` and Utilities

```python
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    permute_classifier_head,
)

rm = RobustnessMetrics(
    epsilon=0.05,    # L∞ perturbation radius (R1)
    n_samples=20,    # perturbation draws per image (use 50 for production)
    seed=42,         # RNG seed for reproducibility
    ssim_window=3,   # Gaussian kernel size for SSIM (R2)
    ssim_sigma=1.0,  # Gaussian sigma for SSIM (R2)
)

# Pre-compute randomised model copies once — NOT per sample
f_rand = randomise_model_weights(model, seed=0)        # every layer re-initialised
f_shuf = permute_classifier_head(model, seed=0)        # head columns permuted only

# Per sample
att_orig = explainer(model,  image.unsqueeze(0))[0]
att_rand = explainer(f_rand, image.unsqueeze(0))[0]
att_shuf = explainer(f_shuf, image.unsqueeze(0))[0]

# R1
ms = rm.max_sensitivity(explainer, model, image, att_orig)

# R2
mr = rm.model_randomisation(att_orig, att_rand)

# R3
hp = rm.head_permutation(att_orig, att_shuf)

# All three at once
scores = rm.compute_all(explainer, model, image, att_orig, att_rand, att_shuf)
# → {'max_sensitivity': 0.31, 'model_randomisation': 0.87, 'head_permutation': 0.46}
```

**`randomise_model_weights(model, seed=0)`**
— Deep copy; every layer re-initialised with its standard initialisation (see R2).

**`permute_classifier_head(model, seed=0, head_attr='head')`** (alias `randomise_classifier_labels`)
— Deep copy; only the head's weight columns are permuted.
Column permutation of $W \in \mathbb{R}^{C \times D}$ reassigns which class each feature
maps to while preserving weight norms (stricter than zeroing the head).
Fallback chain for `head_attr`: `'head'`, `'classifier'`, `'fc'`, `'linear'`, each also
looked up as `<name>.fc` and under `backbone.` (model-zoo wrappers, Swin's `head.fc`).

---

### 2.3.2 Axiomatic Analysis (R1–R3)

#### Theorem 2.3.A — MaxSens satisfies the Lipschitz-Sensitivity Axiom

**Lipschitz-Sensitivity Axiom.** $\exists L < \infty$ such that $\text{MaxSens}(\phi, x) \leq L \cdot \epsilon$.

_Proof._ For a Lipschitz explainer with constant $\Lambda$:

$$\|\phi(f, x+\delta) - \phi(f, x)\|_2 \leq \Lambda\|\delta\|_2 \leq \Lambda\epsilon\sqrt{d}$$

Setting $L = \Lambda\sqrt{d} / (\|\phi(f,x)\|_2 + \varepsilon_\text{num})$ completes the bound. $\square$

**Corollary.** MaxSens is linear in $\epsilon$ for smooth explainers. Always report $\epsilon$
alongside MaxSens scores — cross-$\epsilon$ comparisons are meaningless.

---

#### Theorem 2.3.B — ModelRand satisfies the Sanity Check Axiom

**Sanity Check Axiom** (Adebayo et al., 2018). A faithful attribution $\phi$ must produce
structurally different maps when model weights are fully randomised:

$$\lim_{\text{randomisation} \to 1} \text{ModelRand}(\phi, x) = 1$$

_Proof._ For fully randomised $f_\text{rand}$, attribution maps approach i.i.d. uniform
noise (by CLT on random projections). The SSIM between a learned spatial map and uniform
noise → 0, so ModelRand → 1. $\square$

**Counterexample (sanity-check failure).**

```python
def pixel_explainer(model, image_4d):
    return [image_4d[0, 0].detach()]   # Uses only input, ignores model!

# att_orig == att_rand → ModelRand = 0 → FAIL
```

---

#### Remark 2.3.C — HeadPerm is a diagnostic, not a distance

While the Spearman correlation $\rho$ is numerically symmetric, HeadPerm is
**semantically asymmetric**: the roles of $\phi_\text{orig}$ (trusted reference) and
$\phi_\text{shuf}$ (perturbed) cannot be swapped without changing the interpretation.

**Implication.** Always specify which model is "original" vs. "permuted" when reporting
HeadPerm. Do not compare across architectures with different head sizes (different
permutation degrees). Report alongside R1 and R2, never in isolation.

---

### 2.3.3 Unit Tests (R1–R3)

File: `tests/test_robustness.py` — **19 tests**, all passing (collected by pytest
through `test_registered`).

| ID  | Category    | Test                              | Assertion                                   |
| --- | ----------- | --------------------------------- | ------------------------------------------- |
| R01 | Bounds      | `max_sensitivity_nonneg`          | MaxSens ≥ 0 for 20 trials                   |
| R02 | Bounds      | `model_randomisation_in_0_1`      | ModelRand ∈ [0,1] for 20 trials             |
| R03 | Bounds      | `label_randomisation_in_0_0p5`    | HeadPerm ∈ [0, 0.5] for 20 trials           |
| R04 | Bounds      | `compute_all_keys`                | compute_all() returns exactly 3 keys        |
| R05 | Sensitivity | `sensitivity_increases_with_eps`  | Larger ε → ≥ mean MaxSens (30 trials)       |
| R06 | Sensitivity | `model_rand_orthogonal_maps`      | ModelRand > 0.5 for top-half vs bottom-half |
| R07 | Sensitivity | `label_rand_orthogonal_maps`      | Mean HeadPerm ≈ 0.5 for 200 random pairs    |
| R08 | Zero        | `sensitivity_zero_constant_expl`  | MaxSens = 0 for constant explainer          |
| R09 | Zero        | `model_rand_identical_maps`       | ModelRand = 0 for identical maps            |
| R10 | Zero        | `label_rand_identical_maps`       | HeadPerm = 0 for identical maps             |
| R11 | Utility     | `randomise_model_changes_weights` | Every Linear parameter re-drawn             |
| R12 | Utility     | `randomise_labels_only_head`      | Backbone unchanged; head changed            |
| R13 | Utility     | `ssim_self_consistency`           | \_ssim(t, t) = 1.0 for 10 trials            |
| R14 | Edge case   | `spearman_constant_map`           | Constant map → ρ = 0, no NaN                |
| R15 | Edge case   | `max_sensitivity_n_samples_one`   | n_samples = 1 works                         |
| R16 | Edge case   | `epsilon_constructor_validation`  | ε ≤ 0 raises ValueError                     |
| R17 | Layer curve | `spearman_layer_curve_keys`       | One key per block, last block first         |
| R18 | Layer curve | `randomise_model_cascade_depth`   | Only the last n blocks change               |
| R19 | Layer curve | `layer_curve_rho_bounds`          | Every ρ ∈ [-1, 1]                           |

```
Results: 19/19 passed, 0 failed
```

Run all tests together:

```bash
uv run pytest tests/ -v
```

---

### 2.3.4 Performance Guidance

| Metric       | Cost per sample        | Recommendation          |
| ------------ | ---------------------- | ----------------------- |
| L1–L3        | O(1) tensor ops        | Full dataset            |
| L4 CalibGap  | O(N) at dataset end    | Full dataset            |
| R1 MaxSens   | K explainer calls      | Fixed subset (`--robustness-subset`, default 100) |
| R2 ModelRand | 1 extra explainer call | Same subset as R1       |
| R3 HeadPerm  | 1 extra explainer call | Same subset as R1       |

> [!IMPORTANT]
> MaxSens costs $K$ extra explainer calls per image (≈ 4,000 masks each for RISE).
> `run_phase3.py` therefore computes R1–R3 on the first `--robustness-subset` images of
> each combination (default 100; the evaluation subsets are spread across classes).
> Report the subset size in the table footnote (`combination_summary.csv` has
> `n_max_sensitivity` etc.).

---

## 2.4 Complexity Metrics (C1–C3)

> [!NOTE]
> Complexity metrics measure the **parsimony** of an explanation: a good
> explanation should identify a _compact_ set of decisive patches. Diffuse
> attributions are harder to interpret and weaker evidence of understanding.

Three complementary metrics measure different facets of concentration:

| ID  | Name                 | Formula (short)   | Range  | Higher =        |
| --- | -------------------- | ----------------- | ------ | --------------- |
| C1  | Gini coefficient     | See §2.4.1        | [0, 1] | Sparser         |
| C2  | Attribution entropy  | H(e) / log N      | [0, 1] | **Less** sparse |
| C3  | Effective mass ratio | k\* / N at θ=0.90 | [0, 1] | **Less** sparse |

> [!IMPORTANT]
> C1 is **higher-is-better**; C2 and C3 are **lower-is-better**. Report all
> three alongside a uniform-random baseline to give reviewers a calibration
> reference.

---

### 2.4.1 C1 — Gini Coefficient

$$C1 = \text{Gini}(e) = \frac{2 \sum_{i=1}^{N} i \cdot e_{(i)}}{N \sum_{i=1}^{N} e_{(i)}} - \frac{N+1}{N}$$

where $e_{(1)} \leq \ldots \leq e_{(N)}$ is the **ascending**-sorted map.

**Properties.**

- _Normalisation:_ minmax before computation (scale-invariant by construction).
- _Zero-map convention:_ returns 0 (uniform mass convention).
- _One-hot exact maximum:_ Gini = $(N-1)/N \to 1$ as $N \to \infty$.

---

### 2.4.2 C2 — Attribution Entropy

$$C2 = H_{\text{norm}}(e) = -\frac{1}{\log N} \sum_{p=1}^{N} \tilde{e}_p \log \tilde{e}_p$$

where $\tilde{e}_p = |e_p| / \sum_q |e_q|$ is the map's own mass distribution.

**Properties.**

- _Normalisation:_ sum of absolute values (signed maps are handled by $|\cdot|$).
  An earlier version used $\mathrm{softmax}(e)$; on $[0,1]$ maps softmax values lie in
  $[1, e]$ before normalisation, so the distribution is nearly flat and C2 ≈ 1 for any
  map (a map with all mass on 10 of 196 patches scored ≈ 0.99). It also disagreed with
  the standalone `attribution_entropy()` and with the T6 proof, which both use this
  sum normalisation.
- _Zero-map convention:_ entropy = log N, norm = 1 (flags uninformative maps).
- _Contrast with Gini:_ lower entropy $\Leftrightarrow$ higher Gini for the same map.

---

### 2.4.3 C3 — Effective Mass Ratio (EMR)

Let $k^*(\theta)$ be the minimum number of patches (sorted by attribution, descending)
required to capture $\theta$ fraction of total attribution mass:

$$k^*(\theta) = \min\left\{ k : \sum_{i=1}^{k} e_{(N+1-i)} \geq \theta \cdot \sum_{i=1}^{N} e_i \right\}$$

$$C3 = \text{EMR}(e, \theta) = \frac{k^*(\theta)}{N}$$

**Cross-architecture reporting.** Always report $k^*_{90}$ (absolute patch count)
alongside fractional EMR for cross-architecture comparability:

| Model           | N   | EMR(0.90)=0.05 means… |
| --------------- | --- | --------------------- |
| ViT-B/16        | 196 | 10 patches            |
| DINO-ViT-B/8    | 784 | 39 patches            |
| DINOv2-ViT-B/14 | 256 | 13 patches            |

**Default thresholds:** {0.50, 0.90, 0.95}. Primary: $\theta = 0.90$.

---

### 2.4.4 Implementation — `ComplexityMetrics`

```python
from metrics.complexity import ComplexityMetrics

cm = ComplexityMetrics()   # emr_thresholds=[0.5, 0.9, 0.95]

# Single attribution map (numpy array or torch.Tensor)
result = cm.compute(attribution_map, model_name="vit_b16", explainer_name="rollout")
print(result.gini, result.entropy_norm, result.emr_90, result.k_star_90)

# Batch (list of arrays or (B, N) tensor)
results = cm.compute_batch(attribution_batch)

# Aggregate over a dataset
agg = cm.aggregate(results)
# → {'gini_mean': 0.42, 'gini_std': 0.11, 'emr_90_mean': 0.17, ...}

# Standalone functions (no class instance required)
from metrics.complexity import (
    gini_coefficient,
    attribution_entropy,
    effective_mass_ratio,
    normalise_attribution,
)
g = gini_coefficient(att_map)              # float in [0, 1]
e = attribution_entropy(att_map)           # {'entropy_raw', 'entropy_norm', 'n_patches'}
r = effective_mass_ratio(att_map, 0.9)     # {'emr', 'k_star', 'n_patches', 'threshold'}
n = normalise_attribution(att_map, 'minmax')  # [0, 1]
```

---

### 2.4.5 Unit Tests (C1–C3)

**File: `tests/test_complexity.py`** — **45 tests**, all passing (pure-numpy, no torch required).

| Class                        | Tests | Category                                         |
| ---------------------------- | ----- | ------------------------------------------------ |
| `TestGiniCoefficient`        | 9     | C1 boundary, range, ordering, arch sizes         |
| `TestAttributionEntropy`     | 7     | C2 boundary, range, ordering                     |
| `TestEffectiveMassRatio`     | 7     | C3 monotone, range, scale-invariance             |
| `TestComplexityMetricsClass` | 12    | Integration: compute, batch, aggregate, warnings |
| `TestNormaliseAttribution`   | 6     | Utility: minmax, softmax, percentile, error      |
| `TestTheoremT6`              | 3     | Anti-alignment (Gini/Entropy/EMR × Symmetry)     |
| `TestSanityCheck`            | 1     | End-to-end sanity check ordering                 |

**File: `tests/test_torch_complexity.py`** — **7 PyTorch-specific tests** (skipped when torch absent).

| ID    | Class                          | Tests | What is tested                                           |
| ----- | ------------------------------ | ----- | -------------------------------------------------------- |
| PT_C1 | `TestGiniBatchTorchShape`      | 3     | `gini_batch_torch` output shape (B,), flat, dtype        |
| PT_C2 | `TestGiniBatchTorchValues`     | 3     | Values match numpy reference; one-hot; uniform           |
| PT_C3 | `TestEntropyBatchTorchRange`   | 2     | `entropy_batch_torch` range [0,1]; spiky map → near 0    |
| PT_C4 | `TestEntropyBatchTorchUniform` | 2     | Uniform map → H_norm = 1.0; batch of uniforms            |
| PT_C5 | `TestEmrBatchTorchOneHot`      | 3     | One-hot → EMR=1/N; uniform → EMR≈0.9; batch shape        |
| PT_C6 | `TestComputeBatchTorchTensor`  | 3     | `compute_batch` accepts `(B,H_p,W_p)` torch.Tensor       |
| PT_C7 | `TestDownsampleAttribution`    | 4     | `downsample_attribution` shape, non-negativity, 2D guard |

```
Historical results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0; current run in §1.4):
  test_complexity.py      : 45 passed, 0 failed
  test_torch_complexity.py: 20 passed, 0 failed, 1 skipped (CUDA)
  Combined Task 2.4       : 65 passed, 0 failed
```

---

### 2.4.6 Sanity Check

```bash
uv run python -c "from metrics.complexity import run_sanity_check; run_sanity_check()"
```

Expected output (N=196; C2 now sum-normalised):

```
Map                      Gini   H_norm    EMR90   k*90
one_hot                0.9949   0.0000   0.0051      1
uniform                0.0000   1.0000   1.0000    196
concentrated           0.9031   0.5717   0.0918     18
random                 0.4034   0.9513   0.6633    130
exponential            0.7964   0.7569   0.2347     46
✓ All sanity check assertions passed.
```

---

## 2.5 Axiomatic Analysis

The four Shapley-value axioms (Dummy, Completeness, Symmetry, Linearity) are used
as a theoretical lens to characterise the representational biases of the 13
benchmark metrics. This section consolidates the formal definitions, the
satisfaction table, and six results (T1–T6).

> [!NOTE]
> Axiomatic analysis does not prescribe which metrics are "best". It provides
> a principled vocabulary for the paper's discussion section and enables
> reviewers to verify that we understand the limitations of our own metrics.

---

### 2.5.1 Axiom Definitions

For an explanation method $\phi$ and model $f$, let $\phi_p(f, x)$ be the
attribution assigned to patch $p$.

**A1 — Dummy.** If $f$ does not depend on patch $p$ — i.e. $f(x) = f(x')$ whenever
$x$ and $x'$ differ only inside patch $p$ — then $\phi_p(f, x) = 0$.

> In a trained ViT every token attends to every other, so exact dummy patches
> essentially never occur in practice; A1 is checked on constructed models whose
> output provably ignores a patch (e.g. `verify_dummy_axiom()`, and the masked-patch
> test for Integrated Gradients in `tests/test_phase4_analysis.py`).

**A2 — Completeness.** $\sum_p \phi_p(f, x) = f(x) - f(x_\text{baseline})$.

**A3 — Symmetry.** If two patches $p, q$ make identical contributions to $f$
for all inputs, then $\phi_p(f, x) = \phi_q(f, x)$.

**A4 — Linearity.** If $f = \alpha g + \beta h$, then
$\phi(f, x) = \alpha\,\phi(g, x) + \beta\,\phi(h, x)$.

> [!NOTE]
> **ViT-specific caveats.** Dummy (A1) and Completeness (A2) require a baseline;
> ViT patch embeddings naturally support the zero-patch-embedding baseline.
> Symmetry (A3) assumes identical attention-path contributions, which requires
> careful control of positional encodings.

---

### 2.5.2 Axiom Satisfaction Table

Symbol key: ✓ rewards compliance · ∼ partially sensitive · ✗ insensitive · ⊗ designed to test axiom · †anti-aligned

| Metric                 | Dummy (A1) | Completeness (A2) | Symmetry (A3) | Linearity (A4) |
| ---------------------- | ---------- | ----------------- | ------------- | -------------- |
| F1 — Sufficiency       | ✓          | ✗ (T1)            | ✗             | ✗              |
| F2 — Comprehensiveness | ✓          | ✗ (T1)            | ✗             | ✗              |
| F3 — Log-odds drop     | ✓          | ✗ (T1)            | ✗             | ✗              |
| L1 — mIoU              | ∼          | ✗ (T1)            | ✗             | ✗              |
| L2 — Pointing Game     | ✗          | ✗ (T1)            | ✗             | ✗              |
| L3 — EGT               | ✓ (T2)     | ✗ (T1)            | ✗             | ✗              |
| L4 — Calibration Gap   | ✓          | ✗ (T1)            | ✗             | ✗              |
| R1 — Max-Sensitivity   | ✗          | ✗ (T1)            | ∼             | ✗              |
| R2 — Model Rand.       | ⊗          | ✗ (T1)            | ✗             | ✗              |
| R3 — Head Perm.        | ⊗          | ✗ (T1)            | ✗             | ✗              |
| **C1 — Gini**          | ✗          | ✗ (T1)            | **∼†**        | ✗              |
| **C2 — Entropy**       | ✗          | ✗ (T1)            | **∼†**        | ✗              |
| **C3 — EMR**           | ✗          | ✗ (T1)            | **∼†**        | ✗              |

Rows F1–F3, R1–R3 and L4 are theoretical (they need a trained classifier or a dataset);
`AxiomVerifier.run_all()` checks L1–L3 and C1–C3 empirically on synthetic models.

> [!NOTE]
> **Why the A2 and A4 columns are empty.** This is a result, not an omission.
> Theorem T1 shows that *every* metric in the suite is invariant to positive rescaling
> of the attribution, so none can detect Completeness violations. Linearity (A4) is a
> property of an explainer *across* models ($\phi(\alpha g + \beta h) = \alpha\phi(g) +
> \beta\phi(h)$), which no metric evaluated on a single $(f, x, \phi)$ can test.
> A2 and A4 must be verified on the explainer directly: `verify_completeness()` and
> the Integrated Gradients completeness gap (T4). The informative columns for metric
> design are A1 and A3.

> [!IMPORTANT]
> **†Anti-alignment (Theorem T6):** C1–C3 reward A3 _violations_. Concentrating
> all mass on one of two equal-contribution patches increases Gini and decreases
> Entropy/EMR, even though this breaks symmetry. This is a structural property
> of sparsity metrics, not a bug. It is reported explicitly in the paper.

---

### 2.5.3 Theorems

#### Theorem T1 — No metric in the suite can detect Completeness (A2) violations

**Claim.** For every metric $m$ among F1–F3, L1–L4, R1–R3, C1–C3 and every $c > 0$:
$m(c\,\phi) = m(\phi)$. Consequently no metric distinguishes a complete attribution
from an incomplete one.

_Proof._ Per family:

- **F1–F3** depend on $\phi$ only through the set of top-$k$ patches, which is invariant
  under any strictly increasing transformation of $\phi$.
- **L1** thresholds the min-max normalised map, invariant under positive affine maps;
  **L2** uses $\arg\max_p \phi_p$; **L3** is a ratio of positive masses (scale-invariant
  by construction, §2.2); **L4** is a difference of mean L3 values.
- **R1** is $\|\phi' - \phi\|_2 / \|\phi\|_2$, homogeneous of degree 0 when both maps are
  rescaled by the same explainer; **R2** compares min-max normalised maps; **R3** uses
  Spearman ranks.
- **C1–C3** apply min-max or sum normalisation before computation.

Completeness fixes the *scale*: if $\sum_p \phi_p = f(x) - f(x')\neq 0$, then
$\sum_p c\,\phi_p \neq f(x) - f(x')$ for every $c \neq 1$, yet all metrics agree on
$\phi$ and $c\,\phi$. $\square$

**Implication.** A2 must be verified on the explainer itself (`verify_completeness()`;
the IG completeness gap). The pipeline's own min-max normalisation (Task 3.2) would
remove completeness information anyway.

---

#### Theorem T2 — EGT (L3) is unaffected by non-positive attribution

**Claim.** Let $Z = \{p : \phi_p \leq 0\}$. Changing $\phi_p$ to any non-positive value for
$p \in Z$ leaves EGT unchanged. In particular, an attribution that satisfies A1
($\phi_p = 0$ on dummy patches) is never penalised for its dummy patches, and if
$\phi_p \leq 0$ for all $p \notin M^{GT}$ (with some positive mass), then EGT $= 1$.

_Proof._ With $e^+_p = \max(\phi_p, 0)$, every $p \in Z$ contributes $0$ to both the
numerator $\sum_{p \in M^{GT}} e^+_p$ and the denominator $\sum_p e^+_p$. If no positive
mass lies outside $M^{GT}$, numerator and denominator coincide. $\square$

**Implication.** L3 rewards A1 compliance on out-of-GT dummy patches (a violation that
puts positive mass on such a patch strictly lowers EGT), but cannot detect a violation
inside the GT region. (Under the earlier softmax definition this result was false:
softmax is strictly positive, so EGT never reached 1 — see §2.2 L3.)

---

#### Theorem T3 — Attention Rollout never assigns zero attribution

**Setting.** Rollout uses $\hat A_l = \tfrac12 A_l + \tfrac12 I$ for each layer's
head-averaged attention $A_l$ (row-stochastic, from a softmax), and
$R = \hat A_L \hat A_{L-1} \cdots \hat A_1$; the attribution of patch $p$ is $R_{\text{cls},p}$.

**Claim.** For every patch $p$:

$$R_{\text{cls},p} \;\geq\; \left(\tfrac12\right)^{L} \, (A_L)_{\text{cls},p} \;>\; 0.$$

_Proof._ All entries of $\hat A_l$ are non-negative, so $R_{\text{cls},p}$ is at least
the contribution of any single path. Take the path that stays on the CLS token through
layers $1, \dots, L-1$ (diagonal entries $(\hat A_l)_{\text{cls},\text{cls}} \geq \tfrac12$)
and moves to $p$ at layer $L$ ($(\hat A_L)_{\text{cls},p} = \tfrac12 (A_L)_{\text{cls},p}$).
Softmax attention is strictly positive, so $(A_L)_{\text{cls},p} > 0$. $\square$

**Note.** The bound depends on the attention entry; it is not a constant floor. The
earlier statement $\min_p \phi_p \geq (1/2)^L$ only holds for diagonal (token-to-itself)
entries, not for CLS-to-patch entries. Softmax positivity alone already makes raw
attention (E1) strictly positive; the point of T3 is that rollout's composition keeps
this property and adds the $(1/2)^L$ residual factor, so rollout can satisfy A1 only
in the limit where the last layer's CLS attention to $p$ vanishes.

**Implication.** On a constructed model with a dummy patch, rollout violates A1 by
at least $(1/2)^L (A_L)_{\text{cls},p}$ (`verify_rollout_dummy_violation()`).

---

#### Theorem T4 — Integrated Gradients satisfies A1, A2 and A4; exact Shapley values satisfy all four

**Claim.** (i) Exact Shapley values over patches are the unique attribution satisfying
A1–A4 (Shapley, 1953). (ii) Integrated Gradients (Sundararajan et al., 2017) satisfies
Dummy, Completeness and Linearity, and is symmetry-preserving for patches that are
symmetric in $f$ and equal in $x$ and the baseline. Summing pixel attributions within
each patch (E8) preserves all of these.

_Proof._ (i) is the classical result. (ii) is Propositions 1–2 and §3 of Sundararajan
et al.; patch aggregation is a fixed linear map, so sums (A2), zeros (A1) and linear
combinations (A4) are preserved. $\square$

**What is not claimed.** No E1–E7 explainer computes Shapley values, and approximate
KernelSHAP (Lundberg & Lee, 2017) satisfies the axioms only in expectation / in the
limit of infinitely many coalition samples, not exactly. Exact Shapley values over
196+ patches are intractable. IG's Riemann-sum approximation also has a completeness
error; `IntegratedGradientsExplainer.last_completeness_gap` reports it, and it decreases
with `n_steps` (tested).

**Implication.** IG (E8, optional explainer) is the benchmark's axiomatic reference
for A1/A2/A4. It is model-agnostic, so it also covers Swin-B.

---

#### Theorem T5 — GradCAM satisfies A1 approximately via ReLU

**Claim.** GradCAM (post-ReLU) satisfies the Dummy axiom for patches where
the gradient is 0, which occurs when that patch does not activate any feature
map that contributes positively to the target class.

_Proof sketch._ GradCAM weights each feature map $A^k$ by
$\alpha^k_c = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$.
If patch $p$ contributes to a feature map only with negative gradient,
the ReLU zeroes the contribution. $\square$

**Implication.** GradCAM's implicit ReLU is a practical approximation of A1.
Report the fraction of patches with zero GradCAM attributions as a
"dummy satisfaction rate" in Table T4.

---

#### Theorem T6 — Complexity metrics (C1–C3) are incompatible with Symmetry (A3)

**Setting.** Let $e \in \mathbb{R}^N_{\geq 0}$ with $\min_p e_p = 0$ (some patch carries no
attribution, so the implementation's min-max normalisation is a positive rescaling and,
by T1, can be ignored). For $p \neq q$ with $e_p = e_q = a > 0$ and $0 < \delta \leq a$,
let $e^{(\delta)}$ be $e$ with $e_p \mapsto a + \delta$, $e_q \mapsto a - \delta$.

**Claim.** $e$ is majorised by $e^{(\delta)}$ ($e \prec e^{(\delta)}$), and

$$\text{C1}(e^{(\delta)}) > \text{C1}(e), \qquad \text{C2}(e^{(\delta)}) < \text{C2}(e), \qquad \text{C3}_\theta(e^{(\delta)}) \leq \text{C3}_\theta(e)\ \ \forall \theta.$$

_Proof._ $e^{(\delta)}$ is obtained from $e$ by a transfer from a no-larger to a no-smaller
entry with the total unchanged, so $e \prec e^{(\delta)}$ (Marshall, Olkin & Arnold, 2011,
§2.B). The Gini coefficient is strictly Schur-convex and Shannon entropy of the
sum-normalised vector is strictly Schur-concave on vectors that are not permutations of
one another, giving the strict inequalities for C1 and C2. For C3, majorisation means
every top-$k$ partial sum of $e^{(\delta)}$ is at least that of $e$, so $k^*(\theta)$ can
only decrease. $\square$

**Corollary (incompatibility).** Suppose $f$ has two patches $p, q$ that are symmetric
in $f$ (A3 applies) and receive non-zero attribution. Among all attributions with the
same values elsewhere and the same total mass on $\{p, q\}$, the unique A3-compliant one
($e_p = e_q$) is the **strict minimiser** of C1 and the strict maximiser of C2: any
optimum of C1 or C2 violates A3.

**Example.** $e_\text{sym} = (\tfrac12, \tfrac12, 0, \ldots)$ vs. $e_\text{asym} = (1, 0, \ldots)$
($\delta = \tfrac12$): C2 $= \log 2 / \log N$ vs. $0$; C3$_{0.9}$ $= 2/N$ vs. $1/N$;
C1 is higher for $e_\text{asym}$.

**Why this is not a tautology.** "Sparsity metrics reward sparsity" says nothing about
axioms. T6 identifies a specific axiom that concentration conflicts with, and shows the
conflict is strict for C1–C2 at *every* symmetric pair. (C1–C3 have no consistent
relation to A1: removing a dummy patch's mass increases concentration only if that
mass moves to larger entries.)
`AxiomVerifier.run_all()` confirms the sign empirically (C1–C3 × A3: `delta < 0`).

**Implication.** Never use C1–C3 as the sole criterion for comparing methods that
differ primarily in symmetry. Report alongside L1–L4.

---

### 2.5.4 Implementation — `AxiomVerifier`

```python
from metrics.axiom_verifier import AxiomVerifier
from metrics.complexity import gini_coefficient, attribution_entropy, effective_mass_ratio

# Standalone (C1–C3 only, no MetricSuite required)
verifier = AxiomVerifier(n_patches=16, seed=42)

# Test Gini vs. Symmetry (Theorem T6: expected anti-alignment)
result = verifier.test_a3(
    lambda e, x, m: gini_coefficient(e),
    "C1-Gini",
    higher_is_better=True,
)
print(result.satisfies)               # False (anti-aligned)
print(result.delta)                   # < 0

# Run all 12 (C1–C3 × A1–A4) tests; with metric_suite=MetricSuite() also L1–L3 (24 tests)
results = verifier.run_all()          # list of 12 AxiomTestResult
table   = verifier.build_satisfaction_table(results)
print(table)                          # Markdown table with ✓ / ∼† / ✗

# Completeness verification (one sample)
from metrics.axiom_verifier import verify_completeness
cv = verify_completeness(explainer_fn, model, x, target_class, x_baseline)
print(cv['satisfies_ce'], cv['completeness_error'])

# Generate Figure F1: axiom satisfaction heatmap
from metrics.axiom_verifier import generate_axiom_satisfaction_heatmap
generate_axiom_satisfaction_heatmap(output_path="figures/axiom_satisfaction.pdf")
```

---

### 2.5.5 Unit Tests (Axiomatic Analysis)

**File: `tests/test_axiom_verifier.py`** — **20 tests**, all passing (14 torch-dependent via `@requires_torch`, 6 pure-numpy).

| Class                     | Tests | Category                             |
| ------------------------- | ----- | ------------------------------------ |
| `TestLinearPatchModel`    | 4     | Toy model forward pass (torch)       |
| `TestXORInteractionModel` | 3     | Multiplicative interaction (torch)   |
| `TestAxiomVerifierC1C3`   | 11    | A1–A4 tests, run_all, table building |
| `TestAxiomTestResult`     | 2     | Dataclass serialisation              |
| `TestTheoremT6Canonical`  | 1     | Canonical T6 empirical proof         |

**File: `tests/test_torch_axioms.py`** — **7 PyTorch-specific tests** (skipped when torch absent).

| ID    | Class                                  | Tests | What is tested                                                 |
| ----- | -------------------------------------- | ----- | -------------------------------------------------------------- |
| PT_A1 | `TestVerifyCompletenessLinear`         | 2     | `verify_completeness` returns 4 keys; finite error             |
| PT_A2 | `TestVerifyCompletenessXOR`            | 1     | XOR model illustrates Theorem T1 counterexample                |
| PT_A3 | `TestVerifyDummyAxiomPasses`           | 1     | `verify_dummy_axiom` returns 7 keys; values in [0,1]           |
| PT_A4 | `TestVerifyDummyAxiomFails`            | 1     | Wrong attribution → non-zero dummy_attribution_ratio           |
| PT_A5 | `TestAxiomVerifierA3GiniAntialignment` | 1     | Theorem T6: Gini A3 delta < 0 via verifier (torch backend)     |
| PT_A6 | `TestVerifyRolloutDummyViolation`      | 1     | `verify_rollout_dummy_violation` interface contract + keys     |
| PT_A7 | `TestGiniBatchTorchCPUGPUConsistency`  | 2     | `gini_batch_torch` deterministic on CPU; GPU agree (skip CUDA) |

```
Historical results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0; current run in §1.4):
  test_axiom_verifier.py : 20 passed, 0 failed
  test_torch_axioms.py   : 8 passed, 0 failed, 1 skipped (CUDA)
  Combined Task 2.5      : 28 passed, 0 failed
```

---

## 2.6 BenchmarkRunner — Unified Evaluation Loop

File: `metrics/runner.py`. Integrates L1–L4 and (optionally) R1–R3 into a single
dataset-level loop. C1–C3 integration is in progress (Phase 3).

### Constructor

```python
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
)

# Localization only
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_explainer,
)

# Localization + Robustness
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_explainer,
    robustness=RobustnessMetrics(epsilon=0.05, n_samples=50),
    randomised_model=randomise_model_weights(model, seed=0),
    label_randomised_model=randomise_classifier_labels(model, seed=0),
)

results = runner.evaluate(model, val_loader, dataset_name="cub200")
```

---

---

# Phase 3 — Baseline Evaluation Pipeline

Phase 3 wraps the 13 metrics (F1–F3, L1–L4, R1–R3, C1–C3) and 6 explainers (E1–E6; E7 DIME is a placeholder, see §3.1.2)
into a **single reproducible evaluation command** with:

- Fixed random seeds (per-run seed logged in metadata)
- Checkpoint-and-resume for long GPU runs
- Unified CSV output that feeds directly into Phase 4 result tables

> [!IMPORTANT]
> Every number in the TPAMI result tables **must come from this pipeline** with
> fixed seeds. Do not manually insert numbers computed outside this framework.

---

## 3.1 Standardised Explainer Interface

All explainers (E1–E6, the E7 placeholder and optional E8) share a common `BaseExplainer` contract defined in
`explainers/base.py`.

### 3.1.1 BaseExplainer Contract

```python
from explainers.base import BaseExplainer

class MyExplainer(BaseExplainer):
    def explain(self, x: torch.Tensor, target_class: int, **kwargs) -> torch.Tensor:
        # x: (3, H, W) float32 in [0, 1]
        # returns: (H // patch_size, W // patch_size) float32 tensor
        ...
```

Key invariants:

- Output shape is always `(H_img // patch_size, W_img // patch_size)`.
- Values are **un-normalised** (raw attribution scores; normalisation is Task 3.2).
- `explain_batch(xs, target_classes)` defaults to a loop; override for amortised methods.
- `UnsupportedArchitectureError` is raised by E1/E2/E4 on models without a global CLS token (Swin-B).

### 3.1.2 Explainer Index

| #   | Class                       | Method                     | File               | Swin-B | Note                           |
| --- | --------------------------- | -------------------------- | ------------------ | ------ | ------------------------------ |
| E1  | `RawAttentionExplainer`     | Raw CLS attention          | `raw_attention.py` | ✗      | Last block, mean over heads    |
| E2  | `AttentionRolloutExplainer` | Attention Rollout          | `rollout.py`       | ✗      | Recursive Â product            |
| E3  | `GradCAMExplainer`          | GradCAM                    | `gradcam.py`       | ✓      | Hook-based, ReLU gated         |
| E4  | `CheferLRPExplainer`        | Chefer et al. LRP          | `chefer_lrp.py`    | ✗      | Pure-PyTorch reimplementation  |
| E5  | `RISEExplainer`             | RISE (Petsiuk et al. 2018) | `rise.py`          | ✓      | 4000 float16 masks; chunked    |
| E6  | `LIMEExplainer`             | LIME (patch-grid)          | `lime.py`          | ✓      | Ridge regression on P² patches |
| E7  | `DIMEExplainer`             | DIME                       | `dime.py`          | —      | ⚠ Placeholder (see below)      |
| E8  | `IntegratedGradientsExplainer` | Integrated Gradients    | `integrated_gradients.py` | ✓ | Optional; 50-step path, patch sums |

**Effective method count.** The main matrix runs **six** explainers (E1–E6); E7 is a
stub, so the benchmark does not claim seven. On Swin-B only E3, E5, E6 (and E8) apply.
E8 Integrated Gradients is available with `--explainers ... integrated_gradients` as the
axiomatic reference (Theorem T4). Newer ViT-specific methods (e.g. AttnLRP,
TransAttribution variants) are not implemented and are listed as future work.

Ablation-only variants (Phase 4, A1/A2; `--ablations`): `RawPatchAttentionExplainer`
(patch-token attention), `RawAttentionLast3Explainer`, `RawAttentionAllLayersExplainer`.

> [!NOTE]
> **E7 — DIME Guide Inconsistency.** The implementation guide lists DIME as
> Explainer 7, but DIME (Differently Interpreted Multimodal Explanations) is a
> VQA/multimodal method that does not produce spatial attribution maps for
> single-image ViT classification. `DIMEExplainer` is a documented placeholder
> that raises `NotImplementedError` referencing this section. Deviation will be
> documented in the paper's Appendix.

### 3.1.3 Design Decisions

**E4 — CheferLRP:** Pure-PyTorch reimplementation (not a wrapper of the
original repo). The guide allows "implement or wrap" and self-contained code is
more auditable and dependency-free for TPAMI reproducibility review.

**E5 — RISE vectorisation:** Pre-generates all M=4000 masks as float16
`(M, 1, H, W)` at `__init__` time. Inference loops over chunks of 100, giving
~40 forward passes instead of 4000 serial passes. Consistent with the guide's
_"vectorise over mask dimension"_ requirement.

**Gradient handling:** Outer `torch.no_grad()` context + inner
`torch.enable_grad()` for gradient methods (GradCAM, CheferLRP). Hook cleanup
is guaranteed via `try … finally: hook.remove()`.

### 3.1.4 Unit Tests (Task 3.1)

File: `tests/test_explainers.py` — **26 tests passing, 1 documented skip**
(E14: DIME no-NaN test, pending guide resolution).

| Category   | Tests | What is tested                                              |
| ---------- | ----- | ----------------------------------------------------------- |
| A – Shape  | 7     | Every explainer returns exactly `(P, P)` shape              |
| B – Finite | 7     | `torch.isfinite(output).all()` for all explainers           |
| C – Batch  | 3     | `explain_batch` ≡ loop over `explain` (E1, E2, E3)          |
| D – Var    | 3     | Output `std > 0` on non-trivial input (E1, E5, E6)          |
| E – Swin   | 2     | E1/E2 raise `UnsupportedArchitectureError` on `_MockSwinB`  |
| F – RISE   | 2     | Non-negative output; mask count stored correctly            |
| G – LIME   | 2     | Finite output; exactly P² = 16 coefficients                 |
| H – DIME   | 1     | `is_resolved=False`; `NotImplementedError` ∋ "BENCHMARK.md" |

```
Historical results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0; current run in §1.4):
  test_explainers.py : 26 passed, 1 skipped (E14 documented), 0 failed
```

### 3.1.5 Production Notes

| Concern              | Recommendation                                                    |
| -------------------- | ----------------------------------------------------------------- |
| RISE cost (M=4000)   | Sub-sample to 500 val images per model; report in table footnote  |
| LIME cost            | `n_samples=500` for production; 20 for dev/CI                     |
| Seed reproducibility | Pass `seed=42` to E5, E6; fix `torch.manual_seed` before each run |
| Swin-B explainers    | Use E3 (GradCAM) only; E1/E2/E4 auto-raise at runtime             |

---

## 3.2 Attribution Normalisation

File: `metrics/normalize.py` — Guide Listing 3 canonical normalisation pipeline.
Sits **between** explainers (raw maps) and metrics (expect [0, 1] inputs).

### 3.2.1 API

```python
from metrics.normalize import normalize_attribution, normalize_batch, NormMode

# Single map (Hp, Wp)
norm = normalize_attribution(att_map, mode='minmax')

# Batch (B, Hp, Wp) — each sample normalised independently
batch_norm = normalize_attribution(batch_att, mode='percentile')

# Convenience: enforces 3-D input strictly
batch_norm = normalize_batch(batch_att, mode='softmax')
```

### 3.2.2 Mode Specifications (Guide §3.2)

| Mode         | Formula                                           | Use case                            |
| ------------ | ------------------------------------------------- | ----------------------------------- |
| `minmax`     | `(att − min) / (max − min)` ; zeros if degenerate | Default; all fidelity + loc metrics |
| `percentile` | clamp at 99th percentile → minmax                 | Heavy-tailed maps (RISE, LIME)      |
| `softmax`    | `softmax(att.flatten())` ; numerically stable     | Optional; not used by any metric    |

- Output always `float32`, values in `[0, 1]` (`softmax` sums to 1.0). The pipeline
  uses `minmax` by default; metrics that need a distribution (L3, C2) apply their own
  positive-mass / sum normalisation, which is scale-invariant (Theorem T1).
- Degenerate (constant) maps return all-zeros for `minmax` and `percentile`.
- Batch input `(B, Hp, Wp)`: each sample is normalised **independently**.

### 3.2.3 Unit Tests (Task 3.2)

File: `tests/test_normalize.py` — **24 tests**, all passing.

| Class                         | Tests | Category                                                             |
| ----------------------------- | ----- | -------------------------------------------------------------------- |
| `TestMinmax`                  | 6     | range, exact [0,1], degenerate, shape, batch, dtype                  |
| `TestPercentile`              | 6     | range, outlier suppression, rank-preservation, batch                 |
| `TestSoftmax`                 | 6     | sum=1, all-positive, batch, stability, uniform, shape                |
| `TestIntegrationAndEdgeCases` | 6     | pipeline, invalid mode, 1-D rejection, batch guard, idempotent, enum |

```
Historical results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0; current run in §1.4):
  test_normalize.py : 24 passed, 0 failed
  Full suite        : 149 passed, 2 skipped (CUDA), 0 failed
```

---

## 3.3 BenchmarkRunner (Phase3Runner)

File `metrics/runner.py` now contains **two classes**:

| Class             | Purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `BenchmarkRunner` | Task 2.x L1–L4 loop (unchanged, backward-compatible)        |
| `Phase3Runner`    | Guide Listing 4 — full `dataset × model × explainer` matrix |

### 3.3.1 Phase3Runner API

```python
from metrics.runner import Phase3Runner
from explainers import AttentionRolloutExplainer, GradCAMExplainer

runner = Phase3Runner(
    models     = {'vit-b16': model_vit, 'swin-b': model_swin},
    explainers = {'rollout': AttentionRolloutExplainer,
                  'gradcam': GradCAMExplainer},
    datasets   = {'imagenet': val_loader, 'cub200': cub_loader},
    device     = 'cuda',
    norm_mode  = 'minmax',    # Task 3.2 normalisation
    patch_size = 16,
)
results = runner.run(checkpoint_dir='results/phase3', seed=42)
```

### 3.3.2 Checkpoint Design

One `.pkl` file per `(dataset, model, explainer)` combination:

```
results/phase3/
    imagenet-vit-b16-rollout.pkl     ← dict: key, n_samples, attributions, labels, gt_masks, norm_mode
    imagenet-vit-b16-gradcam.pkl
    imagenet-swin-b-gradcam.pkl      ← skipped automatically if exists (resume)
    cub200-vit-b16-rollout.pkl
```

- **Atomic write** via `.pkl.tmp` → rename to prevent corruption on interrupt.
- **Resume**: re-running skips any existing `.pkl` automatically.
- **Seed**: `torch.manual_seed(seed)` applied once at `run()` start and again before each combination, ensuring reproducible explainer random state.

### 3.3.3 CLI Entry Point

```bash
# Dry-run validation (1 batch per combination)
python -m metrics.runner --dry-run --checkpoint-dir results/phase3 --seed 42

# List existing checkpoints
python -m metrics.runner --list-checkpoints --checkpoint-dir results/phase3

# Full run with percentile normalisation
python -m metrics.runner --norm-mode percentile --checkpoint-dir results/phase3
```

All flags: `--checkpoint-dir`, `--seed`, `--max-batches`, `--norm-mode`, `--patch-size`, `--dry-run`, `--list-checkpoints`.

### 3.3.4 Unit Tests (Task 3.3)

File: `tests/test_runner.py` — **36 tests**, all passing.

| Class                  | Tests | Category                                              |
| ---------------------- | ----- | ----------------------------------------------------- |
| `TestPhase3RunnerInit` | 9     | Constructor attributes, device, norm_mode, patch_size |
| `TestRunCombination`   | 9     | Batch flow, schema, normalisation, masks, pkl write   |
| `TestRun`              | 9     | Dir creation, one pkl/combo, skip, seed repro, resume |
| `TestCLI`              | 9     | --help, defaults, flags, invalid mode, --list-ckpts   |

```
Historical results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0; current run in §1.4):
  test_runner.py : 36 passed, 0 failed
  Full suite     : 185 passed, 2 skipped (CUDA), 0 failed
```


---

## 3.4 Sanity Checks (S1–S3)

Mandatory model/explainer verification (Guide §3.4) implemented in `metrics/sanity.py`, to run before generating the final results matrix.

| Check | Name | Description | Export |
|---|---|---|---|
| **S1** | Random Baseline | Verifies that a generated uniform random map yields chance-level Gini, max Entropy, and uniform EMR90. | `run_s1_random_baseline` |
| **S2** | Model Parameter Randomisation | R3 validation (Adebayo et al. 2018): cascading randomisation from top to bottom transformer blocks monotonically reduces Spearman $\rho$ with the base map. | `run_s2_model_randomisation` |
| **S3** | Label Permutation | Compares attribution of the true class vs a permuted (wrong) class to verify dependence behavior (high vs low sensitivity). | `run_s3_label_permutation` |

### 3.4.1 Public API

```python
from metrics.sanity import run_all_sanity_checks

results = run_all_sanity_checks(
    explainer_cls=GradCAMExplainer,
    model=model_vit,
    patch_size=16,
    img_size=224,
    n_images=10,
    seed=42
)
# Returns Dict[str, SanityResult]: {'S1': S1Result, 'S2': S2Result, 'S3': S3Result}
```

### 3.4.2 Unit Tests (Task 3.4)

File: `tests/test_sanity.py` — **16 tests**, all passing.

| Class                      | Tests | Category                                               |
| -------------------------- | ----- | ------------------------------------------------------ |
| `TestS1RandomBaseline`     | 5     | S1 Result schema, entropy/gini empirical thresholds    |
| `TestS2ModelRandomisation` | 5     | Monotonic decrease check, slack factor for tiny models |
| `TestS3LabelPermutation`   | 4     | Independence vs gradient-dependence, Spearman scores   |
| `TestSanityIntegration`    | 2     | Dictionary extraction and JSON-serialisation safety    |

```
Historical results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0; current run in §1.4):
  test_sanity.py : 16 passed, 0 failed
  Full suite     : 201 passed, 2 skipped (CUDA), 0 failed
```

---

# Appendix A — Project Layout

```
vit-explainability-benchmark/
├── BENCHMARK.md                        ← THIS FILE — single authoritative reference
│
├── model_zoo/
│   ├── __init__.py                     # load_model() dispatcher
│   ├── vit_b16.py                      # Model 1 — ViT-B/16 (timm augreg_in21k)
│   ├── deit_b16.py                     # Model 2 — DeiT-B/16 distilled
│   ├── swin_b.py                       # Model 3 — Swin-B
│   ├── beit_b16.py                     # Model 4 — BEiT-B/16
│   ├── dino_vitb8.py                   # Model 5 — DINO-ViT-B/8
│   └── dinov2_vitb14.py                # Model 6 — DINOv2-ViT-B/14
│
├── training/
│   ├── __init__.py
│   ├── transforms.py                   # RandAugment + random erasing pipeline
│   ├── mixup.py                        # Batch-level Mixup α=0.8
│   ├── optimizer.py                    # AdamW + LR warmup + cosine decay
│   ├── loss.py                         # SoftTargetCrossEntropy / BCE (NIH ChestX-ray14)
│   └── trainer.py                      # Full fine-tune loop with grad accumulation
│
├── explainers/
│   ├── __init__.py                     # All explainer exports
│   ├── base.py                         # BaseExplainer ABC + shared helpers
│   ├── raw_attention.py                # E1 — Raw CLS attention (last block)
│   ├── rollout.py                      # E2 — Attention Rollout (recursive Â)
│   ├── gradcam.py                      # E3 — GradCAM (hook-based, ReLU)
│   ├── chefer_lrp.py                   # E4 — Chefer et al. LRP (pure-PyTorch)
│   ├── rise.py                         # E5 — RISE (4000 float16 masks, chunked)
│   ├── lime.py                         # E6 — LIME (patch-grid, ridge regression)
│   └── dime.py                         # E7 — DIME (placeholder — see §3.1.2)
│
├── metrics/
│   ├── __init__.py                     # All public exports (torch-free first)
│   ├── fidelity.py                     # F1–F3 Sufficiency / Comprehensiveness / Log-odds
│   ├── localization.py                 # L1–L4 (requires torch)
│   ├── robustness.py                   # R1–R3 + model utilities (requires torch)
│   ├── complexity.py                   # C1–C3 (torch-free; optional GPU batch ops)
│   ├── axiom_verifier.py               # AxiomVerifier, verify_completeness, Figure F1
│   ├── normalize.py                    # Attribution normalisation (minmax/percentile/softmax)
│   ├── sanity.py                       # Sanity checks S1–S3
│   └── runner.py                       # BenchmarkRunner + Phase3Runner
│
├── tests/
│   ├── test_fidelity.py                #  5 unit tests — F1–F3
│   ├── test_localization.py            # 12 unit tests — L1–L4
│   ├── test_robustness.py              # 16 unit tests — R1–R3
│   ├── test_complexity.py              # 45 unit tests — C1–C3 + Theorem T6
│   ├── test_torch_complexity.py        #  7 PyTorch-specific C1–C3 tests
│   ├── test_axiom_verifier.py          # 20 unit tests — AxiomVerifier + T6
│   ├── test_torch_axioms.py            #  7 PyTorch-specific axiom tests
│   ├── test_normalize.py               # 24 unit tests — normalisation pipeline
│   ├── test_explainers.py              # 28 tests (+1 skip) — E1–E7 explainers, GradCAM layout
│   ├── test_runner.py                  # 36 unit tests — Phase3Runner + CLI
│   └── test_sanity.py                  # 16 unit tests — S1–S3
│
├── figures/                            # Generated PDFs (gitignored; on demand)
│   ├── axiom_satisfaction.pdf          # Figure F1 — 13×4 axiom heatmap
│   └── complexity_distributions.pdf
│
├── scripts/
│   ├── record_model_hashes.py          # SHA-256 logging for reproducibility
│   ├── pilot_finetune.py               # 5-epoch sanity check
│   ├── verify_datasets.py              # Dataset integrity verification
│   ├── create_cub_val_split.py         # Stratified val split for CUB-200-2011
│   ├── phase4_correlation_analysis.py  # Task 4.1 — Spearman correlation + PCA
│   ├── phase4_interaction_analysis.py  # Task 4.2 — Kendall τ concordance
│   └── phase4_ablations.py             # Task 4.3 — Ablation A1–A4 + Cohen's d
│
├── configs/
│   ├── cub200.yaml                     # 50 epochs, fine-grained, 200 classes
│   ├── pascal_voc.yaml                 # 30 epochs, multi-label, 20 classes
│   ├── imagenet.yaml                   # ImageNet base config
│   ├── imagenet_s50.yaml               # 30 epochs, 50-class subset
│   └── nih_chestxray.yaml              # 30 epochs, class-weighted BCE, 14 classes
│
├── results/
│   └── phase4/                         # Phase 4 analytics output directory
│
├── pyproject.toml                      # uv project file (Python 3.13)
├── model_hashes.txt                    # SHA-256 of all pre-trained checkpoints
├── requirements.txt                    # Pinned runtime dependency versions
├── docs/                               # Project documentation
│   ├── implementation_guide.md         # Step-by-step operational guide
│   ├── phase4_instructions.md          # Phase 4 analytical instructions
│   ├── project_status_checklist.md     # Full task checklist/audit
│   ├── compute_budget_breakdown.md     # GPU budget estimates
│   ├── anaconda_compatibility_and_runtime_analysis.md # Anaconda/runtime analysis
│   └── literature_review.md            # Lit review & gap analysis
└── .gitignore
```

---

# Appendix B — Complete Metric Index

| ID  | Name              | Formula (short)                                                       | Range        | File               | Better    |
| --- | ----------------- | --------------------------------------------------------------------- | ------------ | ------------------ | --------- |
| F1  | Sufficiency       | $f(x)_y - f(\tilde x_\text{keep})_y$                                  | $(-1,1)$     | `fidelity.py`      | **Lower** |
| F2  | Comprehensiveness | $f(x)_y - f(\tilde x_\text{drop})_y$                                  | $(-1,1)$     | `fidelity.py`      | Higher    |
| F3  | Log-odds Drop     | $\operatorname{logit} f(x)_y - \operatorname{logit} f(\tilde x_\text{drop})_y$ | $\mathbb{R}$ | `fidelity.py` | Higher    |
| L1  | mIoU              | $\text{mean}_\tau \text{IoU}(\hat{M}_\tau, M^{GT})$                   | $[0,1]$      | `localization.py`  | Higher    |
| L2  | Pointing Game     | $\mathbf{1}[p^* \in M^{GT}]$                                          | $\{0,1\}$    | `localization.py`  | Higher    |
| L3  | EGT               | $\sum_{p \in M^{GT}} e^+_p / \sum_p e^+_p$                            | $[0,1]$      | `localization.py`  | Higher    |
| L4  | CalibGap          | $\mathbb{E}[\text{EGT}\mid\text{correct}] - \mathbb{E}[\text{EGT}\mid\text{wrong}]$ | $(-1,1)$ | `aggregate.py` | Higher |
| R1  | MaxSens           | $\max_k \lVert\phi(x+\delta_k)-\phi(x)\rVert_2 / \lVert\phi(x)\rVert_2$ | $[0,\infty)$ | `robustness.py`  | **Lower** |
| R2  | ModelRand         | $1 - \text{SSIM}(\phi_\text{orig}, \phi_\text{rand})$                 | $[0,1]$      | `robustness.py`    | Higher    |
| R3  | HeadPerm          | $1 - (\lvert\rho(\phi_\text{orig}, \phi_\text{perm})\rvert + 1)/2$    | $[0,0.5]$    | `robustness.py`    | Higher    |
| C1  | Gini              | $(2\sum_i i \cdot e_{(i)}) / (N \sum e) - (N+1)/N$                    | $[0,1]$      | `complexity.py`    | Higher    |
| C2  | Entropy (norm)    | $H(\lvert e\rvert / \sum \lvert e\rvert) / \log N$                    | $[0,1]$      | `complexity.py`    | **Lower** |
| C3  | EMR               | $k^*(0.90) / N$                                                       | $[0,1]$      | `complexity.py`    | **Lower** |

Optional / ablation-only columns: `pgd_robustness` (cosine similarity before/after PGD,
higher is better; `--metrics adversarial`) and fidelity columns suffixed `[zero]`/`[blur]`
(ablation A3).

> [!NOTE]
> **F1, R1, C2, C3 direction is inverted** (lower = better explanation). For all
> other metrics, higher = better. Clearly flag direction in all paper tables.

---

# Appendix C — Master Checklist

## Phase 1

```
☑ Model zoo: 6 ViT variants selected with controlled variation
☑ ViT-B/16 wrapper — timm augreg_in21k, get_last_selfattention()
☑ DeiT-B/16 wrapper — CLS+distillation tokens documented
☑ Swin-B wrapper — GradCAM only; CLS limitation documented
☑ BEiT-B/16 wrapper — MIM pre-training, standard attention
☑ DINO-ViT-B/8 wrapper — torch.hub, 28×28 patch grid
☑ DINOv2-ViT-B/14 wrapper — 16×16 patch grid, timm/HF
☑ training/transforms.py — RandAugment M=9 N=2, random erasing
☑ training/mixup.py — α=0.8 batch Mixup
☑ training/optimizer.py — AdamW + warmup + cosine
☑ training/loss.py — SoftTargetCrossEntropy + class-weighted BCE
☑ training/trainer.py — full loop, grad accumulation, AMP
☑ configs/*.yaml — 4 datasets, all hyperparameters explicit
☑ scripts/record_model_hashes.py — SHA-256 logging
☑ scripts/pilot_finetune.py — 5-epoch sanity check, ≥65% top-1
☑ scripts/verify_datasets.py — integrity checks for all 4 datasets
☑ scripts/create_cub_val_split.py — stratified CUB val split
☑ model_hashes.txt — committed
☑ requirements.txt — pinned versions
```

## Phase 2 — Localization Metrics (L1–L4)

```
☑ L1 formal definition: mIoU at τ ∈ {0.25, 0.50, 0.75}
☑ L2 formal definition: PG = 1[argmax_p e_p ∈ M_GT], tie-breaking seeded
☑ L3 formal definition: EGT = Σ_{p∈M_GT} e⁺_p / Σ_p e⁺_p (positive-mass fraction)
☑ L4 formal definition: CalibGap = E[EGT|correct] − E[EGT|incorrect]
☑ LocalizationMetrics class — metrics/localization.py
☑ 12 unit tests — tests/test_localization.py — 12/12 passing
☑ BenchmarkRunner — metrics/runner.py
```

## Phase 2 — Robustness Metrics (R1–R3)

```
☑ R1 formal definition: MaxSens
☑ R2 formal definition: ModelRand
☑ R3 formal definition: HeadPerm (head permutation; not Adebayo's retraining test)
☑ RobustnessMetrics class — metrics/robustness.py
☑ randomise_model_weights(), randomise_classifier_labels()
☑ Theorems 2.3.A, 2.3.B, 2.3.C
☑ 16 unit tests — tests/test_robustness.py — 16/16 passing
☑ BenchmarkRunner extended — fully backward-compatible
```

## Phase 2 — Complexity Metrics (C1–C3)

```
☑ C1 formal definition: Gini coefficient, O(N log N), scale-invariant
☑ C2 formal definition: Shannon entropy of |e|/Σ|e|, norm ∈ [0,1]
☑ C3 formal definition: EMR at θ∈{0.50, 0.90, 0.95}; k*_90 cross-arch
☑ ComplexityMetrics class — metrics/complexity.py
☑ gini_coefficient() — standalone, numpy, O(N log N), zero-map convention
☑ attribution_entropy() — standalone, sum normalisation, H/log(N)
☑ effective_mass_ratio() — standalone, shared sort, multi-threshold
☑ normalise_attribution() — minmax / softmax / percentile
☑ ComplexityResult dataclass — to_dict() for CSV/W&B
☑ compute_batch(), compute_batch_as_dict(), aggregate()
☑ PyTorch vectorised batch ops: gini_batch_torch(), entropy_batch_torch(),
   emr_batch_torch()
☑ downsample_attribution() — bilinear align for cross-arch comparison
☑ plot_complexity_distributions() — PDF figure generator
☑ plot_complexity_vs_fidelity() — scatter against F1
☑ run_sanity_check() — validates ordering on 5 synthetic maps
☑ 45 unit tests — tests/test_complexity.py — 45/45 passing
```

## Phase 2 — Axiomatic Analysis

```
☑ Axioms A1–A4 formally defined with ViT-specific caveats
☑ 13×4 axiom satisfaction table (§2.5.2) — the 13 implemented metrics
☑ Theorem T1: all 13 metrics are scale-invariant ⇒ none detects A2 violations
☑ Theorem T2: EGT is unaffected by non-positive attribution (exact, positive-mass EGT)
☑ Theorem T3: rollout CLS→patch entries ≥ (1/2)^L · A_L[cls,p] > 0 (corrected bound)
☑ Theorem T4: IG satisfies A1/A2/A4 (E8 reference); exact Shapley unique for A1–A4
☑ Theorem T5: GradCAM satisfies A1 approximately via ReLU (proof sketch)
☑ Theorem T6: majorisation — C1/C2 strictly prefer A3 violations (Schur-convexity)
☑ AxiomVerifier class — metrics/axiom_verifier.py
☑ LinearPatchModel, XORInteractionModel — toy models for controlled tests
☑ AxiomTestResult dataclass — to_dict() for serialisation
☑ test_a1(), test_a2(), test_a3(), test_a4() — per-axiom empirical tests
☑ run_all() — 12 (C1–C3 × A1–A4) tests; 24 with MetricSuite (adds L1–L3; F/R/L4 are theoretical)
☑ build_satisfaction_table() — Markdown with ✓ / ∼† / ✗
☑ verify_completeness() — A2 check for one sample
☑ run_completeness_verification() — dataset-level A2 report
☑ generate_completeness_error_table() — LaTeX Table T3
☑ verify_dummy_axiom() — empirical A1 approximation via patch masking
☑ verify_rollout_dummy_violation() — empirical Theorem T3 check
☑ generate_axiom_satisfaction_heatmap() — Figure F1 PDF
☑ 20 unit tests — tests/test_axiom_verifier.py — 20 passing
☑ 7 PyTorch tests — tests/test_torch_axioms.py — 7 passing (1 CUDA skip)
☑ metrics/__init__.py updated — torch-free imports first
☑ pyproject.toml created — uv project (Python 3.13)
```

### Phase 3 — Task 3.1 Checklist
```

☑ BaseExplainer ABC — explain(x, target_class) → (P, P) tensor contract
☑ explain_batch() default loop; overridable for amortised methods
☑ UnsupportedArchitectureError — raised by E1/E2/E4 on Swin-B
☑ \_get_timm_blocks(), \_has_cls_token(), \_to_patch_grid() — shared helpers
☑ \_capture_attn_weights() — fused_attn-safe attention hook
☑ E1 RawAttentionExplainer — last-block CLS row, mean over heads (explainers/raw_attention.py)
☑ E2 AttentionRolloutExplainer — recursive A-hat product, drop-residual option (explainers/rollout.py)
☑ E3 GradCAMExplainer — gradient hooks, ReLU, bilinear upsample (explainers/gradcam.py)
☑ E4 CheferLRPExplainer — pure-PyTorch LRP, attn + mlp contrib layers (explainers/chefer_lrp.py)
☑ E5 RISEExplainer — 4000 float16 masks pre-generated; chunked batching (explainers/rise.py)
☑ E6 LIMEExplainer — patch-grid superpixels, ridge regression (explainers/lime.py)
☑ E7 DIMEExplainer — placeholder stub, NotImplementedError + BENCHMARK.md ref (explainers/dime.py)
☑ 26 unit tests passing + 1 documented skip — tests/test_explainers.py
☑ Full test suite: 125 passed, 2 skipped, 0 failed (2026-04-10)

```

### Phase 3 — Task 3.2 Checklist
```

☑ normalize_attribution(att_map, mode) — Guide Listing 3 canonical API
☑ normalize_batch(att_maps, mode) — strict (B,Hp,Wp) enforcer
☑ NormMode enum — 'minmax', 'percentile', 'softmax'
☑ AttributionNormError — raised on invalid mode or shape
☑ minmax: (att−min)/(max−min); degenerate → zeros
☑ percentile: clamp at 99th percentile then minmax
☑ softmax: numerically stable (max-shifted); sums to 1.0
☑ Batch (B,Hp,Wp): each sample normalised independently
☑ Output always float32; input accepts float16/32/64
☑ metrics/**init**.py updated — normalize_attribution in torch-free block
☑ 24 unit tests — tests/test_normalize.py — 24 passing
☑ Full suite: 149 passed, 2 skipped (CUDA), 0 failed (2026-04-10)

```

### Phase 3 — Task 3.3 Checklist
```

☑ Phase3Runner class — dataset × model × explainer matrix loop
☑ Checkpoint design — one .pkl per (dataset, model, explainer) combination
☑ Atomic write via .pkl.tmp → rename (corruption prevention)
☑ Resume — re-running skips existing .pkl automatically
☑ Seed reproducibility — torch.manual_seed(seed) per combination
☑ CLI entry point — python -m metrics.runner with all flags
☑ --dry-run, --list-checkpoints, --norm-mode, --patch-size flags
☑ Backward-compatible BenchmarkRunner preserved
☑ 36 unit tests — tests/test_runner.py — 36 passing
☑ Full suite: 185 passed, 2 skipped (CUDA), 0 failed (2026-04-10)

```

### Phase 3 — Task 3.4 Checklist
```

☑ S1 Random Baseline — run_s1_random_baseline()
☑ S2 Model Parameter Randomisation — run_s2_model_randomisation()
☑ S3 Label Permutation — run_s3_label_permutation()
☑ run_all_sanity_checks() — unified entry point
☑ SanityResult dataclass — JSON-serialisable
☑ 16 unit tests — tests/test_sanity.py — 16 passing
☑ Full suite: 201 passed, 2 skipped (CUDA), 0 failed (2026-04-10)

```

---

# Phase 4 — Analysis, Ablations & Theoretical Grounding

This section documents the experimental results constructed from the Phase 3 evaluation pipeline.

## 4.1 Inter-Metric Correlation

> **[PLACEHOLDER: Insert Inter-metric correlation heatmap here]**

*Summary of findings:*
- [PLACEHOLDER: Detail if metrics measure orthogonal properties or if any are highly correlated (|?| > 0.9)]
- [PLACEHOLDER: Factor Analysis latent variance breakdown]

## 4.2 Task-Metric Interaction

> **[PLACEHOLDER: Insert Kendall tau concordance matrix and Decision Tree]**

*Summary of findings:*
- [PLACEHOLDER: Discordances identified across datasets]
- [PLACEHOLDER: Practitioner decision tree recommendations]

## 4.3 Ablation Studies

| Ablation | Variable Changed | Effect Size (Cohen's d) | Key Observation |
|----------|------------------|-------------------------|-----------------|
| A1 | Token Resolution (patch vs CLS) | [PLACEHOLDER] | [PLACEHOLDER] |
| A2 | Layer Depth (last vs last-3 / all / rollout) | [PLACEHOLDER] | [PLACEHOLDER] |
| A3 | Masking (mean vs zero vs blur) + masking sensitivity | [PLACEHOLDER] | [PLACEHOLDER] |
| A4 | Pre-training (Sup vs MAE) | [PLACEHOLDER] | [PLACEHOLDER] |
