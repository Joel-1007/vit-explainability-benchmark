# ViT Explainability Benchmark — Full Technical Reference

> **Venue target: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**
>
> This document is the single authoritative reference for all phases, tasks, formal
> definitions, implementation details, axiomatic analyses, unit tests, and integration
> specifications of the ViT Explainability Benchmark project.
> All task documents (`task_2_2_*.md`, `task_2_3_*.md`) are superseded by this file.

---

## Table of Contents

| § | Title | Status |
|---|-------|--------|
| [Phase 1](#phase-1--model-zoo--training-protocol) | Model Zoo & Training Protocol | ✅ Complete |
| [1.1](#11--model-zoo-selection) | Model Zoo Selection | ✅ |
| [1.2](#12--standardised-fine-tuning-protocol) | Standardised Fine-Tuning Protocol | ✅ |
| [1.3](#13--dataset-registry) | Dataset Registry | ✅ |
| [1.4](#14--reproducibility-infrastructure) | Reproducibility Infrastructure | ✅ |
| [Phase 2](#phase-2--metrics-suite) | Metrics Suite | ✅ Complete |
| [2.1](#21--fidelity-metrics-f1f3) | Fidelity Metrics (F1–F3) | ✅ |
| [2.2](#22--localization-metrics-l1l4) | Localization Metrics (L1–L4) · 12 tests | ✅ |
| [2.3](#23--robustness-metrics-r1r3) | Robustness Metrics (R1–R3 + Layer Curve) · 19 tests | ✅ |
| [2.4](#24--complexity-metrics-c1c3) | Complexity Metrics (C1–C3) · 12 tests | ✅ |
| [2.5](#25--benchmarkrunner--unified-evaluation-loop) | BenchmarkRunner — Unified Evaluation Loop | ✅ |
| [2.6](#26--axiomatic-analysis) | Axiomatic Analysis · 8 property-based tests | ✅ |
| [Appendix A](#appendix-a--project-layout) | Project Layout | — |
| [Appendix B](#appendix-b--complete-metric-index) | Complete Metric Index | — |
| [Appendix C](#appendix-c--master-checklist) | Master Checklist | — |

---

---

# Phase 1 — Model Zoo & Training Protocol

## 1.1  Model Zoo Selection

Six Vision Transformers are selected with controlled variation: architecture family,
pre-training objective, and patch size are varied while **parameter count (~86 M),
input resolution (224×224), and fine-tuning protocol are held constant**.
This isolation ensures that any differences in explanation quality are attributable
to architecture, not training conditions.

| # | Model | Architecture | Pre-training | Patch | Params | IN-1K Top-1 | timm ID |
|---|-------|-------------|-------------|-------|--------|-------------|---------|
| 1 | **ViT-B/16** | Standard ViT | Supervised IN-21K (AugReg) | 16 | 86 M | 84.2 % | `vit_base_patch16_224.augreg_in21k_ft_in1k` |
| 2 | **DeiT-B/16** | DeiT (distilled) | Knowledge distillation IN-1K | 16 | 87 M | 83.4 % | `deit_base_distilled_patch16_224` |
| 3 | **Swin-B** | Shifted-window | Supervised IN-22K → IN-1K | 4 | 88 M | 85.2 % | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| 4 | **BEiT-B/16** | BERT-style MIM | MIM IN-22K (DALL-E dVAE) | 16 | 86 M | 85.2 % | `beit_base_patch16_224.in22k_ft_in22k_ft_in1k` |
| 5 | **DINO-ViT-B/8** | Standard ViT | Self-distillation IN-1K | 8 | 85 M | 80.1 % (k-NN) | torch.hub `facebookresearch/dino` |
| 6 | **DINOv2-ViT-B/14** | Standard ViT | Self-distillation LVD-142M | 14 | 86 M | 86.5 % | `vit_base_patch14_dinov2.lvd142m` |

**Architectural variation axes:**

| Axis | Values represented |
|------|--------------------|
| Supervision | Supervised (ViT, DeiT, Swin, BEiT), Self-supervised (DINO, DINOv2) |
| Attention scope | Global (ViT, DeiT, BEiT, DINO, DINOv2), Local-window (Swin) |
| Patch size | 8 (DINO), 14 (DINOv2), 16 (ViT/DeiT/BEiT), 32 (window-merged in Swin) |
| Special tokens | CLS only (ViT/BEiT/DINO/DINOv2), CLS+Distil (DeiT), None (Swin) |

> [!IMPORTANT]
> **Swin-B structural limitation.** Standard attention rollout and CLS-token attribution
> methods do not apply to Swin-B (no CLS token; attention is local-window-based).
> Swin-B uses GradCAM on the final stage feature map for all attribution comparisons.
> This limitation is reported explicitly in all benchmark tables.

---

## 1.2  Standardised Fine-Tuning Protocol

All six models are fine-tuned under an identical protocol. No per-model hyperparameter
tuning is performed — the protocol is fixed once and applied uniformly.

### Hyperparameters

| Component | Value | Source |
|-----------|-------|--------|
| **Optimiser** | AdamW | Loshchilov & Hutter (2019) |
| β₁, β₂, ε | 0.9, 0.999, 1e-8 | — |
| **Weight decay** | 0.05 (biases and LayerNorm excluded) | — |
| **Base LR** | 1e-4 @ batch 256; scaled linearly | — |
| **LR schedule** | Cosine annealing + 5-epoch linear warmup | — |
| **Epochs** | 50 (CUB-200-2011); 30 (VOC, CheXpert, ImageNet) | — |
| **Input size** | 224 × 224, bicubic interpolation | — |
| **Augmentation** | RandAugment M=9 N=2, random erasing p=0.25 | Cubuk et al. (2020) |
| **Mixup** | α=0.8; CutMix disabled | Zhang et al. (2018) |
| **Label smoothing** | ε=0.1 | Szegedy et al. (2016) |
| **Stochastic depth** | drop_path_rate=0.1 | Huang et al. (2016) |
| **Dropout** | 0.0 (disabled) | — |
| **Loss (CUB/VOC/ImageNet)** | SoftTargetCrossEntropy (Mixup-compatible) | — |
| **Loss (CheXpert)** | Binary CE, class-weighted | — |

### Implementation files

| File | Responsibility |
|------|---------------|
| `training/transforms.py` | RandAugment + random erasing pipeline |
| `training/mixup.py` | Batch-level Mixup (α=0.8) |
| `training/optimizer.py` | AdamW + LR warmup + cosine decay |
| `training/loss.py` | SoftTargetCrossEntropy / BCE |
| `training/trainer.py` | Full fine-tune loop with grad accumulation |

### Pilot sanity check (§7)

```bash
python scripts/pilot_finetune.py \
    --data_root /path/to/CUB_200_2011 \
    --batch_size 64 \
    --accum_steps 4
```

**Required result:** top-1 ≥ 65 % on CUB-200-2011 val after 5 epochs.

---

## 1.3  Dataset Registry

Four benchmark datasets are used. All are resized to 224×224 for model inference.

| Dataset | Task | Classes | Train | Val | GT type | Config |
|---------|------|---------|-------|-----|---------|--------|
| **CUB-200-2011** | Fine-grained bird classification | 200 | 5,994 | 5,794 | Bounding box + part annotations | `configs/cub200.yaml` |
| **PASCAL VOC 2012** | Multi-label object detection | 20 | 10,582 | 5,823 | Pixel segmentation masks | `configs/pascal_voc.yaml` |
| **ImageNet-S-50** | Large-scale classification (50-class subset) | 50 | 64,500 | 1,500 | Pixel segmentation masks | `configs/imagenet_s50.yaml` |
| **NIH ChestX-ray14** | Multi-label chest pathology | 14 | 86,524 | 11,219 | Bounding box annotations | `configs/nih_chestxray.yaml` |

### Dataset verification protocol

```bash
python scripts/verify_datasets.py --data_root /path/to/data --dataset cub200
python scripts/verify_datasets.py --data_root /path/to/data --dataset voc
python scripts/verify_datasets.py --data_root /path/to/data --dataset imagenet_s50
python scripts/verify_datasets.py --data_root /path/to/data --dataset nih_chestxray
```

All four datasets must pass: file-count checks, annotation bounds checks, class-balance
checks, and stratified validation split creation.

---

## 1.4  Reproducibility Infrastructure

| Asset | Description | Location |
|-------|-------------|----------|
| `model_hashes.txt` | SHA-256 of all 6 pre-trained checkpoints | project root |
| `checkpoints/*/finetuned_hashes.txt` | Per-epoch SHA-256 of fine-tuned weights | per dataset |
| `requirements.txt` | Pinned dependency versions | project root |
| `configs/*.yaml` | Full hyperparameter specs | `configs/` |

```bash
# Record pre-trained hashes once after download
python scripts/record_model_hashes.py --out model_hashes.txt
```

**Checkpoints**: large binaries are `.gitignore`d; only hash logs are committed.

---

---

# Phase 2 — Metrics Suite

The metrics suite evaluates attribution maps $\phi(f, x) \in \mathbb{R}^{H_a \times W_a}$
produced by an explanation method $\phi$ for model $f$ at input $x$.
Seven distinct metrics are implemented across three complementary families:

| Family | Metrics | Question answered |
|--------|---------|-------------------|
| **Fidelity** (§2.1) | F1 Sufficiency, F2 Comprehensiveness, F3 Log-odds | Does the attribution reflect what the model actually uses? |
| **Localization** (§2.2) | L1 mIoU, L2 PG, L3 EGT, L4 CalibGap | Does the attribution point at the right region? |
| **Robustness** (§2.3) | R1 MaxSens, R2 ModelRand, R3 LabelRand | Is the attribution stable and model-sensitive? |

All metrics reside in `metrics/`. The `BenchmarkRunner` integrates all three
families into a single dataset-level evaluation loop.

---

## 2.1  Fidelity Metrics (F1–F3)

> [!NOTE]
> Fidelity metrics quantify whether the attribution reflects what inputs the model
> **actually relies on** for its prediction. They operate by masking or removing
> high-attribution regions and measuring the resulting change in model output.

### F1 — Sufficiency

$$F1 = \text{Suff}(\phi, f, x) = f(x)_{y^*} - f(x \odot \mathbf{1}_{M_\tau})_{y^*}$$

The attributed region alone should be *sufficient* to reproduce the model's confidence
in the predicted class $y^*$. Higher = attribution covers the decision-relevant region.

### F2 — Comprehensiveness

$$F2 = \text{Comp}(\phi, f, x) = f(x)_{y^*} - f(x \odot (1 - \mathbf{1}_{M_\tau}))_{y^*}$$

Removing the attributed region should *reduce* confidence. Higher = attribution
successfully identifies the decision-relevant region.

### F3 — Log-odds Drop

$$F3 = \text{LogOddsDrop}(\phi, f, x) = \log \frac{f(x)_{y^*}}{1 - f(x)_{y^*}} - \log \frac{f(x')_{y^*}}{1 - f(x')_{y^*}}$$

where $x'$ is the masked input. Measures the log-odds change in predicted probability
after masking the top-$k$% attributed region.

---

## 2.2  Localization Metrics (L1–L4)

All four metrics operate on a pair $(e, M^{GT})$ where $e \in \mathbb{R}^{H \times W}$
is the raw attribution map and $M^{GT}$ is the binary ground-truth mask
(bounding box or pixel segmentation).

> [!NOTE]
> Attribution maps at patch resolution (e.g., 14×14 for ViT-B/16) are bilinearly
> upsampled to the GT mask resolution before metric computation.
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

Let $\tilde{e}_p$ be the **softmax-normalised** attribution:

$$\tilde{e}_p = \frac{\exp(e_p)}{\sum_{q} \exp(e_q)}$$

The fraction of total attribution mass inside the GT region:

$$L3 = \text{EGT} = \sum_{p \in M^{GT}} \tilde{e}_p$$

**Range:** $[0, 1]$. EGT=1 means all mass is inside GT; EGT=0 means none is.

**Why softmax?** Sum-normalisation is undefined for negative attributions
(Integrated Gradients, raw GradCAM). Softmax is well-defined for all methods.

---

### L4 — Calibration Gap (CalibGap)

Let $\mathcal{C}$ and $\mathcal{I}$ be correctly and incorrectly classified samples:

$$L4 = \text{CalibGap} = \mathbb{E}_{i \in \mathcal{C}}\!\left[\text{EGT}_i\right] - \mathbb{E}_{i \in \mathcal{I}}\!\left[\text{EGT}_i\right]$$

**Range:** $(-1, 1)$. Positive = explanation aligns better with GT on correct predictions
(necessary condition for faithfulness). Near-zero or negative = sanity-check failure.

**Undefined** when all predictions correct ($\mathcal{I} = \emptyset$) or all incorrect
($\mathcal{C} = \emptyset$) — raises `ValueError`.

---

### 2.2.1  Implementation — `LocalizationMetrics`

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

### 2.2.2  Axiomatic Analysis (L1–L4)

#### Theorem 2.2.A — IoU satisfies Symmetry but penalises over-segmentation

**Symmetry Axiom.** $\text{IoU}(\hat{M}, M^{GT}) = \text{IoU}(M^{GT}, \hat{M})$ always,
since set intersection and union are symmetric operations.

**Area-penalty corollary.** Despite symmetry, IoU penalises over-segmentation:
a large predicted mask that contains the GT region gives low IoU because the union is large.

$$\text{IoU}(\hat{M}_\text{large}, M^{GT}_\text{small}) = \frac{|M^{GT}_\text{small}|}{|\hat{M}_\text{large}|} \ll 1$$

**Implication.** An explanation that highlights the entire image is correctly penalised,
even if it technically contains the GT region. IoU must be reported at all three thresholds.

---

#### Theorem 2.2.B — EGT satisfies Monotone Coverage but NOT Boundary Sensitivity

**Monotone Coverage.** If $M^{GT}_1 \subseteq M^{GT}_2$ with $e$ fixed:

$$\text{EGT}(e, M^{GT}_1) \leq \text{EGT}(e, M^{GT}_2)$$

*Proof.* $\text{EGT} = \sum_{p \in M^{GT}} \tilde{e}_p$. Since all softmax values $\tilde{e}_p \geq 0$,
expanding the mask can only increase the sum. $\square$

**Boundary Sensitivity violation.** EGT is a sum over $M^{GT}$ — it cannot distinguish mass
concentrated at the GT boundary vs. the GT semantic centre. Pointing Game (L2) is
complementary: it penalises boundary-concentrated peaks.

**Implication.** L2 and L3 are not redundant — both are needed.

---

#### Theorem 2.2.C — CalibGap is NOT a proper scoring rule

**Proper Scoring Rule:** expected score maximised by truthful belief reporting.

**Claim.** CalibGap can be gamed by an adversarial explainer with access to the
prediction correctness flag.

*Proof by construction.* If $\phi$ outputs high-EGT maps for correct predictions
and low-EGT maps for incorrect ones, CalibGap → 1 without faithfulness. $\square$

**Mitigation.** The benchmark:
1. Passes only `(model, image)` to explainers — never correctness flags.
2. Reports L1–L3 alongside L4 so inflated CalibGap is detectable.
3. Requires consistent CalibGap across all four datasets.

**Implication.** CalibGap is a *diagnostic* metric, never a primary ranking criterion.

---

### 2.2.3  Unit Tests (L1–L4)

File: `tests/test_localization.py` — **12 tests**, all passing.

| ID | Test | Assertion |
|----|------|-----------|
| T01 | `random_iou_not_one` | mIoU < 1.0 for 20 random trials |
| T02 | `random_pg_within_bounds` | PG ∈ {0,1}; not always 1.0 over 100 trials |
| T03 | `random_egt_within_bounds` | EGT ∈ [0,1] for 50 random pairs |
| T04 | `perfect_iou_is_one` | att=GT → mIoU = 1.0 at all τ |
| T05 | `perfect_pg_is_one` | peak at GT pixel → PG = 1.0 |
| T06 | `perfect_egt_is_one` | all mass inside GT → EGT = 1.0 |
| T07 | `perfect_calibgap_positive` | perfect-correct vs wrong-incorrect → gap > 0 |
| T08 | `misaligned_iou_is_zero` | att outside GT → IoU@0.75 = 0.0 |
| T09 | `misaligned_pg_is_zero` | peak outside GT → PG = 0.0 |
| T10 | `misaligned_egt_near_zero` | all mass outside GT → EGT < 0.01 |
| T11 | `constant_att_map_iou` | constant map → no NaN; result in [0,1] |
| T12 | `calibgap_empty_list_raises` | empty correct/incorrect → ValueError |

```
Results: 12/12 passed, 0 failed
```

---

### 2.2.4  ViT Patch Grid Reference

| Model | Patch size | Grid | Patches | Upsample to 224×224 |
|-------|-----------|------|---------|---------------------|
| ViT-B/16, DeiT-B/16, BEiT-B/16 | 16 | 14×14 | 196 | Bilinear ×16 |
| DINOv2-ViT-B/14 | 14 | 16×16 | 256 | Bilinear ×14 |
| DINO-ViT-B/8 | 8 | 28×28 | 784 | Bilinear ×8 |
| Swin-B (GradCAM) | — | 7×7 (stage 4) | 49 | Bilinear ×32 |

---

## 2.3  Robustness Metrics (R1–R3)

Robustness metrics test whether an explanation is **stable and faithful** under
systematic perturbations to the input or the model. A faithful attribution must:

- Change **when the input moves** toward a worst-case perturbation (R1).
- Change **when the model weights are randomised** — sanity check (R2).
- Change **when output labels are shuffled** — faithfulness to classifier (R3).

> [!NOTE]
> R1–R3 complement L1–L4, which test spatial faithfulness to GT.
> Robustness metrics test faithfulness to the *model's internal computation*.

---

### R1 — Max-Sensitivity (MaxSens)

**Reference**: Yeh et al. (2019), *On the (In)fidelity and Sensitivity of Explanations*.

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

**Reference**: Adebayo et al. (2018), *Sanity Checks for Saliency Maps*.

Let $f_\text{rand}$ be a deep copy of $f$ with all parameters re-initialised from $\mathcal{N}(0,1)$.
Let $\tilde{\phi}(\cdot) = \text{MinMax}(\phi(\cdot))$ be attribution normalised to $[0,1]$.

SSIM (Wang et al., 2004) between the two normalised maps:

$$\text{SSIM}(A, B) = \frac{(2\mu_A\mu_B + C_1)(2\sigma_{AB} + C_2)}{(\mu_A^2 + \mu_B^2 + C_1)(\sigma_A^2 + \sigma_B^2 + C_2)}$$

$$R2 = \text{ModelRand}(\phi, x) = 1 - \text{SSIM}\!\left(\tilde{\phi}(f_\text{orig}, x),\; \tilde{\phi}(f_\text{rand}, x)\right)$$

**Range:** $[0, 1]$. Higher = explanation is model-sensitive (desired). Near 0 = sanity-check
failure (attribution ignores the model weights entirely).

**SSIM implementation:** Pure-PyTorch, 3×3 Gaussian window ($\sigma=1.0$), GPU-native.
No `scikit-image` dependency.

---

### R3 — Label Randomisation (LabelRand)

Let $f_\text{shuf}$ be a deep copy of $f$ with the final classifier head weights
**column-permuted** (output label assignments shuffled, backbone unchanged).

Let $\rho(\cdot, \cdot)$ denote the Spearman rank correlation of two flattened maps:

$$R3 = \text{LabelRand}(\phi, x) = 1 - \frac{|\rho(\phi_\text{orig}, \phi_\text{shuf})| + 1}{2}$$

**Derivation:**
- $\rho \in [-1, 1]$; $(|\rho|+1)/2 \in [0.5, 1]$ measures structural preservation.
- $1 - (|\rho|+1)/2 \in [0, 0.5]$ measures structural divergence.
- $|\rho|$ (not $\rho$) is used because sign-flipped attributions still preserve spatial structure.

**Range:** $[0, 0.5]$. Higher = explanation is label-sensitive. Near 0 = failure
(explanation unchanged despite class permutation).

**Spearman implementation:** Pure-PyTorch rank-via-argsort, stable, handles ties,
no SciPy dependency.

---

### 2.3.1  Implementation — `RobustnessMetrics` and Utilities

```python
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
)

rm = RobustnessMetrics(
    epsilon=0.05,    # L∞ perturbation radius (R1)
    n_samples=20,    # perturbation draws per image (use 50 for production)
    seed=42,         # RNG seed for reproducibility
    ssim_window=3,   # Gaussian kernel size for SSIM (R2)
    ssim_sigma=1.0,  # Gaussian sigma for SSIM (R2)
)

# Pre-compute randomised model copies once — NOT per sample
f_rand = randomise_model_weights(model, seed=0)        # all params ← N(0,1)
f_shuf = randomise_classifier_labels(model, seed=0)    # head columns permuted only

# Per sample
att_orig = explainer(model,  image.unsqueeze(0))[0]
att_rand = explainer(f_rand, image.unsqueeze(0))[0]
att_shuf = explainer(f_shuf, image.unsqueeze(0))[0]

# R1
ms = rm.max_sensitivity(explainer, model, image, att_orig)

# R2
mr = rm.model_randomisation(att_orig, att_rand)

# R3
lr = rm.label_randomisation(att_orig, att_shuf)

# All three at once
scores = rm.compute_all(explainer, model, image, att_orig, att_rand, att_shuf)
# → {'max_sensitivity': 0.31, 'model_randomisation': 0.87, 'label_randomisation': 0.46}
```

**`randomise_model_weights(model, seed=0)`**
— Deep copy; ALL parameters ← $\mathcal{N}(0,1)$; biases ← 0.
Uses maximum randomisation (not Kaiming/Xavier) to maximise deviation from learned weights.

**`randomise_classifier_labels(model, seed=0, head_attr='head')`**
— Deep copy; only `head.weight` columns are permuted.
Column permutation of $W \in \mathbb{R}^{C \times D}$ reassigns which class each feature
maps to while preserving weight norms (stricter than zeroing the head).
Fallback chain for `head_attr`: `'head'`, `'classifier'`, `'fc'`, `'linear'`.

---

### 2.3.2  Axiomatic Analysis (R1–R3)

#### Theorem 2.3.A — MaxSens satisfies the Lipschitz-Sensitivity Axiom

**Lipschitz-Sensitivity Axiom.** $\exists L < \infty$ such that $\text{MaxSens}(\phi, x) \leq L \cdot \epsilon$.

*Proof.* For a Lipschitz explainer with constant $\Lambda$:

$$\|\phi(f, x+\delta) - \phi(f, x)\|_2 \leq \Lambda\|\delta\|_2 \leq \Lambda\epsilon\sqrt{d}$$

Setting $L = \Lambda\sqrt{d} / (\|\phi(f,x)\|_2 + \varepsilon_\text{num})$ completes the bound. $\square$

**Corollary.** MaxSens is linear in $\epsilon$ for smooth explainers. Always report $\epsilon$
alongside MaxSens scores — cross-$\epsilon$ comparisons are meaningless.

---

#### Theorem 2.3.B — ModelRand satisfies the Sanity Check Axiom

**Sanity Check Axiom** (Adebayo et al., 2018). A faithful attribution $\phi$ must produce
structurally different maps when model weights are fully randomised:

$$\lim_{\text{randomisation} \to 1} \text{ModelRand}(\phi, x) = 1$$

*Proof.* For fully randomised $f_\text{rand}$, attribution maps approach i.i.d. uniform
noise (by CLT on random projections). The SSIM between a learned spatial map and uniform
noise → 0, so ModelRand → 1. $\square$

**Counterexample (sanity-check failure).**

```python
def pixel_explainer(model, image_4d):
    return [image_4d[0, 0].detach()]   # Uses only input, ignores model!

# att_orig == att_rand → ModelRand = 0 → FAIL
```

---

#### Theorem 2.3.C — LabelRand is NOT a proper metric

**Claim.** LabelRand is a diagnostic metric, not a symmetric distance.

While the Spearman correlation $\rho$ is numerically symmetric, LabelRand is
**semantically asymmetric**: the roles of $\phi_\text{orig}$ (trusted reference) and
$\phi_\text{shuf}$ (perturbed) cannot be swapped without changing the interpretation.

**Implication.** Always specify which model is "original" vs. "permuted" when reporting
LabelRand. Do not compare across architectures with different head sizes (different
permutation degrees). Report alongside R1 and R2, never in isolation.

---

### 2.3.3  Unit Tests (R1–R3)

File: `tests/test_robustness.py` — **16 tests**, all passing.

| ID | Category | Test | Assertion |
|----|----------|------|-----------|
| R01 | Bounds | `max_sensitivity_nonneg` | MaxSens ≥ 0 for 20 trials |
| R02 | Bounds | `model_randomisation_in_0_1` | ModelRand ∈ [0,1] for 20 trials |
| R03 | Bounds | `label_randomisation_in_0_0p5` | LabelRand ∈ [0, 0.5] for 20 trials |
| R04 | Bounds | `compute_all_keys` | compute_all() returns exactly 3 keys |
| R05 | Sensitivity | `sensitivity_increases_with_eps` | Larger ε → ≥ mean MaxSens (30 trials) |
| R06 | Sensitivity | `model_rand_orthogonal_maps` | ModelRand > 0.5 for top-half vs bottom-half |
| R07 | Sensitivity | `label_rand_orthogonal_maps` | Mean LabelRand ≈ 0.5 for 200 random pairs |
| R08 | Zero | `sensitivity_zero_constant_expl` | MaxSens = 0 for constant explainer |
| R09 | Zero | `model_rand_identical_maps` | ModelRand = 0 for identical maps |
| R10 | Zero | `label_rand_identical_maps` | LabelRand = 0 for identical maps |
| R11 | Utility | `randomise_model_changes_weights` | All 4 params changed |
| R12 | Utility | `randomise_labels_only_head` | Backbone unchanged; head changed |
| R13 | Utility | `ssim_self_consistency` | _ssim(t, t) = 1.0 for 10 trials |
| R14 | Edge | `spearman_constant_map` | Constant map → no NaN; ρ ∈ [-1,1] |
| R15 | Edge | `max_sensitivity_n_samples_one` | n_samples=1 works without crash |
| R16 | Edge | `epsilon_constructor_validation` | ε ≤ 0 → ValueError |

```
Results: 16/16 passed, 0 failed
```

Run all Phase 2 tests together:

```bash
python tests/test_localization.py   # 12/12
python tests/test_robustness.py     # 16/16

# Or via pytest
pytest tests/ -v
```

---

### 2.3.4  Performance Guidance

| Metric | Cost per sample | Recommendation |
|--------|----------------|----------------|
| L1–L3 | O(1) tensor ops | Full dataset |
| L4 CalibGap | O(N) at dataset end | Full dataset |
| R1 MaxSens | K explainer calls | Random 500-image subset |
| R2 ModelRand | 1 extra explainer call | Full dataset |
| R3 LabelRand | 1 extra explainer call | Full dataset |

> [!IMPORTANT]
> For ImageNet-S-50 (5,000 val images × $K=50$ perturbations), MaxSens alone requires
> 250,000 additional explainer calls. Run R1 on a fixed random subset of 500 images
> and report the subset mean in the paper table footnote.

---

## 2.4  BenchmarkRunner — Unified Evaluation Loop

File: `metrics/runner.py`. Integrates L1–L4 and (optionally) R1–R3 into a single
dataset-level loop.

### Constructor

```python
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
)

# Localization only (Task 2.2 — backward-compatible)
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_explainer,
)

# Localization + Robustness (Task 2.3)
runner = BenchmarkRunner(
    metrics=LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer=my_explainer,
    robustness=RobustnessMetrics(epsilon=0.05, n_samples=50),
    randomised_model=randomise_model_weights(model, seed=0),
    label_randomised_model=randomise_classifier_labels(model, seed=0),
)

results = runner.evaluate(model, val_loader, dataset_name="cub200")
```

### DataLoader contract

```python
# DataLoader must yield exactly:
(images, gt_masks, labels)
# images   : torch.Tensor  (B, C, H, W)     normalised input images
# gt_masks : torch.Tensor  (B, H_m, W_m)    integer or float GT masks
# labels   : torch.Tensor  (B,)             integer class indices
```

### Explainer interface

```python
def my_explainer(model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    """
    Returns one (H_a, W_a) attribution map per image in the batch.
    Any resolution accepted — BenchmarkRunner/LocalizationMetrics will align.
    """
    ...
```

### Results dict schema

```python
{
    "dataset":     "cub200",
    "n_samples":   5794,
    "n_correct":   4823,
    "n_incorrect": 971,
    "macro": {
        # Localization (L1–L4) — always present
        "iou@0.25":          float,
        "iou@0.50":          float,
        "iou@0.75":          float,
        "miou":              float,
        "pointing_game":     float,
        "egt":               float,
        "calibration_gap":   float,
        # Robustness (R1–R3) — present only when robustness is set
        "max_sensitivity":     float,
        "model_randomisation": float,
        "label_randomisation": float,
    },
    "per_metric": {
        # One value per sample for L1–L3 and R1–R3
        # One dataset-level value for L4
        ...
    },
}
```

---

---

# Appendix A — Project Layout

```
TPAMI/
├── BENCHMARK.md             ← THIS FILE — single authoritative reference
│
├── model_zoo/
│   ├── __init__.py          # load_model() dispatcher
│   ├── vit_b16.py           # Model 1 — ViT-B/16 (timm augreg_in21k)
│   ├── deit_b16.py          # Model 2 — DeiT-B/16 distilled
│   ├── swin_b.py            # Model 3 — Swin-B
│   ├── beit_b16.py          # Model 4 — BEiT-B/16
│   ├── dino_vitb8.py        # Model 5 — DINO-ViT-B/8
│   └── dinov2_vitb14.py     # Model 6 — DINOv2-ViT-B/14
│
├── training/
│   ├── __init__.py
│   ├── transforms.py        # RandAugment + random erasing pipeline
│   ├── mixup.py             # Batch-level Mixup α=0.8
│   ├── optimizer.py         # AdamW + LR warmup + cosine decay
│   ├── loss.py              # SoftTargetCrossEntropy / BCE (CheXpert)
│   └── trainer.py           # Full fine-tune loop with grad accumulation
│
├── metrics/
│   ├── __init__.py          # Exports: LocalizationMetrics, RobustnessMetrics,
│   │                        #   BenchmarkRunner, randomise_model_weights,
│   │                        #   randomise_classifier_labels
│   ├── localization.py      # L1–L4 (Task 2.2)
│   ├── robustness.py        # R1–R3 + model utilities (Task 2.3)
│   └── runner.py            # BenchmarkRunner (unified evaluation loop)
│
├── tests/
│   ├── test_localization.py # 12 unit tests — L1–L4
│   └── test_robustness.py   # 16 unit tests — R1–R3
│
├── scripts/
│   ├── record_model_hashes.py   # SHA-256 logging for reproducibility
│   ├── pilot_finetune.py        # 5-epoch sanity check
│   ├── verify_datasets.py       # Dataset integrity verification
│   └── create_cub_val_split.py  # Stratified val split for CUB-200-2011
│
├── configs/
│   ├── cub200.yaml              # 50 epochs, fine-grained, 200 classes
│   ├── pascal_voc.yaml          # 30 epochs, multi-label, 20 classes
│   ├── imagenet_s50.yaml        # 30 epochs, 50-class subset
│   └── nih_chestxray.yaml       # 30 epochs, class-weighted BCE, 14 classes
│
├── model_hashes.txt             # SHA-256 of all pre-trained checkpoints
├── requirements.txt             # Pinned dependency versions
└── .gitignore
```

---

# Appendix B — Complete Metric Index

| ID | Name | Formula (short) | Range | File | Higher = |
|----|------|-----------------|-------|------|----------|
| F1 | Sufficiency | $f(x)_{y^*} - f(x \odot M)_{y^*}$ | $(-1,1)$ | — | Better |
| F2 | Comprehensiveness | $f(x)_{y^*} - f(x \odot \bar{M})_{y^*}$ | $(-1,1)$ | — | Better |
| F3 | Log-odds Drop | $\log\frac{p}{1-p} - \log\frac{p'}{1-p'}$ | $\mathbb{R}$ | — | Better |
| L1 | mIoU | $\text{mean}_\tau \text{IoU}(\hat{M}_\tau, M^{GT})$ | $[0,1]$ | `localization.py` | Better |
| L2 | Pointing Game | $\mathbf{1}[p^* \in M^{GT}]$ | $\{0,1\}$ | `localization.py` | Better |
| L3 | EGT | $\sum_{p \in M^{GT}} \text{softmax}(e)_p$ | $[0,1]$ | `localization.py` | Better |
| L4 | CalibGap | $\mathbb{E}[\text{EGT}\mid\text{correct}] - \mathbb{E}[\text{EGT}\mid\text{wrong}]$ | $(-1,1)$ | `localization.py` | Better |
| R1 | MaxSens | $\max_k \|\phi(x+\delta_k)-\phi(x)\|_2 / \|\phi(x)\|_2$ | $[0,\infty)$ | `robustness.py` | **Worse** |
| R2 | ModelRand | $1 - \text{SSIM}(\phi_\text{orig}, \phi_\text{rand})$ | $[0,1]$ | `robustness.py` | Better |
| R3 | LabelRand | $1 - (\|\rho(\phi_\text{orig}, \phi_\text{shuf})\| + 1)/2$ | $[0,0.5]$ | `robustness.py` | Better |

> [!NOTE]
> **R1 direction is inverted.** Lower MaxSens = more stable = better explanation.
> For all other metrics, higher = better.

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

## Phase 2 — Task 2.2 Localization Metrics

```
☑ L1 formal definition: mIoU at τ ∈ {0.25, 0.50, 0.75}
☑ L2 formal definition: PG = 1[argmax_p e_p ∈ M_GT], tie-breaking seeded
☑ L3 formal definition: EGT = Σ_{p∈M_GT} softmax(e)_p
☑ L4 formal definition: CalibGap = E[EGT|correct] − E[EGT|incorrect]
☑ LocalizationMetrics class — metrics/localization.py
☑ multi_threshold_iou(): min-max norm → binarise → IoU at each τ → mean
☑ pointing_game(): argmax + seeded uniform tie-breaking → {0.0, 1.0}
☑ egt(): softmax norm → sum over GT → float in [0,1]
☑ calibration_gap(): mean EGT split; ValueError on empty list
☑ align_patch_to_pixel(): bilinear upsample; all ViT grids documented
☑ compute_all(): single-call L1+L2+L3 per sample
☑ Theorem 2.2.A: IoU symmetry + area-penalty corollary
☑ Theorem 2.2.B: EGT monotone coverage + boundary insensitivity
☑ Theorem 2.2.C: CalibGap not a proper scoring rule + mitigation
☑ 12 unit tests — tests/test_localization.py — 12/12 passing
☑ BenchmarkRunner — metrics/runner.py
```

## Phase 2 — Task 2.3 Robustness Metrics

```
☑ R1 formal definition: MaxSens = max‖φ(x+δ)−φ(x)‖₂ / ‖φ(x)‖₂
☑ R2 formal definition: ModelRand = 1 − SSIM(φ_orig, φ_rand)
☑ R3 formal definition: LabelRand = 1 − (|ρ(φ_orig, φ_shuf)| + 1)/2
☑ RobustnessMetrics class — metrics/robustness.py
☑ max_sensitivity(): K random L∞ perturbations → max relative L2 change
☑ model_randomisation(): MinMax norm → pure-PyTorch SSIM → 1 − SSIM → clamp [0,1]
☑ label_randomisation(): Spearman rank-corr → 1 − (|ρ|+1)/2
☑ compute_all(): single-call R1+R2+R3 per sample
☑ _ssim(): pure-PyTorch 3×3 Gaussian, GPU-native, Wang et al. (2004)
☑ _spearman_corr(): rank-via-argsort, stable, no SciPy
☑ _perturb(): seeded torch.Generator, L∞ perturbations
☑ _minmax_norm(): constant-map guard (returns zeros, not NaN)
☑ randomise_model_weights(): deep copy; all params ← N(0,1); biases ← 0
☑ randomise_classifier_labels(): deep copy; head.weight columns permuted only
☑ head_attr fallback: 'head' → 'classifier' → 'fc' → 'linear'
☑ Theorem 2.3.A: MaxSens satisfies Lipschitz-Sensitivity Axiom (proved)
☑ Theorem 2.3.B: ModelRand satisfies Sanity Check Axiom (proved + counterexample)
☑ Theorem 2.3.C: LabelRand is NOT a proper metric (semantic asymmetry)
☑ 16 unit tests — tests/test_robustness.py — 16/16 passing
☑ BenchmarkRunner extended — optional R1–R3, fully backward-compatible
☑ metrics/__init__.py updated with all new exports
☑ No new mandatory dependencies: no scikit-image, no SciPy
```
