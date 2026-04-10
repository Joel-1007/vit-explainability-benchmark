# ViT Explainability Benchmark — Full Technical Reference

> **Venue target: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**
>
> This document is the single authoritative reference for all phases, formal
> definitions, implementation details, axiomatic analyses, unit tests, and integration
> specifications of the ViT Explainability Benchmark project.

---

## Table of Contents

| §                                                 | Title                             | Status      |
| ------------------------------------------------- | --------------------------------- | ----------- |
| [Phase 1](#phase-1--model-zoo--training-protocol) | Model Zoo & Training Protocol     | ✅ Complete |
| [1.1](#11--model-zoo-selection)                   | Model Zoo Selection               | ✅          |
| [1.2](#12--standardised-fine-tuning-protocol)     | Standardised Fine-Tuning Protocol | ✅          |
| [1.3](#13--dataset-registry)                      | Dataset Registry                  | ✅          |
| [1.4](#14--reproducibility-infrastructure)        | Reproducibility Infrastructure    | ✅          |
| [Phase 2](#phase-2--metrics-suite)                | Metrics Suite                     | ✅ Complete |
| [2.1](#21--fidelity-metrics-f1f3)                 | Fidelity Metrics (F1–F3)          | ✅          |
| [2.2](#22--localization-metrics-l1l4)             | Localization Metrics (L1–L4)      | ✅          |
| [2.3](#23--robustness-metrics-r1r3)               | Robustness Metrics (R1–R3)        | ✅          |
| [2.4](#24--complexity-metrics-c1c3)               | Complexity Metrics (C1–C3)        | ✅          |
| [2.5](#25--axiomatic-analysis)                    | Axiomatic Analysis                | ✅          |
| [Phase 3](#phase-3--baseline-evaluation-pipeline) | Baseline Evaluation Pipeline      | 🔄 In Progress |
| [3.1](#31--standardised-explainer-interface)      | Standardised Explainer Interface  | ✅          |
| [3.2](#32--attribution-normalisation)             | Attribution Normalisation         | ✅          |
| [3.3](#33--benchmarkrunner)                       | BenchmarkRunner                   | ⬜ Planned  |
| [Appendix A](#appendix-a--project-layout)         | Project Layout                    | —           |
| [Appendix B](#appendix-b--complete-metric-index)  | Complete Metric Index             | —           |
| [Appendix C](#appendix-c--master-checklist)       | Master Checklist                  | —           |

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
| 1   | **ViT-B/16**        | Standard ViT     | Supervised IN-21K (AugReg)   | 16    | 86 M   | 84.2 %        | `vit_base_patch16_224.augreg_in21k_ft_in1k`     |
| 2   | **DeiT-B/16**       | DeiT (distilled) | Knowledge distillation IN-1K | 16    | 87 M   | 83.4 %        | `deit_base_distilled_patch16_224`               |
| 3   | **Swin-B**          | Shifted-window   | Supervised IN-22K → IN-1K    | 4     | 88 M   | 85.2 %        | `swin_base_patch4_window7_224.ms_in22k_ft_in1k` |
| 4   | **BEiT-B/16**       | BERT-style MIM   | MIM IN-22K (DALL-E dVAE)     | 16    | 86 M   | 85.2 %        | `beit_base_patch16_224.in22k_ft_in22k_ft_in1k`  |
| 5   | **DINO-ViT-B/8**    | Standard ViT     | Self-distillation IN-1K      | 8     | 85 M   | 80.1 % (k-NN) | torch.hub `facebookresearch/dino`               |
| 6   | **DINOv2-ViT-B/14** | Standard ViT     | Self-distillation LVD-142M   | 14    | 86 M   | 86.5 %        | `vit_base_patch14_dinov2.lvd142m`               |

**Architectural variation axes:**

| Axis            | Values represented                                                    |
| --------------- | --------------------------------------------------------------------- |
| Supervision     | Supervised (ViT, DeiT, Swin, BEiT), Self-supervised (DINO, DINOv2)    |
| Attention scope | Global (ViT, DeiT, BEiT, DINO, DINOv2), Local-window (Swin)           |
| Patch size      | 8 (DINO), 14 (DINOv2), 16 (ViT/DeiT/BEiT), 32 (window-merged in Swin) |
| Special tokens  | CLS only (ViT/BEiT/DINO/DINOv2), CLS+Distil (DeiT), None (Swin)       |

> [!IMPORTANT]
> **Swin-B structural limitation.** Standard attention rollout and CLS-token attribution
> methods do not apply to Swin-B (no CLS token; attention is local-window-based).
> Swin-B uses GradCAM on the final stage feature map for all attribution comparisons.
> This limitation is reported explicitly in all benchmark tables.

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
| **Epochs**                  | 50 (CUB-200-2011); 30 (VOC, CheXpert, ImageNet) | —                          |
| **Input size**              | 224 × 224, bicubic interpolation                | —                          |
| **Augmentation**            | RandAugment M=9 N=2, random erasing p=0.25      | Cubuk et al. (2020)        |
| **Mixup**                   | α=0.8; CutMix disabled                          | Zhang et al. (2018)        |
| **Label smoothing**         | ε=0.1                                           | Szegedy et al. (2016)      |
| **Stochastic depth**        | drop_path_rate=0.1                              | Huang et al. (2016)        |
| **Dropout**                 | 0.0 (disabled)                                  | —                          |
| **Loss (CUB/VOC/ImageNet)** | SoftTargetCrossEntropy (Mixup-compatible)       | —                          |
| **Loss (CheXpert)**         | Binary CE, class-weighted                       | —                          |

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

Four benchmark datasets are used. All are resized to 224×224 for model inference.

| Dataset              | Task                                         | Classes | Train  | Val    | GT type                         | Config                       |
| -------------------- | -------------------------------------------- | ------- | ------ | ------ | ------------------------------- | ---------------------------- |
| **CUB-200-2011**     | Fine-grained bird classification             | 200     | 5,994  | 5,794  | Bounding box + part annotations | `configs/cub200.yaml`        |
| **PASCAL VOC 2012**  | Multi-label object detection                 | 20      | 10,582 | 5,823  | Pixel segmentation masks        | `configs/pascal_voc.yaml`    |
| **ImageNet-S-50**    | Large-scale classification (50-class subset) | 50      | 64,500 | 1,500  | Pixel segmentation masks        | `configs/imagenet_s50.yaml`  |
| **NIH ChestX-ray14** | Multi-label chest pathology                  | 14      | 86,524 | 11,219 | Bounding box annotations        | `configs/nih_chestxray.yaml` |

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

---

---

# Phase 2 — Metrics Suite

The metrics suite evaluates attribution maps $\phi(f, x) \in \mathbb{R}^{H_a \times W_a}$
produced by an explanation method $\phi$ for model $f$ at input $x$.
Seven distinct metrics are implemented across three complementary families:

| Family                  | Metrics                                           | Question answered                                          |
| ----------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| **Fidelity** (§2.1)     | F1 Sufficiency, F2 Comprehensiveness, F3 Log-odds | Does the attribution reflect what the model actually uses? |
| **Localization** (§2.2) | L1 mIoU, L2 PG, L3 EGT, L4 CalibGap               | Does the attribution point at the right region?            |
| **Robustness** (§2.3)   | R1 MaxSens, R2 ModelRand, R3 LabelRand            | Is the attribution stable and model-sensitive?             |

All metrics reside in `metrics/`. The `BenchmarkRunner` integrates all three
families into a single dataset-level evaluation loop.

---

## 2.1 Fidelity Metrics (F1–F3)

> [!NOTE]
> Fidelity metrics quantify whether the attribution reflects what inputs the model
> **actually relies on** for its prediction. They operate by masking or removing
> high-attribution regions and measuring the resulting change in model output.

### F1 — Sufficiency

$$F1 = \text{Suff}(\phi, f, x) = f(x)_{y^*} - f(x \odot \mathbf{1}_{M_\tau})_{y^*}$$

The attributed region alone should be _sufficient_ to reproduce the model's confidence
in the predicted class $y^*$. Higher = attribution covers the decision-relevant region.

### F2 — Comprehensiveness

$$F2 = \text{Comp}(\phi, f, x) = f(x)_{y^*} - f(x \odot (1 - \mathbf{1}_{M_\tau}))_{y^*}$$

Removing the attributed region should _reduce_ confidence. Higher = attribution
successfully identifies the decision-relevant region.

### F3 — Log-odds Drop

$$F3 = \text{LogOddsDrop}(\phi, f, x) = \log \frac{f(x)_{y^*}}{1 - f(x)_{y^*}} - \log \frac{f(x')_{y^*}}{1 - f(x')_{y^*}}$$

where $x'$ is the masked input. Measures the log-odds change in predicted probability
after masking the top-$k$% attributed region.

---

### 2.1.1 Implementation — `FidelityMetrics`

```python
from metrics.fidelity import FidelityMetrics

fm = FidelityMetrics(mask_mode="zero", k_fractions=(0.1, 0.2, 0.5))

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

| ID  | Test | Assertion |
| --- | --- | --- |
| F01 | `test_generate_mask` | Masks top `k_frac` elements correctly. |
| F02 | `test_apply_mask_zero` | Selected regions retain content, others zeroed. |
| F03 | `test_apply_mask_mean` | Drops selected regions, interpolates with baseline mean. |
| F04 | `test_metrics_standalone` | Individual metrics execution shapes are correct. |
| F05 | `test_compute_all` | Computes F1, F2, F3 recursively over configured fractions. |

```
Results: 5/5 passed, 0 failed
```

---

## 2.2 Localization Metrics (L1–L4)

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

_Proof._ $\text{EGT} = \sum_{p \in M^{GT}} \tilde{e}_p$. Since all softmax values $\tilde{e}_p \geq 0$,
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

_Proof by construction._ If $\phi$ outputs high-EGT maps for correct predictions
and low-EGT maps for incorrect ones, CalibGap → 1 without faithfulness. $\square$

**Mitigation.** The benchmark:

1. Passes only `(model, image)` to explainers — never correctness flags.
2. Reports L1–L3 alongside L4 so inflated CalibGap is detectable.
3. Requires consistent CalibGap across all four datasets.

**Implication.** CalibGap is a _diagnostic_ metric, never a primary ranking criterion.

---

### 2.2.3 Unit Tests (L1–L4)

File: `tests/test_localization.py` — **12 tests**, all passing.

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

```
Results: 12/12 passed, 0 failed
```

---

### 2.2.4 ViT Patch Grid Reference

| Model                          | Patch size | Grid          | Patches | Upsample to 224×224 |
| ------------------------------ | ---------- | ------------- | ------- | ------------------- |
| ViT-B/16, DeiT-B/16, BEiT-B/16 | 16         | 14×14         | 196     | Bilinear ×16        |
| DINOv2-ViT-B/14                | 14         | 16×16         | 256     | Bilinear ×14        |
| DINO-ViT-B/8                   | 8          | 28×28         | 784     | Bilinear ×8         |
| Swin-B (GradCAM)               | —          | 7×7 (stage 4) | 49      | Bilinear ×32        |

---

## 2.3 Robustness Metrics (R1–R3)

Robustness metrics test whether an explanation is **stable and faithful** under
systematic perturbations to the input or the model. A faithful attribution must:

- Change **when the input moves** toward a worst-case perturbation (R1).
- Change **when the model weights are randomised** — sanity check (R2).
- Change **when output labels are shuffled** — faithfulness to classifier (R3).

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

### 2.3.1 Implementation — `RobustnessMetrics` and Utilities

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

#### Theorem 2.3.C — LabelRand is NOT a proper metric

**Claim.** LabelRand is a diagnostic metric, not a symmetric distance.

While the Spearman correlation $\rho$ is numerically symmetric, LabelRand is
**semantically asymmetric**: the roles of $\phi_\text{orig}$ (trusted reference) and
$\phi_\text{shuf}$ (perturbed) cannot be swapped without changing the interpretation.

**Implication.** Always specify which model is "original" vs. "permuted" when reporting
LabelRand. Do not compare across architectures with different head sizes (different
permutation degrees). Report alongside R1 and R2, never in isolation.

---

### 2.3.3 Unit Tests (R1–R3)

File: `tests/test_robustness.py` — **16 tests**, all passing.

| ID  | Category    | Test                              | Assertion                                   |
| --- | ----------- | --------------------------------- | ------------------------------------------- |
| R01 | Bounds      | `max_sensitivity_nonneg`          | MaxSens ≥ 0 for 20 trials                   |
| R02 | Bounds      | `model_randomisation_in_0_1`      | ModelRand ∈ [0,1] for 20 trials             |
| R03 | Bounds      | `label_randomisation_in_0_0p5`    | LabelRand ∈ [0, 0.5] for 20 trials          |
| R04 | Bounds      | `compute_all_keys`                | compute_all() returns exactly 3 keys        |
| R05 | Sensitivity | `sensitivity_increases_with_eps`  | Larger ε → ≥ mean MaxSens (30 trials)       |
| R06 | Sensitivity | `model_rand_orthogonal_maps`      | ModelRand > 0.5 for top-half vs bottom-half |
| R07 | Sensitivity | `label_rand_orthogonal_maps`      | Mean LabelRand ≈ 0.5 for 200 random pairs   |
| R08 | Zero        | `sensitivity_zero_constant_expl`  | MaxSens = 0 for constant explainer          |
| R09 | Zero        | `model_rand_identical_maps`       | ModelRand = 0 for identical maps            |
| R10 | Zero        | `label_rand_identical_maps`       | LabelRand = 0 for identical maps            |
| R11 | Utility     | `randomise_model_changes_weights` | All 4 params changed                        |
| R12 | Utility     | `randomise_labels_only_head`      | Backbone unchanged; head changed            |
| R13 | Utility     | `ssim_self_consistency`           | \_ssim(t, t) = 1.0 for 10 trials ```        |

Results: 16/16 passed, 0 failed

````

Run all Phase 2 tests together:

```bash
uv run pytest tests/ -v
````

---

### 2.3.4 Performance Guidance

| Metric       | Cost per sample        | Recommendation          |
| ------------ | ---------------------- | ----------------------- |
| L1–L3        | O(1) tensor ops        | Full dataset            |
| L4 CalibGap  | O(N) at dataset end    | Full dataset            |
| R1 MaxSens   | K explainer calls      | Random 500-image subset |
| R2 ModelRand | 1 extra explainer call | Full dataset            |
| R3 LabelRand | 1 extra explainer call | Full dataset            |

> [!IMPORTANT]
> For ImageNet-S-50 (5,000 val images × $K=50$ perturbations), MaxSens alone requires
> 250,000 additional explainer calls. Run R1 on a fixed random subset of 500 images
> and report the subset mean in the paper table footnote.

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

where $\tilde{e} = \text{softmax}(e)$ (proper probability distribution).

**Properties.**

- _Normalisation:_ softmax (well-defined for negative attributions).
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

| ID     | Class                          | Tests | What is tested                                           |
| ------ | ------------------------------ | ----- | -------------------------------------------------------- |
| PT_C1  | `TestGiniBatchTorchShape`      | 3     | `gini_batch_torch` output shape (B,), flat, dtype        |
| PT_C2  | `TestGiniBatchTorchValues`     | 3     | Values match numpy reference; one-hot; uniform           |
| PT_C3  | `TestEntropyBatchTorchRange`   | 2     | `entropy_batch_torch` range [0,1]; spiky map → near 0   |
| PT_C4  | `TestEntropyBatchTorchUniform` | 2     | Uniform map → H_norm = 1.0; batch of uniforms            |
| PT_C5  | `TestEmrBatchTorchOneHot`      | 3     | One-hot → EMR=1/N; uniform → EMR≈0.9; batch shape       |
| PT_C6  | `TestComputeBatchTorchTensor`  | 3     | `compute_batch` accepts `(B,H_p,W_p)` torch.Tensor      |
| PT_C7  | `TestDownsampleAttribution`    | 4     | `downsample_attribution` shape, non-negativity, 2D guard |

```
Results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0):
  test_complexity.py      : 45 passed, 0 failed
  test_torch_complexity.py: 20 passed, 0 failed, 1 skipped (CUDA)
  Combined Task 2.4       : 65 passed, 0 failed
```

---

### 2.4.6 Sanity Check

```bash
uv run python -c "from metrics.complexity import run_sanity_check; run_sanity_check()"
```

Expected output (N=196):

```
Map                  Gini   H_norm    EMR90   k*90
one_hot            0.9949   0.0000   0.0051      1
uniform            0.0000   1.0000   0.9000    176
concentrated       0.8739   0.1563   0.1020     20
random             0.3801   0.9247   0.7704    151
exponential        0.7612   0.4823   0.2500     49
✓ All sanity check assertions passed.
```

---

## 2.5 Axiomatic Analysis

The four Shapley-value axioms (Dummy, Completeness, Symmetry, Linearity) are used
as a theoretical lens to characterise the representational biases of all 15
benchmark metrics. This section consolidates the formal definitions, the
satisfaction table, and six key theorems.

> [!NOTE]
> Axiomatic analysis does not prescribe which metrics are "best". It provides
> a principled vocabulary for the paper's discussion section and enables
> reviewers to verify that we understand the limitations of our own metrics.

---

### 2.5.1 Axiom Definitions

For an explanation method $\phi$ and model $f$, let $\phi_p(f, x)$ be the
attribution assigned to patch $p$.

**A1 — Dummy.** If patch $p$ does not affect $f(x)$ for any input,
$\phi_p(f, x) = 0$.

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
| F1 — Insertion AUC     | ∼          | ✗                 | ✗             | ✗              |
| F2 — Deletion AUC      | ∼          | ✗                 | ✗             | ✗              |
| F3 — Comprehensiveness | ✓          | ∼                 | ✗             | ✗              |
| F4 — Log-odds shift    | ✓          | ∼                 | ✗             | ✗              |
| L1 — IoU with GT       | ∼          | ✗                 | ✗             | ✗              |
| L2 — Pointing Game     | ✗          | ✗                 | ✗             | ✗              |
| L3 — Energy on GT      | ✓          | ✗                 | ✗             | ✗              |
| L4 — Calibration Gap   | ✓          | ✗                 | ✗             | ✗              |
| R1 — Max-Sensitivity   | ✗          | ✗                 | ∼             | ✗              |
| R2 — Avg-Sensitivity   | ✗          | ✗                 | ∼             | ✗              |
| R3 — Param. Rand.      | ⊗          | ✗                 | ✗             | ✗              |
| R4 — Label Rand.       | ⊗          | ✗                 | ✗             | ✗              |
| **C1 — Gini**          | ✗          | ✗                 | **∼†**        | ✗              |
| **C2 — Entropy**       | ✗          | ✗                 | **∼†**        | ✗              |
| **C3 — EMR**           | ✗          | ✗                 | **∼†**        | ✗              |

> [!IMPORTANT]
> **†Anti-alignment (Theorem T6):** C1–C3 reward A3 _violations_. Concentrating
> all mass on one of two equal-contribution patches increases Gini and decreases
> Entropy/EMR, even though this breaks symmetry. This is a structural property
> of sparsity metrics, not a bug. It is reported explicitly in the paper.

---

### 2.5.3 Theorems

#### Theorem T1 — Fidelity metrics are blind to A2 (Completeness)

**Claim.** Insertion AUC / Deletion AUC / Comprehensiveness cannot distinguish
an attribution $\phi$ from $2\phi$ (or any positive scalar multiple).

_Proof._ All fidelity metrics are computed by ranking patches by attribution
magnitude and computing AUC over a sweep. Multiplication by a positive scalar
preserves the ranking and thus all fidelity scores. $\square$

**Implication.** A2 cannot be empirically verified using fidelity metrics alone;
use the dedicated `verify_completeness()` function.

---

#### Theorem T2 — EGT (L3) satisfies A1 for zero-attribution patches

**Claim.** If $\phi_p(f, x) = 0$ for all patches $p \notin M^{GT}$, then
$\text{EGT}(\phi, M^{GT}) = \sum_{p \in M^{GT}} \tilde{\phi}_p = 1$.

_Proof._ Softmax normalisation gives $\tilde{\phi}_p > 0$ for exactly the
patches with nonzero attribution. If all nonzero mass is inside $M^{GT}$,
the sum over $M^{GT}$ covers all probability mass. $\square$

**Implication.** L3 partially rewards A1 compliance (attributions confined to
GT have higher EGT), but does not enforce exact zero on out-of-GT patches.

---

#### Theorem T3 — Attention Rollout violates A1

**Claim.** For any ViT with $L$ attention layers, Attention Rollout assigns
non-zero attribution to every patch, including informationally irrelevant ones:

$$\min_p \phi^{\text{rollout}}_p \geq \left(\frac{1}{2}\right)^L > 0$$

_Proof._ Rollout adds a residual term $0.5 \cdot I$ at each layer, making
all attributions strictly positive by induction. For $L=12$, the floor is
$0.5^{12} \approx 2.4 \times 10^{-4}$. $\square$

**Implication.** Report the rollout dummy violation empirically using
`verify_rollout_dummy_violation()`. GradCAM does not have this floor.

---

#### Theorem T4 — SHAP satisfies A1, A2, A3, and A4 exactly

**Claim.** Shapley values (SHAP) are the unique attribution method satisfying
all four axioms simultaneously.

_Proof._ Classical result (Shapley 1953; Lundberg & Lee 2017). $\square$

**Implication.** SHAP is included as a reference explainer in Phase 3.
All other methods are measured _against_ the SHAP baseline in the axiom analysis.

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

#### Theorem T6 — Complexity metrics (C1–C3) are anti-aligned with A3 (Symmetry)

**Claim.** For two patches $p, q$ with equal model contributions, concentrating
all attribution on $p$ (A3 violation) always improves C1, C2, and C3 scores
compared to distributing equally.

_Proof._

Let $e_\text{sym} = (0.5, 0.5, 0, \ldots, 0)$ (A3 satisfied) and
$e_\text{asym} = (1.0, 0.0, 0, \ldots, 0)$ (A3 violated).

- **Gini:** sorted ascending $(0, \ldots, 0, 0.5, 0.5)$ vs. $(0, \ldots, 0, 1.0)$.
  The asymmetric version has a single-value non-zero rank sum, yielding higher Gini.
- **Entropy:** $H(0.5, 0.5) = \log 2 > 0 = H(1)$. Lower entropy for the asymmetric map.
- **EMR:** $k^*(0.9)$ for the symmetric map requires 2 patches; the asymmetric map needs 1.

All three metrics score the A3-violating attribution better. $\square$

**Implication.** Never use C1–C3 as the sole criterion for comparing methods
that differ primarily in symmetry. Report alongside L1–L4.

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

# Run all 12 (C1–C3 × A1–A4) tests
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

| ID     | Class                                    | Tests | What is tested                                                  |
| ------ | ---------------------------------------- | ----- | --------------------------------------------------------------- |
| PT_A1  | `TestVerifyCompletenessLinear`           | 2     | `verify_completeness` returns 4 keys; finite error              |
| PT_A2  | `TestVerifyCompletenessXOR`              | 1     | XOR model illustrates Theorem T1 counterexample                 |
| PT_A3  | `TestVerifyDummyAxiomPasses`             | 1     | `verify_dummy_axiom` returns 7 keys; values in [0,1]            |
| PT_A4  | `TestVerifyDummyAxiomFails`              | 1     | Wrong attribution → non-zero dummy_attribution_ratio            |
| PT_A5  | `TestAxiomVerifierA3GiniAntialignment`   | 1     | Theorem T6: Gini A3 delta < 0 via verifier (torch backend)      |
| PT_A6  | `TestVerifyRolloutDummyViolation`        | 1     | `verify_rollout_dummy_violation` interface contract + keys       |
| PT_A7  | `TestGiniBatchTorchCPUGPUConsistency`   | 2     | `gini_batch_torch` deterministic on CPU; GPU agree (skip CUDA)  |

```
Results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0):
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

Phase 3 wraps the 13 metrics (F1–F4, L1–L4, R1–R4, C1–C3) and 7 explainers (E1–E7)
into a **single reproducible evaluation command** with:
- Fixed random seeds (per-run seed logged in metadata)
- Checkpoint-and-resume for long GPU runs
- Unified CSV output that feeds directly into Phase 4 result tables

> [!IMPORTANT]
> Every number in the TPAMI result tables **must come from this pipeline** with
> fixed seeds. Do not manually insert numbers computed outside this framework.

---

## 3.1 Standardised Explainer Interface

All 7 explainers share a common `BaseExplainer` contract defined in
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

| #   | Class                        | Method                     | File                  | Swin-B | Note                           |
| --- | ---------------------------- | -------------------------- | --------------------- | ------ | ------------------------------ |
| E1  | `RawAttentionExplainer`      | Raw CLS attention          | `raw_attention.py`    | ✗      | Last block, mean over heads    |
| E2  | `AttentionRolloutExplainer`  | Attention Rollout          | `rollout.py`          | ✗      | Recursive Â product            |
| E3  | `GradCAMExplainer`           | GradCAM                    | `gradcam.py`          | ✓      | Hook-based, ReLU gated         |
| E4  | `CheferLRPExplainer`         | Chefer et al. LRP          | `chefer_lrp.py`       | ✗      | Pure-PyTorch reimplementation  |
| E5  | `RISEExplainer`              | RISE (Petsiuk et al. 2018) | `rise.py`             | ✓      | 4000 float16 masks; chunked    |
| E6  | `LIMEExplainer`              | LIME (patch-grid)          | `lime.py`             | ✓      | Ridge regression on P² patches |
| E7  | `DIMEExplainer`              | DIME                       | `dime.py`             | —      | ⚠ Placeholder (see below)     |

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
*"vectorise over mask dimension"* requirement.

**Gradient handling:** Outer `torch.no_grad()` context + inner
`torch.enable_grad()` for gradient methods (GradCAM, CheferLRP). Hook cleanup
is guaranteed via `try … finally: hook.remove()`.

### 3.1.4 Unit Tests (Task 3.1)

File: `tests/test_explainers.py` — **26 tests passing, 1 documented skip**
(E14: DIME no-NaN test, pending guide resolution).

| Category  | Tests | What is tested                                              |
| --------- | ----- | ----------------------------------------------------------- |
| A – Shape | 7     | Every explainer returns exactly `(P, P)` shape              |
| B – Finite| 7     | `torch.isfinite(output).all()` for all explainers           |
| C – Batch | 3     | `explain_batch` ≡ loop over `explain` (E1, E2, E3)          |
| D – Var   | 3     | Output `std > 0` on non-trivial input (E1, E5, E6)          |
| E – Swin  | 2     | E1/E2 raise `UnsupportedArchitectureError` on `_MockSwinB`  |
| F – RISE  | 2     | Non-negative output; mask count stored correctly            |
| G – LIME  | 2     | Finite output; exactly P² = 16 coefficients                 |
| H – DIME  | 1     | `is_resolved=False`; `NotImplementedError` ∋ "BENCHMARK.md" |

```
Results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0):
  test_explainers.py : 26 passed, 1 skipped (E14 documented), 0 failed
```

### 3.1.5 Production Notes

| Concern              | Recommendation                                                  |
| -------------------- | --------------------------------------------------------------- |
| RISE cost (M=4000)   | Sub-sample to 500 val images per model; report in table footnote |
| LIME cost            | `n_samples=500` for production; 20 for dev/CI                   |
| Seed reproducibility | Pass `seed=42` to E5, E6; fix `torch.manual_seed` before each run |
| Swin-B explainers    | Use E3 (GradCAM) only; E1/E2/E4 auto-raise at runtime           |

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

| Mode          | Formula                                          | Use case                              |
| ------------- | ------------------------------------------------ | ------------------------------------- |
| `minmax`      | `(att − min) / (max − min)` ; zeros if degenerate | Default; all fidelity + loc metrics   |
| `percentile`  | clamp at 99th percentile → minmax               | Heavy-tailed maps (RISE, LIME)        |
| `softmax`     | `softmax(att.flatten())` ; numerically stable    | EGT (L3), Effective Mass Ratio (C3)   |

- Output always `float32`, values in `[0, 1]` (`softmax` sums to 1.0).
- Degenerate (constant) maps return all-zeros for `minmax` and `percentile`.
- Batch input `(B, Hp, Wp)`: each sample is normalised **independently**.

### 3.2.3 Unit Tests (Task 3.2)

File: `tests/test_normalize.py` — **24 tests**, all passing.

| Class                          | Tests | Category                                          |
| ------------------------------ | ----- | ------------------------------------------------- |
| `TestMinmax`                   | 6     | range, exact [0,1], degenerate, shape, batch, dtype |
| `TestPercentile`               | 6     | range, outlier suppression, rank-preservation, batch |
| `TestSoftmax`                  | 6     | sum=1, all-positive, batch, stability, uniform, shape |
| `TestIntegrationAndEdgeCases`  | 6     | pipeline, invalid mode, 1-D rejection, batch guard, idempotent, enum |

```
Results (2026-04-10, CPU, Python 3.11.8 / PyTorch 2.10.0):
  test_normalize.py : 24 passed, 0 failed
  Full suite        : 149 passed, 2 skipped (CUDA), 0 failed
```

---

## 3.3 BenchmarkRunner

> [!NOTE]
> **Status: Planned.** Extension of `metrics/runner.py` to include all 13
> metrics + 7 explainers + checkpointing (pickle-based) + deterministic seed
> injection. CLI: `python -m benchmark.run --config configs/run.yaml`.

---

# Appendix A — Project Layout

```
vit-explainability-benchmark/
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
│   ├── __init__.py          # All public exports (torch-free first)
│   ├── localization.py      # L1–L4 (requires torch)
│   ├── robustness.py        # R1–R3 + model utilities (requires torch)
│   ├── complexity.py        # C1–C3 (torch-free; optional GPU batch ops)
│   ├── axiom_verifier.py    # AxiomVerifier, verify_completeness, Figure F1
│   └── runner.py            # BenchmarkRunner unified evaluation loop
│
├── tests/
│   ├── test_localization.py   # 12 unit tests — L1–L4
│   ├── test_robustness.py     # 16 unit tests — R1–R3
│   ├── test_complexity.py     # 45 unit tests — C1–C3 + Theorem T6
│   └── test_axiom_verifier.py # 21 unit tests — AxiomVerifier + T6 canonical
│
├── figures/                   # Generated PDFs (gitignored; generated on demand)
│   ├── axiom_satisfaction.pdf # Figure F1 — 15×4 axiom heatmap
│   └── complexity_distributions.pdf
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
├── pyproject.toml               # uv project file (Python 3.13, dev: pytest, numpy)
├── model_hashes.txt             # SHA-256 of all pre-trained checkpoints
├── requirements.txt             # Pinned runtime dependency versions
└── .gitignore
```

---

# Appendix B — Complete Metric Index

| ID  | Name              | Formula (short)                                                                     | Range                                    | File              | Higher =  |
| --- | ----------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- | ----------------- | --------- | --------------- | ------ |
| F1  | Sufficiency       | $f(x)_{y^*} - f(x \odot M)_{y^*}$                                                   | $(-1,1)$                                 | —                 | Better    |
| F2  | Comprehensiveness | $f(x)_{y^*} - f(x \odot \bar{M})_{y^*}$                                             | $(-1,1)$                                 | —                 | Better    |
| F3  | Log-odds Drop     | $\log\frac{p}{1-p} - \log\frac{p'}{1-p'}$                                           | $\mathbb{R}$                             | —                 | Better    |
| L1  | mIoU              | $\text{mean}_\tau \text{IoU}(\hat{M}_\tau, M^{GT})$                                 | $[0,1]$                                  | `localization.py` | Better    |
| L2  | Pointing Game     | $\mathbf{1}[p^* \in M^{GT}]$                                                        | $\{0,1\}$                                | `localization.py` | Better    |
| L3  | EGT               | $\sum_{p \in M^{GT}} \text{softmax}(e)_p$                                           | $[0,1]$                                  | `localization.py` | Better    |
| L4  | CalibGap          | $\mathbb{E}[\text{EGT}\mid\text{correct}] - \mathbb{E}[\text{EGT}\mid\text{wrong}]$ | $(-1,1)$                                 | `localization.py` | Better    |
| R1  | MaxSens           | $\max_k \|\phi(x+\delta_k)-\phi(x)\|_2 / \|\phi(x)\|_2$                             | $[0,\infty)$                             | `robustness.py`   | **Worse** |
| R2  | ModelRand         | $1 - \text{SSIM}(\phi_\text{orig}, \phi_\text{rand})$                               | $[0,1]$                                  | `robustness.py`   | Better    |
| R3  | LabelRand         | $1 - (                                                                              | \rho(\phi*\text{orig}, \phi*\text{shuf}) | + 1)/2$           | $[0,0.5]$ | `robustness.py` | Better |
| C1  | Gini              | $(2\sum_i i \cdot e_{(i)}) / (N \sum e) - (N+1)/N$                                  | $[0,1]$                                  | `complexity.py`   | Better    |
| C2  | Entropy (norm)    | $H(\text{softmax}(e)) / \log N$                                                     | $[0,1]$                                  | `complexity.py`   | **Worse** |
| C3  | EMR               | $k^*(0.90) / N$                                                                     | $[0,1]$                                  | `complexity.py`   | **Worse** |

> [!NOTE]
> **R1, C2, C3 direction is inverted** (lower = better explanation). For all
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
☑ L3 formal definition: EGT = Σ_{p∈M_GT} softmax(e)_p
☑ L4 formal definition: CalibGap = E[EGT|correct] − E[EGT|incorrect]
☑ LocalizationMetrics class — metrics/localization.py
☑ 12 unit tests — tests/test_localization.py — 12/12 passing
☑ BenchmarkRunner — metrics/runner.py
```

## Phase 2 — Robustness Metrics (R1–R3)

```
☑ R1 formal definition: MaxSens
☑ R2 formal definition: ModelRand
☑ R3 formal definition: LabelRand
☑ RobustnessMetrics class — metrics/robustness.py
☑ randomise_model_weights(), randomise_classifier_labels()
☑ Theorems 2.3.A, 2.3.B, 2.3.C
☑ 16 unit tests — tests/test_robustness.py — 16/16 passing
☑ BenchmarkRunner extended — fully backward-compatible
```

## Phase 2 — Complexity Metrics (C1–C3)

```
☑ C1 formal definition: Gini coefficient, O(N log N), scale-invariant
☑ C2 formal definition: Shannon entropy, softmax-normalised, norm ∈ [0,1]
☑ C3 formal definition: EMR at θ∈{0.50, 0.90, 0.95}; k*_90 cross-arch
☑ ComplexityMetrics class — metrics/complexity.py
☑ gini_coefficient() — standalone, numpy, O(N log N), zero-map convention
☑ attribution_entropy() — standalone, softmax normalisation, H/log(N)
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
☑ 15×4 axiom satisfaction table (§2.5.2)
☑ Theorem T1: Fidelity metrics blind to A2 (scale-invariance proof)
☑ Theorem T2: EGT (L3) satisfies A1 approximately (proof)
☑ Theorem T3: Attention Rollout violates A1 — floor (0.5)^L proof
☑ Theorem T4: SHAP satisfies all 4 axioms exactly (reference)
☑ Theorem T5: GradCAM satisfies A1 approximately via ReLU (proof sketch)
☑ Theorem T6: C1–C3 anti-aligned with A3 (Symmetry) — full proof
☑ AxiomVerifier class — metrics/axiom_verifier.py
☑ LinearPatchModel, XORInteractionModel — toy models for controlled tests
☑ AxiomTestResult dataclass — to_dict() for serialisation
☑ test_a1(), test_a2(), test_a3(), test_a4() — per-axiom empirical tests
☑ run_all() — 12 (C1–C3 × A1–A4) tests; 60 tests with full MetricSuite
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

### Phase 3 — Task 3.1 Checklist
```
☑ BaseExplainer ABC — explain(x, target_class) → (P, P) tensor contract
☑ explain_batch() default loop; overridable for amortised methods
☑ UnsupportedArchitectureError — raised by E1/E2/E4 on Swin-B
☑ _get_timm_blocks(), _has_cls_token(), _to_patch_grid() — shared helpers
☑ _capture_attn_weights() — fused_attn-safe attention hook
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
☑ metrics/__init__.py updated — normalize_attribution in torch-free block
☑ 24 unit tests — tests/test_normalize.py — 24 passing
☑ Full suite: 149 passed, 2 skipped (CUDA), 0 failed (2026-04-10)
```
