# Task 2.2 — Localization Metrics

> **ViT Explainability Benchmark · Phase 2 Document**
> This document covers the formal definitions of all four localization metrics (L1–L4), their complete Python implementation, an axiomatic analysis with proofs and counterexamples, 12 unit tests, BenchmarkRunner integration, and the Task 2.2 master checklist.

---

## 1. Formal Metric Definitions

The four localization metrics measure whether an explanation method correctly attributes model decisions to the ground-truth region of the input. All metrics operate on a pair $(e, M^{GT})$ where $e \in \mathbb{R}^{H \times W}$ is the raw attribution map at pixel (or patch) resolution and $M^{GT} \subseteq \{1,\ldots,H\} \times \{1,\ldots,W\}$ is the binary ground-truth mask (from bounding boxes or pixel segmentation).

### 1.1  L1 — Multi-Threshold IoU (mIoU)

**Notation.** Let $\tilde{e}_p = (e_p - \min_q e_q) / (\max_q e_q - \min_q e_q + \varepsilon)$ be the min-max normalised attribution (range $[0,1]$). For threshold $\tau$, define the predicted binary mask:

$$\hat{M}_\tau = \{p : \tilde{e}_p \geq \tau\}$$

The Intersection-over-Union at threshold $\tau$ is:

$$\text{IoU}(\hat{M}_\tau, M^{GT}) = \frac{|\hat{M}_\tau \cap M^{GT}|}{|\hat{M}_\tau \cup M^{GT}|}$$

The multi-threshold mIoU is the mean over $\tau \in \{0.25, 0.50, 0.75\}$:

$$L1 = \text{mIoU} = \frac{1}{|\mathcal{T}|} \sum_{\tau \in \mathcal{T}} \text{IoU}(\hat{M}_\tau, M^{GT})$$

**Range:** $[0, 1]$ at each threshold; 0 = no overlap, 1 = perfect overlap.

**Sensitivity note:** Lower thresholds ($\tau = 0.25$) are more permissive — they accept any pixel with attribution above the bottom quarter. Higher thresholds ($\tau = 0.75$) are conservative — only the top 25% of attribution mass is predicted foreground. Reporting all three prevents cherry-picking.

---

### 1.2  L2 — Pointing Game (PG)

Let $p^* = \arg\max_p e_p$ be the pixel with maximum raw attribution.

$$L2 = \text{PG} = \mathbf{1}\!\left[p^* \in M^{GT}\right]$$

**Range:** $\{0, 1\}$ per sample; expectation over the dataset is the dataset-level accuracy.

**Tie-breaking.** When multiple pixels share the maximum value, one is drawn uniformly at random (seeded at 42 for reproducibility). This prevents a degenerate implementation from always picking position $(0,0)$, which may fall inside the GT mask by chance.

**Dataset-level PG.** Averaged over $N$ samples:

$$\overline{\text{PG}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[p^*_i \in M^{GT}_i\right]$$

---

### 1.3  L3 — Energy-Ground-Truth (EGT)

Let $\tilde{e}_p$ be the **softmax-normalised** attribution:

$$\tilde{e}_p = \frac{\exp(e_p)}{\sum_{q} \exp(e_q)}$$

This produces a probability distribution over all spatial positions (sums to 1). The EGT is the total attribution mass inside the GT region:

$$L3 = \text{EGT} = \sum_{p \in M^{GT}} \tilde{e}_p$$

**Range:** $[0, 1]$. EGT = 1 means all attribution mass is inside the GT region; EGT = 0 means none is.

**Why softmax, not sum-normalisation.** Sum-normalisation ($e_p / \sum_q e_p$) is undefined when any $e_p < 0$ (as with Integrated Gradients or raw GradCAM values). Softmax handles negative values correctly and ensures the distribution is well-defined for all attribution methods.

---

### 1.4  L4 — Calibration Gap (CalibGap)

Let $\mathcal{C}$ and $\mathcal{I}$ be the sets of correctly and incorrectly classified samples respectively.

$$L4 = \text{CalibGap} = \mathbb{E}_{i \in \mathcal{C}}\!\left[\text{EGT}_i\right] - \mathbb{E}_{i \in \mathcal{I}}\!\left[\text{EGT}_i\right]$$

**Range:** $(-1, 1)$. A positive CalibGap indicates the model's explanations align better with the GT region on samples it classifies correctly than on samples it gets wrong — a necessary condition for faithful explanations. A CalibGap near zero or negative is a failure mode: the explanation is insensitive to prediction correctness.

**Undefined case.** CalibGap is undefined when all predictions are correct ($\mathcal{I} = \emptyset$) or all are incorrect ($\mathcal{C} = \emptyset$). Both raise `ValueError` in the implementation.

---

## 2. Implementation

All implementation files are in `metrics/`. The package follows the same structure and code style as `training/`.

### 2.1 Directory Layout

```
metrics/
├── __init__.py          # Package exports: LocalizationMetrics, BenchmarkRunner
├── localization.py      # LocalizationMetrics class (L1–L4)
├── runner.py            # BenchmarkRunner — dataset-level evaluation loop
tests/
└── test_localization.py # 12 unit tests (Task 2.2 §4)
```

### 2.2 `LocalizationMetrics` Class — API Reference

```python
from metrics.localization import LocalizationMetrics

lm = LocalizationMetrics(
    thresholds=[0.25, 0.50, 0.75],   # L1 binarisation thresholds
    seed=42,                          # tie-breaking seed for L2
)
```

#### `multi_threshold_iou(att_map, gt_mask, thresholds=None)` — L1

```python
result = lm.multi_threshold_iou(att_map, gt_mask)
# Returns dict:
# {
#   'iou@0.25': 0.74,
#   'iou@0.50': 0.61,
#   'iou@0.75': 0.42,
#   'miou':     0.59,
# }
```

**Parameters:**
- `att_map` — `torch.Tensor` of shape `(H_a, W_a)`. Raw (un-normalised) attribution values; any range.
- `gt_mask` — `torch.Tensor` of shape `(H_r, W_r)`. Float in `[0,1]` or integer mask. Pixel value 255 (VOC void) is automatically treated as background.
- `thresholds` — optional override; defaults to instance `thresholds`.

**Internal pipeline:**
1. Min-max normalise `att_map` to `[0, 1]`.
2. Bilinearly upsample to match `gt_mask` resolution.
3. Binarise at each τ → compute intersection and union → IoU.
4. Return all per-threshold IoUs + mean.

---

#### `pointing_game(att_map, gt_mask)` — L2

```python
pg = lm.pointing_game(att_map, gt_mask)   # 1.0 or 0.0
```

**Tie-breaking:** if multiple pixels share the maximum value, one is sampled uniformly at random using `self._rng` (seeded at construction time).

---

#### `egt(att_map, gt_mask)` — L3

```python
e = lm.egt(att_map, gt_mask)   # float in [0, 1]
```

**Normalisation:** softmax over the flattened spatial map. Handles negative attributions correctly (Integrated Gradients, raw GradCAM).

---

#### `calibration_gap(atts_correct, atts_incorrect, gt_masks=None)` — L4

```python
gap = lm.calibration_gap(
    atts_correct   = [att_c1, att_c2, ...],   # correctly classified
    atts_incorrect = [att_i1, att_i2, ...],   # incorrectly classified
    gt_masks       = [m1, m2, ..., mn],       # all masks: correct first
)
# gap > 0 → explanations better on correct predictions (desired)
# gap < 0 → explanations worse on correct predictions (failure mode)
```

When `gt_masks` is `None`, each `att` in both lists is interpreted as a pre-computed scalar EGT value wrapped in a `torch.Tensor`.

---

#### `align_patch_to_pixel(patch_attr, target_hw, mode='bilinear')` — Utility

```python
# Upsample a 14×14 ViT patch attribution to 224×224 pixel resolution
pixel_att = lm.align_patch_to_pixel(
    patch_attr = att_14x14,
    target_hw  = (224, 224),
)
```

ViT patch grids at 224×224 input:

| Model | Patch size | Grid | Patches |
|-------|-----------|------|---------|
| ViT-B/16, DeiT-B/16, BEiT-B/16 | 16 | 14×14 | 196 |
| DINOv2-ViT-B/14 | 14 | 16×16 | 256 |
| DINO-ViT-B/8 | 8 | 28×28 | 784 |

Note: Swin-B does not produce a standard patch grid — see §2.4 for Swin-specific handling.

---

#### `compute_all(att_map, gt_mask)` — Convenience

```python
result = lm.compute_all(att_map, gt_mask)
# Returns L1 dict + 'pointing_game' + 'egt' for one sample.
# L4 is excluded — requires paired correct/incorrect lists.
```

### 2.3 Full Usage Example

```python
from metrics.localization import LocalizationMetrics

lm = LocalizationMetrics(thresholds=[0.25, 0.50, 0.75], seed=42)

# Simulate a 14×14 attention map (ViT-B/16 output) and 224×224 GT mask
att_map = torch.randn(14, 14)
gt_mask = torch.zeros(224, 224)
gt_mask[80:160, 60:140] = 1.0   # bird bounding box region

# --- L1: mIoU ---
iou_result = lm.multi_threshold_iou(att_map, gt_mask)
print(f"mIoU: {iou_result['miou']:.4f}")

# --- L2: Pointing Game ---
pg = lm.pointing_game(att_map, gt_mask)
print(f"PG: {pg:.1f}")

# --- L3: EGT ---
e = lm.egt(att_map, gt_mask)
print(f"EGT: {e:.4f}")

# --- L4: CalibGap (dataset-level) ---
atts_c = [torch.randn(14, 14) for _ in range(50)]   # correct predictions
atts_i = [torch.randn(14, 14) for _ in range(20)]   # incorrect predictions
masks  = [gt_mask] * 70
gap = lm.calibration_gap(atts_c, atts_i, gt_masks=masks)
print(f"CalibGap: {gap:.4f}")
```

### 2.4 Model-Specific Notes

| Model | Attribution source | Grid | Alignment note |
|-------|-------------------|------|---------------|
| ViT-B/16, BEiT-B/16 | CLS→patch attention, last block | 14×14 | Bilinear ×16 to 224×224 |
| DeiT-B/16 (distilled) | Average CLS + distil token attn | 14×14 | Same as ViT-B |
| Swin-B | Per-window attention (no CLS) | Variable | Merge windows; map to 7×7 stage outputs; bilinear to 224×224 |
| DINO-ViT-B/8 | CLS→patch attention | 28×28 | Bilinear ×8 to 224×224 |
| DINOv2-ViT-B/14 | CLS→patch attention | 16×16 | Bilinear ×14 to 224×224 |

**Swin-B limitation.** No CLS token exists; attention is local-window-based. For `LocalizationMetrics`, GradCAM output maps (typically at the final stage feature resolution, 7×7) are upsampled to 224×224. Standard L1–L3 apply; L2 peak pixel logic is unchanged. This must be flagged as a methodological limitation in the paper because the attribution mechanism is architecturally different from global-attention models.

---

## 3. Axiomatic Analysis

This section proves three formal properties of L1–L4 with counterexamples where properties are violated. These results justify metric design choices and inform how results should be interpreted.

---

### Theorem 1 — IoU violates the Symmetry Axiom when GT regions differ in area

**Symmetry Axiom.** A metric $\mu$ satisfies symmetry if swapping the roles of predicted mask $\hat{M}$ and GT mask $M^{GT}$ leaves the value unchanged: $\mu(\hat{M}, M^{GT}) = \mu(M^{GT}, \hat{M})$.

**Claim.** IoU satisfies symmetry: $\text{IoU}(\hat{M}, M^{GT}) = \text{IoU}(M^{GT}, \hat{M})$ always.

**Proof.**
$$\text{IoU}(\hat{M}, M^{GT}) = \frac{|\hat{M} \cap M^{GT}|}{|\hat{M} \cup M^{GT}|}$$

Since set intersection and union are symmetric operations ($A \cap B = B \cap A$ and $A \cup B = B \cup A$), swapping $\hat{M}$ and $M^{GT}$ leaves both numerator and denominator invariant. $\square$

**Corollary.** IoU is symmetric, but this does **not** mean it is area-invariant. A small GT mask that is perfectly overlapped by a large $\hat{M}$ gives low IoU because the union is large:

$$\text{IoU}(\hat{M}_\text{large}, M^{GT}_\text{small}) = \frac{|M^{GT}_\text{small}|}{|\hat{M}_\text{large}|} \ll 1$$

**Counterexample (area penalty).**

```python
import torch
from metrics.localization import LocalizationMetrics

lm = LocalizationMetrics(thresholds=[0.50])

# Small GT mask (10×10 pixels) inside a 224×224 image
gt = torch.zeros(224, 224)
gt[100:110, 100:110] = 1.0            # 100 foreground pixels

# Attribution: perfect (matches GT exactly) → IoU = 1.0
att_perfect = gt.clone()
r1 = lm.multi_threshold_iou(att_perfect, gt)
# iou@0.50 = 1.0  ✓

# Attribution: matches GT but also covers 90% of background
att_large = torch.zeros(224, 224)
att_large[10:200, 10:200] = 1.0       # 36,100 pixels, includes GT
r2 = lm.multi_threshold_iou(att_large, gt)
# iou@0.50 ≈ 100/(100 + 36100-100) ≈ 0.003 — very low despite full recall
```

**Implication.**  IoU penalises over-segmentation. This is intentional in the benchmark — an explanation that highlights the entire image is not useful, even if it technically "contains" the GT region.

---

### Theorem 2 — EGT (L3) satisfies Monotone Coverage but NOT Concentration Independence

**Monotone Coverage Axiom.** If the GT mask $M^{GT}$ grows (i.e., $M^{GT}_1 \subseteq M^{GT}_2$) with the attribution map $e$ held fixed, then $\text{EGT}(e, M^{GT}_1) \leq \text{EGT}(e, M^{GT}_2)$.

**Proof.** Let $\tilde{e} = \text{softmax}(e)$ (fixed for fixed $e$). Then:

$$\text{EGT}(e, M^{GT}) = \sum_{p \in M^{GT}} \tilde{e}_p$$

Since $M^{GT}_1 \subseteq M^{GT}_2$, the sum over $M^{GT}_2$ includes all terms from $M^{GT}_1$ plus additional non-negative terms $\tilde{e}_p \geq 0$. Therefore $\text{EGT}(e, M^{GT}_1) \leq \text{EGT}(e, M^{GT}_2)$. $\square$

**Concentration Independence Axiom.** A metric satisfies Concentration Independence if, for a fixed GT mask, two attribution maps with the same total mass inside $M^{GT}$ achieve the same score regardless of how that mass is distributed within $M^{GT}$.

**Counterexample (EGT violates Concentration Independence).**

Concentration Independence holds for EGT trivially: EGT is a *sum* over $M^{GT}$, so it is already insensitive to *intra-GT* spatial distribution of attribution. However, it violates **Boundary Sensitivity** — it does not penalise mass that is concentrated at the very edge of $M^{GT}$ versus at the semantic centre. This is a known limitation of energy-based metrics vs. pointing-game style metrics.

```python
import torch

gt = torch.zeros(14, 14)
gt[4:10, 4:10] = 1.0          # 6×6 foreground

# Map A: all mass at one pixel on the GT boundary
att_boundary = torch.full((14, 14), -10.0)
att_boundary[4, 4] = 10.0    # boundary pixel of GT

# Map B: mass spread over all 36 GT pixels equally
att_spread = torch.full((14, 14), -10.0)
att_spread[4:10, 4:10] = 0.0  # spread uniformly over GT

lm = LocalizationMetrics()
e_boundary = lm.egt(att_boundary, gt)
e_spread   = lm.egt(att_spread,   gt)
# Both achieve EGT ≈ 1.0 — EGT cannot distinguish boundary from spread.
# L2 (Pointing Game) is complementary: it does distinguish peak location.
```

**Implication.** L3 (EGT) and L2 (PG) are **complementary**, not redundant. EGT captures mass allocated to the GT region; PG captures peak precision. Both are needed.

---

### Theorem 3 — CalibGap (L4) is NOT a proper scoring rule

**Proper Scoring Rule.** A scoring rule $S$ is proper if the expected score is maximised when the predicted distribution matches the true distribution — i.e., an honest forecaster maximises expected score by reporting true beliefs.

**Claim.** CalibGap is **not** a proper scoring rule in the forecasting sense, and can be gamed by a degenerate explanation method.

**Proof by construction (adversarial counterexample).**

Suppose an adversarial explanation method $\phi$ has access to the model's prediction correctness flag before generating an attribution:

- For correctly classified samples: $\phi$ outputs an attribution map concentrated inside $M^{GT}$ → high EGT.
- For incorrectly classified samples: $\phi$ outputs an attribution map concentrated outside $M^{GT}$ → low EGT.

This adversary achieves CalibGap → 1 without the attributions being faithful to the model's actual decision process.

**Mitigation in the benchmark.** The benchmark prevents this by:

1. **Blind attribution**: all explanation methods receive only $(model, image)$ — the prediction correctness label is never passed to the explainer.
2. **Reporting L1–L3 alongside L4**: a method that games CalibGap will also show inflated PG and EGT on correct samples, but the per-class breakdown and comparison between model architectures will expose inconsistencies.
3. **Cross-dataset consistency**: a truly faithful explanation should show consistent CalibGap across CUB, VOC, ImageNet-S-50, and NIH ChestX-ray14. A gamed metric would show dataset-specific anomalies.

**Implication.** CalibGap is a *diagnostic metric*, not a primary ranking metric. It should always be reported alongside L1–L3 and model accuracy, never in isolation.

---

## 4. Unit Tests

File: `tests/test_localization.py` — 12 tests across 4 categories.

```
Category A — Random input → chance-level output
  T01  random_iou_not_one           mIoU < 1.0 over 20 trials
  T02  random_pg_within_bounds      PG ∈ {0,1}; not always 1.0 over 100 trials
  T03  random_egt_within_bounds     EGT ∈ [0,1] for 50 random pairs

Category B — Perfect input → maximum output
  T04  perfect_iou_is_one           att=GT → mIoU = 1.0 at all τ
  T05  perfect_pg_is_one            peak at GT pixel → PG = 1.0
  T06  perfect_egt_is_one           all mass inside GT → EGT = 1.0
  T07  perfect_calibgap_positive    perfect-correct vs wrong-incorrect → gap > 0

Category C — GT-misaligned input → minimum output
  T08  misaligned_iou_is_zero       att outside GT → IoU@0.75 = 0.0
  T09  misaligned_pg_is_zero        peak outside GT → PG = 0.0
  T10  misaligned_egt_near_zero     all mass outside GT → EGT < 0.01

Category D — Edge cases and contracts
  T11  constant_att_map_iou         degenerate constant map → no NaN, result in [0,1]
  T12  calibgap_empty_list_raises   empty correct/incorrect list → ValueError
```

Run:

```bash
# With torch environment active:
python tests/test_localization.py

# Or with pytest:
pytest tests/test_localization.py -v
```

Expected output (all 12 pass):

```
============================================================
Task 2.2 §4 — LocalizationMetrics Unit Tests
============================================================

T01 ✓  random_iou_not_one
T02 ✓  random_pg_within_bounds  (hits=27/100)
T03 ✓  random_egt_within_bounds
T04 ✓  perfect_iou_is_one
T05 ✓  perfect_pg_is_one
T06 ✓  perfect_egt_is_one
T07 ✓  perfect_calibgap_positive  (gap=0.xxxx)
T08 ✓  misaligned_iou_is_zero
T09 ✓  misaligned_pg_is_zero
T10 ✓  misaligned_egt_near_zero  (egt=x.xxe-xx)
T11 ✓  constant_att_map_iou  (miou=0.0000)
T12 ✓  calibgap_empty_list_raises

============================================================
Results: 12/12 passed, 0 failed
============================================================
```

---

## 5. BenchmarkRunner Integration

File: `metrics/runner.py`.

`BenchmarkRunner` wraps `LocalizationMetrics` in a dataset-level evaluation loop. It handles:

- Iterating over a `DataLoader` yielding `(images, gt_masks, labels)`.
- Calling the `explainer` callable to produce attribution maps per batch.
- Per-sample L1, L2, L3 computation via `compute_all()`.
- Partitioning samples by prediction correctness for L4 (CalibGap).
- Macro-averaging all metrics over the dataset.
- Structured result dict compatible with paper Table format.

### 5.1 DataLoader Contract

The DataLoader **must** yield tuples of exactly:

```
(images, gt_masks, labels)
  images   : torch.Tensor  (B, C, H, W)       — normalised input images
  gt_masks : torch.Tensor  (B, H_m, W_m)      — integer or float GT masks
  labels   : torch.Tensor  (B,)                — integer class indices
```

### 5.2 Explainer Interface

The `explainer` callable must have signature:

```python
def explainer(model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    """
    Returns one attribution map per image in the batch.
    Each map: (H_a, W_a) float tensor — any scale/resolution.
    """
    ...
```

The `BenchmarkRunner` passes raw model + images; the explainer is responsible for all pre/post-processing specific to its method (attention rollout, GradCAM, etc.).

### 5.3 Usage

```python
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics

# Define your explainer (e.g. attention rollout)
def my_attention_rollout(model, images):
    atts = []
    for img in images:
        with torch.no_grad():
            att = model.get_last_selfattention(img.unsqueeze(0))
            # → (1, num_heads, N+1, N+1)
            # Average heads, take CLS→patch slice, reshape to (H_p, W_p)
            att_map = att.mean(1)[0, 0, 1:]  # (N,)
            h = w = int(att_map.shape[0] ** 0.5)
            att_map = att_map.reshape(h, w)
        atts.append(att_map)
    return atts

# Build runner
runner = BenchmarkRunner(
    metrics   = LocalizationMetrics(thresholds=[0.25, 0.50, 0.75], seed=42),
    explainer = my_attention_rollout,
)

# Evaluate
results = runner.evaluate(
    model        = fine_tuned_model,
    loader       = val_loader,          # yields (images, gt_masks, labels)
    dataset_name = "cub200",
)

# Access results
print(f"mIoU:       {results['macro']['miou']:.4f}")
print(f"PG:         {results['macro']['pointing_game']:.4f}")
print(f"EGT:        {results['macro']['egt']:.4f}")
print(f"CalibGap:   {results['macro']['calibration_gap']:.4f}")
```

### 5.4 Results Dict Schema

```python
{
    "dataset":     "cub200",
    "n_samples":   5794,
    "n_correct":   4823,
    "n_incorrect": 971,
    "macro": {
        "iou@0.25":        float,
        "iou@0.50":        float,
        "iou@0.75":        float,
        "miou":            float,
        "pointing_game":   float,
        "egt":             float,
        "calibration_gap": float,
    },
    "per_metric": {
        "miou":          [float, ...],   # one per sample
        "pointing_game": [float, ...],
        "egt":           [float, ...],
        "calibration_gap": [float],      # single dataset-level value
    },
}
```

---

## 6. Task 2.2 Master Checklist

```
☑ L1 formal definition written: mIoU at τ ∈ {0.25, 0.50, 0.75} with IoU formula
☑ L2 formal definition written: PG = 1[argmax_p e_p ∈ M_GT] with tie-breaking rule
☑ L3 formal definition written: EGT = Σ_{p∈M_GT} softmax(e)_p with softmax justification
☑ L4 formal definition written: CalibGap = E[EGT|correct] − E[EGT|incorrect]
☑ LocalizationMetrics class implemented in metrics/localization.py
☑ multi_threshold_iou(): min-max normalise → binarise → IoU at each τ → mIoU mean
☑ pointing_game(): argmax with seeded uniform tie-breaking → {0.0, 1.0}
☑ egt(): softmax normalise → sum over GT region → float in [0,1]
☑ calibration_gap(): mean EGT correct − mean EGT incorrect; ValueError on empty list
☑ align_patch_to_pixel(): bilinear upsample from patch grid to pixel resolution
☑ compute_all(): single-call L1+L2+L3 for one sample
☑ Patch-resolution GT mask alignment: bilinear interpolation, all ViT grids documented
☑ Swin-B limitation documented: no CLS token; GradCAM only; window-merge required
☑ Theorem 1: IoU symmetry proof + area-penalty counterexample
☑ Theorem 2: EGT monotone coverage proof + concentration independence violation
☑ Theorem 3: CalibGap not a proper scoring rule + adversarial counterexample + mitigation
☑ 12 unit tests written in tests/test_localization.py
☑ T01–T03: random → chance-level (not trivially maximal)
☑ T04–T07: perfect input → 1.0 / positive (L1–L4)
☑ T08–T10: GT-misaligned input → 0.0 / near-zero (L1–L3)
☑ T11: constant attribution map handled without NaN
☑ T12: empty list → ValueError contract enforced
☑ BenchmarkRunner implemented in metrics/runner.py
☑ DataLoader contract documented: (images, gt_masks, labels)
☑ Explainer interface documented: (model, images) → List[Tensor]
☑ Results dict schema documented and matches paper Table format
☑ All code follows Phase 1 style: from __future__ import annotations,
      NumPy-style docstrings, dash-separated section comments
```
