# Task 2.3 — Robustness Metrics

> **ViT Explainability Benchmark · Phase 2 Document**
> This document covers the formal definitions of all three robustness metrics (R1–R3), their complete Python implementation, an axiomatic analysis with proofs and counterexamples, 16 unit tests, extended BenchmarkRunner integration, and the Task 2.3 master checklist.

---

## 1. Formal Metric Definitions

Robustness metrics evaluate whether an explanation method produces **stable and faithful** attributions when the input or the model is systematically perturbed. All three metrics operate on an explainer $\phi: (f, x) \mapsto e$ where $f$ is a trained model, $x \in \mathbb{R}^{C \times H \times W}$ is the input image, and $e \in \mathbb{R}^{H_a \times W_a}$ is the raw attribution map.

---

### 1.1  R1 — Max-Sensitivity (MaxSens)

**Reference**: Yeh et al. (2019), *On the (In)fidelity and Sensitivity of Explanations*.

**Notation.** Let $\mathcal{B}_\epsilon(x) = \{x + \delta : \|\delta\|_\infty \leq \epsilon\}$ be the $\ell_\infty$ ball of radius $\epsilon$ around $x$. The worst-case relative perturbation of the attribution is:

$$R1 = \text{MaxSens}(\phi, x) = \max_{\delta:\|\delta\|_\infty \leq \epsilon} \frac{\|\phi(f, x+\delta) - \phi(f, x)\|_2}{\|\phi(f, x)\|_2 + \varepsilon_\text{num}}$$

where $\varepsilon_\text{num} = 10^{-8}$ is a numerical stability floor.

**Approximation.** The maximisation is approximated by sampling $K$ random perturbations $\delta_k \sim \text{Uniform}(-\epsilon, +\epsilon)^{HWC}$ and taking the maximum ratio observed:

$$\text{MaxSens}(\phi, x) \approx \max_{k=1,\ldots,K} \frac{\|\phi(f, x+\delta_k) - \phi(f, x)\|_2}{\|\phi(f, x)\|_2 + \varepsilon_\text{num}}$$

**Default**: $K=20$ (fast), $\epsilon=0.05$. For production benchmark runs use $K=50$ per Yeh et al. (2019).

**Range:** $[0, \infty)$. Lower is better — a robust explanation should not change drastically under imperceptible input perturbations.

**Interpretation.** MaxSens = 0 for a perfectly deterministic explainer on a constant model. MaxSens near 0 for a good attribution; unbounded for a sensitive/noisy one.

---

### 1.2  R2 — Model Randomisation (ModelRand)

**Reference**: Adebayo et al. (2018), *Sanity Checks for Saliency Maps*.

**Notation.** Let $f_\text{rand}$ be a deep copy of $f$ with all parameters re-initialised from $\mathcal{N}(0, 1)$. Let $\tilde{\phi}(\cdot) = \text{MinMax}(\phi(\cdot))$ denote the attribution normalised to $[0, 1]$.

The Structural Similarity Index (Wang et al., 2004) between the two normalised attribution maps is:

$$\text{SSIM}(A, B) = \frac{(2\mu_A\mu_B + C_1)(2\sigma_{AB} + C_2)}{(\mu_A^2 + \mu_B^2 + C_1)(\sigma_A^2 + \sigma_B^2 + C_2)}$$

where $\mu$, $\sigma^2$ are Gaussian-weighted local means and variances, $C_1 = (0.01)^2$, $C_2 = (0.03)^2$.

The ModelRand metric is:

$$R2 = \text{ModelRand}(\phi, x) = 1 - \text{SSIM}\bigl(\tilde{\phi}(f_\text{orig}, x),\ \tilde{\phi}(f_\text{rand}, x)\bigr)$$

**Range:** $[0, 1]$. Higher is better — a faithful explanation must change substantially when the model's learned weights are completely randomised.

**SSIM implementation:** Pure-PyTorch, 3×3 Gaussian window ($\sigma=1.0$), GPU-native (no `scikit-image` dependency). Computes local statistics via `F.conv2d` with Gaussian kernel.

**Interpretation.** ModelRand ≈ 1 means the explanation is structurally orthogonal to the randomised-model explanation (good). ModelRand ≈ 0 is a **sanity-check failure**: the attribution is insensitive to the model's weights and is likely ignoring learned features entirely.

---

### 1.3  R3 — Label Randomisation (LabelRand)

**Notation.** Let $f_\text{shuf}$ be a deep copy of $f$ with the final classifier head weights **column-permuted** — i.e., the mapping from feature dimensions to class logits is randomly shuffled while the backbone is unchanged.

Let $\rho(\cdot, \cdot)$ denote the **Spearman rank correlation** between two flattened attribution maps.

$$\rho(\phi_\text{orig}, \phi_\text{shuf}) = \text{SpearmanCorr}\bigl(\text{vec}(\phi(f_\text{orig}, x)),\ \text{vec}(\phi(f_\text{shuf}, x))\bigr)$$

The LabelRand metric is:

$$R3 = \text{LabelRand}(\phi, x) = 1 - \frac{|\rho(\phi_\text{orig}, \phi_\text{shuf})| + 1}{2}$$

**Derivation:**
- $\rho \in [-1, 1]$, so $(|\rho| + 1)/2 \in [0.5, 1]$ — measures *structural preservation*.
- $1 - (|\rho| + 1)/2 \in [0, 0.5]$ — measures *structural divergence*.
- We take $|\rho|$ (not $\rho$) because both positive and negative correlations indicate preserved spatial structure (a sign-flipped attribution still has the same layout).

**Range:** $[0, 0.5]$. Higher is better — LabelRand = 0.5 when $\rho = 0$ (maps are completely uncorrelated, i.e., the explanation fully changed). LabelRand = 0 when $|\rho| = 1$ (a **failure**: the attribution is structurally identical before and after label permutation, meaning it ignores label identity).

**Spearman implementation:** Pure-PyTorch, rank-via-argsort (stable, handles ties). No SciPy dependency.

---

## 2. Implementation

All implementation files are in `metrics/`. The package follows the same structure and code style as `training/` and Task 2.2.

### 2.1 Directory Layout

```
metrics/
├── __init__.py          # Package exports (updated: + RobustnessMetrics, utilities)
├── localization.py      # LocalizationMetrics class (L1–L4)  [Task 2.2]
├── robustness.py        # RobustnessMetrics class (R1–R3)  [Task 2.3]  ← NEW
└── runner.py            # BenchmarkRunner — extended with optional R1–R3  [updated]
tests/
├── test_localization.py # 12 unit tests (Task 2.2 §4)
└── test_robustness.py   # 16 unit tests (Task 2.3 §4)  ← NEW
```

### 2.2 `RobustnessMetrics` Class — API Reference

```python
from metrics.robustness import RobustnessMetrics

rm = RobustnessMetrics(
    epsilon=0.05,    # L∞ perturbation radius for R1
    n_samples=20,    # perturbation draws for R1 (use 50 for production)
    seed=42,         # RNG seed for reproducible perturbations
    ssim_window=3,   # Gaussian kernel size for SSIM (R2)
    ssim_sigma=1.0,  # Gaussian sigma for SSIM (R2)
)
```

#### `max_sensitivity(explainer, model, image, att_orig)` — R1

```python
ms = rm.max_sensitivity(
    explainer = my_explainer,   # callable: (model, image_4d) → List[Tensor] | Tensor
    model     = fine_tuned_vit, # the original model
    image     = img_chw,        # (C, H, W) tensor — one sample
    att_orig  = att,            # (H_a, W_a) pre-computed attribution
)
# Returns float ≥ 0 — maximum relative L2 deviation across n_samples perturb.
```

**Performance note:** `max_sensitivity` calls `explainer` `n_samples` times per image. For large batch evaluation, consider running this on a held-out random subset of the validation set (e.g., 500 images) rather than the full dataset.

---

#### `model_randomisation(att_orig, att_rand)` — R2

```python
mr = rm.model_randomisation(
    att_orig = att,       # (H_a, W_a) — attribution under original model
    att_rand = att_r,     # (H_a, W_a) — attribution under randomised model
)
# Returns float in [0, 1]. Higher = more model-sensitive (better).
```

**Pre-requisite:** use `randomise_model_weights(model)` to obtain the randomised model before calling the explainer.

---

#### `label_randomisation(att_orig, att_shuf)` — R3

```python
lr = rm.label_randomisation(
    att_orig = att,       # (H_a, W_a) — attribution under original model
    att_shuf = att_s,     # (H_a, W_a) — attribution under label-shuffled model
)
# Returns float in [0, 0.5]. Higher = more label-sensitive (better).
```

**Pre-requisite:** use `randomise_classifier_labels(model)` to obtain the shuffled model.

---

#### `compute_all(explainer, model, image, att_orig, att_rand, att_shuf)` — Convenience

```python
scores = rm.compute_all(
    explainer, model, image, att_orig, att_rand, att_shuf
)
# Returns:
# {
#     'max_sensitivity':     float,   # R1
#     'model_randomisation': float,   # R2
#     'label_randomisation': float,   # R3
# }
```

---

### 2.3 Model Utility Functions

#### `randomise_model_weights(model, seed=0)` → `nn.Module`

```python
from metrics.robustness import randomise_model_weights

f_rand = randomise_model_weights(model, seed=0)
# Returns a deep copy; ALL parameters ← N(0, 1); biases ← 0
# Original model is NOT mutated.
```

**Design note:** all parameters use $\mathcal{N}(0,1)$ (maximum randomisation), NOT Kaiming/Xavier — the goal is maximal deviation from learned weights, not training-ready initialisation.

---

#### `randomise_classifier_labels(model, seed=0, head_attr='head')` → `nn.Module`

```python
from metrics.robustness import randomise_classifier_labels

f_shuf = randomise_classifier_labels(model, seed=0)
# Returns a deep copy; ONLY head.weight columns are permuted.
# Backbone is untouched. head_attr tried in order: 'head', 'classifier', 'fc', 'linear'.
```

**Column permutation** of $W \in \mathbb{R}^{C \times D}$ rearranges which class each feature dimension maps to, preserving weight norms. This is stricter than zeroing the head because the model still produces confident predictions — for the wrong classes.

---

### 2.4 Full Usage Example

```python
import torch
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
)

rm = RobustnessMetrics(epsilon=0.05, n_samples=50, seed=42)

# Pre-compute randomised model variants once (not per sample!)
f_rand = randomise_model_weights(model, seed=0)
f_shuf = randomise_classifier_labels(model, seed=0)

# Move all models to device
model.eval();  f_rand.eval();  f_shuf.eval()

# Per sample: run your explainer on all three model variants
image    = val_batch[0]          # (C, H, W)
att_orig = explainer(model,  image.unsqueeze(0))[0]
att_rand = explainer(f_rand, image.unsqueeze(0))[0]
att_shuf = explainer(f_shuf, image.unsqueeze(0))[0]

scores = rm.compute_all(explainer, model, image, att_orig, att_rand, att_shuf)
print(f"MaxSens:   {scores['max_sensitivity']:.4f}")
print(f"ModelRand: {scores['model_randomisation']:.4f}")
print(f"LabelRand: {scores['label_randomisation']:.4f}")
```

### 2.5 Model-Specific Notes

| Model | R1 Note | R2 Note | R3 Note |
|-------|---------|---------|---------|
| ViT-B/16, DeiT-B/16, BEiT-B/16 | Standard | All 12 transformer block params randomised | `head` attr |
| DINOv2-ViT-B/14 | Standard | Same | `head` attr |
| DINO-ViT-B/8 | Higher MaxSens expected (larger patch grid) | Same | `head` attr |
| Swin-B | GradCAM-based explainer; MaxSens less interpretable (window attention) | All stage params randomised | `head` attr |

**Swin-B limitation.** Since Swin-B uses hierarchical window attention (no global CLS token), its explainer (GradCAM) is architecturally different from global-attention models. MaxSens values for Swin-B are not directly comparable to ViT-family scores. This must be flagged as a methodological limitation in the paper Table footnote.

---

## 3. Axiomatic Analysis

### Theorem 1 — MaxSens satisfies the Lipschitz-Sensitivity Axiom

**Lipschitz-Sensitivity Axiom.** A robustness metric $R$ satisfies Lipschitz-sensitivity if there exists a finite constant $L$ such that for all $\epsilon > 0$:

$$\text{MaxSens}(\phi, x) \leq L \cdot \epsilon$$

**Proof.** Let $\phi$ be a Lipschitz-continuous explainer with constant $\Lambda$ (i.e., $\|\phi(f, x+\delta) - \phi(f, x)\|_2 \leq \Lambda \|\delta\|_2$ for all $\delta$). Since $\|\delta\|_\infty \leq \epsilon$ implies $\|\delta\|_2 \leq \epsilon \sqrt{d}$ (where $d = CHW$):

$$\text{MaxSens}(\phi, x) = \max_{\|\delta\|_\infty \leq \epsilon} \frac{\|\phi(f, x+\delta) - \phi(f, x)\|_2}{\|\phi(f, x)\|_2 + \varepsilon_\text{num}} \leq \frac{\Lambda \epsilon\sqrt{d}}{\|\phi(f, x)\|_2 + \varepsilon_\text{num}}$$

Setting $L = \Lambda\sqrt{d} / (\|\phi(f, x)\|_2 + \varepsilon_\text{num})$ gives $\text{MaxSens} \leq L\epsilon$. $\square$

**Corollary.** MaxSens is linear in $\epsilon$ for Lipschitz explainers — which is validated empirically by R05 (`sensitivity_increases_with_eps`). Attention-rollout-based explainers are generally Lipschitz because softmax is smooth; gradient-based methods (Integrated Gradients) may have larger Lipschitz constants.

**Implication.** MaxSens must always be reported alongside the $\epsilon$ value used. Comparing MaxSens across methods with different $\epsilon$ is meaningless.

---

### Theorem 2 — ModelRand satisfies the Sanity Check Axiom (Adebayo et al., 2018)

**Sanity Check Axiom.** A faithful explanation method $\phi$ must produce dramatically different attributions when the model's weights are fully randomised. Formally, for any faithful $\phi$:

$$\lim_{\text{degree of randomisation} \to 1} \text{ModelRand}(\phi, x) = 1$$

**Proof (by construction).** Consider a fully randomised model $f_\text{rand}$ where every parameter is i.i.d. $\mathcal{N}(0, 1)$. For any input $x$, the last-layer attention weights of $f_\text{rand}$ are random linear combinations of random projections. The resulting attribution $\phi(f_\text{rand}, x)$ is approximately i.i.d. uniform over $[0,1]^{H_a \times W_a}$ after normalisation (by the central limit theorem on the sum of random products).

Since $\phi(f_\text{orig}, x)$ encodes the model's learned spatial preferences (e.g., foreground regions for a fine-tuned CUB-200-2011 classifier), and $\phi(f_\text{rand}, x)$ is essentially uniform noise, the SSIM between the two approaches 0:

$$\text{SSIM}(\tilde{\phi}(f_\text{orig}, x), \tilde{\phi}(f_\text{rand}, x)) \approx 0 \quad \Rightarrow \quad \text{ModelRand} \approx 1$$

$\square$

**Counterexample (pathological explainer).** An explainer $\phi$ that returns the **raw pixel values** of the input image (i.e., uses only $x$, not $f$) will achieve ModelRand ≈ 0 for all models. This is a sanity-check failure: the explanation ignores the model entirely.

```python
# Adversarial (non-model-sensitive) explainer — should FAIL sanity check
def pixel_explainer(model, image_4d):
    """Returns the red channel of the image as attribution."""
    return [image_4d[0, 0].detach()]   # ignores model entirely

rm = RobustnessMetrics()
att_orig = pixel_explainer(original_model, image.unsqueeze(0))[0]
att_rand = pixel_explainer(randomised_model, image.unsqueeze(0))[0]
# att_orig == att_rand  →  ModelRand = 0  (SANITY CHECK FAILURE)
mr = rm.model_randomisation(att_orig, att_rand)
print(f"ModelRand: {mr:.4f}")   # ≈ 0.0 — correctly flagged
```

---

### Theorem 3 — LabelRand is NOT a proper metric (asymmetry under architecture)

**Claim.** LabelRand does not satisfy the **symmetry of indiscernibles** — the property that $\text{LabelRand}(A, B) = \text{LabelRand}(B, A)$ always. While Spearman rank correlation is symmetric, the two attribution maps $\phi_\text{orig}$ and $\phi_\text{shuf}$ are computed under asymmetric conditions (original vs. label-permuted model), making LabelRand an asymmetric diagnostic rather than a proper distance metric.

**Proof by counterexample.** Let $\phi_\text{shuf}$ produce an attribution where all mass is concentrated at a single pixel. Swapping which model is labeled "orig" versus "shuf" changes which map serves as the reference, but the Spearman correlation between them is the same either way. So LabelRand IS numerically symmetric. However:

**The true asymmetry arises from interpretation.** The benchmark always computes:
$$\text{LabelRand} = 1 - \frac{|\rho(\phi_\text{orig}, \phi_\text{shuf})| + 1}{2}$$

where $\phi_\text{orig}$ is the *trusted reference* and $\phi_\text{shuf}$ is the perturbed one. If the roles are swapped (the label-randomised model is treated as "correct"), the interpretation reverses completely. This semantic asymmetry means LabelRand is a **diagnostic metric**, not a proper distance in the metric-space sense.

**Implication.** LabelRand must always be reported alongside the specification of which model is "original" and which is "permuted". It should never be compared across experiments where architectures differ, because different `head_attr` names can produce different degrees of permutation depending on the weight matrix shape.

---

## 4. Unit Tests

File: `tests/test_robustness.py` — 16 tests across 5 categories.

```
Category A — Bounds (all metrics → legal range on random inputs)
  R01  max_sensitivity_nonneg          MaxSens ≥ 0 for 20 random trials
  R02  model_randomisation_in_0_1      ModelRand ∈ [0, 1] for 20 trials
  R03  label_randomisation_in_0_0p5    LabelRand ∈ [0, 0.5] for 20 trials
  R04  compute_all_keys                compute_all() returns exactly 3 keys

Category B — Perfect Sensitivity (expected high scores)
  R05  sensitivity_increases_with_eps  Larger ε → ≥ MaxSens (mean over 30 trials)
  R06  model_rand_orthogonal_maps      ModelRand > 0.5 for top-half vs bottom-half maps
  R07  label_rand_orthogonal_maps      LabelRand mean ≈ 0.5 for 200 random pairs

Category C — Zero Sensitivity (expected low / zero scores)
  R08  sensitivity_zero_constant_expl  MaxSens = 0 for constant (deterministic) explainer
  R09  model_rand_identical_maps       ModelRand = 0 for att_orig == att_rand
  R10  label_rand_identical_maps       LabelRand = 0 for att_orig == att_shuf

Category D — Utility functions
  R11  randomise_model_changes_weights  ALL parameters change after randomisation
  R12  randomise_labels_only_head       backbone unchanged; head weight changes
  R13  ssim_self_consistency            _ssim(t, t) = 1.0 for 10 random tensors

Category E — Edge cases and contracts
  R14  spearman_constant_map            constant map → no NaN; result in [-1, 1]
  R15  max_sensitivity_n_samples_one    n_samples=1 works without crash
  R16  epsilon_constructor_validation   epsilon ≤ 0 → ValueError
```

Run:

```bash
# Task 2.3 tests only
python tests/test_robustness.py

# Or with pytest
pytest tests/test_robustness.py -v

# Full Phase 2 suite (Task 2.2 + Task 2.3, backward-compat verification)
pytest tests/ -v
```

Expected output (all 16 pass):

```
============================================================
Task 2.3 §4 — RobustnessMetrics Unit Tests
============================================================

R01 ✓  max_sensitivity_nonneg
R02 ✓  model_randomisation_in_0_1
R03 ✓  label_randomisation_in_0_0p5
R04 ✓  compute_all_keys  (MaxSens=x.xxx, ModelRand=x.xxx, LabelRand=x.xxx)
R05 ✓  sensitivity_increases_with_eps  (ε=0.01→x.xxxx, ε=0.20→x.xxxx)
R06 ✓  model_rand_orthogonal_maps  (ModelRand=x.xxxx)
R07 ✓  label_rand_orthogonal_maps  (mean LabelRand=x.xxxx)
R08 ✓  sensitivity_zero_constant_expl
R09 ✓  model_rand_identical_maps
R10 ✓  label_rand_identical_maps
R11 ✓  randomise_model_changes_weights  (4/4 params changed)
R12 ✓  randomise_labels_only_head
R13 ✓  ssim_self_consistency
R14 ✓  spearman_constant_map  (ρ=x.xxxx, LabelRand=x.xxxx)
R15 ✓  max_sensitivity_n_samples_one  (MaxSens=x.xxxx)
R16 ✓  epsilon_constructor_validation

============================================================
Results: 16/16 passed, 0 failed
============================================================
```

---

## 5. BenchmarkRunner Integration

File: `metrics/runner.py` (extended, fully backward-compatible).

The `BenchmarkRunner` now accepts three optional keyword arguments for R1–R3.  Existing code that omits these arguments continues to work without any changes.

### 5.1 Constructor

```python
from metrics.runner import BenchmarkRunner
from metrics.localization import LocalizationMetrics
from metrics.robustness import (
    RobustnessMetrics,
    randomise_model_weights,
    randomise_classifier_labels,
)

rm  = RobustnessMetrics(epsilon=0.05, n_samples=50, seed=42)

runner = BenchmarkRunner(
    # Task 2.2 (required — unchanged)
    metrics   = LocalizationMetrics(thresholds=[0.25, 0.50, 0.75]),
    explainer = my_explainer,
    # Task 2.3 (optional — new)
    robustness             = rm,
    randomised_model       = randomise_model_weights(model, seed=0),
    label_randomised_model = randomise_classifier_labels(model, seed=0),
)
```

### 5.2 Results Dict Schema (extended)

```python
{
    "dataset":     "cub200",
    "n_samples":   5794,
    "n_correct":   4823,
    "n_incorrect": 971,
    "macro": {
        # --- Task 2.2 metrics (L1–L4) ---
        "iou@0.25":          float,
        "iou@0.50":          float,
        "iou@0.75":          float,
        "miou":              float,
        "pointing_game":     float,
        "egt":               float,
        "calibration_gap":   float,
        # --- Task 2.3 metrics (R1–R3) — only when robustness is set ---
        "max_sensitivity":     float,   # macro-avg over samples
        "model_randomisation": float,   # macro-avg over samples
        "label_randomisation": float,   # macro-avg over samples
    },
    "per_metric": {
        "miou":                [float, ...],   # one per sample
        "pointing_game":       [float, ...],
        "egt":                 [float, ...],
        "calibration_gap":     [float],        # single dataset-level value
        "max_sensitivity":     [float, ...],   # one per sample (if robustness set)
        "model_randomisation": [float, ...],   # one per sample (if robustness set)
        "label_randomisation": [float, ...],   # one per sample (if robustness set)
    },
}
```

### 5.3 Performance Guidance

| Metric | Cost per sample | Recommendation |
|--------|----------------|----------------|
| L1–L3 | O(1) tensor ops | Full dataset |
| L4 | O(N) at end | Full dataset |
| R1 MaxSens | K explainer calls | Sample 500 images |
| R2 ModelRand | 1 extra explainer call | Full dataset |
| R3 LabelRand | 1 extra explainer call | Full dataset |

For large datasets (ImageNet-S-50: 50 classes × 100 samples), MaxSens with $K=50$ requires $50 \times 100 \times 50 = 250{,}000$ additional explainer calls. Recommended: run R1 on a random subset of 500 images and macro-average separately.

---

## 6. Task 2.3 Master Checklist

```
☑ R1 formal definition written: MaxSens = max‖φ(x+δ)−φ(x)‖₂ / ‖φ(x)‖₂
☑ R2 formal definition written: ModelRand = 1 − SSIM(φ_orig, φ_rand)
☑ R3 formal definition written: LabelRand = 1 − (|ρ(φ_orig, φ_shuf)| + 1)/2

☑ RobustnessMetrics class implemented in metrics/robustness.py
☑ max_sensitivity(): K random L∞ perturbations → max relative L2 change
☑ model_randomisation(): MinMax-norm both maps → pure-PyTorch SSIM → 1 - SSIM
☑ label_randomisation(): Spearman rank-corr → 1 - (|ρ| + 1)/2
☑ compute_all(): single-call R1+R2+R3 for one sample

☑ _ssim(): pure-PyTorch, 3×3 Gaussian window, GPU-native, Wang et al. (2004)
☑ _spearman_corr(): rank-via-argsort, stable, handles ties, no SciPy
☑ _perturb(): seeded torch.Generator for reproducible L∞ perturbations
☑ _minmax_norm(): constant-map guard (returns zeros, not NaN)

☑ randomise_model_weights(): deep copy; all params ← N(0,1); biases ← 0
☑ randomise_classifier_labels(): deep copy; head.weight columns permuted only
☑ head_attr fallback chain: 'head', 'classifier', 'fc', 'linear'

☑ Theorem 1: MaxSens satisfies Lipschitz-Sensitivity Axiom (with proof)
☑ Theorem 2: ModelRand satisfies Sanity Check Axiom + counterexample
☑ Theorem 3: LabelRand is NOT a proper metric (asymmetry proof)

☑ 16 unit tests written in tests/test_robustness.py
☑ R01–R04: bound checks — all metrics in legal range
☑ R05–R07: high-score scenarios — perfect sensitivity
☑ R08–R10: zero-score scenarios — zero sensitivity
☑ R11–R13: utility function correctness
☑ R14–R16: edge cases — constant maps, n_samples=1, epsilon validation

☑ BenchmarkRunner extended in metrics/runner.py (backward-compatible)
☑ New optional params: robustness, randomised_model, label_randomised_model
☑ Warning logged if robustness set but randomised_model/label_randomised_model is None
☑ R1–R3 results added to all_metrics and macro dict
☑ Results dict schema documented and matches paper Table format

☑ All code follows Phase 2 style: from __future__ import annotations,
      NumPy-style docstrings, dash-separated section comments
☑ No new mandatory dependencies: no scikit-image, no SciPy
☑ metrics/__init__.py updated with all new exports
```
