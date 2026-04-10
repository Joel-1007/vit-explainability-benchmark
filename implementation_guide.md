# ViT Explainability Benchmark
## A Comprehensive Explainability Benchmark for Vision Transformers
### Step-by-step Implementation Guide

**Target venue:** IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

---

## Project Timeline at a Glance

| Phase | Name | Duration | Key Output |
|-------|------|----------|------------|
| 1 | Foundations & Scope | Months 1–2 | Gap table, model/dataset plan |
| 2 | Metric Framework Design | Months 2–4 | Formal metric suite |
| 3 | Baseline Evaluation Pipeline | Months 4–7 | Full results matrix |
| 4 | Analysis, Ablations & Theory | Months 7–10 | Findings & proofs |
| 5 | Writing & Submission | Months 10–13 | Submitted manuscript |

> This document is your working guide. Print it, annotate it, tick off the checklists.

---

## How to Use This Guide

This document is a step-by-step operational guide for executing the ViT explainability benchmark project from first principles through to TPAMI submission. Each phase contains:

- A precise description of what to do and in what order.
- Checklists you can tick off as you complete each sub-task.
- **Reviewer risk boxes** flagging the most common objections at TPAMI and how your process preempts them.
- **Tip boxes** with practical research advice.
- **Code skeletons** where implementation decisions need to be locked in early.
- **Deliverable tags** marking what concrete artefact each task produces.

### Before You Begin: The Core Claim You Are Defending

Your paper makes one central claim: **existing evaluation frameworks for ViT explanations are inconsistent, narrowly scoped, and lack standardised methodology, causing incomparable results across the literature.** Every task in this guide exists to make that claim defensible with mathematical and empirical rigour. Keep it in mind throughout.

---

## Phase 1 — Foundations & Scope Definition *(Months 1–2)*

### Overview

The goal of Phase 1 is to produce a precise, defensible problem statement before writing a single line of experimental code. TPAMI reviewers reject benchmark papers that fail to clearly delineate their contribution from prior work, so your first two months are almost entirely literature-and-planning work. At the end of this phase you should have:

1. A completed gap analysis table
2. A locked-in model zoo with a shared training protocol
3. A dataset plan with justification

---

### Task 1.1 — Literature Review and Gap Analysis

#### What to Do

You need to survey every major XAI evaluation framework and produce a structured gap table. This is not a casual literature review; it is a systematic audit. Work through the following paper categories methodically.

#### Category A: Foundational XAI Methods You Will Benchmark

Read the original papers for each of the following methods, paying attention to what evaluation each paper uses internally (this is often weak or non-existent, which supports your thesis).

- **Raw attention:** Dosovitskiy et al. (2021), *"An Image is Worth 16×16 Words."* Note that the authors themselves do not rigorously evaluate whether the CLS-to-patch attention is interpretable.
- **Attention rollout:** Abnar & Zuidema (2020), *"Quantifying Attention Flow in Transformers."* Understand the recursive averaging formula. Note its assumption that attention is non-negative and that skip connections complicate it.
- **Transformer-LRP (Chefer et al.):** Chefer, Gur & Wolf (2021), *"Transformer Interpretability Beyond Attention Visualization."* This is the current strongest ViT-specific method and your most important baseline.
- **GradCAM & variants:** Selvaraju et al. (2017) and the adaptation for ViTs (grad w.r.t. token embeddings).
- **DIME:** Kokalj et al. (2021), *"DIME: Fine-grained Interpretations of Multimodal Models via Disentangled Local Explanations."*
- **RISE:** Petsiuk et al. (2018). A perturbation-based black-box method. Understand why it is slow but model-agnostic.
- **LIME / SHAP (patch-adapted):** The original papers plus any ViT-specific adaptation. These are less common in vision but reviewers expect you to consider them.

#### Category B: Existing Evaluation Frameworks

These are the papers your work must clearly surpass or complement.

- **Pixel flipping / AOPC:** Samek et al. (2017), *"Evaluating the Visualization of What a Deep Neural Network Has Learned."*
- **ROAR/KAR:** Hooker et al. (2019), *"A Benchmark for Interpretability Methods in Deep Neural Networks."* Read this carefully — it is the most-cited critique of perturbation-based metrics and you will need to address the distribution-shift problem it identifies.
- **Insertion / Deletion:** Petsiuk et al. (2018). Used in their RISE paper but generally applicable.
- **Sensitivity analysis:** Yeh et al. (2019), *"On the (In)fidelity and Sensitivity of Explanations."*
- **Faithfulness and plausibility:** Jacovi & Goldberg (2020), *"Towards Faithfully Interpretable NLP Systems."* NLP-focused but the framework is transferable.
- **BenchXAI:** Find the most recent version. This is your closest competitor; you must explicitly state what you add.
- **Attention critiques:** Jain & Wallace (2019), *"Attention is not Explanation,"* and Wiegreffe & Pinar (2019), *"Attention is not not Explanation."* Understand both sides; your framework should help settle this debate empirically for ViTs.

#### Constructing the Gap Table

Once you have read all the above, fill in the following table. The columns are evaluation properties your benchmark will cover; the rows are prior frameworks. A ✓ means the framework addresses this property adequately; a ~ means it addresses it partially; a blank means it does not.

**Table 1: Gap analysis table — fill this in during Phase 1. Blank cells are your contribution.**

| Framework | Fidelity | Localiz. | Robust. | Complex. | ViT-specific | Multi-arch |
|-----------|----------|----------|---------|----------|--------------|------------|
| Samek et al. (2017) | | | | | | |
| ROAR (Hooker 2019) | | | | | | |
| Ins./Del. (Petsiuk) | | | | | | |
| Sensitivity (Yeh) | | | | | | |
| BenchXAI | | | | | | |
| Chefer et al. eval | | | | | | |
| **This paper** | | | | | | |

> **📦 Deliverable:** Completed gap table (Table 1) with written justification of each entry, approximately 2 pages.

> ⚠️ **Reviewer Risk:** If the gap table reveals that BenchXAI or another recent framework already covers 4 out of 6 columns adequately, your contribution is at risk. In that case, your distinguishing claim must shift to ViT-specific methodology (e.g., the interaction between attention heads and spatial explanation quality) rather than breadth of metrics. Decide this before proceeding.

**Checklist:**
- [ ] Read all Category A papers (methods). Take structured notes: what does each output? what does it assume about the model?
- [ ] Read all Category B papers (evaluations). Note specifically: which models are tested, which metrics, which datasets.
- [ ] Construct Table 1. Write one paragraph per row justifying your entries.
- [ ] Write a 1-page problem statement: what does a comprehensive ViT explanation benchmark require that does not currently exist?

---

### Task 1.2 — Model Zoo Selection and Training Protocol

#### What to Do

Select 5–7 ViT architectures. The guiding principle is **controlled variation**: you want architectures that differ along identifiable axes (attention mechanism design, patch size, pre-training objective) so that differences in explanation quality can be attributed to architecture, not confounds.

**Table 2: Proposed model zoo. Fill in the rightmost column with your confirmed checkpoint sources.**

| Model | Arch. | Pre-training | Patch | Params | Checkpoint Source |
|-------|-------|--------------|-------|--------|-------------------|
| ViT-B/16 | Standard | Supervised (IN-21K) | 16 | 86M | timm / HuggingFace |
| ViT-L/16 | Standard | Supervised (IN-21K) | 16 | 307M | timm |
| DeiT-B/16 | DeiT | Distillation (IN-1K) | 16 | 86M | facebookresearch |
| Swin-B | Shifted-window | Supervised (IN-21K) | 4 | 88M | microsoft/swin |
| BEiT-B/16 | BERT-style | Masked image (IN-21K) | 16 | 86M | microsoft/beit |
| MAE-ViT-B/16 | Standard | Masked autoencoder | 16 | 86M | facebookresearch/mae |

#### Training Protocol

> ⚠️ **Critical for reviewer credibility:** All models must be fine-tuned on each downstream dataset using an identical fine-tuning recipe. Do not use off-the-shelf classification checkpoints for different models trained under different conditions — this is a common mistake that makes results incomparable and will surface immediately in review.

Your fine-tuning recipe must specify:

- **Optimiser:** AdamW, β₁ = 0.9, β₂ = 0.999, weight decay = 0.05
- **Learning rate schedule:** cosine decay with 5-epoch warmup
- **Base learning rate:** 10⁻⁴ with linear scaling rule (scale by batch size / 256)
- **Epochs:** 50 for CUB-200, 30 for ImageNet fine-tuning head only
- **Data augmentation:** RandAugment (M=9, N=2), mixup (α = 0.8), label smoothing = 0.1
- **Input resolution:** 224 × 224 for all models
- **Batch size:** 256 (adjust for memory; accumulate gradients if needed)

Write this protocol down explicitly. It will go verbatim into your paper's implementation details section.

**Checklist:**
- [ ] Confirm access to checkpoints for all 6 models (pre-trained weights only, not fine-tuned).
- [ ] Write the fine-tuning protocol document (it should be specific enough that a third party can replicate it exactly).
- [ ] Run a small pilot fine-tune (5 epochs) on one model, one dataset, to confirm your hardware can handle the batch size and that the recipe produces reasonable accuracy. Target: within 1% of published fine-tuned accuracy.
- [ ] Record exact model hashes / commit SHAs so results are reproducible.

---

### Task 1.3 — Dataset Selection and Justification

#### What to Do

You need datasets that stress different aspects of explanation quality. Each dataset must be justified in terms of what evaluation dimension it uniquely enables.

**Table 3: Dataset plan with justification.**

| Dataset | Classes | GT Masks? | Justification |
|---------|---------|-----------|---------------|
| ImageNet-1K | 1000 | Bounding boxes | Scale; established baselines for comparison |
| CUB-200-2011 | 200 | Part annotations | Fine-grained; forces tight localization |
| PASCAL VOC 2012 (seg. subset) | 20 | Pixel-level masks | Precise IoU-based localization metrics |
| Medical subset (CheXpert 5 findings) | 5 | Expert heatmaps | High-stakes; ground truth from radiologists |

For the medical subset: use the 5-class pathology subset of CheXpert (Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion) with the publicly available radiologist attention maps from the CheXpert paper's localisation task. This gives you clinician-annotated heatmaps as ground truth for your localization metrics — a significant differentiator from prior work.

**Checklist:**
- [ ] Download and verify all datasets. Confirm train/val/test splits.
- [ ] For CUB-200: confirm that part annotation files are intact and that you can compute IoU with them.
- [ ] For PASCAL VOC: confirm that segmentation masks are available for your chosen class subset.
- [ ] For CheXpert: obtain the localisation annotation files. Verify that the coordinate system matches your model's input resolution.
- [ ] Write a one-paragraph justification for each dataset that explains which specific metrics it enables. This becomes your Experiments section setup paragraph.

---

## Phase 2 — Metric Framework Design *(Months 2–4)*

### Overview

This phase is the intellectual core of the paper. You will formally define every metric in your benchmark, prove or explicitly analyse their axiomatic properties, and implement them as a clean, tested Python library. TPAMI expects mathematical rigour: vague operational descriptions of metrics will be rejected. By the end of this phase you should have a self-contained metric specification document (which becomes Sections 3–4 of your paper) and a tested implementation.

---

### Task 2.1 — Fidelity Metrics

#### Conceptual Foundation

Fidelity measures whether an explanation reflects the model's actual computation. An explanation **e** for prediction f(x) has high fidelity if perturbing the input in directions indicated by **e** degrades the prediction in a predictable way. Formally, let f(x) ∈ ℝ^C be the model's logit vector and **e** ∈ ℝ^(Hp×Wp) be a patch-level attribution map (normalised to [0, 1]).

#### Metrics to Implement

**Metric F1: Insertion AUC**

> **Definition (Insertion curve):** Order patches by descending attribution score. For k = 1, …, N (where N is the total number of patches), reveal the top-k patches (replacing all others with a baseline, typically mean-blurred) and record f_c(x^(k)), the model's confidence in the predicted class c. The insertion AUC is the area under this curve, normalised to [0, 1].

The baseline you use for masked patches matters. Use two variants:
- **(a) zero-patch:** x^(k)_mask = 0
- **(b) mean-patch:** x^(k)_mask = x̄, the per-channel dataset mean

Report both. The gap between them quantifies sensitivity to distribution shift — a finding in itself.

**Metric F2: Deletion AUC**

The mirror image: order patches by descending attribution, progressively remove them (mask to baseline). A high-quality explanation should cause rapid confidence collapse as the most important patches are removed. Deletion AUC should be **low** for a good explanation.

**Metric F3: Comprehensiveness and Sufficiency**

Borrowed from the NLP literature (DeYoung et al., 2020, ERASER) and adapted for patches:

```
Comprehensiveness(e, x, f, c) = f_c(x) − f_c(x \ top-k)
Sufficiency(e, x, f, c)       = f_c(x) − f_c(x_top-k)
```

where `x \ top-k` removes the top-k patches and `x_top-k` keeps only the top-k patches. Report at k ∈ {10%, 20%, 50%} of total patches.

**Metric F4: Log-odds Shift**

```
LogOddsShift(e, x, f, c) = log( f_c(x) / (1 − f_c(x)) ) − log( f_c(x \ top-half) / (1 − f_c(x \ top-half)) )
```

This is more interpretable than raw probability differences because it accounts for the non-linearity of the softmax.

#### Implementation Skeleton

```python
class FidelityMetrics:
    def __init__(self, model, mask_mode="mean", k_fractions=(0.1, 0.2, 0.5)):
        self.model = model
        self.mask_mode = mask_mode  # "zero" or "mean"
        self.k_fractions = k_fractions
        self.dataset_mean = None  # set from dataloader

    def _mask_patches(self, x, patch_indices, keep=True):
        """Zero or mean-replace patches not in patch_indices."""
        ...

    def insertion_auc(self, x, attribution_map):
        """Returns scalar AUC over insertion curve."""
        ...

    def deletion_auc(self, x, attribution_map):
        ...

    def comprehensiveness(self, x, attribution_map, k_frac):
        ...

    def sufficiency(self, x, attribution_map, k_frac):
        ...

    def log_odds_shift(self, x, attribution_map):
        ...
```

> ⚠️ **Reviewer Risk:** The insertion/deletion metrics require running the model N times per sample (once per patch step). With N = 196 patches (ViT-B/16 on 224×224) and a large dataset, this is expensive. Use **batched inference**: stack all N masked inputs for a single sample into one batch. Cache the original prediction. This reduces the number of forward passes by ~10×.

---

### Task 2.2 — Localization Metrics

#### Conceptual Foundation

Localization metrics measure spatial alignment between the explanation and human-annotated regions of interest. These require ground-truth annotations and are only computable on datasets with segmentation masks or bounding boxes.

#### Metrics to Implement

**Metric L1: Intersection over Union (IoU) with GT Mask**

Threshold the attribution map at value τ (try τ ∈ {0.25, 0.5, 0.75} of the maximum value) to produce a binary prediction mask M̂. Compute:

```
IoU(M̂, M_GT) = |M̂ ∩ M_GT| / |M̂ ∪ M_GT|
```

Report mean IoU across the threshold range (mIoU) to reduce sensitivity to τ.

**Metric L2: Pointing Game Accuracy**

```
PG = 𝟙[ argmax_p(e_p) ∈ M_GT ]
```

The pointing game asks: does the highest-attribution patch fall inside the ground truth region? Compute this over the full test set and report as a percentage. This is threshold-free, which makes it a useful complement to IoU.

**Metric L3: Energy-on-Ground-Truth (EGT)**

```
EGT(e, M_GT) = Σ_{p ∈ M_GT} e_p  /  Σ_p e_p
```

This is the fraction of total attribution mass inside the GT region. Unlike IoU, it captures graded attribution (a map that gives 80% weight to the GT region and 20% to irrelevant regions scores 0.8, which is informative). Normalise attribution maps to [0, 1] before computing.

**Metric L4: Calibration Gap** *(Novel contribution)*

Compute localization quality separately for correctly classified samples and misclassified samples:

```
CalibGap(e) = EGT(e | correct) − EGT(e | incorrect)
```

A positive calibration gap means the model "looks at the right thing" more when it is correct, which is expected behaviour. A near-zero or negative gap is a red flag — the explanation method is not sensitive to whether the model is reasoning correctly.

**Checklist:**
- [ ] Implement attribution map → binary mask at multiple thresholds.
- [ ] Implement IoU computation with GT masks at patch resolution.
- [ ] Implement pointing game accuracy. Handle ties (random selection from tied patches).
- [ ] Implement EGT with normalised attribution maps.
- [ ] Implement calibration gap, storing correct/incorrect split at inference time.
- [ ] Test all four metrics on a random attribution map (expected: near-chance results) and on a perfect attribution map (expected: near-1.0 results).

---

### Task 2.3 — Robustness Metrics

#### Conceptual Foundation

Robustness measures the stability of explanations under small, semantically irrelevant input changes. A good explanation should not change dramatically if you add slight noise to the input (the decision, and therefore the explanation, should not change). Crucially, robustness also includes randomisation tests, which check whether the explanation method is sensitive to whether the model has actually learned anything.

#### Metrics to Implement

**Metric R1: Max-Sensitivity (MaxSens)**

Following Yeh et al. (2019):

```
MaxSens(e, x) = max_{‖δ‖_∞ ≤ ε} ‖E(x + δ) − E(x)‖_2
```

Approximate via Monte Carlo: sample K = 50 perturbations δ uniformly from [−ε, ε]^d with ε = 0.1, compute the explanation for each, take the maximum ℓ₂ distance to the original explanation.

**Metric R2: Average-Sensitivity (AvgSens)**

The same calculation but taking the mean rather than the maximum. MaxSens captures worst-case instability; AvgSens captures typical instability.

**Metric R3: Model Parameter Randomisation Test** *(Adebayo et al., 2018)*

This is the most important robustness test. Procedure:

1. Compute explanation E_trained(x) for the trained model.
2. Re-initialise the top layer of the model with random weights. Compute E_rand-top(x).
3. Progressively randomise layers from top to bottom (cascading randomisation). At each step, compute the Spearman rank correlation between the current explanation and E_trained(x).
4. A method that produces similar explanations regardless of whether weights are random has **failed** this test — it is not using the model's learned features.

Report the Spearman correlation curve across layers. Methods that maintain high correlation through many layers of randomisation are uninformative.

**Metric R4: Label Randomisation Test**

Train a copy of each model with randomly shuffled labels (model fits noise). Run each explanation method on this model. A good explanation method should produce fundamentally different explanations for the noise-trained model compared to the correctly trained model, because the noise model has not learned meaningful features.

> ⚠️ **Critical — pre-compute and cache for R3 and R4:** Each randomisation test requires multiple forward passes with modified model weights. Do **not** modify your main model in-place. Deep-copy the model, randomise layers, run, discard. Also: label randomisation requires training a separate model per architecture per dataset — plan this into your compute budget early. It is expensive but non-negotiable for TPAMI.

---

### Task 2.4 — Complexity Metrics

#### Conceptual Foundation

Complexity measures parsimony. An explanation that assigns near-uniform weight to all patches is technically non-zero in fidelity but is useless to a human. You want to capture how "focussed" an explanation is.

#### Metrics to Implement

**Metric C1: Sparsity Index (Gini Coefficient)**

```
Gini(e) = Σ_i Σ_j |e_i − e_j| / (2N Σ_i e_i)
```

where N is the number of patches and e_i ≥ 0. A Gini coefficient of 1 means all attribution is concentrated in one patch (maximally sparse); 0 means uniform distribution (maximally diffuse).

**Metric C2: Entropy of Normalised Attribution**

```
H(e) = −Σ_p ẽ_p log ẽ_p,    ẽ_p = e_p / Σ_q e_q
```

Lower entropy = more concentrated explanation. Report both raw entropy and entropy normalised by log N (so it is in [0, 1]).

**Metric C3: Effective Mass Ratio**

The fraction of patches required to capture 90% of total attribution mass. Formally: sort patches by descending e_p, find the smallest set S such that Σ_{p ∈ S} e_p ≥ 0.9 · Σ_q e_q, report |S|/N. A good explanation has a **low** effective mass ratio.

---

### Task 2.5 — Axiomatic Analysis

#### What to Do

For TPAMI, you must go beyond operational definitions. You need to analyse which axiomatic properties (from the Shapley value / attribution literature) your metrics satisfy.

**The key axioms to check for each metric:**

> **Axiom: Dummy.** If a feature has zero influence on f(x) (i.e., removing it does not change the prediction), it should receive zero attribution.

> **Axiom: Completeness / Efficiency.** The sum of all patch attributions equals the difference between the model output and a baseline output: Σ_p e_p = f(x) − f(x_baseline).

> **Axiom: Symmetry.** If two patches contribute identically to all predictions, they receive equal attribution.

> **Axiom: Linearity.** For a model that is a linear combination of two models, the explanation is the linear combination of the individual explanations.

For each of your metrics (F1–F4, L1–L4, R1–R4, C1–C3), determine: (a) does the metric implicitly assume or reward explanations that satisfy each axiom? (b) Can you prove this formally, or can you construct a counterexample?

At minimum, write one theorem or one formal counterexample. For example:

> **Theorem:** Insertion AUC (F1) does not penalise violations of the Completeness axiom: an attribution method that assigns all weight to a single patch while that patch alone is insufficient to recover full model confidence may score highly on Insertion AUC if that patch is highly informative.
>
> *Sketch.* Suppose N = 4 patches and the model's prediction depends on patches p₁ and p₂ jointly (an XOR-like interaction). Assign e_{p₁} = 1, all others = 0. The insertion curve reveals p₁ first; if f_c(x^{(p₁)}) > 0.5, the AUC may still be high, even though p₂ is equally causally responsible. □

Include 3–5 such results. They demonstrate mathematical depth and preempt reviewer objections about whether your metrics are well-founded.

**Checklist:**
- [ ] Formally define all 13 metrics (F1–F4, L1–L4, R1–R4, C1–C3).
- [ ] Implement all 13 metrics in a unified `MetricSuite` class.
- [ ] Write unit tests: known-good and known-bad attribution maps, expected metric ranges.
- [ ] Fill in an axiom satisfaction table (rows = metrics, columns = axioms).
- [ ] Write at least 3 theorems or counterexamples for the axiomatic analysis.

---

## Phase 3 — Baseline Evaluation Pipeline *(Months 4–7)*

### Overview

Phase 3 is engineering-heavy. You will build the standardised pipeline that wraps all explainers, runs all metrics, and produces the full results matrix. The key engineering goal is **reproducibility**: every number in your results tables must be reproducible from a single command with fixed seeds.

---

### Task 3.1 — Standardised Explainer Interface

#### What to Do

Define a base class that every explanation method must implement. This is the most important software design decision in the project; do it carefully before writing any method-specific code.

```python
from abc import ABC, abstractmethod
import torch

class BaseExplainer(ABC):
    """
    All explanation methods must subclass this and implement 'explain'.
    Input: a single image tensor (3, H, W) and a target class index.
    Output: a 2D attribution map of shape (H_patches, W_patches).
    The map is NOT normalised -- normalisation is done downstream in MetricSuite.
    """

    def __init__(self, model: torch.nn.Module, patch_size: int = 16):
        self.model = model
        self.model.eval()
        self.patch_size = patch_size

    @abstractmethod
    def explain(
        self,
        x: torch.Tensor,      # shape: (3, H, W), float32, in [0,1]
        target_class: int,
        **kwargs
    ) -> torch.Tensor:         # shape: (H//patch_size, W//patch_size)
        ...

    def explain_batch(
        self,
        xs: torch.Tensor,              # shape: (B, 3, H, W)
        target_classes: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:                 # shape: (B, H//patch_size, W//patch_size)
        return torch.stack([
            self.explain(xs[i], target_classes[i].item(), **kwargs)
            for i in range(len(xs))
        ])
```

Implement this interface for each of the following methods:

1. **RawAttentionExplainer:** Extract the CLS-to-patch attention weights from the last layer. Average over attention heads. Output shape: (N_patches,) reshaped to (H_p, W_p).
2. **AttentionRolloutExplainer:** Implement the Abnar & Zuidema (2020) recursive formula. Requires access to all attention layers.
3. **GradCAMExplainer:** Compute gradient of the class logit with respect to the last layer's patch token embeddings. Average over the embedding dimension. Apply ReLU.
4. **CheferLRPExplainer:** Implement or wrap the official Chefer et al. (2021) code. This is the most complex — use the authors' released code but wrap it in the `BaseExplainer` interface.
5. **RISEExplainer:** Perturbation-based. Generate M = 4000 random binary masks, measure prediction change for each, weight masks by confidence. This is slow; vectorise over mask dimension.
6. **LIMEExplainer (patch-adapted):** Use superpixels at patch granularity. Fit a linear surrogate model. This is the weakest baseline but establishes a lower bound.
7. **DIMEExplainer:** Implement or wrap available code.

> 💡 **Tip:** For methods that require gradient access (GradCAM, Chefer-LRP), ensure you are running with `torch.no_grad()` in the outer loop but selectively enabling gradients inside `explain()`. Use `torch.enable_grad()` as a context manager. Never forget to zero gradients between samples.

---

### Task 3.2 — Normalisation and Alignment

#### What to Do

Different explainers output attribution maps in different scales and ranges. Before any metric computation, all maps must go through a standardised normalisation pipeline. This is also where you handle the difference between patch-level and pixel-level outputs.

```python
def normalise_attribution(att_map: torch.Tensor,
                           mode: str = "minmax") -> torch.Tensor:
    """
    Normalise attribution map to [0, 1].
    mode: "minmax" or "softmax" or "percentile" (clip at 99th, then minmax)
    """
    if mode == "minmax":
        mn, mx = att_map.min(), att_map.max()
        if (mx - mn).abs() < 1e-8:
            return torch.zeros_like(att_map)
        return (att_map - mn) / (mx - mn)
    elif mode == "percentile":
        p99 = torch.quantile(att_map.flatten(), 0.99)
        att_map = att_map.clamp(max=p99)
        return normalise_attribution(att_map, mode="minmax")
    elif mode == "softmax":
        return att_map.flatten().softmax(0).reshape(att_map.shape)
```

Report which normalisation mode is used for each metric family (e.g., use percentile clipping for fidelity metrics to prevent a single outlier patch from dominating, use softmax for EGT since EGT requires a proper distribution).

---

### Task 3.3 — Metric Computation Engine

#### What to Do

Wrap the `MetricSuite` and all `BaseExplainer` implementations in a top-level `BenchmarkRunner` that:

1. Iterates over dataset × model × explainer.
2. For each combination, runs the explainer to get the attribution map.
3. Runs all applicable metrics on that map.
4. Saves results to a structured dictionary keyed by `(dataset, model, explainer, metric)`.
5. **Checkpoints results after each batch** so a crash does not lose work.

```python
class BenchmarkRunner:
    def __init__(self, models, explainers, metrics, datasets, device):
        self.models    = models     # dict: name -> model
        self.explainers = explainers  # dict: name -> BaseExplainer class
        self.metrics   = metrics    # MetricSuite instance
        self.datasets  = datasets   # dict: name -> DataLoader
        self.device    = device
        self.results   = {}         # populated during run

    def run(self, checkpoint_dir: str, seed: int = 42):
        torch.manual_seed(seed)
        for ds_name, loader in self.datasets.items():
            for model_name, model in self.models.items():
                model = model.to(self.device)
                for exp_name, ExplainerClass in self.explainers.items():
                    explainer = ExplainerClass(model)
                    self._run_combination(
                        ds_name, model_name, exp_name,
                        explainer, loader, checkpoint_dir
                    )

    def _run_combination(self, ds_name, model_name, exp_name,
                          explainer, loader, checkpoint_dir):
        key = f"{ds_name}__{model_name}__{exp_name}"
        ckpt_path = os.path.join(checkpoint_dir, f"{key}.pkl")
        if os.path.exists(ckpt_path):
            print(f"Skipping {key} (checkpoint found)")
            return
        results = defaultdict(list)
        for batch in tqdm(loader, desc=key):
            x, y, gt_mask = batch
            x = x.to(self.device)
            att = explainer.explain_batch(x, y)
            batch_results = self.metrics.compute_all(x, att, y, gt_mask)
            for metric_name, vals in batch_results.items():
                results[metric_name].extend(vals.tolist())
        with open(ckpt_path, "wb") as f:
            pickle.dump(dict(results), f)
        self.results[key] = dict(results)
```

---

### Task 3.4 — Sanity Checks *(Mandatory)*

#### What to Do

Before running any real experiments, run the following sanity checks. These are mandatory for TPAMI and should be reported as a dedicated subsection in your paper.

**Sanity Check S1: Random Explanation Baseline**

For every metric, compute what score a uniformly random attribution map achieves. This establishes the chance-level baseline. If any of your explanation methods score near or below this baseline, it is a significant finding (and should be reported, not hidden).

```python
def random_baseline_metrics(metrics, loader, n_repeats=5):
    results = defaultdict(list)
    for x, y, gt_mask in loader:
        for _ in range(n_repeats):
            random_att = torch.rand(x.shape[0], H_p, W_p)
            batch_results = metrics.compute_all(x, random_att, y, gt_mask)
            for metric_name, vals in batch_results.items():
                results[metric_name].extend(vals.tolist())
    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}
```

**Sanity Check S2: Model Parameter Randomisation**

For each explainer, run the full randomisation test (Metric R3) and confirm that at full randomisation (all layers re-initialised), the Spearman correlation between trained and randomised explanations approaches the random-baseline correlation. If a method passes this check, it is a good sign. If a method shows high correlation even at full randomisation, flag it as potentially non-informative.

**Sanity Check S3: Label Permutation**

Confirm that when you request an explanation for a different class than the model predicted, the attribution map changes in a way that makes sense (for gradient-based methods) or at least changes substantially (for attention-based methods).

**Checklist:**
- [ ] Implement `BaseExplainer` and all 7 explainer classes.
- [ ] Confirm that each explainer outputs shape (H_p, W_p) with correct values (not NaN, not all-zero).
- [ ] Run normalisation pipeline; confirm output in [0, 1].
- [ ] Implement `BenchmarkRunner` with checkpointing.
- [ ] Run sanity checks S1, S2, S3 on a small subset (e.g., 100 samples) before full run.
- [ ] Run the full benchmark (budget: approximately 1 week of GPU time on a single A100).
- [ ] Verify that results tables are complete (no missing cells).

---

## Phase 4 — Analysis, Ablations & Theoretical Grounding *(Months 7–10)*

### Overview

With the results matrix in hand, Phase 4 converts raw numbers into findings. This is where the paper's argument is constructed. Plan approximately 3 months for this: rushing the analysis is the most common reason for weak contributions in benchmark papers.

---

### Task 4.1 — Inter-Metric Correlation Analysis

#### What to Do

Compute the Spearman rank correlation between every pair of metrics across all method × model × dataset combinations. Visualise as a heatmap. This analysis answers a critical question: are your metrics measuring genuinely different things, or are some of them redundant?

```python
import scipy.stats as ss
import pandas as pd

def compute_metric_correlations(results_df):
    """
    results_df: DataFrame with columns = metric names,
                rows = (explainer, model, dataset, sample) combinations.
    Returns a correlation matrix.
    """
    metric_cols = [c for c in results_df.columns
                   if c not in ["explainer", "model", "dataset"]]
    corr_matrix = pd.DataFrame(index=metric_cols, columns=metric_cols, dtype=float)
    for m1 in metric_cols:
        for m2 in metric_cols:
            rho, _ = ss.spearmanr(results_df[m1], results_df[m2])
            corr_matrix.loc[m1, m2] = rho
    return corr_matrix
```

**Interpreting the correlation matrix:**

- If two metrics are correlated at |ρ| > 0.9, one is redundant. You should either remove it from the benchmark or explain why both are needed (e.g., they diverge in specific conditions).
- If fidelity metrics (F1–F4) are uncorrelated with localization metrics (L1–L4), this is a strong finding: fidelity and localization measure genuinely different properties. Articulate what this means for practitioners.
- Apply factor analysis (PCA or varimax rotation) to the metric space. If the top 2–3 factors explain > 80% of variance, you have identified the latent structure of explanation quality.

> **📦 Deliverable:** Inter-metric correlation heatmap (Figure for paper) and factor analysis results (Table for paper).

---

### Task 4.2 — Task-Metric Interaction Analysis

#### What to Do

Answer the question: **does the ranking of explanation methods change across tasks (datasets)?** This is done with a structured concordance analysis.

1. For each dataset d and each metric m, rank the 7 explainers by their mean score on metric m.
2. Compute the Kendall τ concordance between rankings across datasets.
3. A high τ means the ranking is stable across tasks (good: the benchmark measures something fundamental). A low τ means rankings are task-specific (interesting: it means you need task-specific guidance).

Then, for the most interesting discordances (cases where a method ranks 1st on ImageNet but 5th on CUB-200), analyse why. This is your "insight" section. Likely causes: fine-grained tasks require tight localization (favouring methods with high spatial resolution in their maps), while coarse tasks allow diffuse explanations (favouring methods that capture class-discriminative global features).

**The practitioner recommendation:** Based on this analysis, write a **decision tree** (Figure in paper):
> *"If your task is fine-grained with GT masks, prioritise L1–L3. If your task is safety-critical, prioritise R3–R4. If you have no GT masks, use F1–F4 and C1–C3 only."*

---

### Task 4.3 — Ablation Studies

#### What to Do

Run the following ablations. Each ablation changes **exactly one variable** while holding all others fixed.

**Ablation A1: Token Resolution**

Compare two variants of each attention-based explainer: (a) raw patch token attention, (b) CLS token attention (the attention from the [CLS] token to all patch tokens). Hypothesis: CLS attention is more class-discriminative but less spatially precise. Test this with the localization metrics.

**Ablation A2: Layer Depth**

For attention-based methods, compare explanations generated from: last layer only, last 3 layers averaged, all layers averaged (rollout). Measure how fidelity and localization change with depth. This directly informs practitioners about whether to use shallow or deep attention.

**Ablation A3: Masking Strategy**

Compare zero-masking vs. mean-masking vs. blurred-masking for the insertion/deletion metrics. Report the standard deviation of the metric across masking strategies as a "masking sensitivity score." A large standard deviation means the metric is not robust to the choice of baseline, which is a limitation to report.

**Ablation A4: Pre-training Objective**

Compare ViT-B/16 (supervised) vs. MAE-ViT-B/16 (masked autoencoder) with identical fine-tuning protocols. Hypothesis: MAE pre-training may produce richer spatial features, leading to higher localization quality in explanations. Test this hypothesis explicitly and report whether the difference is statistically significant.

#### Effect Size Reporting

For every ablation, report **Cohen's d** as the effect size, not just mean differences. TPAMI reviewers increasingly demand effect size reporting.

```
d = (µ_A − µ_B) / sqrt((σ²_A + σ²_B) / 2)
```

| d value | Interpretation |
|---------|----------------|
| < 0.2 | Negligible |
| 0.2–0.5 | Small |
| 0.5–0.8 | Medium |
| > 0.8 | Large |

**Checklist:**
- [ ] Compute full Spearman correlation matrix. Identify any redundant metrics.
- [ ] Run factor analysis on the metric space. Write the interpretation paragraph.
- [ ] Compute Kendall τ concordance across datasets. Identify top 3 discordances.
- [ ] Write the practitioner decision tree.
- [ ] Run ablations A1–A4. Report means, standard deviations, and Cohen's d.
- [ ] Verify that each ablation is truly single-variable. Double-check controls.
- [ ] Collect all above into a complete "Analysis" section draft.

---

## Phase 5 — Writing & Submission *(Months 10–13)*

### Overview

TPAMI has a strict double-blind review process. The paper must be self-contained: all claims must be backed by either a proof or an experiment in the paper or supplement. Plan for 2–3 rounds of internal review before submission. A typical TPAMI paper in this area is **12–16 pages** in double-column format plus supplementary material.

---

### Task 5.1 — Paper Structure and Section-by-Section Guide

**Section 1: Introduction** *(1–1.5 pages)*

Establish the motivation in the first paragraph: ViTs are now the dominant architecture, but practitioners have no reliable way to compare explanation methods across architectures and tasks. Give 2–3 concrete examples of why this matters (medical imaging, autonomous driving, model debugging).

Paragraph 2: What is the gap? Summarise your gap table (Table 1) in prose. State one sentence: *"We address this gap with [Name of your benchmark]."*

Paragraph 3: Contributions. Three bullets, precisely:
1. A formally defined metric suite of N metrics across four families (fidelity, localization, robustness, complexity) with axiomatic analysis.
2. A comprehensive evaluation of 7 explanation methods across 6 ViT architectures and 4 datasets.
3. Empirically derived practitioner guidelines and a public codebase.

**Section 2: Related Work** *(1–2 pages)*

Organise by: (a) ViT explanation methods, (b) XAI evaluation frameworks, (c) benchmark papers in adjacent areas (GLUE, BenchXAI). For each group, write 2–4 sentences on what they cover and one sentence on what they do not. Do not be dismissive — be precise.

**Section 3: Metric Framework** *(2–3 pages)*

This is your main contribution section. Present all 4 families of metrics with formal definitions. Include a table of metrics (name, family, formula, requires GT mask?, computational cost). Present the axiomatic analysis as a table (metrics × axioms) with a brief paragraph of commentary.

**Section 4: Experimental Setup** *(0.5–1 page)*

Models (Table 2), datasets (Table 3), fine-tuning protocol, implementation details (framework, hardware, seeds, runtime). Keep this dense. Everything that does not fit goes in the supplement.

**Section 5: Results** *(3–4 pages)*

Main results table: rows = explainers, columns = metrics, cells = mean ± std across models and datasets. Separate tables per dataset family. Heatmap figure for inter-metric correlations. Ranking concordance figure. Key findings highlighted in call-out boxes, not buried in prose.

**Section 6: Analysis** *(2 pages)*

Inter-metric correlation findings, task-metric interaction analysis, ablation results. Each ablation in a sub-figure with effect size reported.

**Section 7: Practitioner Guide** *(0.5 page)*

The decision tree from Task 4.2. This section has high practical impact and may attract citations from practitioners.

**Section 8: Conclusion** *(0.5 page)*

State what was learned, state limitations honestly (most importantly: the benchmark does not capture human-interpretability directly), and state future directions.

---

### Task 5.2 — Figure and Table Standards for TPAMI

- All figures must be **vector graphics** (PDF or EPS). No rasterised plots. Use matplotlib with `pdf` backend.
- All tables must use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`). No vertical rules.
- Report mean ± standard deviation across at least 3 random seeds.
- Use **bold** to indicate the best result in each column.
- Every axis must be labelled with units. Every figure must have a caption that is self-contained (a reader should understand the figure without reading the body text).

---

### Task 5.3 — Supplementary Material

The supplement is mandatory and should contain:

1. Full metric derivations (every formula with step-by-step derivation where non-trivial).
2. Extended results tables: every method × model × dataset × metric combination. These are too large for the main paper but reviewers will check them.
3. Additional qualitative visualisations: for each dataset, show the attribution map of all 7 methods on the same 5 images, arranged in a grid.
4. Training curves and fine-tuning accuracy for all models on all datasets.
5. Proof sketches for all theorems stated in the main paper.
6. Implementation details: hyperparameters for each explainer (number of RISE masks, LIME superpixel settings, etc.).

---

### Task 5.4 — Code and Reproducibility

- [ ] Create a GitHub repository (anonymised for submission: use a throwaway account, no identifying information in the code).
- [ ] Provide a `requirements.txt` and a `Dockerfile` that installs all dependencies and produces a working environment.
- [ ] Provide a `run_benchmark.sh` script that runs the full benchmark from raw data to results tables with a single command.
- [ ] All random seeds must be fixed. Run the full pipeline twice with the same seed and verify identical results.
- [ ] Write a `README.md` that explains: how to install, how to add a new explanation method, how to add a new metric, how to reproduce each figure in the paper.
- [ ] After acceptance: make the repository public, release pip-installable package (`vit-bench`).

---

### Task 5.5 — Pre-Submission Checklist

- [ ] Every claim in the paper is backed by either a citation, a proof, or a table/figure in the paper or supplement.
- [ ] All standard deviations and statistical tests are reported (use Wilcoxon signed-rank test for pairwise method comparisons; report p-values with Bonferroni correction for multiple comparisons).
- [ ] The paper is within the TPAMI page limit. Check the current author guidelines on the IEEE TPAMI website before submission.
- [ ] All figures are legible at 100% zoom and when printed in black-and-white. Use colour-blind-safe palettes (viridis, cividis).
- [ ] All acronyms are defined on first use.
- [ ] The abstract states the problem, the approach, and the key finding (one sentence each).
- [ ] A colleague outside the project has proofread the paper.
- [ ] The supplementary material is complete and the main paper is self-contained without it.
- [ ] Double-blind: no author names, affiliations, or identifying acknowledgements in the submission version.

---

## Appendix A: Common TPAMI Reviewer Objections and How to Address Them

| Objection | How Your Process Addresses It |
|-----------|-------------------------------|
| "The metrics are not theoretically grounded." | Phase 2, Task 2.5: You have an axiomatic analysis table and at least 3 formal theorems or counterexamples. |
| "Why these specific datasets?" | Phase 1, Task 1.3: Each dataset is chosen to uniquely enable a specific class of metrics, documented in Table 3 and the text. |
| "Results are not statistically significant." | Phase 4 and Phase 5: You report standard deviations, Cohen's d effect sizes, and Wilcoxon signed-rank tests with Bonferroni correction. |
| "The sanity checks were not run." | Phase 3, Task 3.4: Sanity checks S1–S3 are a dedicated subsection in the paper. |
| "The models were not trained under the same conditions." | Phase 1, Task 1.2: All models use an identical fine-tuning recipe documented verbatim in the paper. |
| "The distribution shift in insertion/deletion metrics invalidates results." | Phase 3, Task 3.2: You report results under two masking strategies and cite Hooker et al. (2019) explicitly. You acknowledge this as a limitation in the conclusion. |
| "The benchmark does not measure human interpretability." | You acknowledge this in the conclusion as a limitation and future work direction. You do not overclaim: your metrics measure faithfulness and spatial alignment, not human utility directly. |
| "The code is not publicly available." | Phase 5, Task 5.4: The anonymised repository is linked in the submission. Full release on acceptance. |

---

## Appendix B: Compute Budget Estimate

**Table 5: Approximate GPU hours per major task (single A100 80GB or equivalent).**

| Task | Estimate | Notes |
|------|----------|-------|
| Fine-tuning all 6 models × 4 datasets | 120–180 hrs | Can parallelise; use 4 GPUs |
| Pilot benchmark (100 samples) | 2–4 hrs | Sanity check before full run |
| Full benchmark (all combos) | 250–400 hrs | Most expensive task; checkpoint |
| Randomisation tests (R3, R4) | 80–120 hrs | Per architecture |
| Ablation runs (A1–A4) | 60–100 hrs | Selective re-runs |
| **Total estimate** | **512–804 hrs** | Budget for 2× overrun |

If you have access to a university HPC cluster, distribute fine-tuning and full benchmark runs across nodes using a simple job array. Use SLURM array jobs with one job per (model, dataset) combination.
