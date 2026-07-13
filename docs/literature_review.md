# Task 1.1 – Literature Review and Gap Analysis for a ViT Explainability Benchmark

## 1. Overview of Task 1.1

Task 1.1 requires a systematic literature review of major explainability methods to be benchmarked (Category A) and existing evaluation frameworks for explanations (Category B), followed by the construction of a structured gap table and a concise problem statement for a comprehensive Vision Transformer (ViT) explanation benchmark.
The central claim to be defended is that existing evaluation frameworks for ViT explanations are inconsistent, narrowly scoped, and lack a standardised methodology, which leads to incomparable results across the literature.[^1]

[^1]

***

## 2. Category A – Foundational XAI Methods to Benchmark

### 2.1 Vision Transformers and Raw Attention (Dosovitskiy et al., 2020/2021)

The Vision Transformer (ViT) introduced the idea of applying a pure Transformer encoder directly to sequences of image patches, replacing convolutional inductive biases with global self‑attention over patch tokens.
An image of size 224×224 is split into fixed 16×16 patches (14×14 = 196 patches for ViT‑B/16), each flattened and linearly projected, with a prepended class (CLS) token whose representation is used for classification.
When interpretability is considered, many works extract CLS‑to‑patch attention weights from the last layer as a "raw attention" explanation, averaging attention over heads, but the original ViT paper does not provide a systematic evaluation of whether these attention maps are faithful explanations of the model’s decisions.




### 2.2 Attention Rollout and Attention Flow (Abnar & Zuidema, 2020)

Abnar and Zuidema study how information flows through multi‑layer self‑attention and propose attention rollout and attention flow as post‑hoc methods to approximate token‑level relevance to the input.[^2][^3]
Attention rollout accounts for residual connections by adding an identity matrix to each attention matrix and re‑normalising, then multiplying attention matrices across layers so that CLS‑to‑input relevance incorporates contributions from all layers.[^2]
Their experiments in NLP show that rollout and flow correlate better with ablation‑based importance measures than raw attention, but the work does not provide a full benchmark of evaluation metrics, and is mostly focused on textual models rather than ViTs.[^3]

[^2]
[^3]

### 2.3 Transformer-LRP for Transformers and ViTs (Chefer et al., 2021)

Chefer et al. propose a method for Transformer interpretability based on Layer‑wise Relevance Propagation (LRP) and Deep Taylor Decomposition, adapted to handle self‑attention, skip connections, and non‑positive activations.[^4][^5]
Their algorithm assigns local relevance to outputs and propagates it backwards through attention and feed‑forward layers while conserving total relevance, producing input‑level attribution maps for both NLP transformers and ViTs that empirically outperform attention‑based baselines on localization and faithfulness metrics.[^5][^6]
Although they do provide quantitative comparisons on a few tasks, the evaluation uses a limited set of metrics and architectures, and does not establish a broad, standardised benchmark across multiple ViT variants and datasets.

[^4]
[^5]
[^6]

### 2.4 Grad-CAM and its Adaptation to ViTs (Selvaraju et al., 2017)

Grad‑CAM (Gradient‑weighted Class Activation Mapping) produces class‑specific localization maps by back‑propagating gradients of a target logit to the final convolutional feature maps and weighting each channel by the global average of its gradients.[^7][^8]
The method is model‑agnostic over CNN architectures, requires no retraining, and has been shown to produce coarse but effective visual explanations that outperform earlier saliency methods on weakly‑supervised localization tasks and human trust evaluations.[^8][^7]
To apply Grad‑CAM to ViTs, later works treat patch token embeddings or intermediate feature maps as analogous to convolutional feature maps, but there is no canonical adaptation or unified evaluation framework assessing how such ViT‑Grad‑CAM variants compare in fidelity, localization, robustness, and complexity across models.

[^7]
[^8]

### 2.5 DIME – Disentangled Local Explanations for Multimodal Models (Lyu et al., 2022)

DIME is a framework for interpreting multimodal models that decomposes predictions into unimodal contributions and multimodal interactions, yielding fine‑grained local explanations that generalise across architectures and modalities.[^9][^10]
The method is evaluated on both synthetic and real‑world multimodal tasks (e.g., affective computing, multimodal classification) and is shown to produce accurate and disentangled explanations that improve debugging and understanding.[^9]
While DIME is not ViT‑specific and primarily targets multimodal settings, its decomposition of contributions is conceptually relevant for ViT‑based vision‑language models, yet there is no ViT‑focused benchmark that situates DIME‑style explanations among other saliency and attribution methods.

[^9]
[^10]

### 2.6 RISE – Randomized Input Sampling for Explanation of Black-box Models (Petsiuk et al., 2018)

RISE is a black‑box explanation method that estimates pixel‑wise importance by sampling many random binary masks over the input image, computing the model output for each masked image, and correlating prediction changes with mask values to obtain an importance map.[^11][^12]
The method is model‑agnostic, requires only forward passes, and achieves state‑of‑the‑art performance on causal metrics such as deletion/insertion curves and competitive results on human "pointing game" evaluations compared to gradient‑based saliency maps.[^12][^13]
RISE is slow due to the need for thousands of masks per image and has typically been evaluated on CNNs; there is no unified ViT‑oriented benchmark that compares RISE to gradient‑ and attention‑based ViT explanations across diverse datasets and architectures.

[^11]
[^12]
[^13]

### 2.7 LIME and SHAP for Vision (Ribeiro et al., 2016; Lundberg & Lee, 2017)

LIME (Local Interpretable Model‑Agnostic Explanations) approximates the local decision boundary of any black‑box model by fitting an interpretable surrogate (often sparse linear) model on perturbed samples around a point, using an interpretable representation such as superpixels for images.[^14][^15]
This approach provides local fidelity and model‑agnosticism but is sensitive to the choice of perturbation distribution and interpretable representation, and its explanations can be unstable.[^14]
SHAP (SHapley Additive exPlanations) unifies several additive feature attribution methods using Shapley values from cooperative game theory, providing feature importance values with desirable properties such as local accuracy, symmetry, and consistency.[^16][^17]
For images, SHAP typically groups pixels into superpixels, but computing exact Shapley values is expensive, so approximations and specialised algorithms are used.[^17]
Neither LIME nor SHAP has been systematically benchmarked for patch‑level ViT explanations across multiple datasets and architectures with a dedicated ViT‑specific metric suite.

[^14]
[^15]
[^16]
[^17]

***

## 3. Category B – Existing Evaluation Frameworks for Explanations

### 3.1 Pixel Flipping / AOPC – Samek et al., 2017

Samek et al. propose a general methodology for evaluating heatmaps based on region perturbation: pixels or regions are ranked by importance, then progressively removed (or replaced) and the resulting degradation in classifier output is measured.[^18][^19]
They introduce the Area Over the Perturbation Curve (AOPC) metric, showing that heatmaps produced by Layer‑wise Relevance Propagation (LRP) yield higher AOPC values than sensitivity analysis or deconvolution, indicating better identification of relevant pixels.[^18]
This work focuses mainly on perturbation‑based fidelity for CNNs, uses limited datasets (e.g., SUN397, ILSVRC2012, MIT Places), and does not cover robustness, complexity, ViT‑specific concerns, or multi‑architecture standardisation.

[^18]
[^19]

### 3.2 ROAR/KAR – A Benchmark for Interpretability Methods in DNNs (Hooker et al., 2019)

Hooker et al. introduce the ROAR (RemOve And Retrain) framework, which evaluates feature importance estimates by removing features deemed important, retraining the model on the modified dataset, and measuring performance degradation compared to the original model.[^20][^21]
They show that many popular saliency methods perform no better than random baselines under this retraining‑based evaluation, while some ensemble variants such as SmoothGrad‑Squared and VarGrad perform significantly better, highlighting distribution‑shift issues in standard deletion‑based metrics.[^21][^20]
ROAR provides a strong fidelity‑oriented benchmark but is computationally heavy, designed primarily for CNNs and tabular models, and does not integrate localization, robustness, complexity, or ViT‑specific architectural factors.

[^20]
[^21]

### 3.3 Insertion / Deletion Curves – Petsiuk et al., 2018

In the RISE paper, Petsiuk et al. popularise the insertion and deletion metrics for visual explanations, where high‑importance regions are either gradually added to a baseline image (insertion) or removed from the original image (deletion) while tracking model confidence.[^22][^12]
Good explanations cause rapid confidence increases in insertion and steep confidence drops in deletion, and area under these curves is used as an automatic causal evaluation of explanation faithfulness.[^12]
These metrics focus exclusively on fidelity and are typically applied to CNNs, with less attention to localisation against ground truth masks, robustness to perturbations, complexity/sparsity, or ViT‑specific patch structure.

[^12]
[^22]

### 3.4 Infidelity and Sensitivity Metrics – Yeh et al., 2019

Yeh et al. formalise two objective measures for evaluating explanations: infidelity (how well an explanation accounts for output changes under input perturbations) and sensitivity (how much an explanation changes under small input perturbations).[^23][^24]
They show that many existing explanation methods can be viewed as optimal solutions for infidelity under specific perturbation distributions and propose robust variants that trade off sensitivity and fidelity.[^24][^23]
While this framework is powerful for defining theoretical desiderata, it targets explanation methods in general (mostly gradient‑type explanations) and does not instantiate a full, task‑level benchmark across architectures, datasets, or visual localisation tasks, particularly not for ViTs.

[^23]
[^24]

### 3.5 Faithfulness and Plausibility in NLP – Jacovi & Goldberg, 2020

Jacovi and Goldberg survey interpretability evaluation in NLP, emphasising the distinction between faithfulness (whether explanations reflect the true reasoning of the model) and plausibility (whether explanations align with human intuition).[^25][^26]
They argue that current evaluation practices often conflate these aspects and call for more rigorous definitions and graded notions of faithfulness rather than binary labels, providing guidelines for proper evaluation design.[^25]
Although this work is NLP‑focused, its conceptual separation of faithfulness and plausibility, and its emphasis on evaluation methodology, are directly relevant to designing a ViT benchmark that integrates both causal fidelity metrics and human‑aligned localisation metrics.

[^25]
[^26]

### 3.6 BenchXAI – Comprehensive Benchmarking of Post-hoc XAI (Metsch & Hauschild, 2025)

BenchXAI is a biomedical‑focused benchmark package that evaluates fifteen post‑hoc XAI methods across multi‑modal biomedical data (clinical, imaging, and signal data) using a set of metrics to study robustness, suitability, and limitations.[^27][^28]
The framework is designed to be easy to use and supports comparative evaluation of multiple methods, but its metrics are tuned to biomedical tasks and modalities, and it does not provide ViT‑centric design, patch‑level metrics, or an integrated emphasis on attention‑specific phenomena.

[^27]
[^28]

### 3.7 Attention as (Non-)Explanation – Jain & Wallace (2019) and Wiegreffe & Pinter (2019)

Jain and Wallace argue that standard attention weights in NLP models often do not provide faithful explanations: attention distributions can be decorrelated from gradient‑based importance measures, and adversarially modified attention maps can yield identical predictions, suggesting that attention is "not explanation".[^29]
Wiegreffe and Pinter counter that this negative result depends on the definition and evaluation of explanation, proposing alternative tests (uniform baselines, variance‑based calibration across random seeds, frozen‑weight diagnostics, adversarial training) and arguing that attention can sometimes function as explanation when carefully evaluated.[^30][^31]
Together, these works motivate a ViT benchmark that can empirically study when attention‑based explanations (raw attention, rollout, etc.) align with causal and localisation metrics and when they fail.

[^29]
[^30]
[^31]

***

## 4. Gap Analysis Table (Task 1.1 Deliverable)

### 4.1 Gap Table Structure

The gap table summarises which evaluation properties are adequately addressed by existing frameworks and which remain uncovered and will form the main contribution of the ViT explainability benchmark.
Columns correspond to:

- Fidelity – causal impact of highlighted regions on predictions.
- Localization – spatial agreement with human or dataset ground truth.
- Robustness – stability to input/parameter/label perturbations.
- Complexity – parsimony and sparsity of explanations.
- ViT‑specific – explicit consideration of ViT patch tokens, attention heads, and architectures.
- Multi‑arch – coverage across multiple architectures within and beyond ViTs.

Rows correspond to key existing frameworks and the proposed benchmark.

### 4.2 Gap Analysis Table

| Framework                 | Fidelity | Localiz. | Robust. | Complex. | ViT-specific | Multi-arch |
|---------------------------|----------|----------|---------|----------|-------------|------------|
| Samek et al. (2017)      | ✓        | ∼        | ∼       |          |             | ∼          |
| ROAR (Hooker 2019)       | ✓        |          | ∼       |          |             | ∼          |
| Ins./Del. (Petsiuk)      | ✓        | ∼        |         |          |             | ∼          |
| Sensitivity (Yeh)        | ✓        |          | ✓       |          |             |            |
| BenchXAI                 | ✓        | ∼        | ✓       | ∼        |             | ✓          |
| Chefer et al. eval       | ✓        | ✓        |         |          | ∼           | ∼          |
| This paper (ViT bench.)  | ✓        | ✓        | ✓       | ✓        | ✓           | ✓          |

### 4.3 Justification – Samek et al. (2017)

Samek et al. focus primarily on perturbation‑based fidelity evaluation via region perturbation, measuring how classifier scores decrease when important pixels are removed and summarising this with the AOPC metric, so fidelity is adequately covered (✓).
[^19][^18]
Localization is partially addressed (∼) because while heatmaps are visualised and qualitative alignment with object regions is discussed, there is no systematic IoU or pointing‑game evaluation against ground truth masks.
[^18]
Robustness is partially present (∼) in the sense that AOPC curves indicate sensitivity to ordered removal, but the framework does not systematically study robustness under different perturbation types, parameter randomisation, or label randomisation.
Complexity is not explicitly measured; there is no metric for sparsity or entropy of explanations.
The framework is not ViT‑specific; experiments target CNNs and do not model patch‑token or attention‑head structures.
Multi‑architecture coverage is partial (∼) because the methodology is applied to several CNN architectures but not across fundamentally different families such as ViTs and hybrid models.

### 4.4 Justification – ROAR (Hooker et al., 2019)

ROAR provides a strong causal fidelity test (✓) by retraining models after removing features ranked as important and measuring how performance degrades compared to the original model, directly tying explanation quality to predictive impact.[^20][^21]
Localization is not evaluated: the framework works at the feature level (pixels or input features) without comparing attribution maps against human or dataset masks.
Robustness is partially present (∼) because ROAR addresses distribution‑shift issues in standard deletion metrics by retraining on modified datasets, but it does not systematically evaluate explanation stability to input noise or model parameter perturbations beyond retraining.
Complexity is not explicitly measured; sparsity or compactness of explanations is not quantified.
The methodology is not ViT‑specific; experiments focus on CNNs and tabular models without considering attention mechanisms or patch‑level explanations.
Multi‑architecture coverage is partial (∼) because the benchmark uses several CNN variants but does not span multiple fundamentally different architectures (e.g., ViTs, Swin, BEiT) in a unified way.

### 4.5 Justification – Insertion / Deletion (Petsiuk et al., 2018)

The insertion and deletion metrics directly quantify fidelity (✓) by tracking how model confidence changes as top‑ranked regions are added to or removed from the input, and have been shown to correlate with explanation quality in RISE experiments.[^22][^12]
Localization is partially covered (∼) via the "pointing game" evaluation reported in RISE, which checks whether the most salient point lies in a ground truth region, but this is a single, coarse measure and not a full suite of localisation metrics like IoU or energy‑on‑ground‑truth.[^22]
Robustness is not systematically considered; insertion/deletion curves are sensitive to mask baselines and perturbation strategies, but the framework does not study stability under noise, parameter randomisation, or label randomisation.
Complexity is not measured; there is no explicit metric for sparsity or focus of explanations.
The framework is not ViT‑specific and is primarily evaluated on CNN‑based models.
Multi‑architecture coverage is partial (∼), reflecting that RISE can be applied to arbitrary models but the published experiments are limited to a small set of architectures.

### 4.6 Justification – Sensitivity / Infidelity (Yeh et al., 2019)

Yeh et al. explicitly define infidelity as an objective measure of how well an explanation accounts for changes in model output under input perturbations, and show that many explanations can be viewed as optimal with respect to specific infidelity measures, so fidelity is well addressed (✓).[^23][^24]
Localization is not considered because their framework is agnostic to spatial structure and focuses on scalar output changes, not spatial overlap with human annotations.
Robustness is well addressed (✓) via the sensitivity measure, which quantifies how much an explanation changes under small perturbations, and via proposed robust variants that reduce sensitivity while maintaining or improving infidelity.[^23]
Complexity is not included; there is no sparsity or entropy‑based measure of explanation parsimony.
The work is not ViT‑specific and does not treat patch‑based architectures or attention explicitly.
Multi‑architecture coverage is not a primary focus; the theory applies broadly, but the experiments are limited to a few model types and do not form a multi‑architecture benchmark.

### 4.7 Justification – BenchXAI (Metsch & Hauschild, 2025)

BenchXAI offers a comprehensive benchmarking package that evaluates fifteen XAI methods on multi‑modal biomedical data, using metrics that include faithfulness, robustness, and complexity‑related measures, so fidelity is covered (✓) and robustness is also addressed (✓).[^28][^27]
Localization is partially covered (∼) because medical imaging tasks often involve region‑of‑interest evaluation, but the description emphasises overall robustness and suitability rather than a systematic family of localisation metrics such as IoU, pointing game, and energy‑on‑ground‑truth across tasks.[^27]
Complexity is partially addressed (∼), as some metrics target explanation sparsity or parsimony, but complexity is not structured as a standalone axis alongside faithfulness and robustness in a way that would generalise to ViT patch‑level explanations.[^28]
The framework is not ViT‑specific; it targets biomedical models and does not design metrics for attention heads, patch grids, or transformer‑specific behaviours.
Multi‑architecture coverage is strong (✓) in the biomedical context, where multiple model families and modalities are evaluated, but does not extend to a curated zoo of ViTs with controlled training protocols.

### 4.8 Justification – Chefer et al. Evaluation (Transformer-LRP)

Chefer et al. evaluate their Transformer‑LRP method by comparing its visual explanations against attention‑based methods and other baselines using both localisation and fidelity‑oriented metrics, so fidelity is addressed (✓) and localisation is reasonably covered (✓).[^6][^5]
However, robustness is not systematically studied; there is no parameter randomisation, label randomisation, or sensitivity analysis, and complexity metrics like Gini sparsity or entropy are not reported.
The method is partially ViT‑specific (∼) in that it is explicitly adapted to ViTs and benchmarked on visual transformer networks, but the evaluation is limited to a small set of models and tasks and does not attempt to systematise ViT explanation evaluation as a general framework.[^5]
Multi‑architecture coverage is partial (∼) because both NLP transformers and ViTs are evaluated, yet there is no standardised multi‑architecture training protocol or metric suite spanning diverse ViT variants.

### 4.9 Justification – This Paper (ViT Explainability Benchmark)

The proposed ViT benchmark is designed to cover fidelity comprehensively (✓) through insertion/deletion AUC, comprehensiveness and sufficiency, and log‑odds shift metrics that explicitly measure how perturbing top‑attributed patches affects model confidence.[^1]
Localization is fully incorporated (✓) via IoU with ground truth masks, pointing game accuracy, energy‑on‑ground‑truth, and calibration gap metrics across datasets with bounding boxes, part annotations, segmentation masks, and expert heatmaps.[^1]
Robustness is a central axis (✓) through max/average sensitivity to input noise, model parameter randomisation tests, and label randomisation tests that check whether explanations change when models do not learn meaningful features.[^1]
Complexity is explicitly measured (✓) with Gini sparsity, entropy of normalised attribution, and effective mass ratio, capturing how focussed or diffuse explanations are.[^1]
The benchmark is ViT‑specific (✓), with a curated model zoo of ViT architectures, patch‑level explanation interfaces, and attention‑centric questions, and is multi‑architecture (✓) by design, with controlled training protocols across several ViT families and datasets.[^1]

[^1]

***

## 5. One-Page Problem Statement

### 5.1 What a Comprehensive ViT Explanation Benchmark Requires

A comprehensive explanation benchmark for Vision Transformers (ViTs) must provide a rigorous, standardised methodology for evaluating and comparing explanation methods across ViT architectures, datasets, and tasks.
Existing evaluation frameworks in explainable AI (XAI) have largely been developed in the context of convolutional neural networks and tabular models, and even when adapted to transformers, they do not jointly address fidelity, localisation, robustness, and complexity in a ViT‑specific, multi‑architecture setting.[^5][^20][^18][^23]
Furthermore, the rapid adoption of ViTs in both vision and multimodal applications has outpaced the development of dedicated benchmarks for their explanations, leading to a fragmented landscape where different works rely on incompatible metrics, datasets, and model choices, making results difficult to compare or reproduce.

First, a ViT explanation benchmark must unify and formalise fidelity metrics that capture causal impact.
Pixel‑flipping AOPC, insertion/deletion curves, and infidelity measures each provide partial views of how strongly explanations influence model outputs under perturbations.[^12][^18][^23]
However, these metrics are not consistently defined at patch granularity, and they are often applied with different baselines, perturbation distributions, and implementation details, which severely limits cross‑paper comparability.
A comprehensive benchmark must standardise fidelity metrics for patch‑level ViT explanations, clearly specify masking baselines (e.g., zero vs. mean patches), and incorporate multiple complementary measures such as insertion AUC, deletion AUC, comprehensiveness, sufficiency, and log‑odds shift.

Second, localisation must be elevated from qualitative visual inspection to a first‑class evaluation axis.
While some works report pointing game accuracy or show qualitative overlays, systematic localisation metrics such as IoU with ground truth masks, energy‑on‑ground‑truth, and calibration gaps between correctly and incorrectly classified samples are rarely used together in a single framework.[^5][^12]
ViTs are increasingly deployed on tasks with rich spatial annotations, including fine‑grained bird classification, object segmentation, and medical imaging with expert heatmaps, yet there is no standard benchmark that exploits these annotations to assess how well explanations align with human‑annotated regions across architectures and tasks.
A comprehensive benchmark must curate datasets with bounding boxes, part annotations, pixel‑level masks, and expert attention maps, and define a unified set of localisation metrics evaluated at patch resolution.

Third, robustness of explanations is largely under‑explored in current ViT work.
Existing frameworks such as ROAR address distribution shift in deletion‑based metrics by retraining models but do not study stability of explanations under small input perturbations, parameter randomisation, or label randomisation.[^20][^23]
NLP research has highlighted the importance of distinguishing faithfulness from plausibility and has proposed graded notions of faithfulness, but this thinking has not yet been systematically transferred to visual transformers.[^25]
A ViT‑specific benchmark must therefore include robustness metrics such as max‑ and average‑sensitivity, model parameter randomisation tests, and label randomisation tests, ensuring that high‑scoring methods genuinely depend on learned features rather than architectural priors or random noise.

Fourth, complexity and parsimony of explanations are seldom quantified in existing visual explanation benchmarks.
Many saliency maps are diffuse, assigning non‑zero importance to most of the image, which can technically achieve reasonable fidelity but are unhelpful for human interpretation.[^18]
Recent XAI benchmark packages such as BenchXAI begin to include complexity‑related measures, but they are not tailored to ViT patch grids and attention heads.[^27][^28]
A comprehensive ViT benchmark must explicitly measure sparsity and focus through metrics like Gini coefficient, entropy of attribution, and effective mass ratio, thereby discouraging trivial near‑uniform explanations and rewarding methods that highlight compact, informative regions.

Fifth, ViT‑specific architectural considerations must be built into the benchmark.
Attention‑based explanations such as raw attention, attention rollout, and Transformer‑LRP are central to ViT interpretability, but their status as faithful explanations is contested in the NLP literature, where attention has been argued to be "not explanation" or only conditionally useful.[^29][^30][^2][^5]
A ViT benchmark must systematically test when attention‑based methods align with causal and localisation metrics and when they fail, across diverse ViT variants that differ in patch size, attention mechanisms (e.g., standard vs. shifted windows), and pre‑training objectives.
This requires a standardised model zoo with controlled fine‑tuning protocols so that explanation differences can be attributed to architectures and methods rather than confounding training differences.[^1]

Finally, comprehensive evaluation demands a unified, reproducible software pipeline.
Current works often rely on ad‑hoc code, incompatible interfaces, and differing normalisation schemes for attribution maps, which makes it difficult to aggregate or reproduce results.[^5][^12]
A ViT explanation benchmark must define a common explainer interface, a normalisation pipeline that standardises attribution maps across methods, and a metric suite implementation that can be invoked via a single benchmark runner, with fixed random seeds and checkpointing.
Such a pipeline should support multiple ViT architectures, datasets, and explanation methods, enabling a full results matrix and facilitating inter‑metric correlation analysis, task‑metric interaction studies, and ablation experiments.

In summary, what does not currently exist is a ViT‑centric, multi‑architecture, multi‑dataset benchmark that jointly covers fidelity, localisation, robustness, and complexity with mathematically formalised metrics, ViT‑aware design, and a reproducible evaluation pipeline.
Existing frameworks each address subsets of these requirements in specific contexts—CNNs, general attribution theory, biomedical models, or individual Transformer explanation methods—but none provide a unified, end‑to‑end standard for evaluating ViT explanations.[^20][^23][^27][^18][^5]
The proposed ViT explainability benchmark is designed to fill this gap by integrating these dimensions into a single, coherent evaluation framework that can support rigorous empirical and theoretical analysis of ViT explanations.

---

## References

1. [vit_benchmark_implementation_guide.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/collection_0a8fd320-12e1-4a45-9333-a05dcd0a510c/aac9f0a4-b60c-4d46-983d-1eb3d4ae039d/vit_benchmark_implementation_guide.pdf?AWSAccessKeyId=ASIA2F3EMEYEVVADXWMJ&Signature=3sSNicKbl0D9EY7hXloHAHFTLPE%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEBgaCXVzLWVhc3QtMSJHMEUCIQDVxZMt9%2FSZNhaHaBjdg3hXN7m7zCJ8Thpgtl2s8KIrGgIgA55eDDSxtF%2B1s4QQW1sWMQaUKd0wR1Q9RSwmLdFCg3oq%2FAQI4f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDJDQWchaFrFj833RTirQBB1raf86de%2FDDGavSt5DMK93qbayH9aoMcUyvhqqGDavLBE8et6whOvlLsYSJ8TB2oSNfmA4sth0zFQ9%2Fj8Q4tZoQHsFwkadsNguRqQptEIrGtuTye%2F0SL9CEkbPLLjy3Ba0djpci8OHVA13ZElca5N534NpUgOrPXtjqSWtQEBSaq2Mnp2bBqNrb%2Fk8L%2BkpNhPZrKnQGMKSEU7Ozn8tXsZWskWeV6NEH%2FPPkQsSxBfrWr%2Fvr20fiCsclazHIzCx6%2FJlJCtkgX9snp7%2Bur7NPnfMHAsXTzJwtssYsAYkF%2Ftir6aRS0x6vO37M%2BHRh0ywgrDZr6L%2FtM1fbQyAV%2FnhrCUL%2B%2FZ3M%2BIVR4UFmMyv05ZrcncLceaAUl5Ysnga831tccBzTqdN2HGGKFGVt05snbRzYKru0rN7xaOitqxTOPyFG5LYFYEHG0rqH9sCzamsheE8qNJdALKnl595EgZL3osn689AdPL%2BGQOZI25q%2BLVj13%2Bo87l0wgrK0JklzoO0ff4eTwLSOz%2FGdhGb5C8UivvnBiri36Q1ZENtiZ%2FNysKF44tWTxJCBgi16ZMNiQ%2FzZt386yia0e5v%2BkCrIO5yxPuvNebQ6h6aW%2B1vpxq3NlcKE5qoP8bTerDn%2BKjdgqSIJKAkLAwDs44REvFiRAuxPhpTrD4TUYYTYHmiN8sIsEKLnvONxLSrashE%2Fd4ze8YXLLy%2BJNdv9eUqAFC%2FIZ3hU%2FVdgcNrl0ENtStSN%2BV8v2%2FZMXLirjKXtrqM6fzpMYcfu7d64x4PZrvnk%2Ft%2BFVPrnJkwuOLSzgY6mAH%2FFH5fYyCga07PIiWhy7Kpkgg9R6B7FvNdTnbXBpCxBRwKnjpH6Du9p5bW%2B%2FXy8Y3vnUCbNWCko8l7EQ7peWesH2QCxYX%2FFr0F8B6BKp6vuvVRVqXvVaJkelF7ZOGSM5%2Fw0M4XTMGdkMNfY758H7ReARu99nlR6Z29mJSrXY7Ep%2BmQCOeCW5Mc76wVdE8R68d9483HqXd6mg%3D%3D&Expires=1775550219) - ViT Explainability Benchmark
A Comprehensive Explainability Benchmark
for Vision Transformers
Step-b...

2. [LLM_log #009: An Image is Worth 16×16 Words](https://datahacker.rs/dh-009-an-image-is-worth-16x16-words-from-transformers-to-vision-transformers-and-swin/) - Dosovitskiy et al. (ICLR 2021) introduced the Vision Transformer (ViT) — a pure Transformer applied ...

3. [Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/) - by S Abnar · 2020 · Cited by 1598 — We propose two methods for approximating the attention to input ...

4. [Transformer Interpretability Beyond Attention Visualization](https://openaccess.thecvf.com/content/CVPR2021/papers/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.pdf) - by H Chefer · 2021 · Cited by 1446 — LRP was applied for Transformers based on the premise that cons...

5. [[PDF] An Image is Worth 16x16 Words: Transformers for ...](https://www.semanticscholar.org/paper/An-Image-is-Worth-16x16-Words:-Transformers-for-at-Dosovitskiy-Beyer/268d347e8a55b5eb82fb5e7d2f800e33c75ab18a) - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale ... Focal ViT: image tran...

6. [[PDF] Quantifying Attention Flow in Transformers](https://www.semanticscholar.org/paper/Quantifying-Attention-Flow-in-Transformers-Abnar-Zuidema/76a9f336481b39515d6cea2920696f11fb686451) - This paper proposes two methods for approximating the attention to input tokens given attention weig...

7. [Transformer - CVPR 2021 Open Access Repository](https://openaccess.thecvf.com/content/CVPR2021/html/Chefer_Transformer_Interpretability_Beyond_Attention_Visualization_CVPR_2021_paper.html)

8. [An Image is Worth 16x16 Words: Transformers for ...](https://arxiv.org/abs/2010.11929) - by A Dosovitskiy · 2020 · Cited by 91195 — An Image is Worth 16x16 Words: Transformers for Image Rec...

9. [Attention Flow](https://amsterdamnlp.github.io/blog/attentionflow/) - Samira Abnar and Willem Zuidema. 2020. Quantifying Attention Flow in Transformers. In Proceedings of...

10. [Transformer Interpretability Beyond Attention Visualization](https://www.semanticscholar.org/paper/Transformer-Interpretability-Beyond-Attention-Chefer-Gur/0acd7ff5817d29839b40197f7a4b600b7fba24e4) - This work proposes a novel way to compute relevancy for Transformer networks that assigns local rele...

11. [Dosovitskiy et al (2021) An Image is Worth 16x16 Words](https://www.adrian.idv.hk/2025-03-26-dbkwzudmhguh21-vit/) - This is the paper that introduced the Vision Transformer (ViT), which proposed that transformers can...

12. [Quantifying Attention Flow in Transformers](https://samiraabnar.github.io/articles/2020-04/attention_flow) - I explain two simple but effective methods, called Attention Rollout and Attention Flow, to compute ...

13. [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838) - by H Chefer · 2020 · Cited by 1446 — We propose a novel way to compute relevancy for Transformer net...

14. [Transformers for Image Recognition at Scale](https://research.google/pubs/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale/) - by A Kolesnikov · Cited by 91195 — An Image is Worth 16x16 Words: Transformers for Image Recognition...

15. [Quantifying Attention Flow in Transformers](https://notes-vault.pages.dev/Scientific-Literature-References/literature-notes/@abnarQuantifyingAttentionFlow2020) - We propose two methods for approximating the attention to input tokens given attention weights, atte...

16. [Transformer Interpretability Beyond Attention Visualization](https://cris.bgu.ac.il/en/publications/transformer-interpretability-beyond-attention-visualization)

17. [Selvaraju - Visual Explanations From Deep Networks Via ...](https://www.scribd.com/document/885147204/Selvaraju-Visual-Explanations-from-Deep-Networks-via-Gradient-based-Localization) - The document presents Grad-CAM, a technique for generating visual explanations from Convolutional Ne...

18. [[PDF] DIME: Fine-grained Interpretations of Multimodal Models via Disentangled Local Explanations | Semantic Scholar](https://www.semanticscholar.org/paper/DIME:-Fine-grained-Interpretations-of-Multimodal-Lyu-Liang/0fc5cd83fcc493a764b6b8ab8496a40f46d130d8) - DIME enables accurate and fine-grained analysis of multimodal models while maintaining generality ac...

19. [RISE: Randomized Input Sampling for Explanation of Black ...](https://www.semanticscholar.org/paper/RISE:-Randomized-Input-Sampling-for-Explanation-of-Petsiuk-Das/d00c7fc5201405d5411b5ad3da93c5575ce8f10e) - The problem of Explainable AI for deep neural networks that take images as input and output a class ...

20. [Grad-CAM: Visual Explanations From Deep Networks via ...](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf) - by RR Selvaraju · 2017 · Cited by 31282 — Abstract. We propose a technique for producing 'visual exp...

21. [FINE-GRAINED DISENTANGLED REPRESENTATION LEARNING FOR MULTIMODAL](https://web3.arxiv.org/pdf/2312.13567)

22. [RISE: Randomized Input Sampling for Explanation of Black ...](http://bmvc2018.org/contents/papers/1064.pdf) - by V Petsiuk · Cited by 1966 — PETSIUK, DAS, SAENKO: RISE: RANDOMIZED INPUT SAMPLING FOR EXPLANATION...

23. [Grad-cam: visual explanations from deep - CVF Open Access](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) - by RR Selvaraju · 2017 · Cited by 31282 — We propose a technique for producing 'visual explanations'...

24. [DIME: Fine-grained Interpretations of Multimodal Models via ...](https://papers.app.nz/view/paper?id=424994) - The ability for a human to understand an Artificial Intelligence (AI) model's decision-making proces...

25. [PETSIUK, DAS, SAENKO: RISE: RANDOMIZED INPUT SAMPLING FOR EXPLANATION](http://arxiv.org/pdf/1806.07421.pdf)

26. [Grad-CAM: Visual Explanations from Deep Networks via ...](https://www.semanticscholar.org/paper/Grad-CAM:-Visual-Explanations-from-Deep-Networks-Selvaraju-Das/5582bebed97947a41e3ddd9bd1f284b73f1648c2) - The proposed Grad-CAM technique uses the gradients of any target concept flowing into the final conv...

27. [Novel cross-dimensional coarse-fine-grained complementary ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888920/) - by M Liu · 2025 · Cited by 2 — These models achieved efficient cross-modal retrieval and cross-modal...

28. [RISE: Randomized Input Sampling for Explanation of Black ...](https://arxiv.org/abs/1806.07421) - by V Petsiuk · 2018 · Cited by 1966 — We propose an approach called RISE that generates an importanc...

29. [Grad-CAM: Visual Explanations from Deep Networks via ...](https://dl.acm.org/doi/10.1007/s11263-019-01228-7) - We propose a technique for producing 'visual explanations' for decisions from a large class of Convo...

30. [DIME: Fine-grained Interpretations of Multimodal Models ...](https://dl.acm.org/doi/10.1145/3514094.3534148) - In this paper, we focus on advancing the state-of-the-art in interpreting multimodal models - a clas...

31. [RISE: Randomized Input Sampling for Explanation of Black ...](https://oamonitor.ireland.openaire.eu/rpo/rcsi/search/publication?pid=10.48550%2Farxiv.1806.07421) - RISE: Randomized Input Sampling for Explanation of Black-box Models · Top-Down Visual Saliency Guide...

