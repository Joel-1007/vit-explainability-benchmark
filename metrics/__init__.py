"""
metrics — ViT Explainability Benchmark
=======================================
Public API
----------
Normalisation  (Guide Listing 3 — sits between explainers and metrics)
    normalize_attribution        Normalise (Hp,Wp) or (B,Hp,Wp) to [0,1]
    normalize_batch              Convenience wrapper for (B,Hp,Wp) input
    NormMode                     Enum of valid mode strings
    AttributionNormError         Raised on invalid mode / shape

Sanity Checks  (Guide §3.4 — mandatory validation before experiments)
    run_s1_random_baseline       S1: complexity metrics on random attribution maps
    run_s2_model_randomisation   S2: Spearman ρ curve via cascading layer randomisation
    run_s3_label_permutation     S3: true-label vs wrong-label attribution divergence
    run_all_sanity_checks        Convenience: S1 + S2 + S3 in one call
    SanityResult / S1Result / S2Result / S3Result  typed result dataclasses

Fidelity
    FidelityMetrics              F1 Sufficiency, F2 Comprehensiveness, F3 Log-odds

Localization
    LocalizationMetrics          L1 mIoU, L2 Pointing Game, L3 EGT, L4 CalibGap

Robustness
    RobustnessMetrics            R1 MaxSens, R2 ModelRand, R3 LabelRand
    randomise_model_weights      Utility: deep-copy with N(0,1) weights
    randomise_classifier_labels  Utility: deep-copy with permuted head columns

Complexity
    ComplexityMetrics            C1 Gini, C2 Entropy, C3 EMR — unified class
    ComplexityResult             Dataclass for per-sample complexity scores
    normalise_attribution        Shared normalisation utility (minmax/softmax/percentile)

Axiomatic Analysis
    AxiomVerifier                Empirical (metric × axiom) test suite
    AxiomTestResult              Dataclass for a single test result
    verify_completeness          Standalone A2 verification for one sample
    generate_axiom_satisfaction_heatmap  Figure F1 generator (PDF)

BenchmarkRunner                 Unified dataset-level evaluation loop

Notes
-----
``ComplexityMetrics`` and ``AxiomVerifier`` are importable without ``torch``.
``LocalizationMetrics``, ``RobustnessMetrics``, and ``BenchmarkRunner``
require ``torch`` and are silently skipped if it is not installed.
"""

# ---------------------------------------------------------------------------
# Torch-independent submodules (always importable)
# ---------------------------------------------------------------------------
from .sanity import (
    run_s1_random_baseline,
    run_s2_model_randomisation,
    run_s3_label_permutation,
    run_all_sanity_checks,
    SanityResult,
    S1Result,
    S2Result,
    S3Result,
)
from .normalize import (
    normalize_attribution,
    normalize_batch,
    NormMode,
    AttributionNormError,
)
from .complexity import (
    ComplexityMetrics,
    ComplexityResult,
    normalise_attribution,   # complexity-internal normaliser (kept for back-compat)
)
from .axiom_verifier import (
    AxiomVerifier,
    AxiomTestResult,
    verify_completeness,
    generate_axiom_satisfaction_heatmap,
)

# ---------------------------------------------------------------------------
# Torch-dependent submodules (require torch to be installed)
# ---------------------------------------------------------------------------
try:
    from .fidelity import FidelityMetrics
    from .localization import LocalizationMetrics
    from .runner       import BenchmarkRunner, Phase3Runner
    from .robustness   import (
        RobustnessMetrics,
        randomise_model_weights,
        randomise_classifier_labels,
        randomise_model_cascade,   # Task 2.3 addendum — cascading layer utility
    )
    from .causal_fidelity import CausalMaskingMetric
    from .adversarial_robustness import PGDRobustnessMetric
    from .explainer_interaction import ExplainerInteractionGraph
    _TORCH_SUBMODULES_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_SUBMODULES_AVAILABLE = False

__all__ = [
    # Sanity checks — Guide §3.4 (torch-free)
    "run_s1_random_baseline",
    "run_s2_model_randomisation",
    "run_s3_label_permutation",
    "run_all_sanity_checks",
    "SanityResult",
    "S1Result",
    "S2Result",
    "S3Result",
    # Normalisation pipeline — Guide Listing 3 (torch-free)
    "normalize_attribution",
    "normalize_batch",
    "NormMode",
    "AttributionNormError",
    # Complexity (torch-free)
    "ComplexityMetrics",
    "ComplexityResult",
    "normalise_attribution",   # complexity-internal normaliser
    # Axiomatic analysis (torch-free core)
    "AxiomVerifier",
    "AxiomTestResult",
    "verify_completeness",
    "generate_axiom_satisfaction_heatmap",
    # Fidelity (requires torch)
    "FidelityMetrics",
    # Localization (requires torch)
    "LocalizationMetrics",
    # Runner (requires torch)
    "BenchmarkRunner",
    "Phase3Runner",
    # Robustness (requires torch)
    "RobustnessMetrics",
    "randomise_model_weights",
    "randomise_classifier_labels",
    "randomise_model_cascade",     # Task 2.3 addendum
]
