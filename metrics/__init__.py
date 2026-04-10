"""
metrics — ViT Explainability Benchmark
=======================================
Public API
----------
Fidelity
    (Phase 3 — FidelityMetrics class, not yet implemented)

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
from .complexity import (
    ComplexityMetrics,
    ComplexityResult,
    normalise_attribution,
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
    from .localization import LocalizationMetrics
    from .runner       import BenchmarkRunner
    from .robustness   import (
        RobustnessMetrics,
        randomise_model_weights,
        randomise_classifier_labels,
        randomise_model_cascade,   # Task 2.3 addendum — cascading layer utility
    )
    _TORCH_SUBMODULES_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_SUBMODULES_AVAILABLE = False

__all__ = [
    # Complexity (torch-free)
    "ComplexityMetrics",
    "ComplexityResult",
    "normalise_attribution",
    # Axiomatic analysis (torch-free core)
    "AxiomVerifier",
    "AxiomTestResult",
    "verify_completeness",
    "generate_axiom_satisfaction_heatmap",
    # Localization (requires torch)
    "LocalizationMetrics",
    # Runner (requires torch)
    "BenchmarkRunner",
    # Robustness (requires torch)
    "RobustnessMetrics",
    "randomise_model_weights",
    "randomise_classifier_labels",
    "randomise_model_cascade",     # Task 2.3 addendum
]
