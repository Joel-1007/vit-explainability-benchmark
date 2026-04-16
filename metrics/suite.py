"""
Unified MetricSuite wrapper as specified in Phase 2 implementation guides.
Aggregates Localization, Robustness, Complexity, and Fidelity metrics under one class.
"""

from .fidelity import FidelityMetrics
from .localization import LocalizationMetrics
from .robustness import RobustnessMetrics
from .complexity import gini_coefficient, attribution_entropy, effective_mass_ratio

class MetricSuite:
    """
    Unified access point for all 13 Phase 2 metrics across 4 families:
    Fidelity (F1-F4), Localization (L1-L4), Robustness (R1-R3), Complexity (C1-C3).
    """
    def __init__(self, **kwargs):
        # Configure Fidelity, Localization, Robustness kwargs dynamically
        self.fidelity = FidelityMetrics(
            baseline_value=kwargs.get("baseline_value", 0.0),
            k_fractions=kwargs.get("k_fractions", [0.1, 0.2, 0.3]),
            device=kwargs.get("device", "cpu")
        )
        self.localization = LocalizationMetrics(
            patch_size=kwargs.get("patch_size", 16)
        )
        self.robustness = RobustnessMetrics(
            epsilon=kwargs.get("epsilon", 0.05),
            n_samples=kwargs.get("n_samples", 20),
            seed=kwargs.get("seed", 42),
            ssim_window=kwargs.get("ssim_window", 3),
            ssim_sigma=kwargs.get("ssim_sigma", 1.0)
        )
    
    # Exposing Complexity seamlessly
    @staticmethod
    def complexity_gini(attribution):
        return gini_coefficient(attribution)
        
    @staticmethod
    def complexity_entropy(attribution):
        return attribution_entropy(attribution)["entropy_norm"]
        
    @staticmethod
    def complexity_emr(attribution):
        return effective_mass_ratio(attribution)["emr"]
