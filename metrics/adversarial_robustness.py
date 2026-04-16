# metrics/adversarial_robustness.py
"""
Adversarial Robustness Metric for ViT Explainers.
Evaluates how much an attribution map changes when the input is subjected to a PGD attack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from typing import Callable

from utils.pgd import pgd_attack


class PGDRobustnessMetric:
    """
    Evaluates the adversarial robustness of an explainer.
    
    This metric computes the similarity (e.g., structural or rank-based) between
    the attribution map of an original image and its adversarially perturbed counterpart.
    A robust explainer should output similar explanations for visually indistinguishable inputs,
    even if the classifier's prediction changes.
    """

    def __init__(self, epsilon: float = 0.03, alpha: float = 0.01, steps: int = 10, random_start: bool = True):
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def compute(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        explainer_fn: Callable[[torch.Tensor], torch.Tensor],
        target_class: int | None = None,
    ) -> float:
        """
        Compute adversarial robustness for a single image.

        Parameters
        ----------
        model : torch.nn.Module
            The classifier model.
        x : torch.Tensor
            Original input image tensor of shape (C, H, W) in [0, 1].
        explainer_fn : Callable[[torch.Tensor], torch.Tensor]
            A function or method that takes an image tensor `(C, H, W)` and returns
            the attribution map `(H_m, W_m)`.
        target_class : int | None
            If provided, performs a targeted attack towards this class.
            If None, performs an untargeted attack (maximizing loss against predicted class).

        Returns
        -------
        float
            Cosine similarity score between the original and adversarial attribution maps.
            Higher is better (more robust).
        """
        model.eval()

        # Determine target class for attack
        with torch.no_grad():
            logits = model(x.unsqueeze(0))
            pred_class = logits.argmax(dim=-1).item()
        
        attack_target = target_class if target_class is not None else pred_class

        # Generate adversarial example
        x_adv = pgd_attack(
            model=model,
            x=x,
            target=attack_target,
            epsilon=self.epsilon,
            alpha=self.alpha,
            steps=self.steps,
            random_start=self.random_start
        )

        # Generate explanations
        with torch.no_grad():
            att_orig = explainer_fn(x)
            att_adv = explainer_fn(x_adv)

        # Normalize to measure structural similarity (cosine similarity over flattened maps)
        att_orig_flat = att_orig.flatten()
        att_adv_flat = att_adv.flatten()
        
        # Zero-check
        if torch.all(att_orig_flat == 0) or torch.all(att_adv_flat == 0):
            return 0.0

        similarity = F.cosine_similarity(att_orig_flat.unsqueeze(0), att_adv_flat.unsqueeze(0)).item()
        
        return max(0.0, similarity)
