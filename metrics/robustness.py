"""
robustness.py  —  Task 2.3 Robustness Metrics  (Task 2.3 §2)
=============================================================
Implements the three robustness metrics R1–R3 defined in Task 2.3 §1.

R1  max_sensitivity        — worst-case perturbation response (Yeh et al., 2019)
R2  model_randomisation    — SSIM-based weight-scramble sanity check (Adebayo et al., 2018)
R3  label_randomisation    — Spearman rank-corr under classifier-head permutation
R3+ spearman_layer_curve   — per-layer ρ curve (cascading randomisation, guide §R3)

Also exports three model-utility functions:

    randomise_model_weights(model)       — deep-copy; all params ← N(0, 1)
    randomise_classifier_labels(model)   — deep-copy; only final Linear head
                                           weights are column-permuted
    randomise_model_cascade(model, n)    — deep-copy; last n transformer
                                           blocks randomised (cascading)

Design principles
-----------------
• Mirrors LocalizationMetrics exactly: stateless class, no global state,
  all methods accept torch.Tensor; outputs are Python float.
• Pure-PyTorch SSIM (3×3 Gaussian window, GPU-native) — no scikit-image dep.
• n_samples=20 default (fast unit tests); set to 50 for production benchmark
  runs per Yeh et al. (2019).  Fully configurable via constructor.
• Code style: from __future__ import annotations, NumPy-style docstrings,
  dash-separated section comments — matches all Phase 1 / Task 2.2 modules.

Usage
-----
    from metrics.robustness import (
        RobustnessMetrics,
        randomise_model_weights,
        randomise_classifier_labels,
    )

    rm = RobustnessMetrics(epsilon=0.05, n_samples=20, seed=42)

    # R1 — Max-Sensitivity
    ms = rm.max_sensitivity(explainer, model, image, att_orig)

    # R2 — Model Randomisation  (att_rand from randomise_model_weights copy)
    mr = rm.model_randomisation(att_orig, att_rand)

    # R3 — Label Randomisation  (att_shuf from randomise_classifier_labels copy)
    lr = rm.label_randomisation(att_orig, att_shuf)

    # All three in one call
    scores = rm.compute_all(explainer, model, image, att_rand, att_shuf)
    # → {'max_sensitivity': 0.31, 'model_randomisation': 0.87, 'label_randomisation': 0.54}
"""

from __future__ import annotations

import copy
import math
import random
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPS = 1e-8   # numerical stability floor for division


# ---------------------------------------------------------------------------
# Internal pure-PyTorch helpers
# ---------------------------------------------------------------------------

def _to_2d(t: torch.Tensor) -> torch.Tensor:
    """
    Squeeze tensor to shape (H, W).

    Accepts (H, W), (1, H, W), (1, 1, H, W), (B=1, 1, H, W).
    Matches the helper in localization.py for consistent handling.
    """
    t = t.float()
    while t.dim() > 2:
        if t.shape[0] == 1:
            t = t.squeeze(0)
        else:
            raise ValueError(
                f"Cannot reduce tensor of shape {tuple(t.shape)} to 2D — "
                "batch dimension > 1.  Pass one sample at a time."
            )
    return t


def _minmax_norm(t: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalise tensor to [0, 1].

    If the tensor is constant (max == min), returns a zero tensor of the
    same shape rather than NaN.  This avoids division-by-zero in SSIM
    computation when one attribution map is degenerate.
    """
    t_min = t.min()
    t_max = t.max()
    if (t_max - t_min) < _EPS:
        return torch.zeros_like(t)
    return (t - t_min) / (t_max - t_min + _EPS)


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Return a 1-D Gaussian kernel of the given size and sigma.

    Parameters
    ----------
    size   : kernel width (should be odd)
    sigma  : standard deviation
    device : target device

    Returns
    -------
    Tensor of shape (size,) normalised to sum = 1.
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords = coords - size // 2
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """
    Build a 2-D Gaussian kernel as the outer product of two 1-D kernels.

    Returns (1, 1, size, size) for use with F.conv2d.
    """
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = torch.outer(k1d, k1d)
    return k2d.unsqueeze(0).unsqueeze(0)   # (1, 1, size, size)


def _ssim(
    a: torch.Tensor,
    b: torch.Tensor,
    window_size: int = 3,
    sigma: float   = 1.0,
    data_range: float = 1.0,
) -> float:
    """
    Structural Similarity Index (SSIM) between two (H, W) tensors.

    Pure-PyTorch implementation — GPU-native, no scikit-image dependency.
    Both tensors are assumed to be already in [0, data_range].

    Implementation follows Wang et al. (2004):
        SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2)
                     ────────────────────────────────
                     (μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)

    with Gaussian-weighted local statistics.

    Parameters
    ----------
    a, b        : (H, W) tensors in [0, data_range], same shape.
    window_size : Gaussian kernel size (default 3).
    sigma       : Gaussian standard deviation (default 1.0).
    data_range  : value range of the input (default 1.0 after normalisation).

    Returns
    -------
    float in [-1, 1]; 1.0 = identical images.
    """
    # Wang et al. stability constants
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    device = a.device
    kernel = _gaussian_kernel_2d(window_size, sigma, device)   # (1,1,k,k)
    pad    = window_size // 2

    # Reshape to (1, 1, H, W) for conv2d
    A = a.unsqueeze(0).unsqueeze(0).float()   # (1, 1, H, W)
    B = b.unsqueeze(0).unsqueeze(0).float()   # (1, 1, H, W)

    mu_a  = F.conv2d(A, kernel, padding=pad)
    mu_b  = F.conv2d(B, kernel, padding=pad)
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b

    sig_a2  = F.conv2d(A * A, kernel, padding=pad) - mu_a2
    sig_b2  = F.conv2d(B * B, kernel, padding=pad) - mu_b2
    sig_ab  = F.conv2d(A * B, kernel, padding=pad) - mu_ab

    numerator   = (2.0 * mu_ab + C1) * (2.0 * sig_ab  + C2)
    denominator = (mu_a2 + mu_b2 + C1) * (sig_a2 + sig_b2 + C2)

    ssim_map = numerator / (denominator + _EPS)
    return float(ssim_map.mean().item())


def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Spearman rank correlation between two 1-D (or flattened) tensors.

    Ranking is done via argsort-of-argsort (stable) — no SciPy dependency.

    Parameters
    ----------
    a, b : any-shape tensors; they are flattened internally.

    Returns
    -------
    float in [-1, 1].
    """
    def _rank(x: torch.Tensor) -> torch.Tensor:
        """Return fractional ranks (ties share mean rank)."""
        n = x.numel()
        flat = x.flatten().float()
        # argsort gives indices that would sort the array
        sorted_idx = flat.argsort(stable=True)
        # ranks[sorted_idx[i]] = i  (0-based)
        ranks = torch.empty(n, dtype=torch.float32, device=x.device)
        ranks[sorted_idx] = torch.arange(n, dtype=torch.float32, device=x.device)
        return ranks

    ra = _rank(a)
    rb = _rank(b)

    n    = ra.numel()
    ra_m = ra - ra.mean()
    rb_m = rb - rb.mean()

    numerator   = (ra_m * rb_m).sum()
    denominator = (ra_m.pow(2).sum() * rb_m.pow(2).sum()).sqrt() + _EPS
    return float((numerator / denominator).item())


def _perturb(
    image: torch.Tensor,
    epsilon: float,
    rng: random.Random,
) -> torch.Tensor:
    """
    Add a uniform random perturbation δ ~ U(-ε, +ε) to `image`.

    Parameters
    ----------
    image   : (C, H, W) or (H, W) or (1, C, H, W) — any shape accepted.
    epsilon : ∞-norm bound on the perturbation.
    rng     : seeded Python random.Random instance (for reproducibility).

    Returns
    -------
    Perturbed tensor of identical shape as `image`.
    """
    # Use torch.manual_seed derived from Python RNG for reproducibility
    seed = rng.randint(0, 2 ** 31 - 1)
    gen  = torch.Generator(device=image.device)
    gen.manual_seed(seed)
    delta = torch.empty_like(image.float()).uniform_(-epsilon, epsilon, generator=gen)
    return image.float() + delta


# ---------------------------------------------------------------------------
# Model-utility functions (module-level; exported in __init__.py)
# ---------------------------------------------------------------------------

def randomise_model_weights(
    model: nn.Module,
    seed: int = 0,
) -> nn.Module:
    """
    Return a deep-copy of `model` with **all** parameters re-initialised
    from N(0, 1) (and all biases reset to zero).

    This is the 'fully randomised' model required for R2 (Model Randomisation).
    The original `model` is never mutated.

    Parameters
    ----------
    model : any nn.Module (fine-tuned or pre-trained).
    seed  : integer seed for reproducible randomisation.

    Returns
    -------
    nn.Module — independent deep copy with randomised weights.

    Notes
    -----
    All layer types are handled uniformly: each parameter tensor is replaced
    with a draw from N(0, 1).  No layer-specific Init heuristics (Kaiming,
    Xavier, etc.) are applied — the goal is maximum randomisation, not
    training-ready initialisation.
    """
    rand_model = copy.deepcopy(model)
    gen = torch.Generator()
    gen.manual_seed(seed)
    with torch.no_grad():
        for name, param in rand_model.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.normal_(param, mean=0.0, std=1.0)
    return rand_model


def randomise_classifier_labels(
    model: nn.Module,
    seed: int = 0,
    head_attr: str = "head",
) -> nn.Module:
    """
    Return a deep-copy of `model` with the **classifier head weights
    column-permuted** — i.e., output class assignments are randomly shuffled.

    The backbone parameters are left unchanged.
    This is the 'label-randomised' model required for R3 (Label Randomisation).

    Parameters
    ----------
    model     : any nn.Module with a final Linear classification head.
    seed      : integer seed for reproducible permutation.
    head_attr : attribute name of the head Linear layer (default 'head').
                Common aliases tried automatically if 'head' is absent:
                'classifier', 'fc', 'linear'.

    Returns
    -------
    nn.Module — independent deep copy; only head.weight columns are permuted.

    Raises
    ------
    AttributeError : if no linear head is found via the known attribute names.

    Notes
    -----
    Column permutation of the weight matrix W ∈ R^{C × D} rearranges which
    class each feature dimension is mapped to, but preserves the weight norms.
    This is a stricter test than zeroing the head because the model still
    outputs confident predictions — just for the wrong classes.
    """
    _HEAD_ALIASES = (head_attr, "classifier", "fc", "linear", "head")

    shuf_model = copy.deepcopy(model)

    # Locate the head layer
    head: Optional[nn.Linear] = None
    for alias in _HEAD_ALIASES:
        layer = getattr(shuf_model, alias, None)
        if isinstance(layer, nn.Linear):
            head = layer
            break

    if head is None:
        raise AttributeError(
            f"Could not locate a Linear classification head on model "
            f"{type(model).__name__}.  Tried aliases: {_HEAD_ALIASES}.  "
            "Pass the correct attribute name via head_attr=."
        )

    gen = torch.Generator()
    gen.manual_seed(seed)
    with torch.no_grad():
        n_in = head.weight.shape[1]   # feature dimension
        perm = torch.randperm(n_in, generator=gen)
        head.weight.data = head.weight.data[:, perm]
        # Bias is class-specific, not feature-specific → permute rows
        if head.bias is not None:
            n_out = head.bias.shape[0]
            row_perm = torch.randperm(n_out, generator=gen)
            head.bias.data = head.bias.data[row_perm]

    return shuf_model


def randomise_model_cascade(
    model:               nn.Module,
    n_layers_to_randomise: int,
    seed:                int   = 0,
    block_attr:          str   = "blocks",
) -> nn.Module:
    """
    Return a deep-copy of `model` with the **last `n_layers_to_randomise`
    transformer blocks** re-initialised from N(0, 1).

    Blocks are located via `model.<block_attr>`, a ModuleList or Sequential.
    Randomisation is applied to blocks at indices
    ``[n_total - n_layers_to_randomise, n_total)`` (i.e. from the last
    block backwards) — the cascading protocol used by Adebayo et al. (2018).

    Parameters
    ----------
    model                 : any nn.Module with a block-structured backbone.
    n_layers_to_randomise : number of blocks to randomise (1 = last only).
    seed                  : integer seed for reproducibility.
    block_attr            : attribute name of the block container.  Tried in
                            order: the given name, then 'blocks', 'layers'.
                            For Swin-B use 'layers' (stage-level containers).

    Returns
    -------
    nn.Module — independent deep copy.  Only the last N blocks are
    randomised; the rest of the backbone is intact.

    Raises
    ------
    AttributeError : if no block container is found under any alias.
    ValueError     : if n_layers_to_randomise < 1.
    """
    if n_layers_to_randomise < 1:
        raise ValueError(
            f"n_layers_to_randomise must be ≥ 1, got {n_layers_to_randomise}."
        )

    rand_model = copy.deepcopy(model)

    # Locate the block container (ModuleList / Sequential)
    _BLOCK_ALIASES = (block_attr, "blocks", "layers")
    container = None
    for alias in _BLOCK_ALIASES:
        # Support dotted paths like 'encoder.layers'
        obj = rand_model
        try:
            for part in alias.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):   # ModuleList / Sequential
                container = obj
                break
        except AttributeError:
            continue

    if container is None:
        raise AttributeError(
            f"Could not locate a block container on model "
            f"{type(model).__name__}.  Tried aliases: {_BLOCK_ALIASES}.  "
            "Pass the correct block attribute name via block_attr=."
        )

    blocks  = list(container)          # ordered list of nn.Module
    n_total = len(blocks)
    n_rand  = min(n_layers_to_randomise, n_total)   # clamp
    start   = n_total - n_rand

    gen = torch.Generator()
    gen.manual_seed(seed)
    with torch.no_grad():
        for block in blocks[start:]:
            for name, param in block.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=1.0)

    return rand_model


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RobustnessMetrics:
    """
    Stateless collection of robustness metrics R1–R3.

    All three metrics measure whether an explanation method is **sensitive
    to semantically meaningful changes** (input perturbations, weight
    randomisation, label randomisation) while remaining **insensitive to
    semantically irrelevant changes** — a necessary condition for faithful
    attributions.

    Parameters
    ----------
    epsilon   : float
        L∞ perturbation radius for R1 (Max-Sensitivity).  Default 0.05
        corresponds to ~13/255 in [0, 1]-normalised image space.
    n_samples : int
        Number of random perturbation samples drawn per image for R1.
        Default 20 (fast unit tests); use 50 for production benchmark
        runs per Yeh et al. (2019).
    seed : int | None
        Seed for the internal Python RNG controlling perturbation draws.
        Set to None for non-deterministic behaviour.
    ssim_window : int
        Gaussian kernel size for SSIM in R2.  Default 3.
    ssim_sigma : float
        Gaussian sigma for SSIM in R2.  Default 1.0.
    """

    def __init__(
        self,
        epsilon:     float     = 0.05,
        n_samples:   int       = 20,
        seed:        int | None = 42,
        ssim_window: int       = 3,
        ssim_sigma:  float     = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}.")
        if n_samples < 1:
            raise ValueError(f"n_samples must be at least 1, got {n_samples}.")

        self.epsilon     = epsilon
        self.n_samples   = n_samples
        self.ssim_window = ssim_window
        self.ssim_sigma  = ssim_sigma
        self._rng        = random.Random(seed)

    # ------------------------------------------------------------------
    # R1 — Max-Sensitivity
    # ------------------------------------------------------------------

    def max_sensitivity(
        self,
        explainer: Callable,
        model:     nn.Module,
        image:     torch.Tensor,
        att_orig:  torch.Tensor,
    ) -> float:
        """
        R1 — Max-Sensitivity (Yeh et al., 2019).

        Formal definition (Task 2.3 §1.1)
        -----------------------------------
        For a fixed input x with attribution φ(f, x), draw K random
        perturbations δ_k ~ U(-ε, +ε)^d  (L∞ ball of radius ε):

            MaxSens(φ, x) = max_{k=1..K}  ‖φ(f, x+δ_k) − φ(f, x)‖₂
                                           ─────────────────────────────
                                           ‖φ(f, x)‖₂ + ε_num

        where ε_num = 1e-8 is a numerical stability floor.

        Interpretation: higher MaxSens means the explanation changes
        dramatically under imperceptible input perturbations — a sign of
        instability.  A robust explanation should have low MaxSens.

        Parameters
        ----------
        explainer : callable with signature
                    ``(model, image_4d) -> List[Tensor] | Tensor``
                    where ``image_4d`` is (1, C, H, W).
                    Returns either a list of one (H_a, W_a) tensor or a
                    single (H_a, W_a) / (1, H_a, W_a) tensor.
        model     : nn.Module used for attribution (eval mode recommended).
        image     : (C, H, W)  or (1, C, H, W) — one sample.
        att_orig  : (H_a, W_a) — pre-computed attribution for `image`.
                    Passing this avoids a redundant explainer call.

        Returns
        -------
        float ≥ 0 — maximum relative L2 deviation across n_samples
        perturbations.
        """
        att_base = _to_2d(att_orig).float()
        norm_base = att_base.norm()   # L2 norm of original attribution

        # Ensure image has a batch dimension
        img = image.float()
        if img.dim() == 3:
            img = img.unsqueeze(0)    # (1, C, H, W)

        max_ratio = 0.0

        for _ in range(self.n_samples):
            perturbed = _perturb(img, self.epsilon, self._rng)
            att_perturbed = self._call_explainer(explainer, model, perturbed)
            att_perturbed = _to_2d(att_perturbed).float()

            diff  = (att_perturbed - att_base).norm()
            ratio = float((diff / (norm_base + _EPS)).item())
            if ratio > max_ratio:
                max_ratio = ratio

        return max_ratio

    # ------------------------------------------------------------------
    # R2 — Model Randomisation
    # ------------------------------------------------------------------

    def model_randomisation(
        self,
        att_orig: torch.Tensor,
        att_rand: torch.Tensor,
    ) -> float:
        """
        R2 — Model Randomisation (Adebayo et al., 2018 — Sanity Checks).

        Formal definition (Task 2.3 §1.2)
        -----------------------------------
        Given the original attribution φ(f_orig, x) and the attribution
        φ(f_rand, x) produced by a fully weight-randomised copy of the model:

            ModelRand(φ, x) = 1 − SSIM( norm(φ_orig), norm(φ_rand) )

        where SSIM is the Structural Similarity Index (Wang et al., 2004)
        and norm(·) is min-max normalisation to [0, 1].

        Interpretation: a score near 1 means the explanation changes
        substantially when model weights are randomised — the attribution
        is sensitive to what the model has *learned*.  A score near 0 means
        the explanation is almost identical regardless of the weights —
        a sanity-check failure (attribution ignores the model).

        Use ``randomise_model_weights(model)`` to produce ``f_rand`` and
        then generate ``att_rand`` = explainer(f_rand, image).

        Parameters
        ----------
        att_orig : (H, W) or broadcastable — attribution under original model.
        att_rand : (H, W) or broadcastable — attribution under rand model.

        Returns
        -------
        float in [0, 1].  Higher is better (explanation is model-sensitive).
        """
        a = _minmax_norm(_to_2d(att_orig))
        b = _minmax_norm(_to_2d(att_rand))

        # Ensure both maps have the same spatial resolution
        if a.shape != b.shape:
            b = F.interpolate(
                b.unsqueeze(0).unsqueeze(0),
                size=a.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        s = _ssim(a, b, window_size=self.ssim_window, sigma=self.ssim_sigma)
        # Clamp SSIM to [-1, 1] before subtraction (floating-point overflow guard)
        s = max(-1.0, min(1.0, s))
        # Clamp final result to [0, 1] — 1 − SSIM can drift outside [0, 1] for
        # very small maps where the Gaussian window has boundary effects.
        return float(max(0.0, min(1.0, 1.0 - s)))

    # ------------------------------------------------------------------
    # R3 — Label Randomisation
    # ------------------------------------------------------------------

    def label_randomisation(
        self,
        att_orig: torch.Tensor,
        att_shuf: torch.Tensor,
    ) -> float:
        """
        R3 — Label Randomisation.

        Formal definition (Task 2.3 §1.3)
        -----------------------------------
        Given the original attribution φ(f_orig, x) and the attribution
        φ(f_shuf, x) produced by the model with its classifier head
        column-permuted (output label assignments randomly shuffled):

            LabelRand(φ, x) = 1 − (|ρ(φ_orig, φ_shuf)| + 1) / 2

        where ρ is the Spearman rank correlation of the two flattened maps.

        Derivation:
          • ρ ∈ [-1, 1].
          • (|ρ| + 1) / 2 ∈ [0.5, 1] — measures *structural preservation*.
          • 1 − (|ρ| + 1) / 2 ∈ [0, 0.5] — measures *structural divergence*.
          • Score near 0.5 → explanation completely ignores label identity.
          • Score near 0   → explanation is invariant to label permutation
                             (sanity-check failure).

        Treating both positive AND negative correlation as preservation (via |ρ|)
        is the conventional choice for this family of sanity checks because
        sign-flipped attributions still preserve spatial structure.

        Use ``randomise_classifier_labels(model)`` to produce ``f_shuf``.

        Parameters
        ----------
        att_orig : (H, W) — attribution under original model.
        att_shuf : (H, W) — attribution under label-permuted model.

        Returns
        -------
        float in [0, 0.5].  Higher is better (more sensitive to labels).
        """
        a = _to_2d(att_orig).float()
        b = _to_2d(att_shuf).float()

        # Align spatial resolutions if necessary
        if a.shape != b.shape:
            b = F.interpolate(
                b.unsqueeze(0).unsqueeze(0),
                size=a.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        rho   = _spearman_corr(a, b)
        score = 1.0 - (abs(rho) + 1.0) / 2.0
        return float(score)

    # ------------------------------------------------------------------
    # Convenience: compute R1, R2, and R3 in a single call
    # ------------------------------------------------------------------

    def compute_all(
        self,
        explainer: Callable,
        model:     nn.Module,
        image:     torch.Tensor,
        att_orig:  torch.Tensor,
        att_rand:  torch.Tensor,
        att_shuf:  torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute R1, R2, and R3 in a single call for one sample.

        Parameters
        ----------
        explainer : callable — see max_sensitivity docstring.
        model     : nn.Module (original, unmodified).
        image     : (C, H, W) or (1, C, H, W) — one sample.
        att_orig  : (H_a, W_a) — attribution from original model.
        att_rand  : (H_a, W_a) — attribution from weight-randomised model.
        att_shuf  : (H_a, W_a) — attribution from label-randomised model.

        Returns
        -------
        dict with keys:
            'max_sensitivity'      — R1, float ≥ 0
            'model_randomisation'  — R2, float in [0, 1]
            'label_randomisation'  — R3, float in [0, 0.5]
        """
        return {
            "max_sensitivity":     self.max_sensitivity(
                explainer, model, image, att_orig
            ),
            "model_randomisation": self.model_randomisation(att_orig, att_rand),
            "label_randomisation": self.label_randomisation(att_orig, att_shuf),
        }

    # ------------------------------------------------------------------
    # R3+ — Spearman Layer Curve (guide §R3 reporting requirement)
    # ------------------------------------------------------------------

    def spearman_layer_curve(
        self,
        explainer:  "Callable",
        model:      nn.Module,
        image:      torch.Tensor,
        att_orig:   torch.Tensor,
        block_attr: str = "blocks",
    ) -> Dict[str, float]:
        """
        R3+ — Spearman rank-correlation curve across transformer blocks.

        Cascading randomisation protocol (Adebayo et al., 2018)
        --------------------------------------------------------
        For a model with L transformer blocks, iterate n = 1, 2, …, L:
          1. Create a cascade-randomised copy of the model where the last
             n blocks are re-initialised from N(0, 1); the first (L-n)
             blocks retain their trained weights.
          2. Run the explainer on this cascade model to get att_cascade.
          3. Compute Spearman ρ(att_orig, att_cascade).

        The result is a dict mapping block labels to ρ values:
          { 'blocks.11': ρ₁, 'blocks.10': ρ₂, ..., 'blocks.0': ρ_L }

        Interpretation:
          • ρ near 1 after randomising only the last block → the
            explanation relies almost entirely on earlier layers.
          • ρ near 0 after randomising all blocks → the explanation is
            faithful to the full model's computation.
          • The shape of the curve (slow vs. abrupt drop) characterises
            which layers are most attribution-relevant.

        Parameters
        ----------
        explainer  : callable — same interface as max_sensitivity.
        model      : nn.Module (original, trained).
        image      : (C, H, W) or (1, C, H, W) — one sample.
        att_orig   : (H_a, W_a) — pre-computed attribution for `image`.
        block_attr : attribute name of the transformer block container.
                     Default 'blocks' (ViT, DeiT, BEiT, DINO, DINOv2).
                     For Swin-B use 'layers'.

        Returns
        -------
        dict[str, float] — ordered from last block (n=1) to first (n=L).
        Key format: f"{block_attr}.{block_index}"
        e.g. {'blocks.11': 0.91, 'blocks.10': 0.83, ..., 'blocks.0': 0.04}

        Notes
        -----
        This method makes L explainer calls (one per block depth) so it
        is significantly more expensive than other metrics.  For 12-block
        ViT models it makes 12 calls per image.  Use on a small subset.
        """
        # Locate block container to determine L
        _BLOCK_ALIASES = (block_attr, "blocks", "layers")
        container = None
        for alias in _BLOCK_ALIASES:
            obj = model
            try:
                for part in alias.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "__len__"):
                    container = obj
                    block_attr = alias   # use the alias that worked
                    break
            except AttributeError:
                continue

        if container is None:
            raise AttributeError(
                f"Could not locate block container (tried {_BLOCK_ALIASES}). "
                "For Swin-B, pass block_attr='layers'."
            )

        n_blocks = len(list(container))
        att_base = _to_2d(att_orig).float()

        img = image.float()
        if img.dim() == 3:
            img = img.unsqueeze(0)   # (1, C, H, W)

        curve: Dict[str, float] = {}
        for n in range(1, n_blocks + 1):
            seed = self._rng.randint(0, 2 ** 31 - 1)
            cascade = randomise_model_cascade(
                model,
                n_layers_to_randomise=n,
                seed=seed,
                block_attr=block_attr,
            )
            att_c = self._call_explainer(explainer, cascade, img)
            att_c = _to_2d(att_c).float()

            # Align spatial resolution if necessary
            if att_c.shape != att_base.shape:
                import torch.nn.functional as _F
                att_c = _F.interpolate(
                    att_c.unsqueeze(0).unsqueeze(0),
                    size=att_base.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)

            rho = _spearman_corr(att_base, att_c)
            # Key: which block was the FIRST to be randomised
            key = f"{block_attr}.{n_blocks - n}"
            curve[key] = rho

        return curve

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _call_explainer(
        explainer: Callable,
        model:     nn.Module,
        image_4d:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalise the return value of `explainer(model, image_4d)`.

        Accepts:
          • List[Tensor]  — takes element [0]
          • Tensor (1, H, W) or (H, W)

        Returns (H, W) Tensor.
        """
        result = explainer(model, image_4d)
        if isinstance(result, (list, tuple)):
            result = result[0]
        return result
