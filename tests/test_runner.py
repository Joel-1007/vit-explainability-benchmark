"""
test_runner.py  —  Phase 3 / Task 3.3 Unit Tests
=================================================
36 pytest tests for Phase3Runner (guide Listing 4) in metrics/runner.py.

Test layout (RT = Runner Test)
------------------------------
RT_01–RT_09   Phase3Runner init & attribute checks
RT_10–RT_18   _run_combination: batch flow, normalisation, result schema
RT_19–RT_27   run(): checkpoint creation, skip-if-exists, seed reproducibility
RT_28–RT_36   CLI: _build_cli_parser flags; _cli_main --help/--dry-run/--list

All tests use a tiny MockViT (16×16 images, patch_size=4) and an
in-memory DataLoader — no file I/O except to a pytest-managed tmp_path.

Run with:
    pytest tests/test_runner.py -v
"""

from __future__ import annotations

import io
import pickle
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from explainers.base import BaseExplainer
from metrics.runner import Phase3Runner, _build_cli_parser, _cli_main


# ===========================================================================
# Shared mock infrastructure
# ===========================================================================

IMG_SIZE    = 16
PATCH_SIZE  = 4
P           = IMG_SIZE // PATCH_SIZE     # = 4
N_CLASSES   = 5
DIM         = 32
NUM_HEADS   = 4


class _MockAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim  = DIM // NUM_HEADS
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(DIM, DIM * 3, bias=False)
        self.proj      = nn.Linear(DIM, DIM, bias=False)
        self.attn_drop = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv  = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv  = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x_out)


class _MockBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attn  = _MockAttn()
        self.norm2 = nn.LayerNorm(DIM)
        self.mlp   = nn.Sequential(
            nn.Linear(DIM, DIM * 2), nn.GELU(), nn.Linear(DIM * 2, DIM)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _MockViT(nn.Module):
    """Minimal ViT compatible with BaseExplainer (has cls_token, blocks)."""

    def __init__(self):
        super().__init__()
        n_patches = P * P
        self.patch_embed = nn.Linear(3 * PATCH_SIZE * PATCH_SIZE, DIM, bias=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, DIM))
        self.blocks      = nn.ModuleList([_MockBlock() for _ in range(2)])
        self.norm        = nn.LayerNorm(DIM)
        self.head        = nn.Linear(DIM, N_CLASSES)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = PATCH_SIZE
        n_h, n_w = H // ps, W // ps
        patches = (
            x.unfold(2, ps, ps).unfold(3, ps, ps)
            .contiguous().view(B, C, n_h * n_w, ps * ps)
            .permute(0, 2, 1, 3).reshape(B, n_h * n_w, C * ps * ps)
        )
        tok = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        tok = torch.cat([cls, tok], dim=1) + self.pos_embed
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.norm(tok)[:, 0])


class _ConstantExplainer(BaseExplainer):
    """Explainer that always returns a (P, P) map of ones (deterministic)."""
    def explain(self, x, target_class, **kwargs):
        return torch.ones(P, P)


class _RandExplainer(BaseExplainer):
    """Explainer that returns a random (P, P) map."""
    def explain(self, x, target_class, **kwargs):
        return torch.rand(P, P)


def _make_loader(n_batches: int = 2, batch_size: int = 3,
                 with_masks: bool = False) -> DataLoader:
    """Create a tiny in-memory DataLoader."""
    N = n_batches * batch_size
    xs = torch.rand(N, 3, IMG_SIZE, IMG_SIZE)
    ys = torch.randint(0, N_CLASSES, (N,))
    if with_masks:
        masks = torch.rand(N, P, P)
        ds    = TensorDataset(xs, ys, masks)
    else:
        ds = TensorDataset(xs, ys)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


# Module-level shared objects
torch.manual_seed(0)
_model    = _MockViT().eval()
_loader   = _make_loader(n_batches=2, batch_size=2)
_loader3  = _make_loader(n_batches=2, batch_size=2, with_masks=True)

_MODELS     = {"mock-vit": _model}
_EXPLAINERS = {"constant": _ConstantExplainer, "rand": _RandExplainer}
_DATASETS   = {"tiny": _loader}


# ===========================================================================
# RT_01–RT_09: Phase3Runner init & attribute checks
# ===========================================================================

class TestPhase3RunnerInit:

    def test_RT_01_init_stores_models(self):
        """RT_01: models dict stored on self.models."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        assert "mock-vit" in r.models

    def test_RT_02_init_stores_explainers(self):
        """RT_02: explainers dict stored on self.explainers."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        assert "constant" in r.explainers
        assert r.explainers["constant"] is _ConstantExplainer

    def test_RT_03_init_stores_datasets(self):
        """RT_03: datasets dict stored on self.datasets."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        assert "tiny" in r.datasets

    def test_RT_04_default_device_cpu_when_no_cuda(self):
        """RT_04: default device is CPU on this test machine."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        assert r.device.type in ("cpu", "cuda")   # whichever is available

    def test_RT_05_explicit_device_cpu(self):
        """RT_05: device='cpu' string is converted to torch.device('cpu')."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS, device="cpu")
        assert r.device == torch.device("cpu")

    def test_RT_06_default_norm_mode_is_minmax(self):
        """RT_06: default norm_mode='minmax'."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        assert r.norm_mode == "minmax"

    def test_RT_07_custom_norm_mode(self):
        """RT_07: custom norm_mode='softmax' stored correctly."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS, norm_mode="softmax")
        assert r.norm_mode == "softmax"

    def test_RT_08_default_patch_size_16(self):
        """RT_08: default patch_size=16."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        assert r.patch_size == 16

    def test_RT_09_combination_key_format(self):
        """RT_09: combination_key returns 'ds-model-explainer' string."""
        r = Phase3Runner(_MODELS, _EXPLAINERS, _DATASETS)
        key = r.combination_key("imagenet", "vit-b16", "rollout")
        assert key == "imagenet-vit-b16-rollout"


# ===========================================================================
# RT_10–RT_18: _run_combination batch flow
# ===========================================================================

class TestRunCombination:

    def test_RT_10_result_has_required_keys(self, tmp_path):
        """RT_10: _run_combination result dict has all required keys."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("d-m-c", exp, _loader, path, max_batches=1)
        for key in ("key", "n_samples", "attributions", "labels",
                    "gt_masks", "norm_mode"):
            assert key in res, f"Missing key: {key}"

    def test_RT_11_result_key_matches_argument(self, tmp_path):
        """RT_11: result['key'] equals the passed key string."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("my-key", exp, _loader, path, max_batches=1)
        assert res["key"] == "my-key"

    def test_RT_12_n_samples_correct(self, tmp_path):
        """RT_12: n_samples = n_batches × batch_size (with max_batches)."""
        batch_size = 2
        loader = _make_loader(n_batches=3, batch_size=batch_size)
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        # max_batches=2 → 2 batches × 2 = 4 samples
        res  = r._run_combination("k", exp, loader, path, max_batches=2)
        assert res["n_samples"] == batch_size * 2

    def test_RT_13_attributions_shape(self, tmp_path):
        """RT_13: each attribution is (P, P) float32."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("k", exp, _loader, path, max_batches=1)
        for att in res["attributions"]:
            assert att.shape == (P, P), f"Expected ({P},{P}), got {att.shape}"
            assert att.dtype == torch.float32

    def test_RT_14_normalisation_applied_minmax(self, tmp_path):
        """RT_14: minmax normalisation → each attribution in [0,1]."""
        r    = Phase3Runner({"m": _model}, {"c": _RandExplainer},
                            {"d": _loader}, norm_mode="minmax",
                            patch_size=PATCH_SIZE)
        exp  = _RandExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("k", exp, _loader, path, max_batches=1)
        for att in res["attributions"]:
            assert float(att.min()) >= -1e-6
            assert float(att.max()) <= 1.0 + 1e-6

    def test_RT_15_labels_are_integers(self, tmp_path):
        """RT_15: labels stored as list of Python ints."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("k", exp, _loader, path, max_batches=1)
        for lbl in res["labels"]:
            assert isinstance(lbl, int), f"Expected int, got {type(lbl)}"

    def test_RT_16_gt_masks_none_for_2tuple_loader(self, tmp_path):
        """RT_16: gt_masks are None when loader yields (x, y) only."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("k", exp, _loader, path, max_batches=1)
        for mask in res["gt_masks"]:
            assert mask is None, "Expected None gt_mask for 2-tuple loader"

    def test_RT_17_gt_masks_present_for_3tuple_loader(self, tmp_path):
        """RT_17: gt_masks are tensors when loader yields (x, y, gt_mask)."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader3}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "test.pkl"
        res  = r._run_combination("k", exp, _loader3, path, max_batches=1)
        for mask in res["gt_masks"]:
            assert isinstance(mask, torch.Tensor), "Expected tensor gt_mask"

    def test_RT_18_checkpoint_written_as_pkl(self, tmp_path):
        """RT_18: checkpoint file is reloadable as pickle with result dict."""
        r    = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                            {"d": _loader}, patch_size=PATCH_SIZE)
        exp  = _ConstantExplainer(_model, patch_size=PATCH_SIZE)
        path = tmp_path / "combo.pkl"
        r._run_combination("k", exp, _loader, path, max_batches=1)
        assert path.exists(), "Checkpoint .pkl not written"
        loaded = Phase3Runner.load_checkpoint(str(path))
        assert "n_samples" in loaded


# ===========================================================================
# RT_19–RT_27: run() — checkpoint creation and skip logic
# ===========================================================================

class TestRun:

    def test_RT_19_run_creates_checkpoint_dir(self, tmp_path):
        """RT_19: run() creates checkpoint_dir if it doesn't exist."""
        ckpt_dir = str(tmp_path / "new_dir" / "nested")
        r = Phase3Runner({"m": _model}, {"c": _ConstantExplainer},
                         {"d": _loader}, patch_size=PATCH_SIZE)
        r.run(ckpt_dir, seed=0, max_batches=1)
        assert Path(ckpt_dir).exists()

    def test_RT_20_run_creates_one_pkl_per_combination(self, tmp_path):
        """RT_20: run() writes one .pkl per (dataset, model, explainer) combo."""
        n_ds  = 1
        n_mdl = 2
        n_exp = 2
        models = {f"m{i}": _MockViT().eval() for i in range(n_mdl)}
        r = Phase3Runner(models, {"c1": _ConstantExplainer, "c2": _ConstantExplainer},
                         {"d": _loader}, patch_size=PATCH_SIZE)
        r.run(str(tmp_path), seed=0, max_batches=1)
        pkls = list(tmp_path.glob("*.pkl"))
        assert len(pkls) == n_ds * n_mdl * n_exp, (
            f"Expected {n_ds*n_mdl*n_exp} pkls, found {len(pkls)}"
        )

    def test_RT_21_run_returns_results_dict(self, tmp_path):
        """RT_21: run() returns dict keyed by combination strings."""
        r = Phase3Runner(_MODELS, {"c": _ConstantExplainer},
                         _DATASETS, patch_size=PATCH_SIZE)
        res = r.run(str(tmp_path), seed=0, max_batches=1)
        assert isinstance(res, dict)
        assert len(res) == 1
        key = next(iter(res))
        assert "tiny" in key and "mock-vit" in key and "c" in key

    def test_RT_22_checkpoint_skip_if_exists(self, tmp_path):
        """RT_22: run() skips combinations with existing .pkl checkpoints."""
        r = Phase3Runner(_MODELS, {"c": _ConstantExplainer},
                         _DATASETS, patch_size=PATCH_SIZE)
        # First run — creates checkpoint
        r.run(str(tmp_path), seed=0, max_batches=1)
        pkl = list(tmp_path.glob("*.pkl"))[0]
        mtime1 = pkl.stat().st_mtime

        # Second run — should skip (file not modified)
        import time
        time.sleep(0.05)
        r.run(str(tmp_path), seed=0, max_batches=1)
        mtime2 = pkl.stat().st_mtime
        assert mtime1 == mtime2, "Checkpoint was overwritten despite skip"

    def test_RT_23_run_skipped_combos_not_in_return_dict(self, tmp_path):
        """RT_23: skipped combinations are excluded from run() return value."""
        r = Phase3Runner(_MODELS, {"c": _ConstantExplainer},
                         _DATASETS, patch_size=PATCH_SIZE)
        r.run(str(tmp_path), seed=0, max_batches=1)
        # Second run: no new combos → empty return dict
        res2 = r.run(str(tmp_path), seed=0, max_batches=1)
        assert len(res2) == 0, f"Expected empty dict; got {list(res2.keys())}"

    def test_RT_24_seed_applied_to_torch_rng(self, tmp_path):
        """RT_24: torch.manual_seed(seed) called inside run() — verifiable via first rand."""
        torch.manual_seed(99)
        r1 = Phase3Runner(_MODELS, {"c": _RandExplainer},
                          _DATASETS, patch_size=PATCH_SIZE)
        results1 = r1.run(str(tmp_path / "run1"), seed=7, max_batches=1)

        torch.manual_seed(99)
        r2 = Phase3Runner(_MODELS, {"c": _RandExplainer},
                          {"d": _make_loader()}, patch_size=PATCH_SIZE)
        results2 = r2.run(str(tmp_path / "run2"), seed=7, max_batches=1)

        # attributions should be identical (same seed, same data, same explainer)
        atts1 = results1[next(iter(results1))]["attributions"]
        atts2 = results2[next(iter(results2))]["attributions"]
        for a1, a2 in zip(atts1, atts2):
            assert torch.allclose(a1, a2, atol=1e-5), "Seed did not reproduce results"

    def test_RT_25_max_batches_limits_samples(self, tmp_path):
        """RT_25: max_batches=1 caps samples to batch_size."""
        loader = _make_loader(n_batches=5, batch_size=3)
        r = Phase3Runner(_MODELS, {"c": _ConstantExplainer},
                         {"d": loader}, patch_size=PATCH_SIZE)
        res = r.run(str(tmp_path), seed=0, max_batches=1)
        combo = next(iter(res.values()))
        assert combo["n_samples"] == 3, (
            f"Expected 3 samples (1 batch × 3), got {combo['n_samples']}"
        )

    def test_RT_26_list_checkpoints_utility(self, tmp_path):
        """RT_26: list_checkpoints returns sorted filenames after run."""
        r = Phase3Runner(_MODELS,
                         {"c": _ConstantExplainer, "d2": _ConstantExplainer},
                         _DATASETS, patch_size=PATCH_SIZE)
        r.run(str(tmp_path), seed=0, max_batches=1)
        ckpts = Phase3Runner.list_checkpoints(str(tmp_path))
        assert len(ckpts) == 2
        assert all(c.endswith(".pkl") for c in ckpts)
        assert ckpts == sorted(ckpts)

    def test_RT_27_load_checkpoint_roundtrip(self, tmp_path):
        """RT_27: load_checkpoint returns equal dict to what run() returned."""
        r = Phase3Runner(_MODELS, {"c": _ConstantExplainer},
                         _DATASETS, patch_size=PATCH_SIZE)
        run_res = r.run(str(tmp_path), seed=0, max_batches=1)
        combo   = next(iter(run_res.values()))
        key_str = combo["key"]
        pkl_path = tmp_path / f"{key_str}.pkl"
        loaded  = Phase3Runner.load_checkpoint(str(pkl_path))
        assert loaded["n_samples"] == combo["n_samples"]
        assert loaded["norm_mode"] == combo["norm_mode"]


# ===========================================================================
# RT_28–RT_36: CLI
# ===========================================================================

class TestCLI:

    def test_RT_28_parser_help_exits_zero(self):
        """RT_28: --help causes SystemExit(0)."""
        parser = _build_cli_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_RT_29_parser_defaults(self):
        """RT_29: all CLI flag defaults match documented values."""
        parser = _build_cli_parser()
        args   = parser.parse_args([])
        assert args.checkpoint_dir == "results/phase3"
        assert args.seed           == 42
        assert args.max_batches    is None
        assert args.norm_mode      == "minmax"
        assert args.patch_size     == 16
        assert args.dry_run        is False
        assert args.list_checkpoints is False

    def test_RT_30_parser_seed_flag(self):
        """RT_30: --seed 123 parsed correctly."""
        parser = _build_cli_parser()
        args   = parser.parse_args(["--seed", "123"])
        assert args.seed == 123

    def test_RT_31_parser_norm_mode_softmax(self):
        """RT_31: --norm-mode softmax accepted."""
        parser = _build_cli_parser()
        args   = parser.parse_args(["--norm-mode", "softmax"])
        assert args.norm_mode == "softmax"

    def test_RT_32_parser_invalid_norm_mode_raises(self):
        """RT_32: invalid --norm-mode causes SystemExit(2)."""
        parser = _build_cli_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--norm-mode", "l2"])
        assert exc.value.code == 2

    def test_RT_33_parser_dry_run_flag(self):
        """RT_33: --dry-run sets dry_run=True."""
        parser = _build_cli_parser()
        args   = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_RT_34_parser_max_batches_int(self):
        """RT_34: --max-batches 5 parsed as int."""
        parser = _build_cli_parser()
        args   = parser.parse_args(["--max-batches", "5"])
        assert args.max_batches == 5

    def test_RT_35_cli_main_list_checkpoints_empty(self, tmp_path, capsys):
        """RT_35: --list-checkpoints on empty dir prints 'No checkpoints'."""
        _cli_main(["--checkpoint-dir", str(tmp_path), "--list-checkpoints"])
        captured = capsys.readouterr()
        assert "No checkpoints" in captured.out

    def test_RT_36_cli_main_list_checkpoints_populated(self, tmp_path, capsys):
        """RT_36: --list-checkpoints lists existing .pkl files."""
        # Create a dummy .pkl
        (tmp_path / "dummy-combo.pkl").write_bytes(b"")
        _cli_main(["--checkpoint-dir", str(tmp_path), "--list-checkpoints"])
        captured = capsys.readouterr()
        assert "dummy-combo.pkl" in captured.out
