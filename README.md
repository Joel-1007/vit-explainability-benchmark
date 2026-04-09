# ViT Explainability Benchmark — Phase 1

> **Task 1.2 — Model Zoo & Standardised Fine-Tuning Protocol**

Controlled-variation model zoo of six Vision Transformers (~86 M params each) fine-tuned on
CUB-200-2011, PASCAL VOC 2012, CheXpert, and ImageNet-1K under a single standardised protocol,
enabling causal attribution of explanation-quality differences to architecture, not training conditions.

---

## Model Zoo

| # | Model | Architecture | Pre-training | Patch | Params | IN-1K Top-1 |
|---|-------|-------------|-------------|-------|--------|-------------|
| 1 | ViT-B/16 | Standard ViT | Supervised IN-21K (augreg) | 16 | 86 M | 84.2 % |
| 2 | DeiT-B/16 (distilled) | DeiT | Knowledge distillation IN-1K | 16 | 87 M | 83.4 % |
| 3 | Swin-B | Shifted-window | Supervised IN-22K→IN-1K | 4 | 88 M | 85.2 % |
| 4 | BEiT-B/16 | BERT-style MIM ViT | MIM IN-22K (DALL-E dVAE) | 16 | 86 M | 85.2 % |
| 5 | DINO-ViT-B/8 | Standard ViT | Self-distillation IN-1K | 8 | 85 M | 80.1 % (k-NN) |
| 6 | DINOv2-ViT-B/14 | Standard ViT | Self-distillation LVD-142M | 14 | 86 M | 86.5 % |

---

## Project Layout

```
TPAMI/
├── model_zoo/
│   ├── __init__.py          # load_model() dispatcher
│   ├── vit_b16.py           # Model 1 — ViT-B/16 (timm augreg_in21k)
│   ├── deit_b16.py          # Model 2 — DeiT-B/16 distilled
│   ├── swin_b.py            # Model 3 — Swin-B
│   ├── beit_b16.py          # Model 4 — BEiT-B/16 (pre-train only)
│   ├── dino_vitb8.py        # Model 5 — DINO-ViT-B/8 (torch.hub)
│   └── dinov2_vitb14.py     # Model 6 — DINOv2-ViT-B/14 (timm / HF)
│
├── training/
│   ├── __init__.py
│   ├── transforms.py        # RandAugment pipeline (§5.4)
│   ├── mixup.py             # Batch-level Mixup α=0.8 (§5.4)
│   ├── optimizer.py         # AdamW + LR warmup + cosine decay (§5.5)
│   ├── loss.py              # SoftTargetCrossEntropy / BCE for CheXpert (§5.7)
│   └── trainer.py           # Full fine-tune loop with grad accumulation (§5.6)
│
├── scripts/
│   ├── record_model_hashes.py   # §6.4 — run once after checkpoint download
│   └── pilot_finetune.py        # §7   — 5-epoch pilot with checklist
│
├── configs/
│   ├── cub200.yaml          # 50 epochs, full fine-tune
│   ├── pascal_voc.yaml      # 30 epochs, full fine-tune
│   ├── chexpert.yaml        # 30 epochs, class-weighted loss
│   └── imagenet.yaml        # 30 epochs, linear probe only
│
├── model_hashes.txt         # SHA-256 of all pre-trained checkpoints  ← commit this
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Record pre-trained checkpoint hashes (run once)

```bash
python scripts/record_model_hashes.py --out model_hashes.txt
```

Commit `model_hashes.txt` to the repo immediately.

### 3. Run the pilot fine-tune (§7 sanity check)

```bash
python scripts/pilot_finetune.py \
    --data_root /path/to/CUB_200_2011 \
    --batch_size 64 \
    --accum_steps 4        # 64 × 4 = 256 effective batch size
```

Expected: **top-1 ≥ 65 %** on CUB-200-2011 val after 5 epochs.

### 4. Load any model programmatically

```python
from model_zoo import load_model

# Feature extractor (num_classes=0 strips the head)
model = load_model("vit_b16", num_classes=0, pretrained=True)

# With classification head
model = load_model("dino_vitb8", num_classes=200, pretrained=True)
model = load_model("dinov2_vitb14", num_classes=20,  pretrained=True)
```

### 5. Fine-tune a model

```python
from model_zoo import load_model
from training.trainer import fine_tune
from torch.utils.data import DataLoader

model = load_model("vit_b16", num_classes=200, pretrained=True)

config = {
    "model_name":  "vit_b16",
    "dataset":     "cub200",
    "num_classes": 200,
    "num_epochs":  50,
    "base_lr":     1e-4,
    "batch_size":  256,
    "accum_steps": 4,          # adjust for your GPU
    "save_dir":    "checkpoints/cub200",
}

history = fine_tune(model, train_loader, val_loader, config)
```

---

## Standardised Fine-Tuning Protocol (§5.1)

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW β₁=0.9, β₂=0.999, ε=1e-8 |
| Weight decay | 0.05 (bias/LayerNorm excluded) |
| Base LR | 1e-4 (batch 256); scaled linearly |
| LR schedule | Cosine annealing, 5-epoch linear warmup |
| Epochs | 50 (CUB) · 30 (VOC, CheXpert, ImageNet) |
| Input size | 224 × 224 |
| Augmentation | RandAugment M=9 N=2, random erasing p=0.25 |
| Mixup | α=0.8; CutMix disabled |
| Label smoothing | ε=0.1 |
| Stochastic depth | 0.1 |
| Dropout | 0.0 (disabled) |

---

## Explainability Notes (Phase 3 Preview)

| Model | CLS token | Attention rollout | GradCAM |
|---|---|---|---|
| ViT-B/16 | ✓ | ✓ | ✓ |
| DeiT-B/16 | ✓ + distil token | ✓ (document choice) | ✓ |
| Swin-B | ✗ (none) | ✗ **structural limitation** | ✓ |
| BEiT-B/16 | ✓ | ✓ | ✓ |
| DINO-ViT-B/8 | ✓ | ✓ (784 patches) | ✓ |
| DINOv2-ViT-B/14 | ✓ | ✓ (256 patches) | ✓ |

> **Swin-B**: standard attention rollout does not apply (no CLS token, local windows).  
> Flag as structural limitation in the paper.

---

## Reproducibility

- All pre-trained checkpoint SHA-256 hashes are stored in `model_hashes.txt`.
- All fine-tuned checkpoint hashes are logged per epoch to `checkpoints/*/finetuned_hashes.txt`.
- Pin dependency versions via `requirements.txt`.
- Checkpoints themselves are `.gitignore`d (large binaries); only hash logs are committed.
