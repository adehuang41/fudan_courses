# HW2 · Task 1 · Oxford-IIIT Pet Fine-tuning

Fine-tuning ImageNet-pretrained CNNs (ResNet-18 / ResNet-34, with optional SE / CBAM)
and a lightweight ViT-Tiny on the Oxford-IIIT Pet Dataset (37 classes), plus the
required hyperparameter analysis, pretraining ablation, and attention-module
comparison.

## Environment

Hardware used: 8 × NVIDIA A800 80GB, driver 570 / CUDA 12.8.

```bash
conda create -n cv-hw2 python=3.11 -y
conda activate cv-hw2
# PyTorch wheel matched to CUDA 12.4 (compatible with driver 12.8)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
swanlab login --api-key <YOUR_SWANLAB_KEY>
```

> **cuDNN note** — on this machine cuDNN 9.1 (bundled with torch 2.5.1+cu124)
> errors with `CUDNN_STATUS_NOT_INITIALIZED` on the A800 / driver 570 combo, so
> `src/train.py` disables cuDNN at import time. Native CUDA kernels are still
> fast enough (ResNet-18 / 30 epochs ≈ 2 minutes on a single A800).

## Dataset

Auto-downloaded by `torchvision.datasets.OxfordIIITPet` on first run into
`./data`. Split:

- `trainval` (3680 images) → 90% train / 10% val (random, seed 42)
- `test` (3669 images) → final test set

## Training a single experiment

```bash
python -m src.train --config configs/E1_baseline_r18.yaml
# override knobs ad hoc:
python -m src.train --config configs/E1_baseline_r18.yaml --epochs 10 --no_log
```

Checkpoints land in `checkpoints/<run_name>/{best,last}.pt` plus a
`metrics.json`. Training / val / test curves are streamed to
[swanlab.cn](https://swanlab.cn) under project `hw2-task1`.

## Reproducing all experiments

```bash
bash scripts/run_all_parallel.sh           # E1–E9 on 8 GPUs
# Attention-fix variants (E7b, E8b) launched manually after fixing the
# random-init-disrupts-pretrained issue (see report):
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/E7b_se_r18_fixed.yaml &
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/E8b_cbam_r18_fixed.yaml &
wait
```

## Experiment matrix

| Run | Purpose | Model | Pretrained | head_lr | backbone_lr | attn_lr | epochs |
|-----|---------|-------|:----------:|--------:|------------:|--------:|------:|
| E1 | Baseline | ResNet-18 | ✓ | 1e-3 | 1e-4 | — | 30 |
| E2 | Baseline (deeper) | ResNet-34 | ✓ | 1e-3 | 1e-4 | — | 30 |
| E3 | HP — high LR | ResNet-18 | ✓ | 3e-3 | 3e-4 | — | 30 |
| E4 | HP — low LR | ResNet-18 | ✓ | 3e-4 | 3e-5 | — | 30 |
| E5 | HP — longer | ResNet-18 | ✓ | 1e-3 | 1e-4 | — | 60 |
| E6 | Pretrain ablation | ResNet-18 | ✗ | 1e-2 | 1e-2 | — | 60 |
| E7 | Attention naive | R18 + SE | ✓ | 1e-3 | 1e-4 | — | 30 |
| E7b | Attention fixed | R18 + SE | ✓ + identity-init | 1e-3 | 1e-4 | 1e-3 | 30 |
| E8 | Attention naive | R18 + CBAM | ✓ | 1e-3 | 1e-4 | — | 30 |
| E8b | Attention fixed | R18 + CBAM | ✓ + identity-init | 1e-3 | 1e-4 | 1e-3 | 30 |
| E7c | Attn — full SE pretrained | timm `legacy_seresnet18` | ✓ (incl. SE) | 1e-3 | 1e-4 | — | 30 |
| E8c | Attn — CBAM longer | R18 + CBAM | ✓ + identity-init | 1e-3 | 1e-4 | 1e-3 | 60 |
| E9 | Lightweight ViT | ViT-Tiny | ✓ (timm) | 1e-3 | 1e-4 | — | 30 |

Common to all: SGD (momentum 0.9, wd 5e-4) for ResNets, AdamW (wd 0.05) for
ViT, cosine schedule with 1-2 epoch linear warmup, batch size 64, image size
224, label smoothing 0.0 (0.1 for ViT), seed 42. Every run uses two parameter
groups (head_lr / backbone_lr) plus a third (attn_lr) when SE/CBAM are present.

## Final results

| Run | Test top-1 | Best val |
|---|---:|---:|
| E1 baseline ResNet-18 | 0.866 | 0.902 |
| E2 baseline ResNet-34 | **0.906** | 0.927 |
| E3 lr 3e-3 / 3e-4 | 0.890 | 0.927 |
| E4 lr 3e-4 / 3e-5 | 0.810 | 0.829 |
| E5 60 epochs | 0.887 | 0.927 |
| E6 scratch (no pretrain) | 0.530 | 0.609 |
| E7 R18 + SE (naive) | 0.778 | 0.813 |
| E7b R18 + SE (identity init) | **0.875** | 0.905 |
| E7c timm `legacy_seresnet18` (full SE pretrain) | 0.865 | 0.878 |
| E8 R18 + CBAM (naive) | 0.584 | 0.644 |
| E8b R18 + CBAM (identity init) | 0.869 | 0.889 |
| E8c R18 + CBAM (60 epoch) | 0.889 | 0.905 |
| E9 ViT-Tiny | 0.900 | **0.943** |

## Project layout

```
src/
  data.py                    OxfordIIITPet loader + augmentations
  models/
    resnet_attn.py           ResNet-18/34 with optional SE / CBAM
    build.py                 Model factory + per-group param splitter
  train.py                   Fine-tune loop (AMP, cosine LR, swanlab)
  utils.py
configs/                     One YAML per experiment (E1–E9, E7b, E8b)
scripts/run_all_parallel.sh  Launch 9 experiments across 8 GPUs
checkpoints/<run>/           best.pt, last.pt, metrics.json (gitignored)
logs/<run>.log               Stdout per experiment
```

## Links

- GitHub: <add after pushing>
- Model weights (best.pt for all 11 runs, ~270 MB total): <add network drive link>
