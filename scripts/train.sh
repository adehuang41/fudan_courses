#!/usr/bin/env bash
# Single-GPU YOLOv8s fine-tune on VisDrone (uses CUDA 7).
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA_VISIBLE_DEVICES=7 python -m src.train --config configs/train_yolov8s.yaml "$@"
