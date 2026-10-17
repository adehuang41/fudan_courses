#!/usr/bin/env bash
# Wait until the 3 main runs finish (cards free up), then run E3a + E3b
# sequentially on a single GPU.
set -u
cd "$(dirname "$0")/.."
PY=/home/dechao/.conda/envs/cv-hw2/bin/python
GPU=0

echo "[queue] waiting for E1_ce / E2_dice / E3_ce_dice to finish..."
while ps -ef | grep -v grep \
      | grep -qE 'configs/(E1_ce|E2_dice|E3_ce_dice)\.yaml'; do
  sleep 30
done
echo "[queue] main runs done. starting E3a on GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU $PY -m src.train \
    --config configs/E3a_ce_dice_2to1.yaml \
    > logs/E3a_ce_dice_2to1.log 2>&1
echo "[queue] E3a finished. starting E3b on GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU $PY -m src.train \
    --config configs/E3b_ce_dice_1to2.yaml \
    > logs/E3b_ce_dice_1to2.log 2>&1
echo "[queue] E3b finished. ablation queue done."
