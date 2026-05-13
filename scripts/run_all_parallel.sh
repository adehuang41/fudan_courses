#!/usr/bin/env bash
# Launch all 9 experiments in parallel, one per GPU.
set -u
cd "$(dirname "$0")/.."

CONFIGS=(
  configs/E1_baseline_r18.yaml
  configs/E2_baseline_r34.yaml
  configs/E3_hp_lr_high.yaml
  configs/E4_hp_lr_low.yaml
  configs/E5_hp_long.yaml
  configs/E6_scratch.yaml
  configs/E7_se_r18.yaml
  configs/E8_cbam_r18.yaml
  configs/E9_vit_tiny.yaml
)

mkdir -p logs
PIDS=()
for i in "${!CONFIGS[@]}"; do
  GPU=$((i % 8))
  CFG="${CONFIGS[$i]}"
  NAME=$(basename "$CFG" .yaml)
  LOG="logs/${NAME}.log"
  echo "[launch] GPU $GPU -> $CFG"
  CUDA_VISIBLE_DEVICES=$GPU nohup /home/dechao/.conda/envs/cv-hw2/bin/python -m src.train --config "$CFG" \
      > "$LOG" 2>&1 &
  PIDS+=($!)
  sleep 2  # stagger to avoid simultaneous dataset download
done

echo "Launched ${#PIDS[@]} jobs: ${PIDS[*]}"
echo "Waiting..."
for p in "${PIDS[@]}"; do wait "$p"; done
echo "All done."
