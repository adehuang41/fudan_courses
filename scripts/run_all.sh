#!/usr/bin/env bash
# Launch all 5 experiments in parallel, one per GPU.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

run() {
  local gpu=$1; local cfg=$2; local name=$3
  echo "[GPU $gpu] -> $name"
  CUDA_VISIBLE_DEVICES=$gpu /home/dechao/.conda/envs/cv-hw2/bin/python \
      -m src.train --config "$cfg" > "logs/${name}.log" 2>&1 &
}

run 0 configs/E1_ce.yaml            E1_ce
run 1 configs/E2_dice.yaml          E2_dice
run 2 configs/E3_ce_dice.yaml       E3_ce_dice
run 3 configs/E3a_ce_dice_2to1.yaml E3a_ce_dice_2to1
run 4 configs/E3b_ce_dice_1to2.yaml E3b_ce_dice_1to2

wait
echo "all runs finished"
