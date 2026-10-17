#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

SEQUENCE_DIR=/data/dechao/cv_final_pj2/outputs/baseline/overnight_sequence_20260531
mkdir -p "$SEQUENCE_DIR"

SELECTED_GPU=5
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
export SELECTED_PHYSICAL_GPU="$SELECTED_GPU"

cd "$PROJECT_ROOT"

write_status() {
  local status="$1"
  local detail="$2"
  {
    echo "status: $status"
    echo "detail: \"$detail\""
    echo "updated_at: \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
    echo "physical_gpu: \"$SELECTED_GPU\""
  } > "$SEQUENCE_DIR/sequence_status.yaml"
}

ensure_fresh_output_dir() {
  local output_dir="$1"
  if [ -e "$output_dir" ] && [ "$(find "$output_dir" -mindepth 1 -maxdepth 1 | wc -l)" -ne 0 ]; then
    echo "Refusing to overwrite non-empty output dir: $output_dir"
    exit 3
  fi
  mkdir -p "$output_dir"
}

run_one() {
  local config="$1"
  local output_dir="$2"
  local name="$3"

  ensure_fresh_output_dir "$output_dir"
  mkdir -p "$output_dir/logs" "$output_dir/checkpoints"

  nvidia-smi > "$output_dir/gpu_snapshot_before.txt"

  cat > "$output_dir/train_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"
CUDA_VISIBLE_DEVICES=$SELECTED_GPU SELECTED_PHYSICAL_GPU=$SELECTED_GPU \\
  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/train_act_full.py \\
  --policy-config configs/policy/act_baseline.yaml \\
  --train-config "$config"
EOF
  chmod +x "$output_dir/train_command.sh"

  write_status "running_$name" "training $name"
  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/train_act_full.py \
    --policy-config configs/policy/act_baseline.yaml \
    --train-config "$config" 2>&1 | tee "$output_dir/train_stdout.log"

  nvidia-smi > "$output_dir/gpu_snapshot_after.txt"

  write_status "gate_check_$name" "checking $name"
  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/gate_check_training.py \
    --train-config "$config" 2>&1 | tee "$output_dir/gate_check_stdout.log"

  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/results/render_training_curves.py \
    --output-dir "$output_dir" 2>&1 | tee "$output_dir/render_curves_stdout.log"
}

write_status "starting" "overnight sequence starting"
nvidia-smi > "$SEQUENCE_DIR/gpu_snapshot_sequence_start.txt"

run_one \
  configs/train/full_B_only.yaml \
  /data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline \
  B_only_act_baseline

write_status "b_only_passed_starting_abc" "B-only gate passed; starting ABC-mixed"

run_one \
  configs/train/full_ABC_mixed.yaml \
  /data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline \
  ABC_mixed_act_baseline

nvidia-smi > "$SEQUENCE_DIR/gpu_snapshot_sequence_end.txt"
write_status "completed" "B-only and ABC-mixed completed and passed gate checks"
