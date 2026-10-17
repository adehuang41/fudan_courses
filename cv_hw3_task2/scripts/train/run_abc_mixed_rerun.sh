#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

RERUN_DIR=/data/dechao/cv_final_pj2/outputs/baseline/abc_mixed_rerun_20260601
OUTPUT_DIR=/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline
mkdir -p "$RERUN_DIR"

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
    echo "output_dir: \"$OUTPUT_DIR\""
  } > "$RERUN_DIR/rerun_status.yaml"
}

if [ -e "$OUTPUT_DIR" ] && [ "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 | wc -l)" -ne 0 ]; then
  echo "Refusing to overwrite non-empty output dir: $OUTPUT_DIR"
  exit 3
fi

mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/checkpoints"

write_status "starting" "ABC-mixed rerun starting"
nvidia-smi > "$RERUN_DIR/gpu_snapshot_rerun_start.txt"
nvidia-smi > "$OUTPUT_DIR/gpu_snapshot_before.txt"

cat > "$OUTPUT_DIR/train_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"
CUDA_VISIBLE_DEVICES=$SELECTED_GPU SELECTED_PHYSICAL_GPU=$SELECTED_GPU \\
  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/train_act_full.py \\
  --policy-config configs/policy/act_baseline.yaml \\
  --train-config configs/train/full_ABC_mixed.yaml
EOF
chmod +x "$OUTPUT_DIR/train_command.sh"

write_status "running" "training ABC_mixed_act_baseline"
/data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/train_act_full.py \
  --policy-config configs/policy/act_baseline.yaml \
  --train-config configs/train/full_ABC_mixed.yaml 2>&1 | tee "$OUTPUT_DIR/train_stdout.log"

nvidia-smi > "$OUTPUT_DIR/gpu_snapshot_after.txt"

write_status "gate_check" "checking ABC_mixed_act_baseline"
/data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/gate_check_training.py \
  --train-config configs/train/full_ABC_mixed.yaml 2>&1 | tee "$OUTPUT_DIR/gate_check_stdout.log"

/data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/results/render_training_curves.py \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/render_curves_stdout.log"

nvidia-smi > "$RERUN_DIR/gpu_snapshot_rerun_end.txt"
write_status "completed" "ABC-mixed rerun completed and passed gate check"
