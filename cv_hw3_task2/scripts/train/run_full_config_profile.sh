#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

OUTPUT_DIR=/data/dechao/cv_final_pj2/outputs/profiling/full_config_quick_profile
mkdir -p "$OUTPUT_DIR"

SELECTED_GPU=5
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
export SELECTED_PHYSICAL_GPU="$SELECTED_GPU"

nvidia-smi > "$OUTPUT_DIR/gpu_snapshot_before.txt"

cat > "$OUTPUT_DIR/profile_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"
CUDA_VISIBLE_DEVICES=$SELECTED_GPU SELECTED_PHYSICAL_GPU=$SELECTED_GPU \\
  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/profile_act_full_config.py \\
  --physical-gpu "$SELECTED_GPU" \\
  --output-dir "$OUTPUT_DIR"
EOF
chmod +x "$OUTPUT_DIR/profile_command.sh"

cd "$PROJECT_ROOT"
/data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/profile_act_full_config.py \
  --physical-gpu "$SELECTED_GPU" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/profile_stdout.log"

nvidia-smi > "$OUTPUT_DIR/gpu_snapshot_after.txt"
