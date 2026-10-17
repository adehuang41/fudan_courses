#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

OUTPUT_DIR=/data/dechao/cv_final_pj2/outputs/debug/debug_B_act_smoke
mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/checkpoints"

GPU_PRIORITY=(6 5 4 3 2 1 0)
MIN_FREE_MB=20000
SELECTED_GPU=""

declare -A GPU_FREE_MB
while IFS=, read -r index name total used free util; do
    index="${index// /}"
    free="${free// /}"
    GPU_FREE_MB["$index"]="$free"
done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)

for candidate in "${GPU_PRIORITY[@]}"; do
    free="${GPU_FREE_MB[$candidate]:-0}"
    if [[ "$free" -ge "$MIN_FREE_MB" ]]; then
        SELECTED_GPU="$candidate"
        break
    fi
done

if [[ -z "$SELECTED_GPU" ]]; then
    echo "No GPU has at least ${MIN_FREE_MB} MiB free memory."
    nvidia-smi
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
export SELECTED_PHYSICAL_GPU="$SELECTED_GPU"

nvidia-smi > "$OUTPUT_DIR/gpu_snapshot_before.txt"

cat > "$OUTPUT_DIR/train_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"
CUDA_VISIBLE_DEVICES=$SELECTED_GPU SELECTED_PHYSICAL_GPU=$SELECTED_GPU \\
  /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/debug_act_smoke.py \\
  --policy-config configs/policy/act_baseline.yaml \\
  --train-config configs/train/debug.yaml
EOF
chmod +x "$OUTPUT_DIR/train_command.sh"

cd "$PROJECT_ROOT"
/data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/train/debug_act_smoke.py \
  --policy-config configs/policy/act_baseline.yaml \
  --train-config configs/train/debug.yaml 2>&1 | tee "$OUTPUT_DIR/train_stdout.log"

nvidia-smi > "$OUTPUT_DIR/gpu_snapshot_after.txt"
