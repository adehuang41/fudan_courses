#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 OUT_DIR PHYSICAL_GPU POLICY_CONFIG TRAIN_CONFIG [RESUME_FROM]" >&2
  exit 2
fi

OUT_DIR="$1"
PHYSICAL_GPU="$2"
POLICY_CONFIG="$3"
TRAIN_CONFIG="$4"
RESUME_FROM="${5:-}"

cd /home/dechao/cv_final_pj2

PYTHON="${PYTHON:-/data/dechao/cv_final_pj2/envs/lerobot/bin/python}"
INTERVAL="${RESOURCE_MONITOR_INTERVAL_SECONDS:-30}"
VIDEO_CACHE_SIZE="${CALVIN_VIDEO_FRAME_CACHE_SIZE:-32}"

mkdir -p "$OUT_DIR/logs"

STDOUT_LOG="$OUT_DIR/logs/train_stdout.log"
LAUNCHER_LOG="$OUT_DIR/logs/launcher_stdout.log"
RESOURCE_LOG="$OUT_DIR/logs/resource_usage.tsv"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export SELECTED_PHYSICAL_GPU="$PHYSICAL_GPU"
export TOKENIZERS_PARALLELISM=false
export CALVIN_VIDEO_FRAME_CACHE_SIZE="$VIDEO_CACHE_SIZE"

collect_process_tree() {
  local root_pid="$1"
  echo "$root_pid"
  local children
  children=$(pgrep -P "$root_pid" 2>/dev/null || true)
  local child
  for child in $children; do
    collect_process_tree "$child"
  done
}

monitor_resources() {
  local train_pid="$1"
  printf "timestamp\ttrain_pid\tprocess_count\trss_mib\tpcpu_sum\tgpu_memory_mib\tgpu_util_percent\n" > "$RESOURCE_LOG"
  while kill -0 "$train_pid" >/dev/null 2>&1; do
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local pid_list
    pid_list=$(collect_process_tree "$train_pid" | paste -sd, -)
    local proc_count rss_kib pcpu_sum
    read -r proc_count rss_kib pcpu_sum < <(
      ps -o rss=,pcpu= -p "$pid_list" 2>/dev/null \
        | awk '{count += 1; rss += $1; cpu += $2} END {printf "%d %.0f %.2f\n", count, rss, cpu}'
    )
    local gpu_memory gpu_util
    IFS=',' read -r gpu_memory gpu_util < <(
      nvidia-smi --id="$PHYSICAL_GPU" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 \
        | tr -d ' '
    )
    printf "%s\t%s\t%s\t%.1f\t%s\t%s\t%s\n" \
      "$timestamp" \
      "$train_pid" \
      "${proc_count:-0}" \
      "$(awk -v rss="${rss_kib:-0}" 'BEGIN {print rss / 1024.0}')" \
      "${pcpu_sum:-0}" \
      "${gpu_memory:-NA}" \
      "${gpu_util:-NA}" \
      >> "$RESOURCE_LOG"
    sleep "$INTERVAL"
  done
}

TRAIN_COMMAND=(
  "$PYTHON" scripts/train/train_act_full.py
  --policy-config "$POLICY_CONFIG"
  --train-config "$TRAIN_CONFIG"
  --stdout-log "$STDOUT_LOG"
)
if [[ -n "$RESUME_FROM" ]]; then
  TRAIN_COMMAND+=(--resume-from "$RESUME_FROM")
fi

"${TRAIN_COMMAND[@]}" > "$LAUNCHER_LOG" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$OUT_DIR/train.pid"

monitor_resources "$TRAIN_PID" &
MONITOR_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e

wait "$MONITOR_PID" >/dev/null 2>&1 || true

exit "$TRAIN_STATUS"
