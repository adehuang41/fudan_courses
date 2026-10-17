#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 OUT_DIR PHYSICAL_GPU CARD_NAME -- COMMAND [ARGS...]" >&2
  exit 2
fi

OUT_DIR="$1"
PHYSICAL_GPU="$2"
CARD_NAME="$3"
shift 3

if [[ "${1:-}" != "--" ]]; then
  echo "Expected -- before command" >&2
  exit 2
fi
shift

if [[ $# -lt 1 ]]; then
  echo "Missing command" >&2
  exit 2
fi

cd /home/dechao/cv_final_pj2

INTERVAL="${RESOURCE_MONITOR_INTERVAL_SECONDS:-30}"
STDOUT_LOG="$OUT_DIR/stdout.log"
RESOURCE_LOG="$OUT_DIR/resource_usage.tsv"
STATUS_JSON="$OUT_DIR/status.json"

mkdir -p "$OUT_DIR"

json_escape() {
  python3 -c 'import json, sys; print(json.dumps(sys.stdin.read()))'
}

write_status() {
  local state="$1"
  local root_pid="$2"
  local exit_status="${3:-null}"
  local command_json
  command_json=$(printf "%s\n" "$*" | json_escape)
  cat > "$STATUS_JSON" <<EOF
{
  "card_name": "$CARD_NAME",
  "state": "$state",
  "root_pid": $root_pid,
  "physical_gpu": "$PHYSICAL_GPU",
  "start_time": "$START_TIME",
  "last_update_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "exit_status": $exit_status,
  "command": $COMMAND_JSON
}
EOF
}

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

mem_available_mib() {
  awk '/MemAvailable:/ {printf "%.1f\n", $2 / 1024.0}' /proc/meminfo
}

swap_used_mib() {
  awk '
    /SwapTotal:/ {total=$2}
    /SwapFree:/ {free=$2}
    END {printf "%.1f\n", (total - free) / 1024.0}
  ' /proc/meminfo
}

monitor_resources() {
  local root_pid="$1"
  local swap_baseline_mib="$2"
  printf "timestamp\tcard_name\troot_pid\tprocess_count\trss_mib\tpcpu_sum\tmem_available_mib\tswap_used_mib\tswap_delta_mib\tgpu_memory_mib\tgpu_util_percent\n" > "$RESOURCE_LOG"
  while kill -0 "$root_pid" >/dev/null 2>&1; do
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    local pid_list
    pid_list=$(collect_process_tree "$root_pid" | paste -sd, -)

    local proc_count rss_kib pcpu_sum
    read -r proc_count rss_kib pcpu_sum < <(
      ps -o rss=,pcpu= -p "$pid_list" 2>/dev/null \
        | awk '{count += 1; rss += $1; cpu += $2} END {printf "%d %.0f %.2f\n", count, rss, cpu}'
    )

    local available_mib swap_mib swap_delta_mib
    available_mib=$(mem_available_mib)
    swap_mib=$(swap_used_mib)
    swap_delta_mib=$(awk -v current="$swap_mib" -v base="$swap_baseline_mib" 'BEGIN {printf "%.1f\n", current - base}')

    local gpu_memory gpu_util
    IFS=',' read -r gpu_memory gpu_util < <(
      nvidia-smi --id="$PHYSICAL_GPU" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 \
        | tr -d ' '
    )

    printf "%s\t%s\t%s\t%s\t%.1f\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$timestamp" \
      "$CARD_NAME" \
      "$root_pid" \
      "${proc_count:-0}" \
      "$(awk -v rss="${rss_kib:-0}" 'BEGIN {print rss / 1024.0}')" \
      "${pcpu_sum:-0}" \
      "$available_mib" \
      "$swap_mib" \
      "$swap_delta_mib" \
      "${gpu_memory:-NA}" \
      "${gpu_util:-NA}" \
      >> "$RESOURCE_LOG"
    sleep "$INTERVAL"
  done
}

START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
COMMAND_JSON=$(printf "%s\n" "$*" | json_escape)
SWAP_BASELINE_MIB=$(swap_used_mib)

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export SELECTED_PHYSICAL_GPU="$PHYSICAL_GPU"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

stdbuf -oL -eL "$@" > "$STDOUT_LOG" 2>&1 &
ROOT_PID=$!
echo "$ROOT_PID" > "$OUT_DIR/root.pid"
write_status "running" "$ROOT_PID" "null"

monitor_resources "$ROOT_PID" "$SWAP_BASELINE_MIB" &
MONITOR_PID=$!

set +e
wait "$ROOT_PID"
EXIT_STATUS=$?
set -e

wait "$MONITOR_PID" >/dev/null 2>&1 || true

write_status "exited" "$ROOT_PID" "$EXIT_STATUS"
exit "$EXIT_STATUS"
