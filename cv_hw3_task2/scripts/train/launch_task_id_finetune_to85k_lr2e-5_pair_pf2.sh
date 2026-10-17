#!/usr/bin/env bash
set -euo pipefail

cd /home/dechao/cv_final_pj2

PYTHON=/data/dechao/cv_final_pj2/envs/lerobot/bin/python
POLICY_CONFIG=configs/policy/act_task_id_conditioned.yaml

B_CONFIG=configs/train/finetune_B_only_task_id_from_visual50k_to85k_seed20260531_retry1_lr2e-5_pf2.yaml
ABC_CONFIG=configs/train/finetune_ABC_mixed_task_id_from_visual50k_to85k_seed20260531_retry1_lr2e-5_pf2.yaml

B_OUT=/data/dechao/cv_final_pj2/outputs/task_conditioned/B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1
ABC_OUT=/data/dechao/cv_final_pj2/outputs/task_conditioned/ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1
B_RESUME="$B_OUT/checkpoints/latest.pt"
ABC_RESUME="$ABC_OUT/checkpoints/latest.pt"

B_FROZEN_75K="$B_OUT/frozen_75k_checkpoint_20260610T1645Z_immutable/step_075000.pt"
ABC_FROZEN_75K="$ABC_OUT/frozen_75k_checkpoint_20260610T1645Z_immutable/step_075000.pt"

B_SESSION=cvpj2_b_taskid_finetune_to85k_lr2e5_pf2
ABC_SESSION=cvpj2_abc_taskid_finetune_to85k_lr2e5_pf2

VIDEO_FRAME_CACHE_SIZE=32
RESOURCE_MONITOR_INTERVAL_SECONDS=30

if pgrep -af "scripts/train/train_act_full.py" >/dev/null; then
  echo "Refusing to launch: train_act_full.py is already running." >&2
  pgrep -af "scripts/train/train_act_full.py" >&2
  exit 1
fi

for session in "$B_SESSION" "$ABC_SESSION"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to launch because tmux session already exists: $session" >&2
    exit 1
  fi
done

for required in "$B_CONFIG" "$ABC_CONFIG" "$POLICY_CONFIG" "$B_RESUME" "$ABC_RESUME" "$B_FROZEN_75K" "$ABC_FROZEN_75K"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file is missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$B_OUT/logs" "$ABC_OUT/logs"
: > "$B_OUT/train.pid"
: > "$ABC_OUT/train.pid"

wait_for_pid_file() {
  local pid_file="$1"
  local deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    if [[ -s "$pid_file" ]]; then
      return 0
    fi
    sleep 0.2
  done
  echo "Timed out waiting for pid file: $pid_file" >&2
  exit 1
}

tmux new-session -d -s "$B_SESSION" \
  "cd /home/dechao/cv_final_pj2 && exec env PYTHON='$PYTHON' CALVIN_VIDEO_FRAME_CACHE_SIZE='$VIDEO_FRAME_CACHE_SIZE' RESOURCE_MONITOR_INTERVAL_SECONDS='$RESOURCE_MONITOR_INTERVAL_SECONDS' scripts/train/run_training_with_resource_monitor.sh '$B_OUT' 3 '$POLICY_CONFIG' '$B_CONFIG' '$B_RESUME'"

tmux new-session -d -s "$ABC_SESSION" \
  "cd /home/dechao/cv_final_pj2 && exec env PYTHON='$PYTHON' CALVIN_VIDEO_FRAME_CACHE_SIZE='$VIDEO_FRAME_CACHE_SIZE' RESOURCE_MONITOR_INTERVAL_SECONDS='$RESOURCE_MONITOR_INTERVAL_SECONDS' scripts/train/run_training_with_resource_monitor.sh '$ABC_OUT' 4 '$POLICY_CONFIG' '$ABC_CONFIG' '$ABC_RESUME'"

wait_for_pid_file "$B_OUT/train.pid"
wait_for_pid_file "$ABC_OUT/train.pid"

B_PID=$(cat "$B_OUT/train.pid")
ABC_PID=$(cat "$ABC_OUT/train.pid")

cat <<EOF
Launched task-id fine-tune LR probe extension to 85k.

B-only:
  pid: $B_PID
  tmux: $B_SESSION
  gpu: 3
  resume: $B_RESUME
  frozen_75k: $B_FROZEN_75K
  stdout: $B_OUT/logs/train_stdout.log
  scalars: $B_OUT/logs/scalars.jsonl
  resources: $B_OUT/logs/resource_usage.tsv
  status: $B_OUT/training_status.yaml

ABC-mixed:
  pid: $ABC_PID
  tmux: $ABC_SESSION
  gpu: 4
  resume: $ABC_RESUME
  frozen_75k: $ABC_FROZEN_75K
  stdout: $ABC_OUT/logs/train_stdout.log
  scalars: $ABC_OUT/logs/scalars.jsonl
  resources: $ABC_OUT/logs/resource_usage.tsv
  status: $ABC_OUT/training_status.yaml
EOF
