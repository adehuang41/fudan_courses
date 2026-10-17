#!/usr/bin/env bash
set -euo pipefail

cd /home/dechao/cv_final_pj2

LEROBOT_PY=/data/dechao/cv_final_pj2/envs/lerobot/bin/python
CALVIN_PY=/data/dechao/cv_final_pj2/envs/calvin/bin/python

TASK_D_D_ROOT=/data/dechao/cv_final_pj2/datasets/task_D_D
D_LEROBOT_ROOT=/data/dechao/cv_final_pj2/datasets/calvin_D_D_lerobot

B_VISUAL=/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline
ABC_VISUAL=/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline
B_TASKID=/data/dechao/cv_final_pj2/outputs/task_conditioned/B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1
ABC_TASKID=/data/dechao/cv_final_pj2/outputs/task_conditioned/ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1

NUM_SEQUENCES=${NUM_SEQUENCES:-50}
EP_LEN=${EP_LEN:-360}
TRACK1_GPU=${TRACK1_GPU:-3}
SERVER_GPU=${SERVER_GPU:-3}
TRACK1_PREFIX=${TRACK1_PREFIX:-post_finetune_best_95k}
TRACK1_BATCH_SIZE=${TRACK1_BATCH_SIZE:-32}
TRACK1_NUM_WORKERS=${TRACK1_NUM_WORKERS:-4}
TRACK1_PREFETCH_FACTOR=${TRACK1_PREFETCH_FACTOR:-1}
TRACK1_TABLE_CACHE_SIZE=${TRACK1_TABLE_CACHE_SIZE:-32}
TRACK2_ROOT=${TRACK2_ROOT:-/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_closed_loop}

for path in \
  "$TASK_D_D_ROOT" \
  "$D_LEROBOT_ROOT" \
  "$B_VISUAL/checkpoints/best_val_action_l1.pt" \
  "$ABC_VISUAL/checkpoints/best_val_action_l1.pt" \
  "$B_TASKID/checkpoints/best_val_action_l1.pt" \
  "$ABC_TASKID/checkpoints/best_val_action_l1.pt"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

wait_for_server_ready_log() {
  local log_path="$1"
  local server_pid="$2"
  local deadline=$((SECONDS + 120))
  while (( SECONDS < deadline )); do
    if grep -q "server ready" "$log_path" 2>/dev/null; then
      return 0
    fi
    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "Server exited before readiness. Log:" >&2
      cat "$log_path" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for server readiness. Log:" >&2
  cat "$log_path" >&2 || true
  return 1
}

run_track1_offline() {
  echo "[Track1] Running D offline open-loop Action L1 for visual baselines + task-id fine-tunes"
  CUDA_VISIBLE_DEVICES="$TRACK1_GPU" "$LEROBOT_PY" scripts/eval/evaluate_d_offline_action_l1.py \
    --dataset-root "$D_LEROBOT_ROOT" \
    --include-task-id-finetunes \
    --batch-size "$TRACK1_BATCH_SIZE" \
    --num-workers "$TRACK1_NUM_WORKERS" \
    --prefetch-factor "$TRACK1_PREFETCH_FACTOR" \
    --table-cache-size "$TRACK1_TABLE_CACHE_SIZE" \
    --device cuda \
    --output-prefix "$TRACK1_PREFIX"
}

run_track2_closed_loop_one() {
  local label="$1"
  local server_script="$2"
  local run_dir="$3"
  local checkpoint="$4"
  local port="$5"
  local server_extra_arg="${6:-}"

  local out_dir="$TRACK2_ROOT/${label}_${NUM_SEQUENCES}seq_ep${EP_LEN}"
  local log_dir="$out_dir/logs"
  mkdir -p "$log_dir"

  if pgrep -af "scripts/eval/${server_script}.*--port ${port}" >/dev/null; then
    echo "Refusing to launch: ${server_script} already appears to be using port ${port}" >&2
    pgrep -af "scripts/eval/${server_script}.*--port ${port}" >&2
    exit 1
  fi

  echo "[Track2] Starting server for $label on port $port"
  if [[ -n "$server_extra_arg" ]]; then
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$LEROBOT_PY" "scripts/eval/${server_script}" \
      --host 127.0.0.1 \
      --port "$port" \
      --run-dir "$run_dir" \
      --checkpoint "$checkpoint" \
      --device cuda \
      --log-jsonl "$log_dir/server_actions.jsonl" \
      > "$log_dir/server_stdout.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" "$LEROBOT_PY" "scripts/eval/${server_script}" \
      --host 127.0.0.1 \
      --port "$port" \
      --run-dir "$run_dir" \
      --checkpoint "$checkpoint" \
      --device cuda \
      > "$log_dir/server_stdout.log" 2>&1 &
  fi
  local server_pid=$!
  echo "$server_pid" > "$log_dir/server.pid"

  cleanup_server() {
    local pid="${server_pid:-}"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_server RETURN

  wait_for_server_ready_log "$log_dir/server_stdout.log" "$server_pid"

  echo "[Track2] Running closed-loop $label num_sequences=$NUM_SEQUENCES ep_len=$EP_LEN"
  local eval_status=0
  "$CALVIN_PY" scripts/eval/calvin_closed_loop_video_eval.py \
    --dataset-path "$TASK_D_D_ROOT" \
    --host 127.0.0.1 \
    --port "$port" \
    --num-sequences "$NUM_SEQUENCES" \
    --ep-len "$EP_LEN" \
    --frame-stride 30 \
    --dataset-label "task_D_D / D-environment closed-loop evaluation" \
    --policy-label "$label" \
    --checkpoint-label "best_val_action_l1_post_finetune_eval" \
    --checkpoint-path "$checkpoint" \
    --protocol-tag "task_D_D_post_finetune_${NUM_SEQUENCES}seq_ep${EP_LEN}" \
    --source-note "Post fine-tune task_D_D D-environment closed-loop gate; not full official task_ABC_D 1000-sequence benchmark" \
    --output-dir "$out_dir" \
    --output-json "$out_dir/${label}_${NUM_SEQUENCES}seq_ep${EP_LEN}.json" \
    || eval_status=$?
  cleanup_server
  trap - RETURN
  return "$eval_status"
}

run_track2_closed_loop() {
  echo "[Track2] Running corrected medium-scale closed-loop evaluation for four policies"
  mkdir -p "$TRACK2_ROOT"
  run_track2_closed_loop_one \
    "B_only_visual_baseline" \
    "act_inference_server.py" \
    "$B_VISUAL" \
    "$B_VISUAL/checkpoints/best_val_action_l1.pt" \
    8861
  run_track2_closed_loop_one \
    "ABC_mixed_visual_baseline" \
    "act_inference_server.py" \
    "$ABC_VISUAL" \
    "$ABC_VISUAL/checkpoints/best_val_action_l1.pt" \
    8862
  run_track2_closed_loop_one \
    "B_only_task_id_finetune" \
    "task_conditioned_act_inference_server.py" \
    "$B_TASKID" \
    "$B_TASKID/checkpoints/best_val_action_l1.pt" \
    8863 \
    "log-jsonl"
  run_track2_closed_loop_one \
    "ABC_mixed_task_id_finetune" \
    "task_conditioned_act_inference_server.py" \
    "$ABC_TASKID" \
    "$ABC_TASKID/checkpoints/best_val_action_l1.pt" \
    8864 \
    "log-jsonl"
}

case "${1:-all}" in
  track1)
    run_track1_offline
    ;;
  track2)
    run_track2_closed_loop
    ;;
  all)
    run_track1_offline
    run_track2_closed_loop
    ;;
  *)
    echo "Usage: $0 [track1|track2|all]" >&2
    exit 2
    ;;
esac
