#!/usr/bin/env bash
set -euo pipefail

cd /home/dechao/cv_final_pj2

LEROBOT_PY=${LEROBOT_PY:-/data/dechao/cv_final_pj2/envs/lerobot/bin/python}
CALVIN_PY=${CALVIN_PY:-/data/dechao/cv_final_pj2/envs/calvin/bin/python}

TASK_D_D_ROOT=${TASK_D_D_ROOT:-/data/dechao/cv_final_pj2/datasets/task_D_D}
B_TASKID=${B_TASKID:-/data/dechao/cv_final_pj2/outputs/task_conditioned/B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1}
ABC_TASKID=${ABC_TASKID:-/data/dechao/cv_final_pj2/outputs/task_conditioned/ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1}

NUM_SEQUENCES=${NUM_SEQUENCES:-50}
EP_LEN=${EP_LEN:-360}
FRAME_STRIDE=${FRAME_STRIDE:-60}
N_ACTION_STEPS_LIST=${N_ACTION_STEPS_LIST:-"10 20 50 100"}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
TRACK2_ROOT=${TRACK2_ROOT:-/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_action_chunking/${RUN_TAG}}
B_GPU=${B_GPU:-3}
ABC_GPU=${ABC_GPU:-4}
B_PORT=${B_PORT:-8891}
ABC_PORT=${ABC_PORT:-8892}

for path in \
  "$TASK_D_D_ROOT" \
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
  local deadline=$((SECONDS + 180))
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

run_one() (
  set -euo pipefail
  local label="$1"
  local run_dir="$2"
  local gpu="$3"
  local port="$4"
  local n_action_steps="$5"
  local checkpoint="$run_dir/checkpoints/best_val_action_l1.pt"
  local out_dir="$TRACK2_ROOT/${label}_${NUM_SEQUENCES}seq_ep${EP_LEN}_n${n_action_steps}"
  local log_dir="$out_dir/logs"
  local output_json="$out_dir/${label}_${NUM_SEQUENCES}seq_ep${EP_LEN}_n${n_action_steps}.json"

  mkdir -p "$log_dir"
  echo "[action-chunking] start label=$label n_action_steps=$n_action_steps gpu=$gpu port=$port out=$out_dir"

  CUDA_VISIBLE_DEVICES="$gpu" "$LEROBOT_PY" scripts/eval/task_conditioned_act_inference_server.py \
    --host 127.0.0.1 \
    --port "$port" \
    --run-dir "$run_dir" \
    --checkpoint "$checkpoint" \
    --device cuda \
    --n-action-steps "$n_action_steps" \
    --log-jsonl "$log_dir/server_actions.jsonl" \
    > "$log_dir/server_stdout.log" 2>&1 &
  local server_pid=$!
  echo "$server_pid" > "$log_dir/server.pid"

  cleanup() {
    if kill -0 "$server_pid" >/dev/null 2>&1; then
      kill "$server_pid" >/dev/null 2>&1 || true
      wait "$server_pid" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT

  wait_for_server_ready_log "$log_dir/server_stdout.log" "$server_pid"

  CUDA_VISIBLE_DEVICES="$gpu" EGL_VISIBLE_DEVICES="$gpu" "$CALVIN_PY" scripts/eval/calvin_closed_loop_video_eval.py \
    --dataset-path "$TASK_D_D_ROOT" \
    --host 127.0.0.1 \
    --port "$port" \
    --num-sequences "$NUM_SEQUENCES" \
    --ep-len "$EP_LEN" \
    --frame-stride "$FRAME_STRIDE" \
    --dataset-label "task_D_D / D-environment closed-loop evaluation" \
    --policy-label "$label" \
    --checkpoint-label "best_val_action_l1_post_finetune_action_chunking_ablation" \
    --checkpoint-path "$checkpoint" \
    --policy-chunk-size 100 \
    --policy-n-action-steps "$n_action_steps" \
    --protocol-tag "task_D_D_action_chunking_${NUM_SEQUENCES}seq_ep${EP_LEN}_n${n_action_steps}" \
    --source-note "Action chunking ablation on task_D_D D-environment closed-loop gate; chunk_size=100 fixed, n_action_steps varied; not full official task_ABC_D 1000-sequence benchmark" \
    --output-dir "$out_dir" \
    --output-json "$output_json"

  echo "[action-chunking] done label=$label n_action_steps=$n_action_steps json=$output_json"
)

mkdir -p "$TRACK2_ROOT"
echo "$TRACK2_ROOT" > "$TRACK2_ROOT/RUN_ROOT.txt"
echo "[action-chunking] run_root=$TRACK2_ROOT"

for n_action_steps in $N_ACTION_STEPS_LIST; do
  run_one "B_only_task_id_finetune" "$B_TASKID" "$B_GPU" "$B_PORT" "$n_action_steps" &
  b_pid=$!
  run_one "ABC_mixed_task_id_finetune" "$ABC_TASKID" "$ABC_GPU" "$ABC_PORT" "$n_action_steps" &
  abc_pid=$!

  b_status=0
  abc_status=0
  wait "$b_pid" || b_status=$?
  wait "$abc_pid" || abc_status=$?
  if [[ "$b_status" -ne 0 || "$abc_status" -ne 0 ]]; then
    echo "[action-chunking] failed n_action_steps=$n_action_steps b_status=$b_status abc_status=$abc_status" >&2
    exit 1
  fi
done

echo "[action-chunking] all done run_root=$TRACK2_ROOT"
