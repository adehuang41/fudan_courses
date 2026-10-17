#!/usr/bin/env bash
set -euo pipefail

source /home/dechao/cv_final_pj/scripts/aigc/env_step3.sh

AIGC_ENV=${AIGC_ENV:-/data/dechao/cv_final_pj/conda_envs/cv-final-aigc}
CUDA_DEVICE=${CUDA_DEVICE:-$( "$AIGC_ENV/bin/python" "$PROJECT_ROOT/scripts/aigc/select_cuda_device.py" )}
PROMPT=${AIGC_OBJECT_B_PROMPT:-"a simple yellow rubber duck bath toy, smooth rounded body, small orange beak, black dot eyes, glossy plastic"}
NEGATIVE_PROMPT=${AIGC_OBJECT_B_NEGATIVE_PROMPT-"red stripe, orange band, ring around body, scarf, oversized beak, wide mouth, two beaks, extra parts, extra limbs, feathers, realistic bird, noisy texture, dirty surface, floaters, artifacts"}
SD_MODEL=${AIGC_SD15_MODEL:-"$PROJECT_ROOT/data/interim/cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"}
SEED=${AIGC_OBJECT_B_SEED:-0}
GUIDANCE_SCALE=${AIGC_OBJECT_B_GUIDANCE_SCALE:-70}
GUIDANCE_MIN_STEP_PERCENT=${AIGC_OBJECT_B_MIN_STEP_PERCENT:-0.05}
GUIDANCE_MAX_STEP_PERCENT=${AIGC_OBJECT_B_MAX_STEP_PERCENT:-0.80}
MAX_STEPS=${AIGC_OBJECT_B_MAX_STEPS:-3000}
VAL_INTERVAL=${AIGC_OBJECT_B_VAL_INTERVAL:-1000}
CKPT_INTERVAL=${AIGC_OBJECT_B_CKPT_INTERVAL:-1000}
VIDEO_INTERVAL=${AIGC_OBJECT_B_VIDEO_INTERVAL:-1000}
VAL_VIEWS=${AIGC_OBJECT_B_N_VAL_VIEWS:-4}
ENABLE_VIDEO_MONITOR=${AIGC_OBJECT_B_ENABLE_VIDEO_MONITOR:-0}
RESUME_CKPT=${AIGC_OBJECT_B_RESUME_CKPT:-}
EXP_ROOT="$PROJECT_ROOT/outputs/aigc_assets/object_B_text_to_3d/final"
RUN_TIMESTAMP=${AIGC_RUN_TIMESTAMP:-@$(date +%Y%m%d-%H%M%S)}
TRIAL_TAG=${AIGC_TRIAL_TAG:-rubber_duck_simple_seed${SEED}_g${GUIDANCE_SCALE}_${MAX_STEPS}steps}
TRIAL_DIR="$EXP_ROOT/dreamfusion-sd/${TRIAL_TAG}${RUN_TIMESTAMP}"
TRAIN_LOG="$TRIAL_DIR/train.log"
VIDEO_MONITOR_PID=""
RESUME_ARGS=()
NEGATIVE_PROMPT_ARG="$NEGATIVE_PROMPT"
RUBIKS_WORKSPACE=${AIGC_OBJECT_B_RUBIKS_WORKSPACE:-"$PROJECT_ROOT/object_B_rubiks_cube"}
RUBIKS_WORKSPACE_LINK=${AIGC_OBJECT_B_RUBIKS_WORKSPACE_LINK:-auto}
RUBIKS_RUN_LINK_NAME=${AIGC_OBJECT_B_RUBIKS_RUN_LINK_NAME:-"${TRIAL_TAG}${RUN_TIMESTAMP}"}
RUBIKS_RUN_LINK_NAME=${RUBIKS_RUN_LINK_NAME//\//_}

if [[ -z "$NEGATIVE_PROMPT_ARG" ]]; then
  NEGATIVE_PROMPT_ARG="''"
fi

if [[ ! -d "$SD_MODEL" ]]; then
  echo "Missing local Stable Diffusion 1.5 snapshot: $SD_MODEL" >&2
  exit 1
fi

if [[ -n "$RESUME_CKPT" ]]; then
  if [[ ! -s "$RESUME_CKPT" ]]; then
    echo "Missing resume checkpoint: $RESUME_CKPT" >&2
    exit 1
  fi
  RESUME_ARGS=(resume="$RESUME_CKPT")
fi

if [[ -d "$TRIAL_DIR" ]] && [[ -n "$(find "$TRIAL_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty trial directory: $TRIAL_DIR" >&2
  echo "Set a new AIGC_TRIAL_TAG or AIGC_RUN_TIMESTAMP so previous results stay preserved." >&2
  exit 1
fi

mkdir -p "$TRIAL_DIR"

link_if_missing() {
  local target=$1
  local link=$2
  local link_dir
  local target_rel

  if [[ -e "$link" || -L "$link" ]]; then
    return
  fi

  link_dir=$(dirname "$link")
  target_rel=$(realpath --relative-to="$link_dir" "$target")
  ln -s "$target_rel" "$link"
}

should_link_rubiks_workspace() {
  local label

  case "$RUBIKS_WORKSPACE_LINK" in
    1|true|yes|on)
      return 0
      ;;
    0|false|no|off)
      return 1
      ;;
  esac

  label=$(printf '%s %s\n' "$TRIAL_TAG" "$PROMPT" | tr '[:upper:]' '[:lower:]')
  [[ "$label" == *rubik* ]]
}

create_rubiks_workspace_links() {
  local run_link_dir="$RUBIKS_WORKSPACE/runs/$RUBIKS_RUN_LINK_NAME"

  if ! should_link_rubiks_workspace; then
    return
  fi

  if [[ ! -d "$RUBIKS_WORKSPACE/runs" ]]; then
    echo "[RUN] Rubik workspace link skipped: missing $RUBIKS_WORKSPACE/runs"
    return
  fi

  if [[ -e "$run_link_dir" || -L "$run_link_dir" ]]; then
    echo "[RUN] Rubik workspace link skipped: existing $run_link_dir"
    return
  fi

  mkdir -p "$run_link_dir"
  link_if_missing "$TRIAL_DIR" "$run_link_dir/original_run"
  link_if_missing "$TRIAL_DIR/save" "$run_link_dir/save"
  link_if_missing "$TRIAL_DIR/ckpts" "$run_link_dir/ckpts"
  link_if_missing "$TRIAL_DIR/configs" "$run_link_dir/run_configs"
  link_if_missing "$TRIAL_DIR/cmd.txt" "$run_link_dir/cmd.txt"
  link_if_missing "$TRIAL_DIR/train.log" "$run_link_dir/train.log"
  link_if_missing "$TRIAL_DIR/training.done" "$run_link_dir/training.done"
  link_if_missing "$TRIAL_DIR/training.failed" "$run_link_dir/training.failed"
  link_if_missing "$TRIAL_DIR/training.stopped" "$run_link_dir/training.stopped"
  echo "[RUN] Rubik workspace link=$run_link_dir"
}

create_rubiks_workspace_links

mark_training_exit() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -f "$TRIAL_DIR/training.stopped" ]]; then
    exit "$status"
  fi
  if [[ "$status" -eq 0 ]]; then
    touch "$TRIAL_DIR/training.done"
  else
    printf '%s\n' "$status" > "$TRIAL_DIR/training.failed"
  fi
  exit "$status"
}

mark_training_interrupted() {
  local status=$1
  trap - EXIT INT TERM
  printf '%s\n' "$status" > "$TRIAL_DIR/training.stopped"
  exit "$status"
}

trap mark_training_exit EXIT
trap 'mark_training_interrupted 130' INT
trap 'mark_training_interrupted 143' TERM

if [[ "$ENABLE_VIDEO_MONITOR" == "1" ]]; then
  AIGC_OBJECT_B_TRIAL_DIR="$TRIAL_DIR" \
  AIGC_OBJECT_B_EXP_ROOT="$EXP_ROOT" \
  AIGC_TRIAL_TAG="$TRIAL_TAG" \
  AIGC_RUN_TIMESTAMP="$RUN_TIMESTAMP" \
  AIGC_OBJECT_B_PROMPT="$PROMPT" \
  AIGC_OBJECT_B_NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
  AIGC_OBJECT_B_SEED="$SEED" \
  AIGC_OBJECT_B_GUIDANCE_SCALE="$GUIDANCE_SCALE" \
  AIGC_OBJECT_B_MIN_STEP_PERCENT="$GUIDANCE_MIN_STEP_PERCENT" \
  AIGC_OBJECT_B_MAX_STEP_PERCENT="$GUIDANCE_MAX_STEP_PERCENT" \
  AIGC_OBJECT_B_MAX_STEPS="$MAX_STEPS" \
  AIGC_OBJECT_B_VIDEO_INTERVAL="$VIDEO_INTERVAL" \
  AIGC_OBJECT_B_VIDEO_CUDA_DEVICE="${AIGC_OBJECT_B_VIDEO_CUDA_DEVICE:-$CUDA_DEVICE}" \
  bash "$PROJECT_ROOT/scripts/aigc/render_object_B_checkpoint_videos.sh" \
    > "$TRIAL_DIR/checkpoint_video_monitor.stdout.log" 2>&1 &
  VIDEO_MONITOR_PID=$!
fi

cd "$PROJECT_ROOT/third_party/threestudio"
{
  echo "[RUN] Object B formal DreamFusion training"
  echo "[RUN] CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
  echo "[RUN] seed=$SEED"
  echo "[RUN] prompt=$PROMPT"
  echo "[RUN] negative_prompt=$NEGATIVE_PROMPT"
  echo "[RUN] guidance_scale=$GUIDANCE_SCALE min_step_percent=$GUIDANCE_MIN_STEP_PERCENT max_step_percent=$GUIDANCE_MAX_STEP_PERCENT"
  echo "[RUN] max_steps=$MAX_STEPS val_interval=$VAL_INTERVAL ckpt_interval=$CKPT_INTERVAL video_interval=$VIDEO_INTERVAL val_views=$VAL_VIEWS"
  echo "[RUN] resume_ckpt=${RESUME_CKPT:-none}"
  echo "[RUN] AIGC_TRAIN_LOG_INTERVAL=$AIGC_TRAIN_LOG_INTERVAL"
  echo "[RUN] video_monitor_enabled=$ENABLE_VIDEO_MONITOR video_monitor_pid=${VIDEO_MONITOR_PID:-none}"
  echo "[RUN] train_log=$TRAIN_LOG"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$AIGC_ENV/bin/python" -u launch.py \
    --config configs/dreamfusion-sd.yaml \
    --train \
    seed="$SEED" \
    tag="$TRIAL_TAG" \
    timestamp="'$RUN_TIMESTAMP'" \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.negative_prompt="$NEGATIVE_PROMPT_ARG" \
    system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.guidance_scale="$GUIDANCE_SCALE" \
    system.guidance.min_step_percent="$GUIDANCE_MIN_STEP_PERCENT" \
    system.guidance.max_step_percent="$GUIDANCE_MAX_STEP_PERCENT" \
    data.n_val_views="$VAL_VIEWS" \
    exp_root_dir="$EXP_ROOT" \
    "${RESUME_ARGS[@]}" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.val_check_interval="$VAL_INTERVAL" \
    checkpoint.every_n_train_steps="$CKPT_INTERVAL"
} 2>&1 | tee -a "$TRAIN_LOG"
