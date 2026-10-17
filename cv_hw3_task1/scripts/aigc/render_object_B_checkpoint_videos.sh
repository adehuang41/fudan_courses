#!/usr/bin/env bash
set -euo pipefail

source /home/dechao/cv_final_pj/scripts/aigc/env_step3.sh

AIGC_ENV=${AIGC_ENV:-/data/dechao/cv_final_pj/conda_envs/cv-final-aigc}
TRIAL_DIR=${AIGC_OBJECT_B_TRIAL_DIR:?AIGC_OBJECT_B_TRIAL_DIR is required}
EXP_ROOT=${AIGC_OBJECT_B_EXP_ROOT:-"$PROJECT_ROOT/outputs/aigc_assets/object_B_text_to_3d/final"}
TRIAL_TAG=${AIGC_TRIAL_TAG:?AIGC_TRIAL_TAG is required}
RUN_TIMESTAMP=${AIGC_RUN_TIMESTAMP:?AIGC_RUN_TIMESTAMP is required}
PROMPT=${AIGC_OBJECT_B_PROMPT:-"a simple yellow rubber duck bath toy, smooth rounded body, small orange beak, black dot eyes, glossy plastic"}
NEGATIVE_PROMPT=${AIGC_OBJECT_B_NEGATIVE_PROMPT:-"red stripe, orange band, ring around body, scarf, oversized beak, wide mouth, two beaks, extra parts, extra limbs, feathers, realistic bird, noisy texture, dirty surface, floaters, artifacts"}
SD_MODEL=${AIGC_SD15_MODEL:-"$PROJECT_ROOT/data/interim/cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"}
SEED=${AIGC_OBJECT_B_SEED:-0}
GUIDANCE_SCALE=${AIGC_OBJECT_B_GUIDANCE_SCALE:-70}
GUIDANCE_MIN_STEP_PERCENT=${AIGC_OBJECT_B_MIN_STEP_PERCENT:-0.05}
GUIDANCE_MAX_STEP_PERCENT=${AIGC_OBJECT_B_MAX_STEP_PERCENT:-0.80}
MAX_STEPS=${AIGC_OBJECT_B_MAX_STEPS:-3000}
VIDEO_INTERVAL=${AIGC_OBJECT_B_VIDEO_INTERVAL:-1000}
INCLUDE_FINAL=${AIGC_OBJECT_B_RENDER_FINAL:-0}
POLL_SECONDS=${AIGC_OBJECT_B_VIDEO_POLL_SECONDS:-20}
CUDA_DEVICE=${AIGC_OBJECT_B_VIDEO_CUDA_DEVICE:-${CUDA_DEVICE:-$( "$AIGC_ENV/bin/python" "$PROJECT_ROOT/scripts/aigc/select_cuda_device.py" )}}
MONITOR_LOG="$TRIAL_DIR/checkpoint_video_monitor.log"

if [[ ! -d "$SD_MODEL" ]]; then
  echo "Missing local Stable Diffusion 1.5 snapshot: $SD_MODEL" >&2
  exit 1
fi

mkdir -p "$TRIAL_DIR"

log() {
  printf '[VIDEO_MONITOR] %s\n' "$*" | tee -a "$MONITOR_LOG"
}

wait_for_stable_file() {
  local path=$1
  local size_before=0
  local size_after=0

  if [[ ! -s "$path" ]]; then
    return 1
  fi
  size_before=$(stat -c '%s' "$path")
  sleep 3
  if [[ ! -s "$path" ]]; then
    return 1
  fi
  size_after=$(stat -c '%s' "$path")
  [[ "$size_before" -eq "$size_after" ]]
}

render_checkpoint_video() {
  local step=$1
  local ckpt=$2

  cd "$PROJECT_ROOT/third_party/threestudio"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$AIGC_ENV/bin/python" -u launch.py \
    --config configs/dreamfusion-sd.yaml \
    --test \
    seed="$SEED" \
    tag="$TRIAL_TAG" \
    timestamp="'$RUN_TIMESTAMP'" \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.negative_prompt="$NEGATIVE_PROMPT" \
    system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.guidance_scale="$GUIDANCE_SCALE" \
    system.guidance.min_step_percent="$GUIDANCE_MIN_STEP_PERCENT" \
    system.guidance.max_step_percent="$GUIDANCE_MAX_STEP_PERCENT" \
    exp_root_dir="$EXP_ROOT" \
    resume="$ckpt" \
    trainer.max_steps="$MAX_STEPS"

  log "rendered step=$step video=$TRIAL_DIR/save/it${step}-test.mp4"
}

log "started trial_dir=$TRIAL_DIR cuda=$CUDA_DEVICE interval=$VIDEO_INTERVAL max_steps=$MAX_STEPS include_final=$INCLUDE_FINAL seed=$SEED guidance_scale=$GUIDANCE_SCALE min_step_percent=$GUIDANCE_MIN_STEP_PERCENT max_step_percent=$GUIDANCE_MAX_STEP_PERCENT"

step=$VIDEO_INTERVAL
while [[ "$step" -lt "$MAX_STEPS" || ( "$INCLUDE_FINAL" == "1" && "$step" -le "$MAX_STEPS" ) ]]; do
  ckpt="$TRIAL_DIR/ckpts/epoch=0-step=${step}.ckpt"
  mp4="$TRIAL_DIR/save/it${step}-test.mp4"

  if [[ -s "$mp4" ]]; then
    log "skip existing step=$step video=$mp4"
    step=$((step + VIDEO_INTERVAL))
    continue
  fi

  if wait_for_stable_file "$ckpt"; then
    log "found checkpoint step=$step ckpt=$ckpt"
    if render_checkpoint_video "$step" "$ckpt"; then
      step=$((step + VIDEO_INTERVAL))
    else
      log "render failed step=$step; retrying after ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
    fi
    continue
  fi

  if [[ -f "$TRIAL_DIR/training.done" || -f "$TRIAL_DIR/training.failed" || -f "$TRIAL_DIR/training.stopped" ]]; then
    log "training stopped before checkpoint step=$step; exiting"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done

if [[ "$INCLUDE_FINAL" == "1" ]]; then
  log "finished; rendered or verified checkpoint videos through final step"
else
  log "finished; final step video is produced by the main training script"
fi
