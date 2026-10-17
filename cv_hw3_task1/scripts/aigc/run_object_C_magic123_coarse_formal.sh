#!/usr/bin/env bash
set -euo pipefail

source /home/dechao/cv_final_pj/scripts/aigc/env_step3.sh

AIGC_ENV=${AIGC_ENV:-/data/dechao/cv_final_pj/conda_envs/cv-final-aigc}
CUDA_DEVICE=${CUDA_DEVICE:-$( "$AIGC_ENV/bin/python" "$PROJECT_ROOT/scripts/aigc/select_cuda_device.py" )}
PROMPT=${AIGC_OBJECT_C_PROMPT:-"a simple dark green cylindrical container with a flat lid, smooth glossy surface"}
NEGATIVE_PROMPT=${AIGC_OBJECT_C_NEGATIVE_PROMPT:-""}
IMAGE_PATH=${AIGC_OBJECT_C_IMAGE_PATH:-"$PROJECT_ROOT/data/processed/object_C_green_container/container_rgba.png"}
SD_MODEL=${AIGC_SD15_MODEL:-"$PROJECT_ROOT/data/interim/cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"}
ZERO123_MODEL=${AIGC_ZERO123_MODEL:-"bennyguo/zero123-diffusers"}
SEED=${AIGC_OBJECT_C_SEED:-0}
SD_GUIDANCE_SCALE=${AIGC_OBJECT_C_SD_GUIDANCE_SCALE:-100}
ZERO123_GUIDANCE_SCALE=${AIGC_OBJECT_C_ZERO123_GUIDANCE_SCALE:-5.0}
LAMBDA_SPARSITY=${AIGC_OBJECT_C_LAMBDA_SPARSITY:-0.0}
LAMBDA_OPAQUE=${AIGC_OBJECT_C_LAMBDA_OPAQUE:-0.0}
LAMBDA_NORMAL_SMOOTHNESS_2D=${AIGC_OBJECT_C_LAMBDA_NORMAL_SMOOTHNESS_2D:-1000.0}
LAMBDA_Z_VARIANCE=${AIGC_OBJECT_C_LAMBDA_Z_VARIANCE:-}
DEFAULT_ELEVATION_DEG=${AIGC_OBJECT_C_DEFAULT_ELEVATION_DEG:-0.0}
DEFAULT_AZIMUTH_DEG=${AIGC_OBJECT_C_DEFAULT_AZIMUTH_DEG:-0.0}
DEFAULT_FOVY_DEG=${AIGC_OBJECT_C_DEFAULT_FOVY_DEG:-40.0}
MAX_STEPS=${AIGC_OBJECT_C_COARSE_MAX_STEPS:-10000}
VAL_INTERVAL=${AIGC_OBJECT_C_COARSE_VAL_INTERVAL:-1000}
CKPT_INTERVAL=${AIGC_OBJECT_C_COARSE_CKPT_INTERVAL:-1000}
VIDEO_INTERVAL=${AIGC_OBJECT_C_COARSE_VIDEO_INTERVAL:-1000}
VAL_VIEWS=${AIGC_OBJECT_C_N_VAL_VIEWS:-4}
RENDER_AFTER_TRAIN=${AIGC_OBJECT_C_COARSE_RENDER_AFTER_TRAIN:-1}
RESUME_CKPT=${AIGC_OBJECT_C_RESUME_CKPT:-}
EXP_ROOT=${AIGC_OBJECT_C_EXP_ROOT:-"$PROJECT_ROOT/outputs/aigc_assets/object_C_image_to_3d/green_container/final/coarse"}
RUN_TIMESTAMP=${AIGC_RUN_TIMESTAMP:-@$(date +%Y%m%d-%H%M%S)}
TRIAL_TAG=${AIGC_TRIAL_TAG:-green_container_coarse_${MAX_STEPS}steps}
TRIAL_DIR="$EXP_ROOT/magic123-coarse-sd/${TRIAL_TAG}${RUN_TIMESTAMP}"
TRAIN_LOG="$TRIAL_DIR/train.log"
RESUME_ARGS=()
LOSS_EXTRA_ARGS=()

if [[ ! -d "$SD_MODEL" ]]; then
  echo "Missing local Stable Diffusion 1.5 snapshot: $SD_MODEL" >&2
  exit 1
fi

if [[ "$ZERO123_MODEL" == /* && ! -d "$ZERO123_MODEL" ]]; then
  echo "Missing local Zero123 snapshot: $ZERO123_MODEL" >&2
  exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Missing Object C RGBA image: $IMAGE_PATH" >&2
  exit 1
fi

if [[ -n "$RESUME_CKPT" ]]; then
  if [[ ! -s "$RESUME_CKPT" ]]; then
    echo "Missing resume checkpoint: $RESUME_CKPT" >&2
    exit 1
  fi
  RESUME_ARGS=(resume="$RESUME_CKPT")
fi

if [[ -n "$LAMBDA_Z_VARIANCE" ]]; then
  LOSS_EXTRA_ARGS=(system.loss.lambda_z_variance="$LAMBDA_Z_VARIANCE")
fi

if [[ -d "$TRIAL_DIR" ]] && [[ -n "$(find "$TRIAL_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty trial directory: $TRIAL_DIR" >&2
  echo "Set a new AIGC_TRIAL_TAG or AIGC_RUN_TIMESTAMP so previous results stay preserved." >&2
  exit 1
fi

mkdir -p "$TRIAL_DIR"

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

cd "$PROJECT_ROOT/third_party/threestudio"
{
  echo "[RUN] Object C formal Magic123 coarse training"
  echo "[RUN] CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
  echo "[RUN] seed=$SEED"
  echo "[RUN] prompt=$PROMPT"
  echo "[RUN] negative_prompt=$NEGATIVE_PROMPT"
  echo "[RUN] image_path=$IMAGE_PATH"
  echo "[RUN] default_elevation_deg=$DEFAULT_ELEVATION_DEG default_azimuth_deg=$DEFAULT_AZIMUTH_DEG default_fovy_deg=$DEFAULT_FOVY_DEG"
  echo "[RUN] sd_guidance_scale=$SD_GUIDANCE_SCALE zero123_guidance_scale=$ZERO123_GUIDANCE_SCALE"
  echo "[RUN] lambda_sparsity=$LAMBDA_SPARSITY lambda_opaque=$LAMBDA_OPAQUE lambda_normal_smoothness_2d=$LAMBDA_NORMAL_SMOOTHNESS_2D lambda_z_variance=${LAMBDA_Z_VARIANCE:-unset}"
  echo "[RUN] max_steps=$MAX_STEPS val_interval=$VAL_INTERVAL ckpt_interval=$CKPT_INTERVAL video_interval=$VIDEO_INTERVAL val_views=$VAL_VIEWS"
  echo "[RUN] resume_ckpt=${RESUME_CKPT:-none}"
  echo "[RUN] AIGC_TRAIN_LOG_INTERVAL=$AIGC_TRAIN_LOG_INTERVAL"
  echo "[RUN] render_after_train=$RENDER_AFTER_TRAIN"
  echo "[RUN] train_log=$TRAIN_LOG"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$AIGC_ENV/bin/python" -u launch.py \
    --config configs/magic123-coarse-sd.yaml \
    --train \
    seed="$SEED" \
    tag="$TRIAL_TAG" \
    timestamp="'$RUN_TIMESTAMP'" \
    data.image_path="$IMAGE_PATH" \
    data.default_elevation_deg="$DEFAULT_ELEVATION_DEG" \
    data.default_azimuth_deg="$DEFAULT_AZIMUTH_DEG" \
    data.default_fovy_deg="$DEFAULT_FOVY_DEG" \
    data.random_camera.eval_elevation_deg="$DEFAULT_ELEVATION_DEG" \
    data.random_camera.eval_fovy_deg="$DEFAULT_FOVY_DEG" \
    data.random_camera.fovy_range="[$DEFAULT_FOVY_DEG,$DEFAULT_FOVY_DEG]" \
    data.random_camera.n_val_views="$VAL_VIEWS" \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.negative_prompt="$NEGATIVE_PROMPT" \
    system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.guidance_scale="$SD_GUIDANCE_SCALE" \
    system.guidance_3d.pretrained_model_name_or_path="$ZERO123_MODEL" \
    system.guidance_3d.guidance_scale="$ZERO123_GUIDANCE_SCALE" \
    system.loss.lambda_sparsity="$LAMBDA_SPARSITY" \
    system.loss.lambda_opaque="$LAMBDA_OPAQUE" \
    system.loss.lambda_normal_smoothness_2d="$LAMBDA_NORMAL_SMOOTHNESS_2D" \
    "${LOSS_EXTRA_ARGS[@]}" \
    exp_root_dir="$EXP_ROOT" \
    "${RESUME_ARGS[@]}" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.val_check_interval="$VAL_INTERVAL" \
    checkpoint.every_n_train_steps="$CKPT_INTERVAL"
} 2>&1 | tee -a "$TRAIN_LOG"

if [[ "$RENDER_AFTER_TRAIN" == "1" ]]; then
  AIGC_OBJECT_C_TRIAL_DIR="$TRIAL_DIR" \
  AIGC_OBJECT_C_EXP_ROOT="$EXP_ROOT" \
  AIGC_TRIAL_TAG="$TRIAL_TAG" \
  AIGC_RUN_TIMESTAMP="$RUN_TIMESTAMP" \
  AIGC_OBJECT_C_PROMPT="$PROMPT" \
  AIGC_OBJECT_C_NEGATIVE_PROMPT="$NEGATIVE_PROMPT" \
  AIGC_OBJECT_C_IMAGE_PATH="$IMAGE_PATH" \
  AIGC_OBJECT_C_COARSE_MAX_STEPS="$MAX_STEPS" \
  AIGC_OBJECT_C_COARSE_VIDEO_INTERVAL="$VIDEO_INTERVAL" \
  AIGC_OBJECT_C_COARSE_RENDER_FINAL=1 \
  AIGC_OBJECT_C_VIDEO_CUDA_DEVICE="$CUDA_DEVICE" \
  bash "$PROJECT_ROOT/scripts/aigc/render_object_C_checkpoint_videos.sh" \
    > "$TRIAL_DIR/checkpoint_video_render.stdout.log" 2>&1
fi
