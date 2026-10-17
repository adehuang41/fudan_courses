#!/usr/bin/env bash
set -euo pipefail

source /home/dechao/cv_final_pj/scripts/aigc/env_step3.sh

AIGC_ENV=${AIGC_ENV:-/data/dechao/cv_final_pj/conda_envs/cv-final-aigc}
CUDA_DEVICE=${CUDA_DEVICE:-$( "$AIGC_ENV/bin/python" "$PROJECT_ROOT/scripts/aigc/select_cuda_device.py" )}
PROMPT="a simple dark green cylindrical container with a flat lid, smooth glossy surface"
IMAGE_PATH="$PROJECT_ROOT/data/processed/object_C_green_container/container_rgba.png"
SD_MODEL=${AIGC_SD15_MODEL:-"$PROJECT_ROOT/data/interim/cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"}
ZERO123_MODEL=${AIGC_ZERO123_MODEL:-"bennyguo/zero123-diffusers"}
EXP_ROOT="$PROJECT_ROOT/outputs/aigc_assets/object_C_image_to_3d/green_container/smoke_test/coarse"
RUN_TIMESTAMP=${AIGC_RUN_TIMESTAMP:-@$(date +%Y%m%d-%H%M%S)}
TRIAL_TAG=${AIGC_TRIAL_TAG:-$(basename "$IMAGE_PATH")-${PROMPT// /_}}
TRIAL_DIR="$EXP_ROOT/magic123-coarse-sd/${TRIAL_TAG}${RUN_TIMESTAMP}"
TRAIN_LOG="$TRIAL_DIR/train.log"

if [[ ! -d "$SD_MODEL" ]]; then
  echo "Missing local Stable Diffusion 1.5 snapshot: $SD_MODEL" >&2
  exit 1
fi

if [[ "$ZERO123_MODEL" == /* && ! -d "$ZERO123_MODEL" ]]; then
  echo "Missing local Zero123 snapshot: $ZERO123_MODEL" >&2
  exit 1
fi

if [[ -d "$TRIAL_DIR" ]] && [[ -n "$(find "$TRIAL_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty trial directory: $TRIAL_DIR" >&2
  echo "Set a new AIGC_TRIAL_TAG or AIGC_RUN_TIMESTAMP so previous results stay preserved." >&2
  exit 1
fi

mkdir -p "$TRIAL_DIR"

cd "$PROJECT_ROOT/third_party/threestudio"
{
  echo "[RUN] Object C Magic123 coarse smoke test"
  echo "[RUN] CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
  echo "[RUN] AIGC_TRAIN_LOG_INTERVAL=$AIGC_TRAIN_LOG_INTERVAL"
  echo "[RUN] train_log=$TRAIN_LOG"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$AIGC_ENV/bin/python" -u launch.py \
    --config configs/magic123-coarse-sd.yaml \
    --train \
    data.image_path="$IMAGE_PATH" \
    system.prompt_processor.prompt="$PROMPT" \
    system.prompt_processor.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance.pretrained_model_name_or_path="$SD_MODEL" \
    system.guidance_3d.pretrained_model_name_or_path="$ZERO123_MODEL" \
    exp_root_dir="$EXP_ROOT" \
    timestamp="'$RUN_TIMESTAMP'" \
    trainer.max_steps=25 \
    trainer.val_check_interval=25 \
    checkpoint.every_n_train_steps=25
} 2>&1 | tee -a "$TRAIN_LOG"
