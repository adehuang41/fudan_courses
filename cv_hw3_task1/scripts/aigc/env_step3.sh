#!/usr/bin/env bash

export PROJECT_ROOT=/home/dechao/cv_final_pj
export AIGC_ENV=${AIGC_ENV:-/data/dechao/cv_final_pj/conda_envs/cv-final-aigc}

export HF_HOME=$PROJECT_ROOT/data/interim/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$PROJECT_ROOT/data/interim/cache/torch
export PIP_CACHE_DIR=$PROJECT_ROOT/data/interim/cache/pip
export XDG_CACHE_HOME=$PROJECT_ROOT/data/interim/cache/xdg
export CONDA_PKGS_DIRS=$PROJECT_ROOT/data/interim/cache/conda_pkgs
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/data/interim/cache/torch_extensions
export U2NET_HOME=$PROJECT_ROOT/data/interim/cache/rembg
export REMBG_HOME=$PROJECT_ROOT/data/interim/cache/rembg
export SAM_HOME=$PROJECT_ROOT/data/interim/cache/segment_anything
export AIGC_SD15_MODEL=${AIGC_SD15_MODEL:-$HF_HUB_CACHE/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14}
export AIGC_ZERO123_MODEL=${AIGC_ZERO123_MODEL:-$HF_HUB_CACHE/models--bennyguo--zero123-diffusers/snapshots/b5289c24d8549e3a4737d0c34ab1347e5f074fbe}
export AIGC_TRAIN_LOG_INTERVAL=${AIGC_TRAIN_LOG_INTERVAL:-50}
export THREESTUDIO_TRAIN_LOG_INTERVAL=${THREESTUDIO_TRAIN_LOG_INTERVAL:-$AIGC_TRAIN_LOG_INTERVAL}
export PYTHONUNBUFFERED=1

export CUDA_HOME=${CUDA_HOME:-/home/dechao/.conda/envs/cv-final-2dgs}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib/stubs:$CUDA_HOME/targets/x86_64-linux/lib/stubs:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}
export C_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:${C_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}
export CC=$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++
export MAX_JOBS=${MAX_JOBS:-4}
export GIT_HTTP_VERSION=HTTP/1.1
export GIT_TERMINAL_PROMPT=0

mkdir -p \
  "$HF_HOME" \
  "$HF_HUB_CACHE" \
  "$TORCH_HOME" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$CONDA_PKGS_DIRS" \
  "$TORCH_EXTENSIONS_DIR" \
  "$U2NET_HOME" \
  "$REMBG_HOME" \
  "$SAM_HOME" \
  "$PROJECT_ROOT/data/raw/pretrained_models" \
  "$PROJECT_ROOT/data/raw/object_C_green_container" \
  "$PROJECT_ROOT/data/interim/object_C_green_container" \
  "$PROJECT_ROOT/data/processed/object_C_green_container" \
  "$PROJECT_ROOT/outputs/aigc_assets/object_B_text_to_3d" \
  "$PROJECT_ROOT/outputs/aigc_assets/object_C_image_to_3d/green_container"
