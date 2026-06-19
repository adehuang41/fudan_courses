#!/usr/bin/env bash

export PROJECT_ROOT=/home/dechao/cv_final_pj2
export REAL_DATA_ROOT=/data/dechao/cv_final_pj2
export DATA_ROOT=$PROJECT_ROOT/data

export HF_HOME=$REAL_DATA_ROOT/hf_cache
export HF_HUB_CACHE=$REAL_DATA_ROOT/hf_cache/hub
export HF_DATASETS_CACHE=$REAL_DATA_ROOT/hf_cache/datasets
export TRANSFORMERS_CACHE=$REAL_DATA_ROOT/hf_cache/transformers
export TORCH_HOME=$REAL_DATA_ROOT/hf_cache/torch

export WANDB_DIR=$REAL_DATA_ROOT/wandb
export TMPDIR=$REAL_DATA_ROOT/tmp
export PIP_CACHE_DIR=$REAL_DATA_ROOT/tmp/pip_cache
export CONDA_PKGS_DIRS=$REAL_DATA_ROOT/tmp/conda_pkgs
