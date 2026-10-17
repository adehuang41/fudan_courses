#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

ENV_PREFIX="$REAL_DATA_ROOT/envs/lerobot"

mkdir -p "$REAL_DATA_ROOT/envs" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found. Load conda before running this script." >&2
  exit 1
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  conda create -p "$ENV_PREFIX" python=3.12 -y
fi

conda run -p "$ENV_PREFIX" python -m pip install --upgrade pip
conda run -p "$ENV_PREFIX" python -m pip install -r "$PROJECT_ROOT/requirements.txt"

conda run -p "$ENV_PREFIX" python - <<'PY'
import importlib.util
import sys

required = ["lerobot", "huggingface_hub", "datasets", "pyarrow", "cv2"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing packages after install: {missing}")
print("LeRobot environment is ready:", sys.executable)
PY
