#!/usr/bin/env bash
set -euo pipefail

source /home/dechao/cv_final_pj/scripts/aigc/env_step3.sh

AIGC_ENV=${AIGC_ENV:-/data/dechao/cv_final_pj/conda_envs/cv-final-aigc}
CUDA_DEVICE=${CUDA_DEVICE:-6}
TEXTURE_SIZE=${AIGC_EXPORT_TEXTURE_SIZE:-1024}
FORMAT=${AIGC_EXPORT_FORMAT:-obj-mtl}

CONFIG_PATH=""
CHECKPOINT_PATH=""
EXP_ROOT=""
TAG=""
TIMESTAMP=""
SAVE_NAME=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/aigc/export_threestudio_candidate.sh \
    --config PATH \
    --checkpoint PATH \
    --exp-root PATH \
    --tag TAG \
    --save-name NAME \
    [--timestamp @YYYYMMDD-HHMMSS]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --exp-root)
      EXP_ROOT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    --save-name)
      SAVE_NAME="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG_PATH" || -z "$CHECKPOINT_PATH" || -z "$EXP_ROOT" || -z "$TAG" || -z "$SAVE_NAME" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$TIMESTAMP" ]]; then
  TIMESTAMP=$(date -u +@%Y%m%d-%H%M%S)
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -s "$CHECKPOINT_PATH" ]]; then
  echo "Missing checkpoint: $CHECKPOINT_PATH" >&2
  exit 1
fi

NAME=$("$AIGC_ENV/bin/python" - "$CONFIG_PATH" <<'PY'
import sys
from omegaconf import OmegaConf
print(OmegaConf.load(sys.argv[1]).get("name", "default"))
PY
)

TRIAL_DIR="$EXP_ROOT/$NAME/${TAG}${TIMESTAMP}"
LOG_PATH="$TRIAL_DIR/export.log"
COMMAND_PATH="$TRIAL_DIR/export_command.txt"
STATUS_PATH="$TRIAL_DIR/export_status.txt"
SUMMARY_PATH="$TRIAL_DIR/export_summary.md"

if [[ -d "$TRIAL_DIR" ]] && [[ -n "$(find "$TRIAL_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse non-empty export directory: $TRIAL_DIR" >&2
  exit 1
fi

mkdir -p "$TRIAL_DIR"

CMD=(
  "$AIGC_ENV/bin/python" -u launch.py
  --config "$PROJECT_ROOT/$CONFIG_PATH"
  --export
  resume="$PROJECT_ROOT/$CHECKPOINT_PATH"
  exp_root_dir="$PROJECT_ROOT/$EXP_ROOT"
  tag="$TAG"
  timestamp="'$TIMESTAMP'"
  system.exporter_type=mesh-exporter
  system.exporter.fmt="$FORMAT"
  system.exporter.save_name="$SAVE_NAME"
  system.exporter.save_uv=true
  system.exporter.save_texture=true
  system.exporter.texture_size="$TEXTURE_SIZE"
)

{
  printf 'cwd: %s\n' "$PROJECT_ROOT/third_party/threestudio"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_DEVICE"
  printf 'command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
} > "$COMMAND_PATH"

{
  echo "# ThreeStudio Export Summary"
  echo
  echo "- status: running"
  echo "- config: $CONFIG_PATH"
  echo "- checkpoint: $CHECKPOINT_PATH"
  echo "- export_dir: $TRIAL_DIR"
  echo "- format: $FORMAT"
  echo "- save_name: $SAVE_NAME"
  echo "- texture_size: $TEXTURE_SIZE"
  echo "- command_log: $COMMAND_PATH"
  echo "- runtime_log: $LOG_PATH"
} > "$SUMMARY_PATH"

set +e
(
  cd "$PROJECT_ROOT/third_party/threestudio"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${CMD[@]}"
) 2>&1 | tee "$LOG_PATH"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -eq 0 ]]; then
  printf 'success\n' > "$STATUS_PATH"
  "$AIGC_ENV/bin/python" - "$SUMMARY_PATH" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
path.write_text(text.replace("- status: running", "- status: success"))
PY
else
  printf 'failed:%s\n' "$STATUS" > "$STATUS_PATH"
  "$AIGC_ENV/bin/python" - "$SUMMARY_PATH" "$STATUS" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
status = sys.argv[2]
text = path.read_text()
text = text.replace("- status: running", f"- status: failed ({status})")
path.write_text(text + "\n- failure_reason: see runtime log tail\n")
PY
fi

exit "$STATUS"
