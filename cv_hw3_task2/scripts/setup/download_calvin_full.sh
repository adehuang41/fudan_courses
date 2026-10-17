#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

REPO_ID=${CALVIN_REPO_ID:-huiwon/calvin_task_ABC_D}
REPO_TYPE=dataset
DEST="$REAL_DATA_ROOT/datasets/calvin_task_ABC_D"
LOG_DIR="$DEST/_download_logs"
ENV_PREFIX="$REAL_DATA_ROOT/envs/lerobot"
MAX_DOWNLOAD_GIB=${MAX_DOWNLOAD_GIB:-50}
MANIFEST_TIMEOUT_SECONDS=${MANIFEST_TIMEOUT_SECONDS:-180}
DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES:-3}
DOWNLOAD_MAX_WORKERS=${DOWNLOAD_MAX_WORKERS:-2}

PYTHON_BIN=${PYTHON_BIN:-$ENV_PREFIX/bin/python}
HF_BIN=${HF_BIN:-$ENV_PREFIX/bin/hf}

REQUIRED_PREFIXES=(
  "calvin_task_ABC_D_lerobot_0_4"
  "calvin_task_ABC_D_lerobot_1_4"
  "calvin_task_ABC_D_lerobot_2_4"
  "calvin_task_ABC_D_lerobot_3_4"
)

mkdir -p "$DEST" "$LOG_DIR" "$PROJECT_ROOT/results/tables" "$PROJECT_ROOT/results/figures/dataset_preview"

if [[ "$(readlink -f "$DEST")" == /home/* ]]; then
  echo "Refusing to download CALVIN under /home: $DEST" >&2
  exit 10
fi

if [[ "$(readlink -f "$DEST")" == /data/huggingface* ]]; then
  echo "Refusing to download CALVIN under /data/huggingface: $DEST" >&2
  exit 11
fi

for cache_var in HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TORCH_HOME; do
  value="${!cache_var:-}"
  if [[ "$value" == /home/* || "$value" == /data/huggingface* ]]; then
    echo "Unsafe cache path: $cache_var=$value" >&2
    exit 12
  fi
done

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found at $PYTHON_BIN. Run scripts/setup/create_lerobot_env.sh first." >&2
  exit 13
fi

if [ ! -x "$HF_BIN" ]; then
  echo "hf CLI not found at $HF_BIN. Run scripts/setup/create_lerobot_env.sh first." >&2
  exit 14
fi

manifest_args=()
for prefix in "${REQUIRED_PREFIXES[@]}"; do
  manifest_args+=(--required-prefix "$prefix")
done

start_time=$(date -Is)

manifest_ok=0
if [ "${REUSE_VALIDATED_MANIFEST:-0}" = "1" ] && [ -s "$LOG_DIR/download_manifest.csv" ] && [ -s "$LOG_DIR/manifest_summary.txt" ]; then
  manifest_ok=1
  {
    echo
    echo "Date: $(date -Is)"
    echo "Stage: Part 2"
    echo "Experiment name: calvin_full_repo_manifest_reuse"
    echo "Command: REUSE_VALIDATED_MANIFEST=1 scripts/setup/download_calvin_full.sh"
    echo "Main results: reused existing validated manifest because mirror API is rate limited"
    echo "Artifacts: $LOG_DIR/download_manifest.csv; $LOG_DIR/manifest_summary.txt"
    echo "Exceptions: this skips repeated dry-run/manifest validation only after a prior successful validation"
  } >> "$PROJECT_ROOT/notes/experiment_log.md"
else
  "$HF_BIN" download --help > "$LOG_DIR/hf_download_help.txt" 2>&1 || true
  if grep -q -- "--dry-run" "$LOG_DIR/hf_download_help.txt"; then
    set +e
    timeout "$MANIFEST_TIMEOUT_SECONDS" "$HF_BIN" download "$REPO_ID" \
      --repo-type "$REPO_TYPE" \
      --local-dir "$DEST" \
      --dry-run \
      --json > "$LOG_DIR/hf_dry_run.json" 2> "$LOG_DIR/hf_dry_run.stderr"
    dry_run_status=$?
    set -e
    if [ "$dry_run_status" -ne 0 ]; then
      echo "hf dry-run failed with exit code $dry_run_status; falling back to Python manifest validation." >> "$LOG_DIR/hf_dry_run.stderr"
    fi
  else
    {
      echo "hf download --dry-run is not available in this huggingface_hub version."
      echo "Manifest validation was completed with huggingface_hub Python API instead."
    } > "$LOG_DIR/hf_dry_run.json"
  fi

  for attempt in 1 2 3; do
    if timeout "$MANIFEST_TIMEOUT_SECONDS" "$PYTHON_BIN" "$PROJECT_ROOT/scripts/setup/validate_hf_manifest.py" \
      --repo-id "$REPO_ID" \
      --repo-type "$REPO_TYPE" \
      --output "$LOG_DIR/download_manifest.csv" \
      --summary "$LOG_DIR/manifest_summary.txt" \
      --max-gib "$MAX_DOWNLOAD_GIB" \
      "${manifest_args[@]}" > "$LOG_DIR/manifest_attempt_${attempt}.stdout" 2> "$LOG_DIR/manifest_attempt_${attempt}.stderr"; then
      manifest_ok=1
      break
    fi
    sleep $((attempt * 5))
  done
fi

if [ "$manifest_ok" -ne 1 ]; then
  {
    echo
    echo "Date: $(date -Is)"
    echo "Stage: Part 2"
    echo "Experiment name: calvin_full_repo_manifest_validation"
    echo "Command: scripts/setup/download_calvin_full.sh"
    echo "Data split: full Hugging Face repo $REPO_ID"
    echo "Main results: manifest validation failed; download was not started"
    echo "Artifacts: $LOG_DIR"
    echo "Exceptions: see manifest_attempt_*.stderr and hf_dry_run.stderr"
    echo "Next step: retry after network stabilizes or inspect Hugging Face endpoint/proxy"
  } >> "$PROJECT_ROOT/notes/experiment_log.md"
  echo "Manifest validation failed after 3 attempts. Download was not started." >&2
  exit 15
fi

if [ "${FORCE_DOWNLOAD:-0}" != "1" ] && [ -f "$LOG_DIR/download_complete.flag" ]; then
  echo "Download complete flag already exists. Skipping hf download."
else
  status=1
  : > "$LOG_DIR/download_stdout.log"
  : > "$LOG_DIR/download_stderr.log"
  for attempt in $(seq 1 "$DOWNLOAD_RETRIES"); do
    {
      echo
      echo "download_attempt=$attempt"
      echo "started_at=$(date -Is)"
    } >> "$LOG_DIR/download_stdout.log"

    set +e
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/setup/download_hf_manifest_files.py" \
      --repo-id "$REPO_ID" \
      --repo-type "$REPO_TYPE" \
      --manifest "$LOG_DIR/download_manifest.csv" \
      --local-dir "$DEST" \
      --log-dir "$LOG_DIR" \
      --retries "$DOWNLOAD_RETRIES" \
      --workers "$DOWNLOAD_MAX_WORKERS" >> "$LOG_DIR/download_stdout.log" 2>> "$LOG_DIR/download_stderr.log"
    status=$?
    set -e

    {
      echo "finished_at=$(date -Is)"
      echo "exit_code=$status"
    } >> "$LOG_DIR/download_stdout.log"

    if [ "$status" -eq 0 ]; then
      break
    fi
    sleep $((attempt * 10))
  done

  if [ "$status" -ne 0 ]; then
    {
      echo
      echo "Date: $(date -Is)"
      echo "Stage: Part 2"
      echo "Experiment name: calvin_full_repo_download"
      echo "Command: scripts/setup/download_calvin_full.sh"
      echo "Main results: download failed with exit code $status"
      echo "Artifacts: $LOG_DIR"
      echo "Exceptions: see download_stderr.log"
      echo "Next step: inspect Hugging Face/network/path errors before retrying"
    } >> "$PROJECT_ROOT/notes/experiment_log.md"
    exit "$status"
  fi
  date -Is > "$LOG_DIR/download_complete.flag"
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/data/check_calvin_dataset.py" \
  --dataset-root "$DEST" \
  --report-json "$LOG_DIR/sanity_report.json" \
  --summary-csv "$PROJECT_ROOT/results/tables/dataset_sanity_summary.csv" \
  --preview-dir "$PROJECT_ROOT/results/figures/dataset_preview"

{
  echo
  echo "Date: $(date -Is)"
  echo "Stage: Part 2"
  echo "Experiment name: calvin_full_repo_download"
  echo "Config files: scripts/setup/env_vars.sh, configs/dataset/calvin_abc_d.yaml"
  echo "Command: scripts/setup/download_calvin_full.sh"
  echo "Data split: full Hugging Face repo $REPO_ID after manifest validation"
  echo "Main results: manifest validated, repo downloaded, sanity check completed"
  echo "Artifacts: $LOG_DIR; results/tables/dataset_sanity_summary.csv; results/figures/dataset_preview/"
  echo "Exceptions: none recorded by script"
  echo "Next step: review sanity report before ACT debug training"
} >> "$PROJECT_ROOT/notes/experiment_log.md"

echo "Started: $start_time"
echo "Finished: $(date -Is)"
echo "CALVIN dataset ready at $DEST"
