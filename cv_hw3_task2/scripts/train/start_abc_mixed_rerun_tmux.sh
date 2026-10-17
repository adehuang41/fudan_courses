#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

SESSION_NAME=cvpj2_abc_rerun
RERUN_DIR=/data/dechao/cv_final_pj2/outputs/baseline/abc_mixed_rerun_20260601
mkdir -p "$RERUN_DIR"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    exit 2
  fi
  tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && '$PROJECT_ROOT/scripts/train/run_abc_mixed_rerun.sh' 2>&1 | tee '$RERUN_DIR/rerun_stdout.log'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $RERUN_DIR/rerun_stdout.log"
else
  nohup "$PROJECT_ROOT/scripts/train/run_abc_mixed_rerun.sh" > "$RERUN_DIR/rerun_stdout.log" 2>&1 &
  echo "$!" > "$RERUN_DIR/nohup.pid"
  echo "tmux not found; started nohup PID $(cat "$RERUN_DIR/nohup.pid")"
  echo "Log: $RERUN_DIR/rerun_stdout.log"
fi
