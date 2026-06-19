#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dechao/cv_final_pj2
source "$PROJECT_ROOT/scripts/setup/env_vars.sh"

SESSION_NAME=cvpj2_overnight_train
SEQUENCE_DIR=/data/dechao/cv_final_pj2/outputs/baseline/overnight_sequence_20260531
mkdir -p "$SEQUENCE_DIR"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME"
    exit 2
  fi
  tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && '$PROJECT_ROOT/scripts/train/run_overnight_sequence.sh' 2>&1 | tee '$SEQUENCE_DIR/sequence_stdout.log'"
  echo "Started tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Log: $SEQUENCE_DIR/sequence_stdout.log"
else
  nohup "$PROJECT_ROOT/scripts/train/run_overnight_sequence.sh" > "$SEQUENCE_DIR/sequence_stdout.log" 2>&1 &
  echo "$!" > "$SEQUENCE_DIR/nohup.pid"
  echo "tmux not found; started nohup PID $(cat "$SEQUENCE_DIR/nohup.pid")"
  echo "Log: $SEQUENCE_DIR/sequence_stdout.log"
fi
