#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-adehuang41/cv_hw3_task2}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MS_CLI="${MS_CLI:-}"

if [[ -z "$MS_CLI" ]]; then
  if [[ -x "/home/dechao/.conda/envs/cv-hw2/bin/ms" ]]; then
    MS_CLI="/home/dechao/.conda/envs/cv-hw2/bin/ms"
  else
    MS_CLI="ms"
  fi
fi

if ! command -v "$MS_CLI" >/dev/null 2>&1; then
  echo "ModelScope Hub CLI not found. Install with: pip install modelscope-hub" >&2
  exit 1
fi

"$MS_CLI" whoami >/dev/null
"$MS_CLI" create "$REPO_ID" \
  --repo-type model \
  --visibility public \
  --license apache-2.0 \
  --description "HW3 Task 2 CALVIN ACT checkpoints" \
  --exist-ok

"$MS_CLI" upload "$REPO_ID" "$PROJECT_ROOT/modelscope/README.md" "README.md" \
  --repo-type model \
  --commit-message "Add model card"
"$MS_CLI" upload "$REPO_ID" "$PROJECT_ROOT/modelscope/manifest.json" "manifest.json" \
  --repo-type model \
  --commit-message "Add checkpoint manifest"

"$MS_CLI" upload "$REPO_ID" \
  "/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline/checkpoints/best_val_action_l1.pt" \
  "checkpoints/B_only_visual_ACT_best_val_action_l1.pt" \
  --repo-type model \
  --commit-message "Upload B-only visual ACT checkpoint"

"$MS_CLI" upload "$REPO_ID" \
  "/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline/checkpoints/best_val_action_l1.pt" \
  "checkpoints/ABC_mixed_visual_ACT_best_val_action_l1.pt" \
  --repo-type model \
  --commit-message "Upload ABC-mixed visual ACT checkpoint"

"$MS_CLI" upload "$REPO_ID" \
  "/data/dechao/cv_final_pj2/outputs/task_conditioned/B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1/checkpoints/best_val_action_l1.pt" \
  "checkpoints/B_only_task_id_ACT_best_val_action_l1.pt" \
  --repo-type model \
  --commit-message "Upload B-only task-id ACT checkpoint"

"$MS_CLI" upload "$REPO_ID" \
  "/data/dechao/cv_final_pj2/outputs/task_conditioned/ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1/checkpoints/best_val_action_l1.pt" \
  "checkpoints/ABC_mixed_task_id_ACT_best_val_action_l1.pt" \
  --repo-type model \
  --commit-message "Upload ABC-mixed task-id ACT checkpoint"

echo "Uploaded ModelScope weights to https://modelscope.cn/models/${REPO_ID}"
