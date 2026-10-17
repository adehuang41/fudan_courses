# cv_hw3_task2

This repository hosts the model checkpoints used in the HW3 Task 2 CALVIN ACT cross-environment generalization report.

## Checkpoints

| file | source experiment | role |
| --- | --- | --- |
| `checkpoints/B_only_visual_ACT_best_val_action_l1.pt` | `B_only_act_baseline` | visual ACT baseline trained on environment B |
| `checkpoints/ABC_mixed_visual_ACT_best_val_action_l1.pt` | `ABC_mixed_act_baseline` | visual ACT baseline trained on environments A/B/C |
| `checkpoints/B_only_task_id_ACT_best_val_action_l1.pt` | `B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1` | task-id ACT fine-tuned from the B-only visual checkpoint |
| `checkpoints/ABC_mixed_task_id_ACT_best_val_action_l1.pt` | `ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1` | task-id ACT fine-tuned from the ABC-mixed visual checkpoint |

Each checkpoint is a PyTorch `.pt` file saved by `scripts/train/train_act_full.py`. The runtime used for evaluation was PyTorch `2.10.0+cu128` in the local LeRobot environment.

## Related Links

- GitHub code repository: https://github.com/adehuang41/cv_hw3_task2
- SwanLab curves: https://swanlab.cn/@ade/cv_hw3_task2
- Report: `reports/HW3_task2_23300660009_黄德超.pdf`
