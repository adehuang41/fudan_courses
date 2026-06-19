# ACT 策略跨环境泛化挑战

本仓库用于计算机视觉期末项目 Task 2：基于 LeRobot 的 ACT 策略跨环境泛化挑战。

主线：在 CALVIN 源环境训练模仿学习策略，在 held-out D 环境做 zero-shot 跨环境评估。我们同时报告离线动作误差 Action L1 和额外闭环 Success Rate。

## 作业交付链接

- GitHub repository: https://github.com/adehuang41/cv_hw3_task2
- ModelScope weights: https://modelscope.cn/models/adehuang41/cv_hw3_task2
- SwanLab training curves: https://swanlab.cn/@ade/cv_hw3_task2
- Final report: `reports/HW3_task2_23300660009_黄德超.pdf`

## 当前状态

已完成：

- A/B/C 环境映射与 episode-level split 审计。
- B-only 与 ABC-mixed ACT visual baseline 训练。
- Task-id conditioned ACT 从 visual checkpoint fine-tune 到 95k，并按 best validation Action L1 选 checkpoint。
- D 环境 offline open-loop Action L1。
- `task_D_D / D-environment` 中等规模闭环 rollout Success Rate，50 sequences，`ep_len=360`。
- Action chunking ablation：固定 `chunk_size=100`，改变 `n_action_steps={10,20,50,100}`。

边界：

- 当前闭环结果不是 full official `task_ABC_D` ABC-to-D 1000-sequence benchmark。
- 当前没有真实物理机器人 real-world evaluation。
- 课程若接受 “Success Rate 或 Action Error”，Action L1 已覆盖基本指标；闭环 Success Rate 与 action chunking ablation 是额外工作。

## 核心结果

### D Offline Action L1

协议：offline open-loop `predict_action_chunk`，D observations/actions 使用各自训练 split stats 归一化。

| experiment | normalized Action L1 | raw Action L1 |
| --- | ---: | ---: |
| B_only_act_baseline | 0.6497833514 | 0.2096832675 |
| ABC_mixed_act_baseline | 0.5657362543 | 0.1681750212 |
| B_only_task_id_finetune | 0.6052228280 | 0.1887679086 |
| ABC_mixed_task_id_finetune | 0.5015655940 | 0.1445857702 |

结论：ABC-mixed + task-id fine-tune 在 D 动作误差最低。

### D Closed-Loop Success Rate

协议：`task_D_D / D-environment closed-loop evaluation`，50 sequences，`ep_len=360`。

| policy | avg successful subtasks | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B_only_visual_baseline | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| ABC_mixed_visual_baseline | 0.02 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| B_only_task_id_finetune | 0.28 | 0.20 | 0.06 | 0.02 | 0.00 | 0.00 |
| ABC_mixed_task_id_finetune | 0.68 | 0.42 | 0.18 | 0.08 | 0.00 | 0.00 |

结论：goal/task conditioning 对闭环控制影响远大于对离线 L1 的影响；ABC-mixed task-id 最强。

### Action Chunking Ablation

固定 `chunk_size=100`，只改变每次预测后实际执行步数 `n_action_steps`。更小的 `n_action_steps` 表示更频繁重规划。

| policy | n_action_steps | avg successful subtasks | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B_only_task_id_finetune | 10 | 0.28 | 0.26 | 0.02 | 0.00 | 0.00 | 0.00 |
| B_only_task_id_finetune | 20 | 0.24 | 0.20 | 0.04 | 0.00 | 0.00 | 0.00 |
| B_only_task_id_finetune | 50 | 0.20 | 0.18 | 0.02 | 0.00 | 0.00 | 0.00 |
| B_only_task_id_finetune | 100 | 0.28 | 0.20 | 0.06 | 0.02 | 0.00 | 0.00 |
| ABC_mixed_task_id_finetune | 10 | 0.98 | 0.50 | 0.30 | 0.10 | 0.04 | 0.04 |
| ABC_mixed_task_id_finetune | 20 | 1.36 | 0.64 | 0.38 | 0.18 | 0.08 | 0.08 |
| ABC_mixed_task_id_finetune | 50 | 0.88 | 0.56 | 0.24 | 0.06 | 0.02 | 0.00 |
| ABC_mixed_task_id_finetune | 100 | 0.68 | 0.42 | 0.18 | 0.08 | 0.00 | 0.00 |

结论：ABC-mixed task-id 对 replanning frequency 敏感，`n_action_steps=20` 最优；相对默认 `n_action_steps=100`，avg subtasks 从 0.68 提升到 1.36。B-only 受益有限，说明源环境多样性与 task conditioning 是闭环泛化瓶颈。

## 结果文件

- Track1 Action L1：`results/tables/step4/post_finetune_best_95k_d_evaluation_summary.csv`
- Track2 closed-loop JSON：`/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_closed_loop/`
- Action chunking summary：`results/tables/step4/action_chunking_ablation_50seq_ep360_summary.csv`
- Action chunking per-task：`results/tables/step4/action_chunking_ablation_50seq_ep360_per_task_success.csv`
- Action chunking figures：`results/figures/step4/action_chunking_ablation_50seq_ep360_avg_successful_subtasks.svg`, `results/figures/step4/action_chunking_ablation_50seq_ep360_chain_success_rate.svg`
- Rollout videos：`/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_action_chunking/formal_50seq_ep360_20260613T2050Z/`

## 复现命令

环境变量：

```bash
source scripts/setup/env_vars.sh
```

训练入口：

```bash
# visual ACT baselines: B-only then ABC-mixed
bash scripts/train/start_overnight_training_tmux.sh

# task-id fine-tune from visual checkpoints, first 50k extension
bash scripts/train/launch_task_id_finetune_pair.sh

# later accepted extensions are preserved as explicit resume launchers
bash scripts/train/launch_task_id_finetune_to75k_pair_pf2.sh
bash scripts/train/launch_task_id_finetune_from75k_to85k_lr1e-5_pair_pf2.sh
bash scripts/train/launch_task_id_finetune_from85k_to95k_lr1e-5_pair_pf2.sh
```

D offline Action L1：

```bash
CUDA_VISIBLE_DEVICES=3 /data/dechao/cv_final_pj2/envs/lerobot/bin/python scripts/eval/evaluate_d_offline_action_l1.py \
  --dataset-root /data/dechao/cv_final_pj2/datasets/calvin_D_D_lerobot \
  --include-task-id-finetunes \
  --batch-size 32 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --table-cache-size 32 \
  --device cuda \
  --output-prefix post_finetune_best_95k
```

D closed-loop Success Rate：

```bash
NUM_SEQUENCES=50 EP_LEN=360 bash scripts/eval/launch_post_finetune_track1_track2_eval.sh track2
```

Action chunking ablation：

```bash
NUM_SEQUENCES=50 EP_LEN=360 N_ACTION_STEPS_LIST="10 20 50 100" \
  RUN_TAG=formal_50seq_ep360_20260613T2050Z \
  B_GPU=3 ABC_GPU=4 \
  bash scripts/eval/launch_action_chunking_ablation.sh
```

汇总 ablation：

```bash
python3 scripts/results/summarize_action_chunking_ablation.py \
  --run-root /data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_action_chunking/formal_50seq_ep360_20260613T2050Z
```

上传/复现 SwanLab 曲线：

```bash
/home/dechao/.conda/envs/cv-hw2/bin/python scripts/results/upload_swanlab_curves.py \
  --project cv_hw3_task2 \
  --workspace ade \
  --train-stride 10
```

上传 ModelScope 权重：

```bash
# Requires a ModelScope token; login first with:
# /home/dechao/.conda/envs/cv-hw2/bin/ms login
bash scripts/setup/upload_modelscope_weights.sh
```

## 数据与大文件

代码、配置、小型结果表和图位于：

```text
/home/dechao/cv_final_pj2
```

数据、conda env、checkpoint、rollout 视频位于：

```text
/data/dechao/cv_final_pj2
```

不提交 checkpoint、CALVIN 数据、rollout 视频和 conda 环境。

报告使用的 4 个 best-validation checkpoint 托管在 ModelScope，仓库内的 `modelscope/manifest.json` 记录了远端文件名和本地训练输出来源。
