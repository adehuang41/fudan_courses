# HW2 Task 1：Oxford-IIIT Pet 预训练微调实验

本项目完成了课程作业 Task 1：在 Oxford-IIIT Pet Dataset 上微调 ImageNet 预训练卷积神经网络进行 37 类宠物品种分类，并补充了超参数分析、预训练消融、注意力机制比较和轻量 Transformer 对照。

- 代码仓库：<https://github.com/adehuang41/cv_hw2_task1>
- 模型权重：<https://drive.google.com/file/d/10FoC4MzjHDtPe9WaibzMaE00VgXoNHHN/view?usp=sharing>
- 最终报告：`report/report_task1.md`

## 1. 项目结论

- 最佳最终模型是 `ResNet-34 pretrained baseline`，测试集准确率 `90.57%`。
- 预训练是本任务最重要的性能来源；same-config scratch `ResNet-18/34` 分别只有 `9.13%` 和 `5.29%`。
- 微调时真正值得调的超参数不只是 epoch，而是 `head/backbone lr ratio` 和 `freeze_backbone_epochs`。
- 在预训练模型中加入 `SE/CBAM` 时，关键不只是模块类型，而是新增模块是否以近似恒等映射初始化，从而不破坏已有特征表示。

## 2. 环境配置

实验环境：

- GPU：`NVIDIA A800 80GB`
- Python：`3.11`
- CUDA Driver：`570`

推荐使用如下环境：

```bash
conda create -n cv-hw2 python=3.11 -y
conda activate cv-hw2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
swanlab login --api-key <YOUR_SWANLAB_KEY>
```

说明：

- 数据集会由 `torchvision.datasets.OxfordIIITPet` 首次运行时自动下载到 `./data`
- 当前机器上 `src/train.py` 已兼容处理 cuDNN 初始化问题

## 3. 数据集与划分

- 数据集：Oxford-IIIT Pet Dataset
- 任务：37 类单标签分类
- `trainval`：`3680` 张
- 划分方式：随机种子 `42`，按 `9:1` 划分为训练集 `3312`、验证集 `368`
- `test`：`3669` 张，使用官方测试集

数据处理代码见 [src/data.py](/home/dechao/cv/hw2_task1/src/data.py:1)。

## 4. 代码结构

```text
src/
  data.py                    数据集与数据增强
  train.py                   训练、验证、测试主流程
  utils.py
  models/
    build.py                 模型工厂与参数分组
    resnet_attn.py           ResNet-18/34 + SE/CBAM 实现

configs/                     每个实验对应一个 YAML 配置
scripts/
  run_all_parallel.sh        并行启动基础实验
  generate_artifacts.py
  generate_abc_refresh.py
  generate_final_report_figures.py

report/
  report_task1.md            最终报告
  assets/                    报告图表

checkpoints/<run_name>/      best.pt / last.pt / metrics.json
logs/<run_name>.log          本地训练日志
```

## 5. 训练与测试

### 5.1 单个实验训练

```bash
python -m src.train --config configs/E1_baseline_r18.yaml
```

如果只想快速试跑：

```bash
python -m src.train --config configs/E1_baseline_r18.yaml --epochs 10 --no_log
```

训练完成后会在 `checkpoints/<run_name>/` 下生成：

- `best.pt`
- `last.pt`
- `metrics.json`

同时在 `logs/` 下保存训练日志，并同步到 SwanLab。

### 5.2 测试已有权重

如果权重已经下载并放回对应的 `checkpoints/<run_name>/best.pt`，可以直接重新运行对应配置进行测试。训练脚本在结束时会自动读取 `best.pt` 并在官方测试集上评估。

### 5.3 生成报告图表

最终报告中的整合曲线图来自本地日志重绘：

```bash
conda run -n cv-hw2 python scripts/generate_final_report_figures.py
```

生成文件位于 `report/assets/`，主要包括：

- `final_overview.png`
- `final_core_curves.png`
- `final_pretraining_curves.png`
- `final_attention_curves.png`
- `final_deep_dive.png`

## 6. 主要实验设置

共同设置如下：

- 输入尺寸：`224 x 224`
- Batch size：`64`
- ResNet 优化器：`SGD(momentum=0.9, weight_decay=5e-4)`
- ViT 优化器：`AdamW(weight_decay=0.05)`
- 学习率策略：`warmup + cosine decay`
- 默认分类头学习率：`1e-3`
- 默认 backbone 学习率：`1e-4`
- 随机种子：`42`

模型参数分为三组：

- `head_lr`：分类头
- `backbone_lr`：预训练主干
- `attn_lr`：新加入的注意力模块

## 7. 实验列表

### 7.1 基础实验 E1-E9

| 实验 | 说明 |
| ---- | ---- |
| E1 | `ResNet-18` 预训练 baseline |
| E2 | `ResNet-34` 预训练 baseline |
| E3 | 更大学习率 |
| E4 | 更小学习率 |
| E5 | 更长训练时长（60 epoch） |
| E6 | `ResNet-18` 随机初始化训练 |
| E7 | `ResNet-18 + SE`，naive 插入 |
| E7b | `ResNet-18 + SE`，identity init 修复 |
| E7c | `legacy_seresnet18`，整体 SE 预训练 |
| E8 | `ResNet-18 + CBAM`，naive 插入 |
| E8b | `ResNet-18 + CBAM`，identity init 修复 |
| E8c | `ResNet-18 + CBAM`，修复后训练 60 epoch |
| E9 | `ViT-Tiny` |

### 7.2 深化实验 E10-E19

| 实验 | 说明 |
| ---- | ---- |
| E10 | `head/backbone lr = 1:1` |
| E11 | `head/backbone lr = 20:1` |
| E12 | `freeze_backbone_epochs = 3` |
| E13 | `freeze_backbone_epochs = 5` |
| E14 | 与 E1 同配置，仅关闭预训练 |
| E15 | 与 E2 同配置，仅关闭预训练 |
| E16 | SE：identity init + 低 attn lr |
| E17 | SE：naive init + 高 attn lr |
| E18 | CBAM：identity init + 低 attn lr |
| E19 | CBAM：naive init + 高 attn lr |

## 8. 最终结果

| 实验 | Test Acc | Best Val |
| ---- | -------: | -------: |
| E1 | 86.56% | 90.22% |
| E2 | **90.57%** | 92.66% |
| E3 | 89.02% | 92.66% |
| E4 | 81.00% | 82.88% |
| E5 | 88.66% | 92.66% |
| E6 | 53.01% | 60.87% |
| E7 | 77.81% | 81.25% |
| E7b | 87.46% | 90.49% |
| E7c | 86.45% | 87.77% |
| E8 | 58.38% | 64.40% |
| E8b | 86.89% | 88.86% |
| E8c | 88.93% | 90.49% |
| E9 | 90.02% | **94.29%** |
| E10 | 89.13% | 92.12% |
| E11 | 86.45% | 89.67% |
| E12 | 86.35% | 90.22% |
| E13 | 87.19% | 89.95% |
| E14 | 9.13% | 10.87% |
| E15 | 5.29% | 8.70% |
| E16 | 87.30% | 90.22% |
| E17 | 79.83% | 82.61% |
| E18 | 86.40% | 88.32% |
| E19 | 73.51% | 78.00% |

## 9. 复现命令示例

### 9.1 运行基础实验

```bash
bash scripts/run_all_parallel.sh
```

### 9.2 运行注意力修复实验

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/E7b_se_r18_fixed.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/E8b_cbam_r18_fixed.yaml
```

### 9.3 运行深化实验

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/E10_hp_ratio_1x.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/E11_hp_ratio_20x.yaml
CUDA_VISIBLE_DEVICES=2 python -m src.train --config configs/E12_hp_freeze3.yaml
CUDA_VISIBLE_DEVICES=3 python -m src.train --config configs/E13_hp_freeze5.yaml
CUDA_VISIBLE_DEVICES=4 python -m src.train --config configs/E14_scratch_samecfg_r18.yaml
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/E15_scratch_samecfg_r34.yaml
CUDA_VISIBLE_DEVICES=1 python -m src.train --config configs/E16_se_id_lowlr.yaml
CUDA_VISIBLE_DEVICES=2 python -m src.train --config configs/E17_se_naive_highlr.yaml
CUDA_VISIBLE_DEVICES=3 python -m src.train --config configs/E18_cbam_id_lowlr.yaml
CUDA_VISIBLE_DEVICES=4 python -m src.train --config configs/E19_cbam_naive_highlr.yaml
```

## 10. 报告与展示材料

- 最终报告：`report/report_task1.md`
- 报告图表：`report/assets/`
