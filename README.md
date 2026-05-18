# 任务 3：从零搭建 U-Net 与损失函数工程

本仓库是计算机视觉课程 HW2 的 Task 3。目标是在**不使用任何预训练权重**的前提下，
基于 Oxford-IIIT Pet 的 trimap 分割标签，从零实现 U-Net，并比较不同损失函数在三分类语义分割中的表现。

当前仓库已经完成：

- 手写 U-Net：完整编码器、解码器、Skip Connection
- 手写多类 Dice Loss
- 三组主实验：`CE`、`Dice`、`CE + Dice`
- 两组组合损失权重消融：`1:0.5`、`0.5:1`
- 一组类别不平衡对照实验：`Weighted CE`
- 训练曲线、定性可视化、逐样本误差分析

## 1. 环境配置

如果你已经完成了 Task 1，通常可以直接复用 `cv-hw2` 环境。

```bash
conda activate cv-hw2
pip install -r requirements.txt
```

如果从零创建环境：

```bash
conda create -n cv-hw2 python=3.11 -y
conda activate cv-hw2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

如需在线记录训练曲线，可额外登录 SwanLab：

```bash
swanlab login --api-key <YOUR_SWANLAB_KEY>
```

说明：

- `src/train.py` 中已关闭 cuDNN，以绕过当前机器上 `CUDNN_STATUS_NOT_INITIALIZED` 的兼容性问题
- 训练仍然走 CUDA，单卡 A800 上速度足够

## 2. 数据集

本项目使用 Oxford-IIIT Pet Dataset 的 `segmentation` 标签，也就是 trimap 掩码。

- 原始标签值：`1 / 2 / 3`
- 重映射后类别：`0 / 1 / 2`
- 分别对应：`foreground / background / boundary`

程序会通过 `torchvision.datasets.OxfordIIITPet` 自动下载数据集到 `./data`。
如果本地已经存在数据，也可以通过配置文件中的 `data_root` 改到已有目录。

数据划分：

- 训练/验证：官方 `trainval` 再按 `9:1` 随机切分，`seed=42`
- 测试：官方 `test`

统一预处理：

- 输入尺寸：`256 x 256`
- 图像插值：双线性
- mask 插值：最近邻

## 3. 模型与训练设置

### 模型

- 主干网络：从零实现的 U-Net
- 基础通道数：`64`
- 参数量：约 `31M`
- 输出类别数：`3`

### 训练超参数

- Optimizer：`AdamW`
- Learning rate：`2e-3`
- Weight decay：`1e-4`
- LR scheduler：`cosine decay + 2 epoch warmup`
- Batch size：`64`
- Epoch：`60`
- AMP：开启
- Seed：`42`

### 数据增强

仅训练集使用：

- `RandomResizedCrop(scale=(0.7, 1.0))`
- `HorizontalFlip(p=0.5)`
- `ColorJitter(0.2, 0.2, 0.2)`

## 4. 实验设置

### 4.1 主实验

| Run | 配置文件 | 损失函数 |
|---|---|---|
| `E1_ce` | `configs/E1_ce.yaml` | Cross-Entropy |
| `E2_dice` | `configs/E2_dice.yaml` | Dice Loss |
| `E3_ce_dice` | `configs/E3_ce_dice.yaml` | CE + Dice |

### 4.2 组合损失消融

| Run | 配置文件 | 含义 |
|---|---|---|
| `E3a_ce_dice_2to1` | `configs/E3a_ce_dice_2to1.yaml` | CE 权重大于 Dice |
| `E3b_ce_dice_1to2` | `configs/E3b_ce_dice_1to2.yaml` | Dice 权重大于 CE |

### 4.3 类别不平衡对照实验

| Run | 配置文件 | 含义 |
|---|---|---|
| `E4_weighted_ce` | `configs/E4_weighted_ce.yaml` | 对 CE 加类别权重 |

该实验用于回答：Dice 的优势究竟来自“类别重加权”，还是来自“更贴近 overlap 的优化目标”。

## 5. 如何训练

### 单个实验

```bash
python -m src.train --config configs/E1_ce.yaml
```

不上传 SwanLab，仅本地训练：

```bash
python -m src.train --config configs/E1_ce.yaml --no_log
```

临时覆盖 epoch：

```bash
python -m src.train --config configs/E1_ce.yaml --epochs 5 --no_log
```

### 批量训练主实验

```bash
bash scripts/run_all.sh
```

### 补跑组合消融

```bash
bash scripts/queue_ablation.sh
```

训练产物位于：

```text
checkpoints/<run_name>/
├── best.pt
├── last.pt
└── metrics.json
```

## 6. 如何复现实验图

### 6.1 定性分割结果图

```bash
python -m src.visualize \
  --ckpts E1_ce=checkpoints/E1_ce/best.pt \
          E2_dice=checkpoints/E2_dice/best.pt \
          E3_ce_dice=checkpoints/E3_ce_dice/best.pt \
  --out report/qualitative.png --n 5
```

输出：

- `report/qualitative.png`

### 6.2 消融柱状图

```bash
python -m src.plot_ablation
```

输出：

- `report/ablation_bars.png`

### 6.3 训练曲线

```bash
python src/plot_training_dynamics.py
```

输出：

- `report/training_loss_curves.png`
- `report/validation_metrics.png`
- `report/validation_boundary_iou.png`
- `report/training_dynamics.png`

其中：

- `training_loss_curves.png`：训练集 / 验证集 loss 曲线
- `validation_metrics.png`：验证集 `pixel accuracy` 与 `mIoU`
- `validation_boundary_iou.png`：验证集 `boundary IoU`
- `training_dynamics.png`：旧版总览图

### 6.4 逐样本误差分析

```bash
python -m src.analyze_errors
```

输出：

- `report/error_analysis.csv`
- `report/error_analysis.png`
- `report/hard_cases.png`

## 7. 当前主要结果

主结论如下：

- `Dice only` 是当前最优方案
- `CE + Dice` 优于原始 CE，但没有超过纯 Dice
- `Weighted CE` 没有追平 Dice，说明该任务的关键不只是类别不平衡，还包括边界结构的 overlap 优化

测试集主结果：

| Run | test mIoU | test Boundary IoU |
|---|---:|---:|
| `E1_ce` | 0.7777 | 0.5691 |
| `E2_dice` | **0.7879** | **0.5932** |
| `E3_ce_dice` | 0.7852 | 0.5849 |
| `E4_weighted_ce` | 0.7643 | 0.5594 |

## 8. 目录结构

```text
configs/
├── _base.yaml
├── E1_ce.yaml
├── E2_dice.yaml
├── E3_ce_dice.yaml
├── E3a_ce_dice_2to1.yaml
├── E3b_ce_dice_1to2.yaml
└── E4_weighted_ce.yaml

src/
├── data.py
├── losses.py
├── metrics.py
├── plot_ablation.py
├── plot_training_dynamics.py
├── train.py
├── unet.py
├── utils.py
└── visualize.py

scripts/
├── run_all.sh
└── queue_ablation.sh

report/
├── ablation_bars.png
├── qualitative.png
├── training_loss_curves.png
├── validation_metrics.png
├── validation_boundary_iou.png
├── error_analysis.png
└── hard_cases.png
```
## 9. 相关链接

- GitHub 仓库：`https://github.com/adehuang41/cv_hw2_task3.git`
- 模型权重下载：`https://drive.google.com/file/d/171kHcWQ9D4Yg-OnEjj1QG6UhTIVzd0Aw/view?usp=drive_link`
- SwanLab 项目：`https://swanlab.cn/@ade/hw2-task3/overview`
