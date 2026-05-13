# HW2 任务 1 实验报告草稿

> 这是 Markdown 草稿，最终需导出 PDF。建议工作流：
> 1. 用 typora / pandoc / VSCode "Markdown PDF" 直接导成 PDF；或
> 2. 把内容粘进 Word/LaTeX 模板里加封面排版。

---

## 封面信息（自填）

- 课程：计算机视觉 期中作业
- 任务：任务 1 — 微调 ImageNet 预训练 CNN 实现宠物识别
- 姓名 / 学号：__（请补）__
- 分工：__单人完成 / 与 XXX 合作（XX 负责 …）__
- GitHub repo：__（推到自己 GitHub 后填入）__
- 模型权重网盘：__（百度云 / Google Drive 链接）__

---

## 1 任务与数据集

**目标**：在 Oxford-IIIT Pet Dataset（37 类宠物，约 7349 张图像）上，对 ImageNet
预训练的 ResNet-18 / ResNet-34 进行微调，完成图像分类，并通过一系列对照
实验回答 4 个子问题：(1) baseline 实现，(2) 超参敏感性，(3) 预训练增益，(4)
注意力机制 / 轻量 ViT 增益。

**数据集划分**：

- `trainval` 3680 张：随机 9:1 拆为 train (3312) / val (368)，固定 seed 42。
- `test` 3669 张：仅最终评估时使用。

**数据增广**：

- 训练：`Resize(256) → RandomResizedCrop(224, scale=0.7~1.0) → HFlip → ColorJitter(0.2)`
- 验证 / 测试：`Resize(256) → CenterCrop(224)`
- 归一化：ImageNet 均值方差。

## 2 模型结构

### 2.1 Baseline (ResNet-18 / 34)
来自 `torchvision.models`，加载 `IMAGENET1K_V1` 预训练权重，将 `model.fc`
替换为 `Linear(512, 37)` 并用 Kaiming 重新初始化。

### 2.2 注意力变体 (R18 + SE / CBAM)
在 `BasicBlock` 的第二个 BN 之后、残差相加之前插入注意力子模块（`src/models/resnet_attn.py`）：

- **SE-Block**：Global avg-pool → 1×1 Conv (C → C/16) → ReLU → 1×1 Conv (→ C) → Sigmoid → 通道乘。
- **CBAM**：通道注意力（avg + max 池化 → 共享 MLP → sigmoid 通道乘）→ 空间注意力（沿通道做 avg+max → 7×7 Conv → sigmoid 空间乘）。

预训练权重通过 `strict=False` 加载到对应层，注意力子模块从零初始化。**E7b/E8b 启用 identity init**：把 attn 最后一层 weight 置 0、bias 置 4，使初始 sigmoid 门控 ≈ 0.982 → 起步近似恒等映射，避免随机门扰动预训练特征。

### 2.3 Lightweight ViT (E9)
`timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=37)`。

## 3 实验设置

| 项 | 取值 |
|---|---|
| 输入分辨率 | 224 × 224 |
| Batch size | 64 |
| Optimizer | SGD(momentum=0.9, nesterov, wd=5e-4) for ResNets; AdamW(wd=0.05) for ViT |
| LR schedule | 1–2 epoch 线性 warmup + cosine annealing |
| 损失 | CrossEntropyLoss（ViT 加 label smoothing=0.1） |
| AMP | FP16 自动混合精度 |
| Seed | 42（torch / numpy / python random） |
| 评价指标 | Test set top-1 accuracy（同时记录 best val acc 用于早停 ckpt） |

**差分学习率**（基本要求 1 的关键）：每个 run 把参数分为三组——`head`（新 fc）使用大 LR、`backbone`（其余预训练参数）使用 1/10 LR、`attn`（仅注意力变体存在）单独配置。

**实验矩阵共 11 个 run**，详见 README.md 中的 [Experiment matrix](../README.md) 表格，下方仅汇总结果。

## 4 结果汇总

| Run | 设置 | Test top-1 | Best val |
|---|---|---:|---:|
| E1 | ResNet-18 baseline | 0.866 | 0.902 |
| E2 | ResNet-34 baseline | **0.906** | 0.927 |
| E3 | HP — lr 高 (3e-3 / 3e-4) | 0.890 | 0.927 |
| E4 | HP — lr 低 (3e-4 / 3e-5) | 0.810 | 0.829 |
| E5 | HP — 60 epoch 长跑 | 0.887 | 0.927 |
| E6 | **预训练消融**：随机初始化 R18 | 0.530 | 0.609 |
| E7 | R18 + SE (naive 初始化) | 0.778 | 0.813 |
| E7b | R18 + SE (identity init + attn_lr) | **0.875** | 0.905 |
| E7c | timm `legacy_seresnet18` (整模 ImageNet 预训练) | 0.865 | 0.878 |
| E8 | R18 + CBAM (naive 初始化) | 0.584 | 0.644 |
| E8b | R18 + CBAM (identity init + attn_lr, 30 epoch) | 0.869 | 0.889 |
| E8c | R18 + CBAM (identity init + attn_lr, 60 epoch) | 0.889 | 0.905 |
| E9 | ViT-Tiny (timm 预训练) | 0.900 | **0.943** |

## 5 分析

### 5.1 Baseline 与超参分析（要求 1+2）
- 预训练 ResNet-18 在 30 epoch / `head_lr=1e-3 / bb_lr=1e-4` 下达到 **86.6%**，达到合理基线；ResNet-34 提升 +4.0 pp。
- LR 提高一档（E3）val 提升但 test 几乎持平，提示更激进 LR 在小数据集上略有过拟合；LR 降一档（E4）则明显欠拟合，损失 5.6 pp。
- 训练时长延长至 60 epoch（E5）相对 30 epoch 几乎无收益（0.887 vs 0.866），说明 30 epoch 已基本收敛，cos 退火尾段贡献有限。

### 5.2 预训练消融（要求 3）
随机初始化版（E6）即便用 60 epoch + 较大 LR (1e-2) 训练，test 仅 **53.0%**，比预训练 baseline 低 33.6 pp。Pet 数据集每类约 100 张图，远不足从零学习卷积特征；ImageNet 预训练带来的低/中层特征是关键。

### 5.3 注意力机制（要求 4）
**naive 实现的陷阱**：直接在 BasicBlock 中插入随机初始化的 SE / CBAM 后训练，结果反而比 baseline 差 9 / 28 pp。原因：sigmoid 门控初始输出 ≈ 0.5，等价于把预训练特征"按通道砍半"，扰动过大；30 epoch 不足以让随机注意力恢复被破坏的特征空间。CBAM 因同时含通道+空间两层门控，扰动更大、退化更严重。

**修复方案**：(a) **identity init** —— 将注意力模块最后一层 weight 置 0、bias 置较大正值（如 4），令初始门控 sigmoid≈0.98，整个 attn 模块起步近似恒等；(b) 把 attn 参数单独分组，使用与 head 相同的较高 LR（1e-3，是 backbone 的 10 倍），加快 attn 学习。修复后 SE / CBAM 分别回到 0.875 / 0.869，与 baseline 持平甚至略高。**结论**：注意力机制并非"加上即涨点"，必须考虑预训练→微调间的初始化兼容性。

**进一步对比（E7c）**：为消除"我们手插的 SE 模块本身没有 ImageNet 预训练"这一公平性疑虑，额外引入 `timm` 中的 `legacy_seresnet18`（整套 SE-ResNet-18 在 ImageNet 上完整预训练，attention 模块的权重也是学好的），在相同微调配置下得到 test top-1 = **0.865**，与 baseline E1 0.866 几乎相同（差 0.001 < 噪声），略低于 E7b 0.875。这表明：在 Pet 这种小数据集（每类约 100 张训练图）上，**SE 模块的真实增益就在 0~1 pp 噪声范围内**，与 attention 是否经过完整预训练关系不大；ImageNet 上文献报告的 +0.7~1.1 pp 提升源自 1000 类大规模分类带来的细粒度通道选择需求，下游小数据集很难复现这种增益。

**CBAM 无公开 ImageNet 预训练权重**（timm 整个库都没有 CBAM 系列），无法做与 E7c 等价的对比；技术上也不存在跨架构（如把 R50-CBAM 的 attn 权重迁到 R18-CBAM）的合理迁移方案——CBAM 的通道注意力是 `Conv2d(C, C/16, 1)`，权重形状直接依赖通道数，R50 与 R18 各 stage 通道完全不匹配（256/512/1024/2048 vs 64/128/256/512），无法 `load_state_dict`，且 block 数量也不一致。因此 CBAM 我们额外做 E8c（60 epoch 长训练）观察是否需要更长收敛时间。

E8c 的结果（test 0.889 / val 0.905）相对 E8b（30 epoch）提升 +2.0 pp，相对 E1 baseline（30 epoch）提升 +2.3 pp，已超过 baseline；但与 **同等训练长度** 的 E5（60 epoch baseline，test 0.887）几乎打平（差 0.002，落在噪声内）。这进一步验证了上一段结论：在 Pet 上注意力机制本身没有显著增益，**E8b 与 baseline 的差距大部分来自训练长度而非 CBAM 本身**。延长训练同时使 attention 模块充分学习是 CBAM "看起来涨点" 的真实原因。

**ViT-Tiny vs ResNet**：预训练 ViT-Tiny（5.7M 参数）val 最高 0.943，test 0.900，介于 R18 / R34 之间；考虑参数量，性价比与 R18 接近。在小数据集（3.3k 训练图）上自注意力相对卷积没有显著优势，但充分预训练能弥合架构差异。

## 6 训练曲线（待补 swanlab 截图）

请到 [https://swanlab.cn](https://swanlab.cn) 项目 `hw2-task1`，对每个 run（E1–E9, E7b, E8b）截图：

1. `train/loss_epoch` 与 `val/loss` 同图（每个 run）；
2. `val/acc` 总览（建议把 E1 / E2 / E5 / E6 / E9 / E7b / E8b 叠加在一张图，方便对比）；
3. `train/lr_head` / `train/lr_backbone` 的 cosine 调度曲线（任选 1 个 run 即可，证明调度生效）。

把截图按 figure 1, 2, 3 嵌入到此文档中。

## 7 复现说明

- 代码、配置、脚本：见 README.md 中的「Reproducing all experiments」。
- 模型权重：将 `checkpoints/<run>/best.pt` 11 份打包后上传至网盘，链接补到封面。
- 运行环境：8×A800 80GB，单实验 ResNet-18 30 epoch ≈ 2 分钟，9 实验并行总墙钟 ≈ 8 分钟。
