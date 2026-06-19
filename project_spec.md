# 项目规格

## 项目名称

基于 LeRobot 的 ACT 策略跨环境泛化挑战。

## 课程任务定位

本项目选择计算机视觉期末项目中的 LeRobot + ACT 方向，不做 2DGS/AIGC 方向。

核心研究问题：

> 使用 LeRobot 训练 ACT 模仿学习策略时，模型在训练环境内和跨环境 zero-shot 测试中的表现差距有多大？这种差距与视觉分布变化和 ACT 的 action chunking 有什么关系？

## 课程要求摘要

根据 `要求文件/HW3_计算机视觉.pdf` 当前可识别内容，本项目需要覆盖：

- 使用 LeRobot、ACT 和 CALVIN。
- 使用 Action L1 Loss 作为训练过程中的核心损失记录。
- 在 held-out 环境中进行 zero-shot 测试。
- 用 Success Rate 度量任务成功情况。
- 分析 ACT 的 action chunking 与 visual distribution shift。
- 提交中文 PDF 报告，按 NeurIPS/CVPR 风格组织。
- 提供 Public GitHub Repository。
- 在最终 PDF 报告中写出模型权重外链。

课程 PDF 中可识别 deadline 为 2026-06-23 23:59，最终以课程平台为准。

## 数据与环境设定

默认主线设定：

- CALVIN A/B/C 作为源训练环境。
- CALVIN D 作为 held-out zero-shot 评估环境。
- 如果 Part 2 的数据检查发现实际数据 split 与该设定不一致，以数据 metadata 和老师要求为准，并更新本文档。

Part 2 数据接入策略：

- Hugging Face repo 使用 `huiwon/calvin_task_ABC_D`。
- 下载前先做 manifest/dry-run 验证，确认 repo 存在、包含 `calvin_task_ABC_D_lerobot_0_4` 到 `calvin_task_ABC_D_lerobot_3_4`，并统计真实总大小。
- 如果真实大小处于可接受范围且结构符合预期，则整 repo 下载到 `/data/dechao/cv_final_pj2/datasets/calvin_task_ABC_D`。
- 如果结构不符合预期或大小超过阈值，停止并汇报，不开始下载。
- `/data/huggingface` 是共享 Hugging Face 根目录，本项目不使用它保存私有数据或 cache。

## 交付物

必交：

- 中文 PDF 报告。
- Public GitHub Repository。
- 模型权重外链，并写入最终 PDF 报告。

不交付：

- slides。

## 实验记录

默认使用本地日志：

- 结构化 CSV 或 JSONL 记录训练和评估指标。
- 可选 TensorBoard event，用于后续转换或同步到 SwanLab。
- SwanLab 不作为主线训练依赖，只作为后续可选可视化或同步工具。

## 外部大文件策略

以下内容不进入 Git：

- CALVIN 数据集。
- conda 环境和 HF cache。
- 训练 checkpoint。
- 发布模型权重。
- 大型 rollout 视频。

仓库只保留代码、配置、文档、小型表格、小型图和必要示例。

## 阶段边界

### Part 1：项目需求定义 + 仓库结构初始化

创建项目文档和目录骨架，不安装环境、不下载数据、不写训练或评估代码。

### Part 2：环境配置 + CALVIN 数据接入

配置 LeRobot 环境，定位或下载数据，完成数据 sanity check。

### Part 3：ACT Baseline 训练

先 debug，再 short baseline，最后 full baseline；不改模型结构。

### Part 4：In-domain / Cross-environment 评估

比较训练环境内和 held-out 环境中的成功率，保存失败案例和结果表格。

### Part 5：结果分析 + 主线报告

整理结果，撰写中文学术风格 PDF 报告，明确 GitHub 仓库和模型权重链接。

## 主线外内容

以下内容只在主线完成后再考虑：

- 数据增强。
- Domain randomization。
- 模型结构改动。
- 多 policy 对比。
- 大规模 ablation。
- 复杂可视化系统。
