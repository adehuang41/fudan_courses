# 神经网络与深度学习 Project 1

本仓库对应复旦大学《神经网络与深度学习》课程 Project 1，主题为 MNIST 上的 MLP、CNN 与鲁棒性分析 / 数据增强实验。

仓库中保留了：

- 项目代码
- 报告 LaTeX 源文件
- 报告中使用的小型结果图表与 CSV

仓库中**不包含**：

- MNIST 数据集原始文件
- 训练得到的模型权重 / checkpoint
- 大型训练目录和中间缓存文件

模型权重与 checkpoint 已单独上传到 ModelScope，并在报告中提供链接。

## 仓库结构

- `codes/`
  - NumPy from-scratch 的 MLP / CNN 实现
  - 训练、评估、鲁棒性分析与结果生成脚本
- `project_outputs/part_b/`
  - Part B（MLP vs CNN）在报告中使用的图表与汇总结果
- `project_outputs/part_c/`
  - Part C（robustness diagnosis、target-only augmentation、mixed augmentation、fine-tuning）在报告中使用的图表与汇总结果
- `report/`
  - 报告 LaTeX 源文件 `report.tex`
  - 构建脚本 `build_report.sh`

## 主要实验内容

### Part A

- 从零实现 MLP
- 在 MNIST 上训练 `MLP-clean`

### Part B

- 从零实现 CNN
- 比较 `MLP-clean` 与 `CNN-clean`

### Part C

- 先对 `CNN-clean` 做 validation robustness diagnosis
- 选出主要 weakness `translation-3`
- 比较四个主实验模型：
  - `CNN-AugScratch-target`
  - `CNN-AugScratch-mixed`
  - `CNN-RGFT-target`
  - `CNN-FT-mixed`

## 主要脚本

- `codes/test_train.py`
  - 训练单个模型实验
- `codes/test_model.py`
  - 加载 checkpoint 后在测试集或指定扰动下评估模型
- `codes/run_project_pipeline.py`
  - 生成主线 Part B / Part C 的核心结果与图表
- `codes/run_augscratch_mixed.py`
  - 从头训练 `CNN-AugScratch-mixed`
- `codes/run_mixedft.py`
  - 从 `CNN-clean` checkpoint 出发进行 mixed fine-tuning
- `codes/generate_final_assets.py`
  - 生成报告中使用的部分最终图表与分析产物

## 数据与权重说明

- GitHub 仓库不上传数据集与模型权重，符合课程要求
- `codes/dataset/MNIST/` 中仅保留说明文件 `README.md`
- 训练权重 / checkpoint 请参考报告中的 ModelScope 链接

## 报告编译

在仓库根目录执行：

```bash
cd report
./build_report.sh
```

## 提交说明

本仓库用于提供：

- 课程项目代码
- 报告源码
- 复现实验所需的小型结果文件

最终报告中的 GitHub 链接为：

`https://github.com/adehuang41/deep_learning_pj1`
