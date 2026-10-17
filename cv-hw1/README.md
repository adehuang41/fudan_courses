# HW1: Fashion-MNIST 三层神经网络分类

本项目使用 `NumPy` 从零实现多层感知机，对 Fashion-MNIST 数据集进行 10 类图像分类。

GitHub Repository: https://github.com/adehuang41/cv-hw1.git

## 项目结构

- `final version.ipynb`：最终实验 notebook，包含训练、测试、混淆矩阵、权重可视化和错例分析。
- `src/`：数据处理、模型、损失函数、优化器、训练、测试与超参数搜索代码。
- `data/`：运行时缓存 Fashion-MNIST 数据，默认不提交到仓库。
- `models/final_best_model.pkl`：当前 notebook 训练并保存的最佳模型权重。
- `logs/`：训练过程中生成的报告图和分析图。

## 环境依赖

建议使用 Python 3.9 及以上版本。

核心依赖：

- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nbformat`

如果需要在本地新建环境，可以参考：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行方式

1. 打开 `final version.ipynb`。
2. 从上到下运行全部单元。
3. 运行完成后，模型权重会保存到 `models/final_best_model.pkl`。
4. 混淆矩阵、第一层权重可视化和错例分析图会保存到 `logs/`。

## 当前最优配置

```python
BEST_PARAMS = {
    "hidden_dims": [256, 128],
    "learning_rate": 0.01,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "activation": "relu"
}
```

## 结果文件

- 训练曲线：`final version.ipynb`
- 混淆矩阵：`logs/confusion_matrix.png`
- 第一层权重可视化：`logs/weight_visualization.png`
- 错例分析图：`logs/error_analysis.png`

## 模型权重

当前最佳模型权重已包含在 `models/final_best_model.pkl`。模型权重也已上传到 Google Drive：
https://drive.google.com/file/d/1WgLlZ9yD3csx1ztqtvwvrn2S1ItNH2iw/view?usp=sharing
