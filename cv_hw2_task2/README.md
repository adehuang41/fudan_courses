# 计算机视觉 HW2 Task 2

基于 `VisDrone-DET` 的目标检测、视频多目标跟踪、遮挡分析与虚拟线越线计数。

## 项目简介

本项目围绕课程作业 Task 2 展开，完整实现了以下流程：

1. 在 `VisDrone-DET` 上微调 `YOLOv8` 检测器。
2. 将检测器迁移到无人机交通视频中，结合 `ByteTrack` 完成多目标跟踪。
3. 在轨迹之上进行 `ID switch` 分析与虚拟线越线计数。

项目主要想回答三个问题：

1. `YOLOv8m` 相比 `YOLOv8s` 是否带来稳定提升。
2. 视频中的 `ID switch` 主要来自遮挡，还是来自边界漏检与检测中断。
3. 检测指标的提升能否真正传导到视频级跟踪与计数结果上。

完整报告见 [report/report_task2.md](report/report_task2.md)。

## 主要结论

1. `YOLOv8m` 在验证集和 `test-dev` 上都稳定优于 `YOLOv8s`。
2. `YOLOv8s + ByteTrack` 已经能够在主视频上稳定完成轨迹输出和越线计数。
3. 主视频中的典型 `ID switch` 更像是边界附近的短时漏检导致的关联中断，而不是“只要遮挡就一定换 ID”。
4. 人工遮挡实验表明，在运动连续、重现位置可预测时，`ByteTrack` 可以维持原有 ID。
5. 在域外视频 `fudan_handan.mp4` 上，`YOLOv8m` 更接近人工计数，但类别混淆和泛化退化依然明显。

## 仓库结构

```text
configs/
  train_yolov8s.yaml
  train_yolov8m.yaml
  visdrone.yaml

src/
  prepare_visdrone.py
  train.py
  eval.py
  detect_video.py
  track_video.py
  extract_occlusion.py
  line_counter.py
  conf_ablation.py
  utils.py

scripts/
  train.sh
  run_compare_inference.sh
  postprocess_yolov8m.sh
  backfill_swanlab_yolov8m.py
  generate_presentation_bundle.py

report/
  report_task2.md
  figures/
```

公开仓库不包含大体积产物，以下内容被有意排除：

- 数据集
- 训练权重
- 原始视频与处理后视频
- `outputs/`、`release/` 等中间结果与展示结果

## 环境配置

建议使用 `PyTorch + Ultralytics YOLOv8` 环境。

```bash
conda activate cv-hw2
pip install -r requirements.txt
```

如需使用 `SwanLab` 记录训练过程：

```bash
swanlab login --api-key <YOUR_KEY>
```

## 数据准备

将 `VisDrone-DET` 整理为 YOLO 格式：

```bash
python -m src.prepare_visdrone
python -m src.prepare_visdrone --check
```

预期划分为：

- `train`: 6471 images
- `val`: 548 images
- `test-dev`: 1610 images

## 训练与评估

训练 `YOLOv8s`：

```bash
python -m src.train --config configs/train_yolov8s.yaml
```

训练 `YOLOv8m`：

```bash
python -m src.train --config configs/train_yolov8m.yaml
```

评估模型：

```bash
python -m src.eval \
  --weights checkpoints/yolov8m_visdrone_640/weights/best.pt \
  --also_test
```

报告中使用的主要检测结果如下：

| 模型 | val mAP50 | val mAP50-95 | test-dev mAP50 | test-dev mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8s | `0.4120` | `0.2460` | `0.3260` | `0.1910` |
| YOLOv8m | `0.4295` | `0.2602` | `0.3383` | `0.2004` |

## 视频实验

主视频为 `new_drone_video.mp4`，脚本中使用其工作副本 `data/videos/test_clip.mp4`。

视频跟踪：

```bash
python -m src.track_video \
  --weights checkpoints/yolov8s_visdrone_640/weights/best.pt \
  --video data/videos/test_clip.mp4
```

提取 `ID switch` / 遮挡案例：

```bash
python -m src.extract_occlusion \
  --tracks outputs/tracks.json \
  --video data/videos/test_clip.mp4
```

虚拟线越线计数：

```bash
python -m src.line_counter \
  --weights checkpoints/yolov8s_visdrone_640/weights/best.pt \
  --video data/videos/test_clip.mp4 \
  --line 1280,0,1280,1080
```

置信度阈值消融：

```bash
python -m src.conf_ablation \
  --weights checkpoints/yolov8s_visdrone_640/weights/best.pt \
  --video data/videos/test_clip.mp4 \
  --line 1280,0,1280,1080
```

## 主视频结果摘要

- `YOLOv8s 640 / conf=0.25` 的主线越线总数为 `18`
- 人工计数为 `car = 18, motor = 4, 总计 = 22`
- 提高分辨率后，`motor` 开始被检出，但会出现部分 `car -> van` 的类别偏移
- `YOLOv8m 1280 / conf=0.15` 的总计数达到 `22`，与人工总数一致，但类别误差仍然存在

## 资源链接

- GitHub 仓库：<https://github.com/adehuang41/cv_hw2_task2>
- 主报告：[report/report_task2.md](report/report_task2.md)
- 模型权重：<https://drive.google.com/file/d/1181_K1THxbxi584ephFJekuBFCrbwqRu/view?usp=drive_link>
- SwanLab 记录：<https://swanlab.cn/@ade/hw2-task2/overview>
