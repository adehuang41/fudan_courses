"""Fine-tune YOLOv8 on VisDrone-DET with optional swanlab logging.

The Ultralytics trainer already streams metrics to its own runs dir; we add
a thin per-epoch callback that mirrors loss / mAP into swanlab so the report
can pull screenshots from the same place as task1 / task3.

Usage:
    python -m src.train --config configs/train_yolov8s.yaml
    python -m src.train --config configs/train_yolov8s.yaml --epochs 5 --no_log
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

# cuDNN-9 / A800 / driver-570 hits CUDNN_STATUS_NOT_INITIALIZED (same as task1).
import torch
torch.backends.cudnn.enabled = False

from ultralytics import YOLO  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _attach_swanlab(model, run_name: str, cfg: dict):
    try:
        import swanlab
    except ImportError:
        print("[swanlab] not installed — skipping logging.")
        return None

    run = swanlab.init(
        project="hw2-task2",
        experiment_name=run_name,
        config=cfg,
        logdir=str(REPO_ROOT / "swanlog"),
    )

    def on_fit_epoch_end(trainer):
        # trainer.metrics has keys like 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
        # 'val/box_loss', 'val/cls_loss', 'val/dfl_loss', plus 'lr/pg0', etc.
        log = {}
        for k, v in trainer.metrics.items():
            try:
                log[k.replace("/", ".")] = float(v)
            except (TypeError, ValueError):
                continue
        # train losses live on trainer.loss_items / trainer.tloss
        if hasattr(trainer, "label_loss_items"):
            tlosses = trainer.label_loss_items(trainer.tloss, prefix="train")
            for k, v in tlosses.items():
                try:
                    log[k.replace("/", ".")] = float(v)
                except (TypeError, ValueError):
                    continue
        log["epoch"] = int(trainer.epoch) + 1
        if log:
            swanlab.log(log, step=int(trainer.epoch) + 1)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                    help="Resume training from a checkpoint path using Ultralytics resume mode.")
    ap.add_argument("--no_log", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None: cfg["epochs"] = args.epochs
    if args.batch is not None: cfg["batch"] = args.batch
    if args.imgsz is not None: cfg["imgsz"] = args.imgsz
    if args.device is not None: cfg["device"] = args.device

    os.chdir(REPO_ROOT)  # ultralytics resolves relative paths from CWD

    run_name = cfg.pop("run_name")
    model_id = cfg.pop("model")
    model = YOLO(args.resume or model_id)

    if not args.no_log:
        _attach_swanlab(model, run_name, cfg)

    train_kwargs = dict(cfg)
    train_kwargs["name"] = run_name
    if args.resume:
        train_kwargs["resume"] = args.resume
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
