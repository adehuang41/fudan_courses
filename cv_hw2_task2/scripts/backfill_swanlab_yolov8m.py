from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="checkpoints/yolov8m_visdrone_640/results.csv")
    ap.add_argument("--config", default="configs/train_yolov8m.yaml")
    ap.add_argument("--run_name", default="yolov8m_visdrone_640_backfill")
    args = ap.parse_args()

    import os
    os.chdir(REPO_ROOT)

    import swanlab

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run = swanlab.init(
        project="hw2-task2",
        experiment_name=args.run_name,
        config=cfg,
        logdir=str(REPO_ROOT / "swanlog"),
    )

    with open(args.results, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        epoch = int(float(row["epoch"]))
        log = {
            "epoch": epoch,
            "train.box_loss": float(row["train/box_loss"]),
            "train.cls_loss": float(row["train/cls_loss"]),
            "train.dfl_loss": float(row["train/dfl_loss"]),
            "metrics.precision(B)": float(row["metrics/precision(B)"]),
            "metrics.recall(B)": float(row["metrics/recall(B)"]),
            "metrics.mAP50(B)": float(row["metrics/mAP50(B)"]),
            "metrics.mAP50-95(B)": float(row["metrics/mAP50-95(B)"]),
            "val.box_loss": float(row["val/box_loss"]),
            "val.cls_loss": float(row["val/cls_loss"]),
            "val.dfl_loss": float(row["val/dfl_loss"]),
            "lr.pg0": float(row["lr/pg0"]),
            "lr.pg1": float(row["lr/pg1"]),
            "lr.pg2": float(row["lr/pg2"]),
        }
        swanlab.log(log, step=epoch)

    print(f"Backfilled {len(rows)} epochs into SwanLab run: {args.run_name}")
    run.finish()


if __name__ == "__main__":
    main()
