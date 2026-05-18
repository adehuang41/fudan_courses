"""Conf-threshold ablation on the test clip.

For each `conf` in CONFS, runs YOLO+ByteTrack and the same line-crossing
logic, and records:
  - mean detections per frame
  - unique track-ID count (rough proxy only; not a strict ID-switch metric,
    since both miss-detections and real switches inflate it)
  - line-crossing total

Outputs CSV + bar chart.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
torch.backends.cudnn.enabled = False

from ultralytics import YOLO  # noqa: E402

from src.line_counter import segments_intersect  # noqa: E402

CONFS = [0.15, 0.25, 0.35, 0.50]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_one(weights, video, conf, iou, imgsz, device, line):
    model = YOLO(weights)
    A, B = (line[0], line[1]), (line[2], line[3])

    last_centre = {}; counted = set(); cross = 0
    n_dets_per_frame = []
    unique_ids = set()

    stream = model.track(
        source=video, tracker="bytetrack.yaml", persist=True,
        conf=conf, iou=iou, imgsz=imgsz, device=device,
        stream=True, verbose=False,
    )
    for res in stream:
        if res.boxes is None or len(res.boxes) == 0 or res.boxes.id is None:
            n_dets_per_frame.append(0); continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int).tolist()
        n_dets_per_frame.append(len(ids))
        for box, tid in zip(xyxy, ids):
            unique_ids.add(int(tid))
            cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
            prev = last_centre.get(tid)
            if prev is not None and tid not in counted:
                if segments_intersect(prev, (cx, cy), A, B):
                    counted.add(tid); cross += 1
            last_centre[tid] = (cx, cy)
    return {
        "conf": conf,
        "mean_dets_per_frame": float(np.mean(n_dets_per_frame)) if n_dets_per_frame else 0.0,
        "unique_ids": len(unique_ids),
        "line_crossings": cross,
        "n_frames": len(n_dets_per_frame),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--line", required=True, help="x1,y1,x2,y2")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="7")
    ap.add_argument("--out_csv", default="outputs/conf_ablation.csv")
    ap.add_argument("--out_fig", default="report/figures/conf_ablation.png")
    args = ap.parse_args()

    import os; os.chdir(REPO_ROOT)
    line = [int(v) for v in args.line.split(",")]

    rows = []
    for c in CONFS:
        print(f"=== conf={c} ===")
        r = _run_one(args.weights, args.video, c, args.iou, args.imgsz, args.device, line)
        print(r)
        rows.append(r)

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    confs = [r["conf"] for r in rows]
    for ax, key, title in zip(
        axes,
        ["mean_dets_per_frame", "unique_ids", "line_crossings"],
        ["mean detections / frame", "unique track IDs (rough)", "line crossings"],
    ):
        vals = [r[key] for r in rows]
        bars = ax.bar([str(c) for c in confs], vals, color="steelblue")
        ax.set_title(title); ax.set_xlabel("conf threshold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_fig = Path(args.out_fig); out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150); plt.close(fig)
    print(f"wrote {out_csv} and {out_fig}")


if __name__ == "__main__":
    main()
