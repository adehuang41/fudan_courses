"""Run YOLO + ByteTrack on a test clip, write a tracked mp4 + tracks.json.

Custom renderer overrides ultralytics' default annotator so that the box
color is a deterministic hash of the track ID (easier to follow visually
than the default per-class palette).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
torch.backends.cudnn.enabled = False

from ultralytics import YOLO  # noqa: E402

from src.utils import VISDRONE_NAMES, color_for_id, save_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _draw(frame, boxes_xyxy, ids, clss, confs):
    out = frame.copy()
    for (x1, y1, x2, y2), tid, c, p in zip(boxes_xyxy, ids, clss, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        col = color_for_id(int(tid)) if tid is not None else (180, 180, 180)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        name = VISDRONE_NAMES[int(c)] if 0 <= int(c) < len(VISDRONE_NAMES) else str(int(c))
        label = f"ID{int(tid) if tid is not None else -1}:{name} {p:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_video", default="outputs/tracked_bytetrack.mp4")
    ap.add_argument("--out_json", default="outputs/tracks.json")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="7")
    args = ap.parse_args()

    import os; os.chdir(REPO_ROOT)
    out_video = Path(args.out_video); out_video.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (W, H))

    model = YOLO(args.weights)
    records = []  # list of {frame, id, cls, xyxy, conf}

    stream = model.track(
        source=args.video, tracker=args.tracker, persist=True,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
        stream=True, verbose=False,
    )

    for frame_idx, res in enumerate(stream):
        frame = res.orig_img
        if res.boxes is None or len(res.boxes) == 0:
            writer.write(frame)
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        if res.boxes.id is None:
            ids = [-1] * len(xyxy)
        else:
            ids = res.boxes.id.cpu().numpy().astype(int).tolist()
        for box, tid, c, p in zip(xyxy, ids, clss, confs):
            records.append({
                "frame": frame_idx,
                "id": int(tid),
                "cls": int(c),
                "cls_name": VISDRONE_NAMES[int(c)] if 0 <= int(c) < len(VISDRONE_NAMES) else "?",
                "xyxy": [float(v) for v in box],
                "conf": float(p),
            })
        writer.write(_draw(frame, xyxy, ids, clss, confs))

    writer.release()
    save_json({
        "video": str(args.video),
        "fps": fps, "width": W, "height": H,
        "tracker": args.tracker, "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
        "frame_count": frame_idx + 1,
        "records": records,
    }, out_json)
    print(f"wrote {out_video} ({frame_idx + 1} frames)  + {out_json} ({len(records)} dets)")


if __name__ == "__main__":
    main()
