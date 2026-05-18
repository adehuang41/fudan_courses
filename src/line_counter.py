"""Virtual-line crossing counter on top of YOLO+ByteTrack.

Counts a Track ID once when its detection-box centre crosses a configured
line segment (sign of the 2-D cross product flips between consecutive frames).
Per-class breakdown is overlaid on each frame.

Usage:
    python -m src.line_counter --weights checkpoints/yolov8s_visdrone_640/weights/best.pt \
        --video data/videos/test_clip.mp4 \
        --line 1280,0,1280,1080 \
        --out_video outputs/line_count_demo.mp4 \
        --summary outputs/line_count_summary.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import torch
torch.backends.cudnn.enabled = False

from ultralytics import YOLO  # noqa: E402

from src.utils import VISDRONE_NAMES, color_for_id, save_json  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def side(p, a, b):
    """Sign of (b-a) x (p-a). >0, =0, <0 indicate which side of line a-b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1, p2, a, b):
    """True iff segment p1-p2 intersects segment a-b (cheap predicate)."""
    d1 = side(p1, a, b); d2 = side(p2, a, b)
    d3 = side(a, p1, p2); d4 = side(b, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--line", required=True,
                    help="x1,y1,x2,y2 in pixel coords")
    ap.add_argument("--out_video", default="outputs/line_count_demo.mp4")
    ap.add_argument("--summary", default="outputs/line_count_summary.json")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="7")
    args = ap.parse_args()

    import os; os.chdir(REPO_ROOT)

    x1, y1, x2, y2 = [int(v) for v in args.line.split(",")]
    A, B = (x1, y1), (x2, y2)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out_path = Path(args.out_video); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    model = YOLO(args.weights)

    last_centre = {}        # tid -> (cx, cy)
    counted = set()         # tids already counted
    cls_count = defaultdict(int)
    crossings = []          # list of (frame, tid, cls_name)

    stream = model.track(
        source=args.video, tracker=args.tracker, persist=True,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device,
        stream=True, verbose=False,
    )

    for frame_idx, res in enumerate(stream):
        frame = res.orig_img.copy()
        cv2.line(frame, A, B, (0, 255, 255), 3)

        if res.boxes is not None and len(res.boxes) > 0 and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int).tolist()
            clss = res.boxes.cls.cpu().numpy().astype(int).tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()

            for box, tid, c, p in zip(xyxy, ids, clss, confs):
                bx1, by1, bx2, by2 = box
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                col = color_for_id(int(tid))
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), col, 2)
                cv2.circle(frame, (int(cx), int(cy)), 3, col, -1)
                name = VISDRONE_NAMES[c] if 0 <= c < len(VISDRONE_NAMES) else "?"
                label = f"ID{tid}:{name}"
                cv2.putText(frame, label, (int(bx1), int(by1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

                prev = last_centre.get(tid)
                if prev is not None and tid not in counted:
                    if segments_intersect(prev, (cx, cy), A, B):
                        counted.add(tid)
                        cls_count[name] += 1
                        crossings.append({"frame": frame_idx, "id": int(tid), "cls": name})
                last_centre[tid] = (cx, cy)

        # HUD
        total = sum(cls_count.values())
        cv2.rectangle(frame, (10, 10), (340, 30 + 22 * (len(VISDRONE_NAMES) + 2)),
                      (0, 0, 0), -1)
        cv2.putText(frame, f"frame {frame_idx}  total={total}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, name in enumerate(VISDRONE_NAMES):
            if cls_count[name] > 0:
                cv2.putText(frame, f"{name}: {cls_count[name]}",
                            (18, 56 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 255, 200), 1, cv2.LINE_AA)
        writer.write(frame)

    writer.release()
    summary = {
        "video": args.video, "line": [A, B], "total": sum(cls_count.values()),
        "per_class": dict(cls_count), "crossings": crossings,
        "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz,
    }
    save_json(summary, args.summary)
    print(f"=== line-crossing summary ===")
    print(f"total = {summary['total']}")
    for k, v in summary["per_class"].items():
        print(f"  {k:18s} {v}")
    print(f"wrote {out_path}, {args.summary}")


if __name__ == "__main__":
    main()
