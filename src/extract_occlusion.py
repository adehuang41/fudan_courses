"""Find candidate occlusion / ID-switch segments from tracks.json and dump frames.

Two heuristics:
  A) Dense crossing  — a frame in which the maximum pairwise IoU among track
     boxes exceeds `iou_thresh`.
  B) Suspected ID switch — a track that disappears for <= gap_max frames and
     a *new* track appears within `dist_thresh` pixels of where the old one
     was last seen, of the same class.

The script prints the top-K candidates so you can pick one. Then re-invoke
with --segment <start_frame>,<end_frame> to dump 4 PNGs + a montage.

Usage:
    # 1) list candidates
    python -m src.extract_occlusion --tracks outputs/tracks.json \
         --video data/videos/test_clip.mp4 --topk 10
    # 2) export final frames
    python -m src.extract_occlusion --tracks outputs/tracks.json \
         --video data/videos/test_clip.mp4 --segment 234,240 --stride 2
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.utils import VISDRONE_NAMES, color_for_id, load_json

REPO_ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1); y1 = max(ay1, by1)
    x2 = min(ax2, bx2); y2 = min(ay2, by2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-9, ua + ub - inter)


def _draw(frame, dets):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        col = color_for_id(d["id"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"ID{d['id']}:{d['cls_name']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _frames_from(tracks):
    by_frame = defaultdict(list)
    for r in tracks["records"]:
        by_frame[r["frame"]].append(r)
    return by_frame


def find_candidates(tracks, iou_thresh=0.30, gap_max=15, dist_thresh=80, topk=10):
    by_frame = _frames_from(tracks)
    frames = sorted(by_frame.keys())

    # A) dense-crossing frames
    dense = []
    for f in frames:
        dets = by_frame[f]
        if len(dets) < 2:
            continue
        max_iou = 0.0
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                v = iou(dets[i]["xyxy"], dets[j]["xyxy"])
                if v > max_iou:
                    max_iou = v
        if max_iou >= iou_thresh:
            dense.append((f, max_iou))
    dense.sort(key=lambda x: -x[1])

    # B) suspected ID switches: track ends + a new track of same class starts
    # within gap_max frames and dist_thresh pixels.
    last_seen = {}  # id -> (frame, cx, cy, cls)
    born = []      # list of (id, frame, cx, cy, cls)
    for f in frames:
        for d in by_frame[f]:
            tid = d["id"]
            x1, y1, x2, y2 = d["xyxy"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if tid not in last_seen:
                born.append((tid, f, cx, cy, d["cls"]))
            last_seen[tid] = (f, cx, cy, d["cls"])

    deaths = []  # we approximate "death" as "last_seen frame"; iterate in order
    last_seen_sorted = sorted(last_seen.items(), key=lambda kv: kv[1][0])
    born_by_cls = defaultdict(list)
    for tid, f, cx, cy, c in born:
        born_by_cls[c].append((tid, f, cx, cy))

    switch_candidates = []
    for tid_old, (f_old, cx, cy, c) in last_seen_sorted:
        for tid_new, f_new, cx2, cy2 in born_by_cls[c]:
            if tid_new == tid_old:
                continue
            if 0 < f_new - f_old <= gap_max:
                d = ((cx - cx2) ** 2 + (cy - cy2) ** 2) ** 0.5
                if d <= dist_thresh:
                    switch_candidates.append({
                        "frame_gap": f_new - f_old, "dist_px": d,
                        "old_id": tid_old, "new_id": tid_new,
                        "old_frame": f_old, "new_frame": f_new,
                        "cls": VISDRONE_NAMES[c] if 0 <= c < len(VISDRONE_NAMES) else "?",
                    })
    switch_candidates.sort(key=lambda x: (x["dist_px"], x["frame_gap"]))

    print("=== Top dense-crossing frames (max-pairwise-IoU) ===")
    for f, v in dense[:topk]:
        print(f"  frame {f:4d}   max_iou={v:.3f}   n_dets={len(by_frame[f])}")
    print("\n=== Suspected ID-switch pairs ===")
    for c in switch_candidates[:topk]:
        print(f"  cls={c['cls']:14s} old_id={c['old_id']} -> new_id={c['new_id']}  "
              f"old_frame={c['old_frame']} -> new_frame={c['new_frame']}  "
              f"gap={c['frame_gap']} dist={c['dist_px']:.1f}px")


def export_segment(tracks, video_path, start, end, stride, out_dir):
    by_frame = _frames_from(tracks)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    selected = list(range(start, end + 1, stride))
    print(f"exporting frames: {selected}")
    crops = []
    for f in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            print(f"  skip {f}: read failed")
            continue
        annotated = _draw(frame, by_frame.get(f, []))
        cv2.putText(annotated, f"frame {f}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        out_path = out_dir / f"frame_{f:05d}.png"
        cv2.imwrite(str(out_path), annotated)
        crops.append(annotated)
        print(f"  wrote {out_path}")
    cap.release()

    if crops:
        h = max(c.shape[0] for c in crops); w = max(c.shape[1] for c in crops)
        crops = [cv2.copyMakeBorder(c, 0, h - c.shape[0], 0, w - c.shape[1],
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0)) for c in crops]
        n = len(crops)
        cols = 2 if n > 2 else n
        rows = (n + cols - 1) // cols
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for i, c in enumerate(crops):
            r, cc = divmod(i, cols)
            canvas[r * h:(r + 1) * h, cc * w:(cc + 1) * w] = c
        montage_path = out_dir / "montage.png"
        cv2.imwrite(str(montage_path), canvas)
        print(f"  wrote {montage_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--iou_thresh", type=float, default=0.30)
    ap.add_argument("--gap_max", type=int, default=15)
    ap.add_argument("--dist_thresh", type=float, default=80)
    ap.add_argument("--segment", default=None,
                    help="start,end frame indices to dump (inclusive)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out_dir", default="outputs/occlusion_frames")
    args = ap.parse_args()

    tracks = load_json(args.tracks)
    if args.segment is None:
        find_candidates(tracks, args.iou_thresh, args.gap_max, args.dist_thresh, args.topk)
    else:
        s, e = [int(x) for x in args.segment.split(",")]
        export_segment(tracks, args.video, s, e, args.stride, args.out_dir)


if __name__ == "__main__":
    main()
