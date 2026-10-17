#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a bounded candidate frame pool from a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output_dir", required=True, help="Directory for extracted JPG frames.")
    parser.add_argument("--start_time", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--end_time", type=float, default=180.0, help="End time in seconds.")
    parser.add_argument("--interval_sec", type=float, default=1.0, help="Seconds between extracted frames.")
    parser.add_argument("--max_frames", type=int, default=180, help="Maximum frames to write.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing frame_*.jpg files in output_dir first.")
    parser.add_argument("--summary_path", default=None, help="Optional JSON summary path.")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    return parser.parse_args()


def default_summary_path(output_dir: Path) -> Path:
    if output_dir.parent.name == "object_A_book":
        return output_dir.parent / "metadata" / "frame_extraction_summary.json"
    return output_dir / "frame_extraction_summary.json"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        fail("OpenCV is required. Install opencv in the active environment.")

    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_path) if args.summary_path else default_summary_path(output_dir)

    if not video_path.is_file():
        fail(f"Video not found: {video_path}")
    if args.interval_sec <= 0:
        fail("--interval_sec must be positive.")
    if args.max_frames <= 0:
        fail("--max_frames must be positive.")
    if args.end_time <= args.start_time:
        fail("--end_time must be greater than --start_time.")
    if not 1 <= args.jpeg_quality <= 100:
        fail("--jpeg_quality must be between 1 and 100.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing and not args.overwrite:
        fail(f"{output_dir} already contains frame_*.jpg. Use --overwrite to replace them.")
    if args.overwrite:
        for path in existing:
            path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        fail(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps > 0 else None

    effective_end = args.end_time
    if duration_sec is not None:
        effective_end = min(effective_end, duration_sec)

    timestamps = []
    current = args.start_time
    while current <= effective_end + 1e-6 and len(timestamps) < args.max_frames:
        timestamps.append(round(current, 6))
        current += args.interval_sec

    records = []
    failed_timestamps = []
    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok and fps > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp * fps)))
            ok, frame = cap.read()
        if not ok:
            failed_timestamps.append(timestamp)
            continue

        filename = f"frame_{len(records) + 1:06d}.jpg"
        path = output_dir / filename
        written = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        if not written:
            failed_timestamps.append(timestamp)
            continue
        records.append({"filename": filename, "timestamp_sec": timestamp})

    cap.release()

    summary = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "start_time": args.start_time,
        "end_time": args.end_time,
        "effective_end_time": effective_end,
        "interval_sec": args.interval_sec,
        "max_frames": args.max_frames,
        "actual_frame_count": len(records),
        "frame_filenames": [record["filename"] for record in records],
        "frame_records": records,
        "failed_timestamps": failed_timestamps,
        "video_metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec,
        },
        "role": "candidate_frame_pool",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Extracted {len(records)} frames to {output_dir}")
    print(f"Wrote summary to {summary_path}")
    if failed_timestamps:
        print(f"Warning: failed to extract {len(failed_timestamps)} requested timestamps", file=sys.stderr)


if __name__ == "__main__":
    main()
