#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VIDEO="data/videos/test_clip.mp4"
WEIGHTS="checkpoints/yolov8s_visdrone_640/weights/best.pt"
DEVICE="${1:-7}"

BASE_DIR="outputs/compare/baseline_640_conf025"
HIRES_DIR="outputs/compare/hires_1280_conf010"
MID_DIR="outputs/compare/hires_1280_conf015"

mkdir -p "$BASE_DIR" "$HIRES_DIR" "$MID_DIR"

cp outputs/tracked_bytetrack.mp4 "$BASE_DIR/baseline_640_conf025_tracked_bytetrack.mp4"
cp outputs/tracks.json "$BASE_DIR/baseline_640_conf025_tracks.json"
cp outputs/line_count_demo.mp4 "$BASE_DIR/baseline_640_conf025_line_count_demo.mp4"
cp outputs/line_count_summary.json "$BASE_DIR/baseline_640_conf025_line_count_summary.json"
cp outputs/conf_ablation.csv "$BASE_DIR/baseline_640_conf025_conf_ablation.csv"

python -m src.detect_video \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --conf 0.25 \
  --iou 0.5 \
  --imgsz 640 \
  --device "$DEVICE" \
  --out_video "$BASE_DIR/baseline_640_conf025_detect_demo.mp4" \
  --out_json "$BASE_DIR/baseline_640_conf025_detections.json"

python -m src.detect_video \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --conf 0.10 \
  --iou 0.5 \
  --imgsz 1280 \
  --device "$DEVICE" \
  --out_video "$HIRES_DIR/hires_1280_conf010_detect_demo.mp4" \
  --out_json "$HIRES_DIR/hires_1280_conf010_detections.json"

python -m src.track_video \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --conf 0.10 \
  --iou 0.5 \
  --imgsz 1280 \
  --device "$DEVICE" \
  --out_video "$HIRES_DIR/hires_1280_conf010_tracked_bytetrack.mp4" \
  --out_json "$HIRES_DIR/hires_1280_conf010_tracks.json"

python -m src.line_counter \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --line 1280,0,1280,1080 \
  --conf 0.10 \
  --iou 0.5 \
  --imgsz 1280 \
  --device "$DEVICE" \
  --out_video "$HIRES_DIR/hires_1280_conf010_line_count_demo.mp4" \
  --summary "$HIRES_DIR/hires_1280_conf010_line_count_summary.json"

python -m src.detect_video \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --conf 0.15 \
  --iou 0.5 \
  --imgsz 1280 \
  --device "$DEVICE" \
  --out_video "$MID_DIR/hires_1280_conf015_detect_demo.mp4" \
  --out_json "$MID_DIR/hires_1280_conf015_detections.json"

python -m src.track_video \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --conf 0.15 \
  --iou 0.5 \
  --imgsz 1280 \
  --device "$DEVICE" \
  --out_video "$MID_DIR/hires_1280_conf015_tracked_bytetrack.mp4" \
  --out_json "$MID_DIR/hires_1280_conf015_tracks.json"

python -m src.line_counter \
  --weights "$WEIGHTS" \
  --video "$VIDEO" \
  --line 1280,0,1280,1080 \
  --conf 0.15 \
  --iou 0.5 \
  --imgsz 1280 \
  --device "$DEVICE" \
  --out_video "$MID_DIR/hires_1280_conf015_line_count_demo.mp4" \
  --summary "$MID_DIR/hires_1280_conf015_line_count_summary.json"

echo "Comparison outputs written to:"
echo "  $BASE_DIR"
echo "  $HIRES_DIR"
echo "  $MID_DIR"
