#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEVICE="${1:-7}"
WEIGHTS="checkpoints/yolov8m_visdrone_640/weights/best.pt"

# Default "best" inference choices for the final lightweight comparison.
# test_clip: use the higher-resolution setting already validated in the repo.
# fudan_handan: use the current best-counting setting from prior ablations.
TEST_VIDEO="data/videos/test_clip.mp4"
TEST_CONF="${TEST_CONF:-0.15}"
TEST_IOU="${TEST_IOU:-0.5}"
TEST_IMGSZ="${TEST_IMGSZ:-1280}"
TEST_LINE="${TEST_LINE:-1280,0,1280,1080}"
TEST_OUT_DIR="outputs/yolov8m_compare/test_clip_best"

FUDAN_VIDEO="fudan_handan.mp4"
FUDAN_CONF="${FUDAN_CONF:-0.10}"
FUDAN_IOU="${FUDAN_IOU:-0.5}"
FUDAN_IMGSZ="${FUDAN_IMGSZ:-1280}"
FUDAN_LINE="${FUDAN_LINE:-320,790,1210,620}"
FUDAN_OUT_DIR="outputs/yolov8m_compare/fudan_handan_best"

mkdir -p "$TEST_OUT_DIR" "$FUDAN_OUT_DIR"

echo "[1/7] Evaluate YOLOv8m on val and test-dev"
conda run -n cv-hw2 python -m src.eval \
  --weights "$WEIGHTS" \
  --device "$DEVICE" \
  --also_test

echo "[2/7] test_clip detector demo"
conda run -n cv-hw2 python -m src.detect_video \
  --weights "$WEIGHTS" \
  --video "$TEST_VIDEO" \
  --conf "$TEST_CONF" \
  --iou "$TEST_IOU" \
  --imgsz "$TEST_IMGSZ" \
  --device "$DEVICE" \
  --out_video "$TEST_OUT_DIR/detect_demo.mp4" \
  --out_json "$TEST_OUT_DIR/detections.json"

echo "[3/7] test_clip tracking + counting"
conda run -n cv-hw2 python -m src.track_video \
  --weights "$WEIGHTS" \
  --video "$TEST_VIDEO" \
  --conf "$TEST_CONF" \
  --iou "$TEST_IOU" \
  --imgsz "$TEST_IMGSZ" \
  --device "$DEVICE" \
  --out_video "$TEST_OUT_DIR/tracked_bytetrack.mp4" \
  --out_json "$TEST_OUT_DIR/tracks.json"
conda run -n cv-hw2 python -m src.line_counter \
  --weights "$WEIGHTS" \
  --video "$TEST_VIDEO" \
  --line "$TEST_LINE" \
  --conf "$TEST_CONF" \
  --iou "$TEST_IOU" \
  --imgsz "$TEST_IMGSZ" \
  --device "$DEVICE" \
  --out_video "$TEST_OUT_DIR/line_count_demo.mp4" \
  --summary "$TEST_OUT_DIR/line_count_summary.json"

echo "[4/7] fudan_handan detector demo"
conda run -n cv-hw2 python -m src.detect_video \
  --weights "$WEIGHTS" \
  --video "$FUDAN_VIDEO" \
  --conf "$FUDAN_CONF" \
  --iou "$FUDAN_IOU" \
  --imgsz "$FUDAN_IMGSZ" \
  --device "$DEVICE" \
  --out_video "$FUDAN_OUT_DIR/detect_demo.mp4" \
  --out_json "$FUDAN_OUT_DIR/detections.json"

echo "[5/7] fudan_handan tracking + counting"
conda run -n cv-hw2 python -m src.track_video \
  --weights "$WEIGHTS" \
  --video "$FUDAN_VIDEO" \
  --conf "$FUDAN_CONF" \
  --iou "$FUDAN_IOU" \
  --imgsz "$FUDAN_IMGSZ" \
  --device "$DEVICE" \
  --out_video "$FUDAN_OUT_DIR/tracked_bytetrack.mp4" \
  --out_json "$FUDAN_OUT_DIR/tracks.json"
conda run -n cv-hw2 python -m src.line_counter \
  --weights "$WEIGHTS" \
  --video "$FUDAN_VIDEO" \
  --line "$FUDAN_LINE" \
  --conf "$FUDAN_CONF" \
  --iou "$FUDAN_IOU" \
  --imgsz "$FUDAN_IMGSZ" \
  --device "$DEVICE" \
  --out_video "$FUDAN_OUT_DIR/line_count_demo.mp4" \
  --summary "$FUDAN_OUT_DIR/line_count_summary.json"

echo "[6/7] Backfill complete training curves to SwanLab"
conda run -n cv-hw2 python scripts/backfill_swanlab_yolov8m.py

echo "[7/7] Refresh report bundle"
conda run -n cv-hw2 python scripts/generate_presentation_bundle.py

echo "YOLOv8m postprocessing complete."
echo "Outputs:"
echo "  $TEST_OUT_DIR"
echo "  $FUDAN_OUT_DIR"
