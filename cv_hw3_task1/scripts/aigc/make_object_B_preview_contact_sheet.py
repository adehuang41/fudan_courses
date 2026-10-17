#!/usr/bin/env python3
"""Build contact sheets for Object B DreamFusion checkpoint previews."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_STEPS = (1000, 2000, 3000)
DEFAULT_FRAME_FRACTIONS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)


def read_video_frames(video_path: Path, fractions: tuple[float, ...]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise RuntimeError(f"could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
      raise RuntimeError(f"video has no frames: {video_path}")

    frames: list[np.ndarray] = []
    for fraction in fractions:
        index = min(frame_count - 1, max(0, round(fraction * (frame_count - 1))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read frame {index} from {video_path}")
        frames.append(frame)
    cap.release()
    return frames


def make_tile(frame: np.ndarray, label: str, width: int, label_height: int) -> np.ndarray:
    height = max(1, round(frame.shape[0] * (width / frame.shape[1])))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + label_height, width, 3), 245, dtype=np.uint8)
    canvas[label_height:, :, :] = resized
    cv2.putText(
        canvas,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return canvas


def hstack_same_height(images: list[np.ndarray], pad: int) -> np.ndarray:
    max_height = max(image.shape[0] for image in images)
    padded: list[np.ndarray] = []
    for image in images:
        if image.shape[0] < max_height:
            bottom = max_height - image.shape[0]
            image = cv2.copyMakeBorder(
                image, 0, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)
            )
        padded.append(image)
    spacer = np.full((max_height, pad, 3), 255, dtype=np.uint8)
    row = padded[0]
    for image in padded[1:]:
        row = np.hstack([row, spacer, image])
    return row


def vstack_same_width(images: list[np.ndarray], pad: int) -> np.ndarray:
    max_width = max(image.shape[1] for image in images)
    padded: list[np.ndarray] = []
    for image in images:
        if image.shape[1] < max_width:
            right = max_width - image.shape[1]
            image = cv2.copyMakeBorder(
                image, 0, 0, 0, right, cv2.BORDER_CONSTANT, value=(245, 245, 245)
            )
        padded.append(image)
    spacer = np.full((pad, max_width, 3), 255, dtype=np.uint8)
    sheet = padded[0]
    for image in padded[1:]:
        sheet = np.vstack([sheet, spacer, image])
    return sheet


def build_sheet(
    trial_dirs: list[Path], output_path: Path, steps: tuple[int, ...], tile_width: int
) -> None:
    rows: list[np.ndarray] = []
    for trial_dir in trial_dirs:
        seed_label = trial_dir.name.split("@", 1)[0].replace("rubber_duck_simple_", "")
        for step in steps:
            video_path = trial_dir / "save" / f"it{step}-test.mp4"
            if not video_path.exists():
                continue
            frames = read_video_frames(video_path, DEFAULT_FRAME_FRACTIONS)
            tiles = [
                make_tile(frame, f"{seed_label} step {step} view {index}", tile_width, 30)
                for index, frame in enumerate(frames)
            ]
            rows.append(hstack_same_height(tiles, pad=6))

    if not rows:
        raise RuntimeError("no preview videos found for requested trial dirs/steps")

    sheet = vstack_same_width(rows, pad=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise RuntimeError(f"could not write {output_path}")


def parse_steps(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=",".join(str(step) for step in DEFAULT_STEPS))
    parser.add_argument("--tile-width", default=192, type=int)
    args = parser.parse_args()

    build_sheet(args.trial_dir, args.output, parse_steps(args.steps), args.tile_width)


if __name__ == "__main__":
    main()
