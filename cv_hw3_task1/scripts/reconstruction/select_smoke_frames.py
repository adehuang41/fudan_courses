#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a small, evenly spaced frame subset for COLMAP smoke testing.")
    parser.add_argument("--source_dir", required=True, help="Candidate frame directory, usually images_raw.")
    parser.add_argument("--output_dir", required=True, help="Smoke subset directory, usually images_smoke.")
    parser.add_argument("--count", type=int, default=18, help="Number of frames to copy.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing image files in output_dir before copying.")
    parser.add_argument("--summary_path", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def list_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def default_summary_path(source_dir: Path) -> Path:
    if source_dir.parent.name == "object_A_book":
        return source_dir.parent / "metadata" / "smoke_frame_selection_summary.json"
    return source_dir / "smoke_frame_selection_summary.json"


def uniform_indices(total: int, count: int) -> list[int]:
    if count == 1:
        return [total // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_path) if args.summary_path else default_summary_path(source_dir)

    if not source_dir.is_dir():
        fail(f"Source directory not found: {source_dir}")
    if args.count <= 0:
        fail("--count must be positive.")
    if args.count > 20:
        fail("Smoke test subset should not exceed 20 images.")

    source_images = list_images(source_dir)
    if len(source_images) < args.count:
        fail(f"Need at least {args.count} source images, found {len(source_images)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    existing = list_images(output_dir)
    if existing and not args.overwrite:
        fail(f"{output_dir} already contains images. Use --overwrite to replace them.")
    if args.overwrite:
        for path in existing:
            path.unlink()

    indices = uniform_indices(len(source_images), args.count)
    selected = [source_images[index] for index in indices]

    copied = []
    for source in selected:
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination.name)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_count": len(source_images),
        "selected_count": len(copied),
        "strategy": "uniform_time_axis_sampling_from_sorted_candidate_frames",
        "selected_indices_zero_based": indices,
        "selected_filenames": copied,
        "role": "colmap_small_scale_smoke_input",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Copied {len(copied)} smoke frames to {output_dir}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
