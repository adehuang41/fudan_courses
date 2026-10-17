#!/usr/bin/env python3
"""Build a contact sheet from multi-view validation PNGs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--thumb-width", type=int, default=900)
    return parser.parse_args()


def view_index(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        return 0
    return int(match.group(1))


def add_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 34
    canvas = Image.new("RGB", (image.width, image.height + label_height), "white")
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), label, fill=(20, 20, 20))
    return canvas


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    save_dir = run_dir / "save"
    files = sorted(save_dir.glob(f"it{args.step}-*.png"), key=view_index)
    files = [path for path in files if re.search(rf"it{args.step}-\d+\.png$", path.name)]
    if not files:
        raise FileNotFoundError(f"no validation PNGs found for step {args.step} under {save_dir}")

    labeled_images: list[Image.Image] = []
    for path in files:
        image = Image.open(path).convert("RGB")
        if image.width > args.thumb_width:
            height = round(image.height * args.thumb_width / image.width)
            image = image.resize((args.thumb_width, height), Image.Resampling.LANCZOS)
        labeled_images.append(add_label(image, path.name))

    cols = max(1, args.cols)
    rows = math.ceil(len(labeled_images) / cols)
    cell_w = max(image.width for image in labeled_images)
    cell_h = max(image.height for image in labeled_images)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for idx, image in enumerate(labeled_images):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(image, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
