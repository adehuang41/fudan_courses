#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make a labeled contact sheet for extracted frames.")
    parser.add_argument("--image_dir", required=True, help="Directory containing frame images.")
    parser.add_argument("--output", required=True, help="Output contact sheet image path.")
    parser.add_argument("--cols", type=int, default=6, help="Number of thumbnails per row.")
    parser.add_argument("--thumb_width", type=int, default=180, help="Thumbnail box width.")
    parser.add_argument("--thumb_height", type=int, default=240, help="Thumbnail box height.")
    parser.add_argument("--label_height", type=int, default=28, help="Space reserved for filename labels.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing contact sheet.")
    parser.add_argument("--summary_path", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def default_summary_path(image_dir: Path) -> Path:
    if image_dir.parent.name == "object_A_book":
        return image_dir.parent / "metadata" / "frame_contact_sheet_summary.json"
    return image_dir / "frame_contact_sheet_summary.json"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def draw_centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    x = x0 + max(0, (x1 - x0 - text_w) // 2)
    y = y0 + max(0, (y1 - y0 - text_h) // 2)
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output = Path(args.output)
    summary_path = Path(args.summary_path) if args.summary_path else default_summary_path(image_dir)

    if not image_dir.is_dir():
        fail(f"Image directory not found: {image_dir}")
    if args.cols <= 0:
        fail("--cols must be positive.")
    if args.thumb_width <= 0 or args.thumb_height <= 0:
        fail("Thumbnail dimensions must be positive.")
    if output.exists() and not args.overwrite:
        fail(f"Output already exists: {output}. Use --overwrite to replace it.")

    images = list_images(image_dir)
    if not images:
        fail(f"No images found in {image_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    rows = math.ceil(len(images) / args.cols)
    tile_w = args.thumb_width
    tile_h = args.thumb_height + args.label_height
    sheet = Image.new("RGB", (args.cols * tile_w, rows * tile_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for index, path in enumerate(images):
        row = index // args.cols
        col = index % args.cols
        x = col * tile_w
        y = row * tile_h

        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((args.thumb_width, args.thumb_height), Image.Resampling.LANCZOS)
            paste_x = x + (args.thumb_width - img.width) // 2
            paste_y = y + (args.thumb_height - img.height) // 2
            sheet.paste(img, (paste_x, paste_y))

        draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=(210, 210, 210))
        draw_centered_text(draw, (x, y + args.thumb_height, x + tile_w, y + tile_h), path.name, font)

    sheet.save(output, quality=92)

    summary = {
        "image_dir": str(image_dir),
        "output": str(output),
        "image_count": len(images),
        "columns": args.cols,
        "thumb_width": args.thumb_width,
        "thumb_height": args.thumb_height,
        "image_filenames": [path.name for path in images],
        "human_checks": [
            "blurred frames",
            "duplicate viewpoints",
            "occluded frames",
            "extreme viewpoints",
            "exposure failures",
            "object too small",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote contact sheet for {len(images)} images to {output}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
