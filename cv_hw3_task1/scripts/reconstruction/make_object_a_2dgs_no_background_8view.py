#!/usr/bin/env python3
"""Make transparent-background 8-view evidence from Object A 2DGS renders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rembg import new_session, remove


DEFAULT_INDICES = (0, 19, 37, 56, 74, 93, 112, 130)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/render_train/renders"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/evidence/8view_no_background"
        ),
    )
    parser.add_argument("--indices", default=",".join(str(index) for index in DEFAULT_INDICES))
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--model", default="u2net")
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return (0, 0, image.width, image.height)
    pad = max(4, round(max(image.width, image.height) * 0.025))
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(image.width, x1 + pad),
        min(image.height, y1 + pad),
    )


def checkerboard(size: tuple[int, int], block: int = 12) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, height, block):
        for x in range(0, width, block):
            fill = (226, 226, 226) if ((x // block) + (y // block)) % 2 else (248, 248, 248)
            draw.rectangle((x, y, x + block - 1, y + block - 1), fill=fill)
    return image


def cleanup_alpha_components(image: Image.Image) -> Image.Image:
    rgba = np.array(image)
    alpha = rgba[:, :, 3]
    mask = (alpha > 10).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 2:
        return image

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max()) if len(areas) else 0
    min_area = max(25, int(largest * 0.015))
    keep = np.zeros_like(mask, dtype=bool)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label
    rgba[:, :, 3] = np.where(keep, alpha, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def make_tile(image: Image.Image, label: str, width: int, label_h: int, background: str) -> Image.Image:
    image = image.copy()
    ratio = width / image.width
    height = max(1, round(image.height * ratio))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    if background == "checker":
        bg = checkerboard((width, height))
    else:
        bg = Image.new("RGB", (width, height), background)
    bg.paste(image, (0, 0), image)
    tile = Image.new("RGB", (width, height + label_h), (245, 245, 245))
    tile.paste(bg, (0, label_h))
    ImageDraw.Draw(tile).text((8, 8), label, fill=(20, 20, 20))
    return tile


def build_sheet(images: list[tuple[str, Image.Image]], output: Path, background: str, thumb_width: int) -> None:
    label_h = 28
    pad = 8
    tiles = [make_tile(image, label, thumb_width, label_h, background) for label, image in images]
    width = sum(tile.width for tile in tiles) + pad * (len(tiles) + 1)
    height = max(tile.height for tile in tiles) + pad * 2
    sheet = Image.new("RGB", (width, height), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, (pad + index * (thumb_width + pad), pad))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def build_report_grid(
    images: list[tuple[str, Image.Image]],
    output: Path,
    *,
    transparent: bool,
    cell_size: tuple[int, int] = (280, 360),
    cols: int = 4,
) -> None:
    cell_w, cell_h = cell_size
    rows = (len(images) + cols - 1) // cols
    mode = "RGBA" if transparent else "RGB"
    bg = (255, 255, 255, 0) if transparent else (255, 255, 255)
    sheet = Image.new(mode, (cols * cell_w, rows * cell_h), bg)
    for index, (_, image) in enumerate(images):
        fitted = image.copy()
        scale = min(cell_w * 0.86 / fitted.width, cell_h * 0.88 / fitted.height)
        fitted = fitted.resize(
            (max(1, round(fitted.width * scale)), max(1, round(fitted.height * scale))),
            Image.Resampling.LANCZOS,
        )
        x = (index % cols) * cell_w + (cell_w - fitted.width) // 2
        y = (index // cols) * cell_h + (cell_h - fitted.height) // 2
        sheet.paste(fitted, (x, y), fitted)
    output.parent.mkdir(parents=True, exist_ok=True)
    if transparent:
        sheet.save(output)
    else:
        sheet.save(output, quality=94)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = new_session(args.model)

    selected: list[tuple[str, Image.Image]] = []
    for view_index, render_index in enumerate(parse_indices(args.indices)):
        source = args.input_dir / f"{render_index:05d}.png"
        image = Image.open(source).convert("RGBA")
        cutout = remove(
            image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=6,
        ).convert("RGBA")
        cutout = cleanup_alpha_components(cutout)
        cutout = cutout.crop(alpha_bbox(cutout))
        output = args.output_dir / f"object_A_2dgs_no_bg_view{view_index:03d}_src{render_index:05d}.png"
        cutout.save(output)
        selected.append((f"view {view_index} / render {render_index:03d}", cutout))

    build_sheet(
        selected,
        args.output_dir / "object_A_2dgs_no_background_8view_white_contact_sheet.jpg",
        "white",
        args.thumb_width,
    )
    build_sheet(
        selected,
        args.output_dir / "object_A_2dgs_no_background_8view_checker_contact_sheet.jpg",
        "checker",
        args.thumb_width,
    )
    build_report_grid(
        selected,
        args.output_dir / "object_A_2dgs_no_background_8view_report_grid_white.jpg",
        transparent=False,
    )
    build_report_grid(
        selected,
        args.output_dir / "object_A_2dgs_no_background_8view_report_grid_transparent.png",
        transparent=True,
    )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
