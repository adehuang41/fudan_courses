#!/usr/bin/env python3
"""Create clean Object A book textures from 2DGS render crops."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


PROJECT_ROOT = Path("/home/dechao/cv_final_pj")
TEXTURE_DIR = PROJECT_ROOT / "outputs/renders/blender_fusion/task1_render_level_compositing/textures"
OUTPUT_DIR = TEXTURE_DIR / "object_A_clean_retopo"


def crop_resize(src: Path, box: tuple[int, int, int, int], dst: Path, size: tuple[int, int]) -> None:
    image = Image.open(src).convert("RGB")
    crop = image.crop(box)
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dst)
    print(f"wrote {dst} from {src.name} box={box} size={size}")


def make_inner_page(dst: Path, size: tuple[int, int]) -> None:
    width, height = size
    image = Image.new("RGB", size, (229, 218, 191))
    draw = ImageDraw.Draw(image)
    margin_x = int(width * 0.12)
    margin_y = int(height * 0.10)
    line_h = max(2, int(height * 0.007))
    gap = max(8, int(height * 0.030))
    y = margin_y
    while y < height - margin_y:
        text_w = int(width * (0.55 + 0.28 * ((y // gap) % 3) / 2))
        draw.rounded_rectangle(
            (margin_x, y, min(width - margin_x, margin_x + text_w), y + line_h),
            radius=1,
            fill=(115, 96, 78),
        )
        y += gap
    draw.rectangle((0, 0, max(2, width // 45), height), fill=(178, 42, 58))
    image = image.filter(ImageFilter.GaussianBlur(radius=0.15))
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)
    print(f"wrote {dst} procedural inner page size={size}")


def main() -> None:
    front = TEXTURE_DIR / "object_A_book_front_render_00085_crop.png"
    back = TEXTURE_DIR / "object_A_book_back_render_00020_crop.png"
    pages = TEXTURE_DIR / "object_A_book_pages_render_00120_crop.png"

    crop_resize(front, (7, 4, 188, 333), OUTPUT_DIR / "object_A_front_clean.png", (512, 900))
    crop_resize(back, (3, 76, 171, 324), OUTPUT_DIR / "object_A_back_clean.png", (512, 900))
    crop_resize(pages, (0, 0, 47, 423), OUTPUT_DIR / "object_A_pages_edge_clean.png", (256, 900))
    make_inner_page(OUTPUT_DIR / "object_A_inner_page_clean.png", (512, 900))


if __name__ == "__main__":
    main()
