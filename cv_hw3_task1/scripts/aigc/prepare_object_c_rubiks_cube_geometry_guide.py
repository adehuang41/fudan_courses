#!/usr/bin/env python3
"""Prepare a rectified RGBA geometry guide for Object C Rubik's cube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgba", default="data/processed/object_C_rubiks_cube/rubiks_cube_rgba.png")
    parser.add_argument("--output", default="data/processed/object_C_rubiks_cube/rubiks_cube_geometry_guide_rgba.png")
    parser.add_argument("--mask", default="data/interim/object_C_rubiks_cube/rubiks_cube_geometry_guide_mask.png")
    parser.add_argument(
        "--preview",
        default="outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube/object_C_rubiks_cube_geometry_guide_preview.png",
    )
    parser.add_argument("--summary", default="data/interim/object_C_rubiks_cube/rubiks_cube_geometry_guide_summary.json")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--scale", type=int, default=3)
    return parser.parse_args()


def make_checkerboard(size: tuple[int, int], block: int = 32) -> Image.Image:
    width, height = size
    y, x = np.indices((height, width))
    board = ((x // block + y // block) % 2).astype(np.uint8)
    light = np.array([230, 230, 230], dtype=np.uint8)
    dark = np.array([185, 185, 185], dtype=np.uint8)
    arr = np.where(board[..., None] == 0, light, dark)
    return Image.fromarray(arr, mode="RGB")


def qpoint(face: list[tuple[float, float]], u: float, v: float) -> tuple[float, float]:
    p00 = np.array(face[0], dtype=np.float32)
    p10 = np.array(face[1], dtype=np.float32)
    p11 = np.array(face[2], dtype=np.float32)
    p01 = np.array(face[3], dtype=np.float32)
    p = (1.0 - u) * (1.0 - v) * p00 + u * (1.0 - v) * p10 + u * v * p11 + (1.0 - u) * v * p01
    return float(p[0]), float(p[1])


def scaled(points: list[tuple[float, float]], scale: int) -> list[tuple[int, int]]:
    return [(round(x * scale), round(y * scale)) for x, y in points]


def draw_face(
    draw: ImageDraw.ImageDraw,
    face: list[tuple[float, float]],
    sticker_rgb: tuple[int, int, int],
    scale: int,
    margin: float = 0.045,
) -> None:
    black = (5, 6, 5, 255)
    draw.polygon(scaled(face, scale), fill=black)
    for row in range(3):
        for col in range(3):
            u0 = col / 3.0 + margin
            u1 = (col + 1) / 3.0 - margin
            v0 = row / 3.0 + margin
            v1 = (row + 1) / 3.0 - margin
            cell = [qpoint(face, u0, v0), qpoint(face, u1, v0), qpoint(face, u1, v1), qpoint(face, u0, v1)]
            shade = 1.0 - 0.055 * row + 0.025 * col
            color = tuple(int(np.clip(c * shade, 0, 255)) for c in sticker_rgb) + (255,)
            draw.polygon(scaled(cell, scale), fill=color)


def make_guide(args: argparse.Namespace) -> dict[str, object]:
    size = args.canvas_size
    scale = args.scale
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Three clean projected square faces. Coordinates intentionally follow the
    # source image pose but remove lens/segmentation ambiguity for Magic123.
    top = [(225, 255), (565, 112), (875, 248), (520, 404)]
    left = [(225, 255), (520, 404), (520, 835), (225, 674)]
    right = [(520, 404), (875, 248), (875, 664), (520, 835)]

    draw_face(draw, top, (198, 66, 42), scale)
    draw_face(draw, left, (5, 130, 203), scale)
    draw_face(draw, right, (246, 221, 5), scale)

    # Slight black edge reinforcement, kept within the object silhouette.
    for face in (top, left, right):
        draw.line(scaled(face + [face[0]], scale), fill=(3, 3, 3, 255), width=round(8 * scale), joint="curve")

    if scale != 1:
        image = image.resize((size, size), Image.Resampling.LANCZOS)

    output_path = Path(args.output)
    mask_path = Path(args.mask)
    preview_path = Path(args.preview)
    summary_path = Path(args.summary)
    for path in (output_path, mask_path, preview_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    image.save(output_path)
    image.getchannel("A").save(mask_path)
    checker = make_checkerboard(image.size)
    checker.paste(image, mask=image.getchannel("A"))
    checker.save(preview_path)

    alpha = np.array(image.getchannel("A"))
    summary = {
        "source_rgba": str(args.source_rgba),
        "output": str(output_path),
        "mask": str(mask_path),
        "preview": str(preview_path),
        "method": "rectified hard-geometry Rubik's cube guide; red top, blue left, yellow right, black 3x3 grids",
        "canvas_size": [size, size],
        "alpha_nonzero_fraction": float((alpha > 0).mean()),
        "alpha_opaque_fraction": float((alpha > 240).mean()),
        "visible_faces": {
            "top": "red",
            "left": "blue",
            "right": "yellow",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(make_guide(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
