#!/usr/bin/env python3
"""Prepare a stringless RGBA input for Object C green yoyo body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--margin-fraction", type=float, default=0.07)
    parser.add_argument("--green-red-margin", type=float, default=18.0)
    parser.add_argument("--green-blue-margin", type=float, default=14.0)
    parser.add_argument("--min-green", type=float, default=85.0)
    parser.add_argument("--min-saturation", type=float, default=35.0)
    parser.add_argument("--dilate-iterations", type=int, default=4)
    parser.add_argument("--edge-clean-iterations", type=int, default=7)
    return parser.parse_args()


def make_checkerboard(size: tuple[int, int], block: int = 32) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(0, height, block):
        for x in range(0, width, block):
            color = (232, 232, 232) if ((x // block + y // block) % 2 == 0) else (190, 190, 190)
            draw.rectangle((x, y, x + block - 1, y + block - 1), fill=color)
    return image


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        raise RuntimeError("green body mask is empty")
    sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == int(np.argmax(sizes) + 1)


def square_crop_bbox(mask: np.ndarray, width: int, height: int, margin_fraction: float) -> list[int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise RuntimeError("foreground mask is empty")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    span = max(x1 - x0, y1 - y0)
    margin = int(round(span * margin_fraction))
    x0 = max(0, x0 - margin)
    x1 = min(width, x1 + margin)
    y0 = max(0, y0 - margin)
    y1 = min(height, y1 + margin)

    side = max(x1 - x0, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = cx - side // 2
    y0 = cy - side // 2
    x1 = x0 + side
    y1 = y0 + side

    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    return [int(x0), int(y0), int(x1), int(y1)]


def clean_edge_colors(rgb: np.ndarray, alpha: np.ndarray, edge_clean_iterations: int) -> np.ndarray:
    foreground = alpha > 0
    core = ndi.binary_erosion(alpha > 245, iterations=edge_clean_iterations)
    if not core.any():
        return rgb

    _, nearest = ndi.distance_transform_edt(~core, return_indices=True)
    cleaned = rgb.copy()
    edge = foreground & ~core
    cleaned[edge] = rgb[nearest[0][edge], nearest[1][edge]]
    return cleaned


def prepare_rgba(args: argparse.Namespace) -> dict[str, object]:
    source = Image.open(args.input).convert("RGB")
    rgb = np.asarray(source)
    arr = rgb.astype(np.float32)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    saturation = arr.max(axis=2) - arr.min(axis=2)

    # The string is beige and low-saturation; the yoyo body is the largest saturated green component.
    green_body = (
        (green > args.min_green)
        & (green > red + args.green_red_margin)
        & (green > blue + args.green_blue_margin)
        & (saturation > args.min_saturation)
        & (red < 210.0)
        & (blue < 190.0)
    )
    green_body = ndi.binary_opening(green_body, structure=np.ones((5, 5), dtype=bool))
    green_body = ndi.binary_closing(green_body, structure=np.ones((35, 35), dtype=bool))
    body = largest_component(green_body)
    bbox_before_square = square_crop_bbox(body, source.width, source.height, 0.0)

    body = ndi.binary_fill_holes(body)
    body = ndi.binary_closing(body, structure=np.ones((45, 45), dtype=bool))
    body = ndi.binary_fill_holes(body)
    body = ndi.binary_dilation(body, iterations=args.dilate_iterations)

    crop_bbox = square_crop_bbox(body, source.width, source.height, args.margin_fraction)
    alpha = Image.fromarray((body.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.2)
    )
    alpha_arr = np.asarray(alpha)
    rgb_clean = clean_edge_colors(rgb, alpha_arr, args.edge_clean_iterations)
    rgba = Image.fromarray(np.dstack([rgb_clean, alpha_arr]), mode="RGBA")
    crop = rgba.crop(tuple(crop_bbox)).resize(
        (args.canvas_size, args.canvas_size), Image.Resampling.LANCZOS
    )

    output_path = Path(args.output)
    mask_path = Path(args.mask)
    preview_path = Path(args.preview)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    crop.save(output_path)
    crop.getchannel("A").save(mask_path)
    preview = make_checkerboard(crop.size)
    preview.paste(crop, mask=crop.getchannel("A"))
    preview.save(preview_path)

    crop_alpha = np.asarray(crop.getchannel("A"))
    summary = {
        "input": str(args.input),
        "output": str(output_path),
        "mask": str(mask_path),
        "preview": str(preview_path),
        "method": "largest saturated-green connected component, hole fill, edge color cleanup, square crop",
        "source_size": [source.width, source.height],
        "body_bbox_xyxy_before_square": bbox_before_square,
        "square_crop_xyxy": crop_bbox,
        "canvas_size": [args.canvas_size, args.canvas_size],
        "alpha_nonzero_fraction": float((crop_alpha > 0).mean()),
        "alpha_opaque_fraction": float((crop_alpha > 240).mean()),
        "green_red_margin": args.green_red_margin,
        "green_blue_margin": args.green_blue_margin,
        "min_green": args.min_green,
        "min_saturation": args.min_saturation,
        "margin_fraction": args.margin_fraction,
        "dilate_iterations": args.dilate_iterations,
        "edge_clean_iterations": args.edge_clean_iterations,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    summary = prepare_rgba(parse_args())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
