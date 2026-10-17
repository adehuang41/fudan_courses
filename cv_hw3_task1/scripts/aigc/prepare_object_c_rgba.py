#!/usr/bin/env python3
"""Prepare a square RGBA Object C input."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/object_C_green_container/container_raw.jpg")
    parser.add_argument("--output", default="data/processed/object_C_green_container/container_rgba.png")
    parser.add_argument("--mask", default="data/interim/object_C_green_container/container_mask.png")
    parser.add_argument("--preview", default="outputs/previews/aigc_assets/object_C_image_to_3d/green_container/object_C_green_container_rgba_preview.png")
    parser.add_argument("--summary", default="data/interim/object_C_green_container/container_rgba_summary.json")
    parser.add_argument("--method", choices=["classical", "rembg"], default="rembg")
    parser.add_argument("--rembg-model", default="u2netp")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--mean-threshold", type=float, default=210.0)
    parser.add_argument("--saturation-threshold", type=float, default=18.0)
    parser.add_argument("--row-fraction", type=float, default=0.055)
    parser.add_argument("--col-fraction", type=float, default=0.08)
    parser.add_argument("--margin-fraction", type=float, default=0.08)
    return parser.parse_args()


def largest_contiguous_run(values: np.ndarray) -> tuple[int, int]:
    if values.size == 0:
        raise ValueError("empty foreground index set")
    splits = np.where(np.diff(values) > 1)[0] + 1
    runs = np.split(values, splits)
    best = max(runs, key=len)
    return int(best[0]), int(best[-1]) + 1


def build_initial_mask(rgb: np.ndarray, mean_threshold: float, saturation_threshold: float) -> np.ndarray:
    arr = rgb.astype(np.float32)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    mean = arr.mean(axis=2)
    saturation = arr.max(axis=2) - arr.min(axis=2)
    greenish = (
        (green > red + 5.0)
        & (green > blue + 3.0)
        & (saturation > saturation_threshold)
        & (mean < mean_threshold)
    )
    dark_edges = mean < 55.0
    colored_highlight = (saturation > 55.0) & (mean < 210.0)
    return greenish | dark_edges | colored_highlight


def refine_mask(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    image = image.filter(ImageFilter.MedianFilter(size=5))
    image = image.filter(ImageFilter.MaxFilter(size=11))
    image = image.filter(ImageFilter.MinFilter(size=9))
    image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    return np.array(image, dtype=np.uint8)


def make_checkerboard(size: tuple[int, int], block: int = 32) -> Image.Image:
    width, height = size
    y, x = np.indices((height, width))
    board = ((x // block + y // block) % 2).astype(np.uint8)
    light = np.array([230, 230, 230], dtype=np.uint8)
    dark = np.array([185, 185, 185], dtype=np.uint8)
    arr = np.where(board[..., None] == 0, light, dark)
    return Image.fromarray(arr, mode="RGB")


def load_rembg_rgba(input_path: Path, model: str) -> tuple[Image.Image, dict[str, str]]:
    from rembg import new_session, remove

    session = new_session(model)
    output = remove(input_path.read_bytes(), session=session)
    rgba = Image.open(BytesIO(output)).convert("RGBA")
    return rgba, {"method": "rembg", "model": model}


def load_classical_rgba(
    input_path: Path,
    mean_threshold: float,
    saturation_threshold: float,
    row_fraction: float,
    col_fraction: float,
    margin_fraction: float,
) -> tuple[Image.Image, dict[str, object]]:
    image = Image.open(input_path).convert("RGB")
    rgb = np.array(image)
    initial = build_initial_mask(rgb, mean_threshold, saturation_threshold)

    rows = np.where(initial.mean(axis=1) > row_fraction)[0]
    if rows.size == 0:
        ys, xs = np.where(initial)
        if ys.size == 0:
            raise RuntimeError("foreground mask is empty")
        y0, y1 = int(ys.min()), int(ys.max()) + 1
    else:
        y0, y1 = largest_contiguous_run(rows)

    row_mask = initial[y0:y1, :]
    cols = np.where(row_mask.mean(axis=0) > col_fraction)[0]
    if cols.size == 0:
        ys, xs = np.where(initial)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
    else:
        x0, x1 = largest_contiguous_run(cols)

    height, width = initial.shape
    span = max(y1 - y0, x1 - x0)
    margin = int(round(span * margin_fraction))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)

    bbox_mask = np.zeros_like(initial, dtype=bool)
    bbox_mask[y0:y1, x0:x1] = initial[y0:y1, x0:x1]
    alpha = refine_mask(bbox_mask)
    rgba = Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    return rgba, {
        "method": "classical color/brightness threshold with bbox crop; no model download",
        "bbox_xyxy": [x0, y0, x1, y1],
        "mean_threshold": mean_threshold,
        "saturation_threshold": saturation_threshold,
        "row_fraction": row_fraction,
        "col_fraction": col_fraction,
        "margin_fraction": margin_fraction,
    }


def square_from_alpha(rgba: Image.Image, canvas_size: int, margin_fraction: float) -> tuple[Image.Image, list[int]]:
    alpha = np.array(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 8)
    if ys.size == 0:
        raise RuntimeError("alpha mask is empty")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    span = max(x1 - x0, y1 - y0)
    margin = int(round(span * margin_fraction))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(rgba.width, x1 + margin)
    y1 = min(rgba.height, y1 + margin)

    crop = rgba.crop((x0, y0, x1, y1))
    side = max(crop.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    square = square.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
    return square, [x0, y0, x1, y1]


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    mask_path = Path(args.mask)
    preview_path = Path(args.preview)
    summary_path = Path(args.summary)

    source_image = Image.open(input_path)
    if args.method == "rembg":
        rgba, method_info = load_rembg_rgba(input_path, args.rembg_model)
    else:
        rgba, method_info = load_classical_rgba(
            input_path,
            args.mean_threshold,
            args.saturation_threshold,
            args.row_fraction,
            args.col_fraction,
            args.margin_fraction,
        )
    square, crop_bbox = square_from_alpha(rgba, args.canvas_size, args.margin_fraction)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    square.save(output_path)
    square.getchannel("A").save(mask_path)

    checker = make_checkerboard(square.size)
    checker.paste(square, mask=square.getchannel("A"))
    checker.save(preview_path)

    alpha_arr = np.array(square.getchannel("A"))
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "mask": str(mask_path),
        "preview": str(preview_path),
        "method": method_info,
        "source_size": source_image.size,
        "canvas_size": square.size,
        "crop_bbox_xyxy": crop_bbox,
        "alpha_nonzero_fraction": float((alpha_arr > 0).mean()),
        "alpha_opaque_fraction": float((alpha_arr > 240).mean()),
        "margin_fraction": args.margin_fraction,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
