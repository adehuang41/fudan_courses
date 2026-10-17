#!/usr/bin/env python3
"""Prepare RGBA input for the scrambled Rubik's cube by Keqs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/object_C_rubiks_cube_by_keqs/Rubiks_cube_by_keqs.jpg")
    parser.add_argument("--output", default="data/processed/object_C_rubiks_cube_by_keqs/rubiks_cube_keqs_rgba.png")
    parser.add_argument("--mask", default="data/interim/object_C_rubiks_cube_by_keqs/rubiks_cube_keqs_mask.png")
    parser.add_argument(
        "--preview",
        default="outputs/previews/aigc_assets/object_C_image_to_3d/rubiks_cube_by_keqs/object_C_rubiks_cube_keqs_rgba_preview.png",
    )
    parser.add_argument("--summary", default="data/interim/object_C_rubiks_cube_by_keqs/rubiks_cube_keqs_rgba_summary.json")
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--margin-fraction", type=float, default=0.075)
    parser.add_argument("--saturation-threshold", type=float, default=35.0)
    parser.add_argument("--white-sticker-value", type=float, default=210.0)
    parser.add_argument("--background-value", type=float, default=214.0)
    parser.add_argument("--background-saturation", type=float, default=54.0)
    parser.add_argument("--dilate-iterations", type=int, default=3)
    parser.add_argument("--close-size", type=int, default=21)
    parser.add_argument("--solid-envelope", action="store_true")
    parser.add_argument("--envelope-close-size", type=int, default=111)
    parser.add_argument("--envelope-fill", choices=["nearest", "black"], default="nearest")
    parser.add_argument(
        "--recover-keqs-white-stickers",
        action="store_true",
        help="Recover the two border-connected white stickers in the Keqs source image.",
    )
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


def border_connected(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)

    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]
    if border_labels.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, border_labels)


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        raise RuntimeError("foreground mask is empty")
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


def clean_edge_colors(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    foreground = alpha > 0
    core = alpha > 245
    if not core.any():
        return rgb
    _, nearest = ndi.distance_transform_edt(~core, return_indices=True)
    cleaned = rgb.copy()
    edge = foreground & ~core
    cleaned[edge] = rgb[nearest[0][edge], nearest[1][edge]]
    return cleaned


def fill_added_envelope_colors(
    rgb: np.ndarray, base_mask: np.ndarray, envelope: np.ndarray, fill_mode: str
) -> np.ndarray:
    added = envelope & ~base_mask
    if not added.any():
        return rgb
    if fill_mode == "black":
        filled = rgb.copy()
        filled[added] = np.array([6, 6, 6], dtype=rgb.dtype)
        return filled

    _, nearest = ndi.distance_transform_edt(~base_mask, return_indices=True)
    filled = rgb.copy()
    filled[added] = rgb[nearest[0][added], nearest[1][added]]
    return filled


def make_solid_envelope(mask: np.ndarray, close_size: int) -> np.ndarray:
    envelope = ndi.binary_closing(mask, structure=np.ones((close_size, close_size), dtype=bool))
    envelope = ndi.binary_fill_holes(envelope)
    return envelope


def recover_keqs_white_stickers(size: tuple[int, int]) -> np.ndarray:
    """Recover white stickers that touch the white background in this source photo."""
    width, height = size
    if (width, height) != (1600, 1600):
        raise RuntimeError("Keqs white-sticker recovery is calibrated for the 1600x1600 source image")

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    # Top-left/top-back white sticker; it is connected to the white background in the photo.
    draw.polygon([(526, 226), (716, 162), (850, 227), (669, 337)], fill=255)
    # Right-face middle white sticker; it is also border-connected to the white background.
    draw.polygon([(1308, 613), (1483, 569), (1481, 824), (1295, 866)], fill=255)
    return np.asarray(mask) > 0


def build_mask(rgb: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    mean = rgb.astype(np.float32).mean(axis=2)

    candidate_background = (value > args.background_value) & (saturation < args.background_saturation)
    background = border_connected(candidate_background)

    colored_or_black_grid = ((saturation > args.saturation_threshold) & (value > 35.0)) | (value < 80.0)
    enclosed_white_stickers = (~background) & (value > args.white_sticker_value) & (saturation < args.background_saturation)
    foreground_core = colored_or_black_grid | enclosed_white_stickers

    foreground_core = ndi.binary_opening(foreground_core, structure=np.ones((3, 3), dtype=bool))
    foreground_core = ndi.binary_closing(
        foreground_core, structure=np.ones((args.close_size, args.close_size), dtype=bool)
    )
    body = largest_component(foreground_core)
    body = ndi.binary_fill_holes(body)
    body = ndi.binary_closing(body, structure=np.ones((args.close_size, args.close_size), dtype=bool))
    body = ndi.binary_fill_holes(body)
    body = ndi.binary_dilation(body, iterations=args.dilate_iterations)

    # Exclude neutral low-saturation table shadows below the cube while keeping white stickers.
    shadow_like = (saturation < 26.0) & (value < args.white_sticker_value) & (mean > 90.0)
    body = body & ~shadow_like
    body = ndi.binary_closing(body, structure=np.ones((args.close_size, args.close_size), dtype=bool))
    body = ndi.binary_fill_holes(body)

    recovered_white_stickers = np.zeros_like(body, dtype=bool)
    if args.recover_keqs_white_stickers:
        recovered_white_stickers = recover_keqs_white_stickers((rgb.shape[1], rgb.shape[0]))
        body = body | recovered_white_stickers
        body = ndi.binary_closing(body, structure=np.ones((7, 7), dtype=bool))

    return body, {
        "candidate_background_fraction": float(candidate_background.mean()),
        "border_background_fraction": float(background.mean()),
        "foreground_core_fraction": float(foreground_core.mean()),
        "shadow_like_fraction": float(shadow_like.mean()),
        "recovered_white_sticker_fraction": float(recovered_white_stickers.mean()),
    }


def prepare_rgba(args: argparse.Namespace) -> dict[str, object]:
    source = Image.open(args.input).convert("RGB")
    rgb = np.asarray(source)
    base_mask, mask_stats = build_mask(rgb, args)
    mask = make_solid_envelope(base_mask, args.envelope_close_size) if args.solid_envelope else base_mask
    crop_bbox = square_crop_bbox(mask, source.width, source.height, args.margin_fraction)

    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.GaussianBlur(radius=1.1)
    )
    alpha_arr = np.asarray(alpha)
    rgb_for_alpha = (
        fill_added_envelope_colors(rgb, base_mask, mask, args.envelope_fill)
        if args.solid_envelope
        else rgb
    )
    rgb_clean = clean_edge_colors(rgb_for_alpha, alpha_arr)
    rgba = Image.fromarray(np.dstack([rgb_clean, alpha_arr]), mode="RGBA")
    crop = rgba.crop(tuple(crop_bbox)).resize(
        (args.canvas_size, args.canvas_size), Image.Resampling.LANCZOS
    )

    output_path = Path(args.output)
    mask_path = Path(args.mask)
    preview_path = Path(args.preview)
    summary_path = Path(args.summary)
    for path in (output_path, mask_path, preview_path, summary_path):
        path.parent.mkdir(parents=True, exist_ok=True)

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
        "method": "border-connected white background removal; enclosed white stickers preserved; neutral shadow suppression",
        "source_size": [source.width, source.height],
        "square_crop_xyxy": crop_bbox,
        "canvas_size": [args.canvas_size, args.canvas_size],
        "alpha_nonzero_fraction": float((crop_alpha > 0).mean()),
        "alpha_opaque_fraction": float((crop_alpha > 240).mean()),
        "margin_fraction": args.margin_fraction,
        "saturation_threshold": args.saturation_threshold,
        "white_sticker_value": args.white_sticker_value,
        "background_value": args.background_value,
        "background_saturation": args.background_saturation,
        "dilate_iterations": args.dilate_iterations,
        "close_size": args.close_size,
        "solid_envelope": args.solid_envelope,
        "envelope_close_size": args.envelope_close_size,
        "envelope_fill": args.envelope_fill,
        "recover_keqs_white_stickers": args.recover_keqs_white_stickers,
        "envelope_added_fraction": float((mask & ~base_mask).mean()),
        "mask_stats": mask_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(prepare_rgba(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
