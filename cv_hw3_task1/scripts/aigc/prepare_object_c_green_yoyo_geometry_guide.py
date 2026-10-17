#!/usr/bin/env python3
"""Prepare a geometry-biased RGBA guide for Object C green yoyo body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage as ndi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgba", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--object-width", type=float, default=0.84)
    parser.add_argument("--object-height", type=float, default=0.54)
    parser.add_argument("--waist-pinch", type=float, default=0.14)
    parser.add_argument("--waist-band", type=float, default=0.052)
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


def sample_green_palette(source_rgba: Image.Image) -> dict[str, list[int]]:
    rgba = np.asarray(source_rgba.convert("RGBA"))
    alpha = rgba[..., 3] > 220
    rgb = rgba[..., :3].astype(np.float32)
    green = alpha & (rgb[..., 1] > rgb[..., 0] + 20) & (rgb[..., 1] > rgb[..., 2] + 15)
    if not green.any():
        green = alpha
    samples = rgb[green]
    base = np.percentile(samples, 55, axis=0)
    light = np.percentile(samples, 82, axis=0)
    dark = np.percentile(samples, 20, axis=0)
    return {
        "base": np.clip(base, 0, 255).round().astype(int).tolist(),
        "light": np.clip(light, 0, 255).round().astype(int).tolist(),
        "dark": np.clip(dark, 0, 255).round().astype(int).tolist(),
    }


def make_guide(args: argparse.Namespace) -> dict[str, object]:
    source_rgba = Image.open(args.source_rgba).convert("RGBA")
    palette = sample_green_palette(source_rgba)
    size = args.canvas_size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = (size - 1) / 2.0
    half_height = size * args.object_height / 2.0
    half_width = size * args.object_width / 2.0
    t = (yy - cy) / half_height

    inside_y = np.abs(t) <= 1.0
    superellipse_width = half_width * np.maximum(0.0, 1.0 - np.abs(t) ** 4) ** 0.25
    waist = np.exp(-(t / args.waist_band) ** 2)
    local_half_width = superellipse_width * (1.0 - args.waist_pinch * waist)
    x_norm = np.abs(xx - cx) / np.maximum(local_half_width, 1.0)
    mask = inside_y & (x_norm <= 1.0)

    alpha = ndi.gaussian_filter(mask.astype(np.float32), sigma=1.35)
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    base = np.array(palette["base"], dtype=np.float32)
    light = np.array(palette["light"], dtype=np.float32)
    dark = np.array(palette["dark"], dtype=np.float32)

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    rgb[:] = base

    vertical = 1.02 - 0.18 * (t + 0.15)
    edge_shadow = 1.0 - 0.22 * np.clip(x_norm, 0.0, 1.0) ** 2
    center_shadow = 1.0 - 0.46 * waist
    top_lip = 1.0 + 0.20 * np.exp(-((t + 0.075) / 0.028) ** 2)
    bottom_lip = 1.0 + 0.15 * np.exp(-((t - 0.075) / 0.035) ** 2)
    highlight = 1.0 + 0.26 * np.exp(-(((xx - cx + size * 0.19) / (size * 0.28)) ** 2 + ((yy - cy + size * 0.18) / (size * 0.13)) ** 2))
    shade = vertical * edge_shadow * center_shadow * top_lip * bottom_lip * highlight
    shade = np.clip(shade, 0.35, 1.35)

    rgb = rgb * shade[..., None]
    rgb = rgb * 0.88 + light[None, None, :] * np.clip((highlight - 1.0) / 0.28, 0.0, 1.0)[..., None] * 0.12
    groove_core = np.exp(-(t / (args.waist_band * 0.56)) ** 2)
    rgb = rgb * (1.0 - 0.34 * groove_core[..., None]) + dark[None, None, :] * (0.34 * groove_core[..., None])

    rng = np.random.default_rng(42)
    noise = ndi.gaussian_filter(rng.normal(0.0, 1.0, (size, size)), sigma=1.2)
    rgb *= 1.0 + 0.012 * noise[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Keep RGB green at transparent borders to avoid white halos during Magic123 loading.
    rgba = np.dstack([rgb, alpha])
    image = Image.fromarray(rgba, mode="RGBA")

    output_path = Path(args.output)
    mask_path = Path(args.mask)
    preview_path = Path(args.preview)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    image.save(output_path)
    image.getchannel("A").save(mask_path)
    preview = make_checkerboard(image.size)
    preview.paste(image, mask=image.getchannel("A"))
    preview.save(preview_path)

    summary = {
        "source_rgba": str(args.source_rgba),
        "output": str(output_path),
        "mask": str(mask_path),
        "preview": str(preview_path),
        "method": "geometry-biased side-view yoyo guide with one central waist groove; source image supplies green palette only",
        "canvas_size": [size, size],
        "object_width": args.object_width,
        "object_height": args.object_height,
        "waist_pinch": args.waist_pinch,
        "waist_band": args.waist_band,
        "palette": palette,
        "alpha_nonzero_fraction": float((alpha > 0).mean()),
        "alpha_opaque_fraction": float((alpha > 240).mean()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    print(json.dumps(make_guide(parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
