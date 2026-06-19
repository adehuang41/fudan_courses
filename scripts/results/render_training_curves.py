#!/usr/bin/env python3
"""Render dependency-free SVG loss curves from ACT scalar logs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def downsample(points: list[tuple[float, float]], max_points: int = 1200) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = math.ceil(len(points) / max_points)
    return points[::stride]


def polyline(points: list[tuple[float, float]], x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    width = 980
    height = 420
    left = 70
    top = 40
    inner_w = width - 110
    inner_h = height - 90
    coords = []
    for x, y in downsample(points):
        sx = left + (x - x_min) / max(x_max - x_min, 1e-9) * inner_w
        sy = top + (1.0 - (y - y_min) / max(y_max - y_min, 1e-9)) * inner_h
        coords.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(coords)


def render_svg(title: str, series: list[tuple[str, str, list[tuple[float, float]]]], path: Path) -> None:
    all_points = [point for _, _, points in series for point in points if math.isfinite(point[1])]
    if not all_points:
        return
    x_min = min(x for x, _ in all_points)
    x_max = max(x for x, _ in all_points)
    y_min = min(y for _, y in all_points)
    y_max = max(y for _, y in all_points)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= y_pad
    y_max += y_pad

    lines = []
    legend = []
    for idx, (name, color, points) in enumerate(series):
        if not points:
            continue
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
            f'points="{polyline(points, x_min, x_max, y_min, y_max)}" />'
        )
        legend.append(
            f'<g transform="translate({720},{58 + idx * 24})">'
            f'<rect width="14" height="14" fill="{color}" />'
            f'<text x="22" y="12" font-size="14" fill="#222">{name}</text>'
            f"</g>"
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="420" viewBox="0 0 980 420">
  <rect width="980" height="420" fill="white" />
  <text x="70" y="26" font-size="20" font-family="Arial, sans-serif" fill="#111">{title}</text>
  <line x1="70" y1="370" x2="940" y2="370" stroke="#333" />
  <line x1="70" y1="40" x2="70" y2="370" stroke="#333" />
  <text x="70" y="398" font-size="13" font-family="Arial, sans-serif" fill="#333">step {x_min:.0f} to {x_max:.0f}</text>
  <text x="8" y="48" font-size="13" font-family="Arial, sans-serif" fill="#333">{y_max:.4f}</text>
  <text x="8" y="370" font-size="13" font-family="Arial, sans-serif" fill="#333">{y_min:.4f}</text>
  {''.join(lines)}
  {''.join(legend)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, default=Path("/home/dechao/cv_final_pj2/results/figures"))
    args = parser.parse_args()

    output_dir = args.output_dir
    summary = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    experiment = summary["experiment"]
    scalars = read_jsonl(Path(summary["scalars_jsonl"]))

    train_loss = [(row["step"], row["loss"]) for row in scalars if row.get("phase") == "train"]
    train_action = [(row["step"], row["action_l1_loss"]) for row in scalars if row.get("phase") == "train"]
    val_loss = [(row["step"], row["val_loss"]) for row in scalars if row.get("phase") == "validation"]
    val_action = [(row["step"], row["val_action_l1_loss"]) for row in scalars if row.get("phase") == "validation"]

    figure_dir = args.figures_dir
    render_svg(
        f"{experiment}: total loss",
        [("train loss", "#1f77b4", train_loss), ("validation loss", "#d62728", val_loss)],
        figure_dir / f"{experiment}_loss_curve.svg",
    )
    render_svg(
        f"{experiment}: action L1 loss",
        [("train action L1", "#2ca02c", train_action), ("validation action L1", "#9467bd", val_action)],
        figure_dir / f"{experiment}_action_l1_curve.svg",
    )

    print(f"Wrote curves for {experiment} to {figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
