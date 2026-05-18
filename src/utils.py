"""Shared helpers for hw2_task2."""
from __future__ import annotations

import colorsys
import json
from pathlib import Path

import yaml

VISDRONE_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color from a track id (golden-ratio hue scan)."""
    if track_id is None or track_id < 0:
        return (200, 200, 200)
    h = ((track_id * 0.61803398875) % 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
