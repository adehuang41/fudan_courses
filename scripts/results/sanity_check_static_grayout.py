#!/usr/bin/env python3
"""Sanity checks for the static gray-out visual perturbation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data._utils.collate import default_collate

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.evaluate_visual_chunk_robustness import apply_condition, tensor_to_pil  # noqa: E402
from src.data.calvin_d_lerobot_dataset import DEFAULT_D_DATASET_ROOT, CalvinDDACTDataset  # noqa: E402


IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def same_value(a, b) -> bool:
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return a == b


def build_contact_sheet(path: Path, clean_batch: dict, static_gray_batch: dict, sample_indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    font = load_font(16)
    small_font = load_font(12)
    cell = 112
    label_h = 22
    tile_w = 2 * cell + 12
    tile_h = label_h + 2 * cell + 18
    pad = 18
    cols = 4
    rows = 4
    canvas = Image.new("RGB", (pad + cols * (tile_w + pad), pad + rows * (tile_h + pad)), "white")
    draw = ImageDraw.Draw(canvas)

    labels = (
        ("static clean", clean_batch["observation.images.image"]),
        ("static gray", static_gray_batch["observation.images.image"]),
        ("wrist clean", clean_batch["observation.images.wrist_image"]),
        ("wrist after", static_gray_batch["observation.images.wrist_image"]),
    )
    for idx in range(min(16, len(sample_indices))):
        col = idx % cols
        row = idx // cols
        x0 = pad + col * (tile_w + pad)
        y0 = pad + row * (tile_h + pad)
        draw.text((x0, y0), f"sample {idx + 1}: global {sample_indices[idx]}", fill=(35, 35, 35), font=small_font)
        for j, (label, tensor) in enumerate(labels):
            x = x0 + (j % 2) * (cell + 12)
            y = y0 + label_h + (j // 2) * (cell + 18)
            image = tensor_to_pil(tensor[idx]).resize((cell, cell), Image.BILINEAR)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(210, 210, 210), width=1)
            draw.text((x + 4, y + 4), label, fill=(255, 255, 255), font=font)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D_DATASET_ROOT)
    parser.add_argument("--max-episodes", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results/tables/track2_extra/static_grayout_sanity_checks.json")
    parser.add_argument("--output-image", type=Path, default=PROJECT_ROOT / "figures/track2_extra/static_grayout_sanity_16_examples.png")
    args = parser.parse_args()

    dataset = CalvinDDACTDataset(dataset_root=args.dataset_root, chunk_size=100, max_episodes=args.max_episodes)
    rng = random.Random(args.seed)
    sample_indices = sorted(rng.sample(range(len(dataset)), k=min(args.num_samples, len(dataset))))
    samples = [dataset[idx] for idx in sample_indices]
    clean_batch = default_collate(samples)
    static_gray_batch = apply_condition(clean_batch, "static_grayout", batch_index=0, noise_seed=args.seed, noise_std=0.06)

    static = static_gray_batch["observation.images.image"]
    checks = {
        "dataset_root": str(args.dataset_root),
        "max_episodes": args.max_episodes,
        "num_samples": len(sample_indices),
        "sample_indices": sample_indices,
        "static_image_changed": not torch.equal(clean_batch["observation.images.image"], static),
        "static_image_all_gray_0_5": bool(torch.allclose(static, torch.full_like(static, 0.5))),
        "static_image_min": float(static.min().item()),
        "static_image_max": float(static.max().item()),
        "static_image_mean": float(static.mean().item()),
        "static_image_value_space": "pre-normalization image tensor in [0, 1]",
        "wrist_image_max_abs_diff": max_abs_diff(
            clean_batch["observation.images.wrist_image"], static_gray_batch["observation.images.wrist_image"]
        ),
        "state_max_abs_diff": max_abs_diff(clean_batch["observation.state"], static_gray_batch["observation.state"]),
        "action_max_abs_diff": max_abs_diff(clean_batch["action"], static_gray_batch["action"]),
        "action_is_pad_equal": same_value(clean_batch["action_is_pad"], static_gray_batch["action_is_pad"]),
        "episode_index_equal": same_value(clean_batch["episode_index"], static_gray_batch["episode_index"]),
        "frame_index_equal": same_value(clean_batch["frame_index"], static_gray_batch["frame_index"]),
        "task_index_equal": same_value(clean_batch["task_index"], static_gray_batch["task_index"]),
        "task_text_equal": same_value(clean_batch["task"], static_gray_batch["task"]),
        "environment_equal": same_value(clean_batch["environment"], static_gray_batch["environment"]),
        "canonical_task_id_present": "canonical_task_id" in clean_batch,
    }
    if "canonical_task_id" in clean_batch:
        checks["canonical_task_id_equal"] = same_value(
            clean_batch["canonical_task_id"], static_gray_batch["canonical_task_id"]
        )

    pass_fields = {
        "static_image_changed": checks["static_image_changed"],
        "static_image_all_gray_0_5": checks["static_image_all_gray_0_5"],
        "wrist_image_unchanged": checks["wrist_image_max_abs_diff"] == 0.0,
        "state_unchanged": checks["state_max_abs_diff"] == 0.0,
        "action_unchanged": checks["action_max_abs_diff"] == 0.0,
        "action_is_pad_equal": checks["action_is_pad_equal"],
        "episode_index_equal": checks["episode_index_equal"],
        "frame_index_equal": checks["frame_index_equal"],
        "task_index_equal": checks["task_index_equal"],
        "task_text_equal": checks["task_text_equal"],
        "environment_equal": checks["environment_equal"],
    }
    checks["pass_fields"] = pass_fields
    checks["status"] = "ok" if all(bool(v) for v in pass_fields.values()) else "failed"

    build_contact_sheet(args.output_image, clean_batch, static_gray_batch, sample_indices)
    checks["contact_sheet"] = str(args.output_image)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, ensure_ascii=False)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0 if checks["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
