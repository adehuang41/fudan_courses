#!/usr/bin/env python3
"""Verify Step3 training runs and generate report artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from src.data.calvin_filtered_dataset import (  # noqa: E402
    CalvinACTTrainingDataset,
    EpisodeRecord,
    _decode_video_frame,
    compute_records_stats,
)


RUNS = {
    "B_only_act_baseline": Path("/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline"),
    "ABC_mixed_act_baseline": Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline"),
}
TABLE_DIR = PROJECT_ROOT / "results/tables"
FIGURE_DIR = PROJECT_ROOT / "results/figures"
QUAL_FIGURE_DIR = FIGURE_DIR / "qualitative"
FAILURE_CASE_DIR = PROJECT_ROOT / "results/failure_cases"
QUAL_VIDEO_DIR = Path("/data/dechao/cv_final_pj2/outputs/qualitative")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_split_csv(path: Path, split: str | None = None) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            records.append(
                EpisodeRecord(
                    shard=row["shard"],
                    episode_index=int(row["episode_index"]),
                    original_frame_idx_min=int(row["original_frame_idx_min"]),
                    original_frame_idx_max=int(row["original_frame_idx_max"]),
                    environment=row["environment"],
                    num_frames=int(row["num_frames"]),
                    task=row["task"],
                )
            )
    return records


def finite_scan(rows: list[dict]) -> tuple[bool, str]:
    checked = 0
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                checked += 1
                if not math.isfinite(float(value)):
                    return False, f"non-finite {key} at row step={row.get('step')}: {value}"
    return True, f"checked {checked} numeric values"


def build_policy_config(policy_actual: dict, device: str) -> ACTConfig:
    cfg = dict(policy_actual)
    cfg["device"] = device
    cfg["input_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in cfg["input_features"].items()
    }
    cfg["output_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in cfg["output_features"].items()
    }
    cfg.pop("type", None)
    return ACTConfig(**cfg)


def normalize_batch(batch: dict, stats: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value

    for key in (OBS_STATE, ACTION, "observation.images.image", "observation.images.wrist_image"):
        if key not in out or key not in stats:
            continue
        mean = stats[key]["mean"].to(device)
        std = stats[key]["std"].to(device).clamp_min(1e-6)
        out[key] = (out[key] - mean) / std
    return out


def load_policy(run_dir: Path, checkpoint_path: Path, device: torch.device) -> tuple[ACTPolicy, ACTConfig, dict]:
    config_yaml = read_yaml(run_dir / "config.yaml")
    policy_config = build_policy_config(config_yaml["policy_actual"], str(device))
    policy = ACTPolicy(policy_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, policy_config, checkpoint


def checkpoint_status(path: Path, expected_step: int | None = None) -> dict:
    out = {
        "path": str(path),
        "exists": path.exists(),
        "size_mb": path.stat().st_size / 1024 / 1024 if path.exists() else 0.0,
        "readable": False,
        "step": "",
        "error": "",
    }
    if not path.exists():
        out["error"] = "missing"
        return out
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        out["readable"] = "policy_state_dict" in ckpt
        out["step"] = ckpt.get("step", "")
        if expected_step is not None and ckpt.get("step") != expected_step:
            out["readable"] = False
            out["error"] = f"expected step {expected_step}, got {ckpt.get('step')}"
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def color_for(name: str) -> tuple[int, int, int]:
    return {
        "B_only_act_baseline": (41, 98, 255),
        "ABC_mixed_act_baseline": (220, 88, 42),
        "A": (52, 141, 83),
        "B": (41, 98, 255),
        "C": (220, 88, 42),
    }.get(name, (30, 30, 30))


def downsample(points: list[tuple[float, float]], max_points: int = 1200) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = int(math.ceil(len(points) / max_points))
    return points[::stride]


def draw_line_chart(
    path: Path,
    title: str,
    series: list[tuple[str, list[tuple[float, float]], tuple[int, int, int]]],
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 720
    left, right, top, bottom = 90, 40, 60, 95
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    points_all = [(x, y) for _, pts, _ in series for x, y in pts if math.isfinite(y)]
    if not points_all:
        img.save(path)
        return
    x_min, x_max = min(x for x, _ in points_all), max(x for x, _ in points_all)
    y_min, y_max = min(y for _, y in points_all), max(y for _, y in points_all)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    y_min -= y_pad
    y_max += y_pad
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xy(point):
        x, y = point
        sx = left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w
        sy = top + (1.0 - (y - y_min) / max(y_max - y_min, 1e-9)) * plot_h
        return int(sx), int(sy)

    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(40, 40, 40))
    draw.text((left, 20), title, fill=(0, 0, 0), font=font)
    draw.text((left, height - 42), f"step {x_min:.0f} to {x_max:.0f}", fill=(40, 40, 40), font=font)
    draw.text((8, top), f"{y_max:.4f}", fill=(40, 40, 40), font=font)
    draw.text((8, top + plot_h - 10), f"{y_min:.4f}", fill=(40, 40, 40), font=font)
    draw.text((8, height // 2), ylabel, fill=(40, 40, 40), font=font)

    legend_x, legend_y = width - 360, 30
    for idx, (name, pts, color) in enumerate(series):
        pts = downsample([(x, y) for x, y in pts if math.isfinite(y)])
        if len(pts) >= 2:
            draw.line([xy(point) for point in pts], fill=color, width=3)
        y0 = legend_y + idx * 24
        draw.rectangle((legend_x, y0, legend_x + 14, y0 + 14), fill=color)
        draw.text((legend_x + 20, y0), name, fill=(0, 0, 0), font=font)
    img.save(path)


def draw_bar_chart(path: Path, title: str, rows: list[tuple[str, float, tuple[int, int, int]]], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 620
    left, right, top, bottom = 90, 40, 70, 110
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    max_value = max(value for _, value, _ in rows) * 1.15
    draw.text((left, 25), title, fill=(0, 0, 0), font=font)
    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(40, 40, 40), width=1)
    draw.line((left, top, left, top + plot_h), fill=(40, 40, 40), width=1)
    bar_w = plot_w / max(len(rows), 1) * 0.56
    for idx, (name, value, color) in enumerate(rows):
        cx = left + plot_w / len(rows) * (idx + 0.5)
        h = value / max(max_value, 1e-9) * plot_h
        x0, x1 = int(cx - bar_w / 2), int(cx + bar_w / 2)
        y0, y1 = int(top + plot_h - h), int(top + plot_h)
        draw.rectangle((x0, y0, x1, y1), fill=color)
        draw.text((x0, y0 - 18), f"{value:.4f}", fill=(0, 0, 0), font=font)
        draw.text((x0, y1 + 12), name, fill=(0, 0, 0), font=font)
    draw.text((8, height // 2), ylabel, fill=(40, 40, 40), font=font)
    img.save(path)


def evaluate_per_env(run_dir: Path, device: torch.device) -> list[dict]:
    summary = read_json(run_dir / "metrics_summary.json")
    train_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="train")
    val_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="val")
    stats = compute_records_stats(train_records, dataset_root=Path(read_yaml(run_dir / "config.yaml")["train"]["dataset"]["dataset_root"]))
    policy, policy_config, _ = load_policy(run_dir, Path(summary["best_val_action_l1_checkpoint"]), device)
    policy.train()

    bn_states = []
    for module in policy.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_states.append((module, module.training))
            module.eval()

    rows = []
    try:
        for env in sorted({record.environment for record in val_records}):
            records = [record for record in val_records if record.environment == env]
            dataset = CalvinACTTrainingDataset(
                split_name=summary["split"],
                dataset_root=Path(read_yaml(run_dir / "config.yaml")["train"]["dataset"]["dataset_root"]),
                split_manifest=Path(read_yaml(run_dir / "config.yaml")["train"]["dataset"]["split_manifest"]),
                chunk_size=policy_config.chunk_size,
                image_keys=list(policy_config.image_features),
                episode_records=records,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=False,
            )
            weighted_loss = 0.0
            weighted_l1 = 0.0
            samples = 0
            batches = 0
            start = time.perf_counter()
            with torch.no_grad():
                for batch in dataloader:
                    bs = int(batch["action"].shape[0])
                    batch = normalize_batch(batch, stats, device)
                    loss, output_dict = policy.forward(batch)
                    loss_value = float(loss.detach().cpu().item())
                    l1_value = float(output_dict.get("l1_loss", math.nan))
                    if not math.isfinite(loss_value) or not math.isfinite(l1_value):
                        raise RuntimeError(f"Non-finite per-env metric for {run_dir.name} {env}")
                    weighted_loss += loss_value * bs
                    weighted_l1 += l1_value * bs
                    samples += bs
                    batches += 1
            rows.append(
                {
                    "experiment": summary["experiment"],
                    "split": summary["split"],
                    "environment": env,
                    "checkpoint": "best_val_action_l1.pt",
                    "val_episodes": len(records),
                    "val_frames": sum(record.num_frames for record in records),
                    "val_batches": batches,
                    "val_samples": samples,
                    "val_loss": weighted_loss / samples,
                    "val_action_l1_loss": weighted_l1 / samples,
                    "runtime_seconds": time.perf_counter() - start,
                }
            )
    finally:
        for module, training in bn_states:
            module.train(training)
    return rows


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def draw_action_plot(width: int, height: int, l1_values: np.ndarray) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    left, top, right, bottom = 45, 20, 12, 32
    plot_w, plot_h = width - left - right, height - top - bottom
    draw.rectangle((left, top, left + plot_w, top + plot_h), outline=(50, 50, 50))
    if len(l1_values) > 1:
        y_max = max(float(np.max(l1_values)) * 1.15, 1e-6)
        points = []
        for i, value in enumerate(l1_values):
            x = left + i / (len(l1_values) - 1) * plot_w
            y = top + (1.0 - float(value) / y_max) * plot_h
            points.append((int(x), int(y)))
        draw.line(points, fill=(190, 50, 40), width=2)
        draw.text((left, height - 24), f"chunk action L1 mean={float(np.mean(l1_values)):.4f}", fill=(0, 0, 0), font=font)
        draw.text((4, top), f"{y_max:.3f}", fill=(0, 0, 0), font=font)
    return img


def make_qualitative_for_run(run_dir: Path, device: torch.device) -> list[dict]:
    summary = read_json(run_dir / "metrics_summary.json")
    config = read_yaml(run_dir / "config.yaml")
    train_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="train")
    val_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="val")
    stats = compute_records_stats(train_records, dataset_root=Path(config["train"]["dataset"]["dataset_root"]))
    policy, policy_config, _ = load_policy(run_dir, Path(summary["best_val_action_l1_checkpoint"]), device)
    policy.eval()

    selected: list[EpisodeRecord] = []
    if summary["split"] == "B_only":
        selected = [record for record in val_records if record.environment == "B"][:3]
    else:
        for env in ["A", "B", "C"]:
            selected.extend([record for record in val_records if record.environment == env][:2])

    rows = []
    for record in selected:
        dataset = CalvinACTTrainingDataset(
            split_name=summary["split"],
            dataset_root=Path(config["train"]["dataset"]["dataset_root"]),
            split_manifest=Path(config["train"]["dataset"]["split_manifest"]),
            chunk_size=policy_config.chunk_size,
            image_keys=list(policy_config.image_features),
            episode_records=[record],
        )
        item = dataset[0]
        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
            for key, value in item.items()
        }
        norm_batch = normalize_batch(batch, stats, device)
        with torch.no_grad():
            pred = policy.predict_action_chunk(norm_batch)[0].detach().cpu()
        gt = norm_batch[ACTION][0].detach().cpu()
        valid_mask = ~item["action_is_pad"]
        pred = pred[: len(valid_mask)]
        gt = gt[: len(valid_mask)]
        l1 = torch.mean(torch.abs(pred - gt), dim=-1).numpy()
        l1 = l1[valid_mask.numpy()]
        mean_l1 = float(np.mean(l1)) if len(l1) else math.nan

        image0 = tensor_to_pil(item["observation.images.image"]).resize((256, 256))
        wrist0 = tensor_to_pil(item["observation.images.wrist_image"]).resize((256, 256))
        plot = draw_action_plot(560, 220, l1)
        contact = Image.new("RGB", (560, 520), "white")
        contact.paste(image0, (10, 60))
        contact.paste(wrist0, (286, 60))
        contact.paste(plot, (0, 320))
        draw = ImageDraw.Draw(contact)
        font = ImageFont.load_default()
        title = f"{summary['experiment']} env={record.environment} episode={record.episode_index}"
        draw.text((10, 12), title, fill=(0, 0, 0), font=font)
        draw.text((10, 34), f"open-loop normalized action L1 mean={mean_l1:.4f}", fill=(0, 0, 0), font=font)

        fig_name = f"{summary['experiment']}_env{record.environment}_episode_{record.episode_index:06d}.png"
        fig_path = QUAL_FIGURE_DIR / fig_name
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        contact.save(fig_path)

        video_dir = QUAL_VIDEO_DIR / summary["experiment"]
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"env{record.environment}_episode_{record.episode_index:06d}_open_loop.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (560, 520))
        frames = min(record.num_frames, 50)
        image_path = str(dataset._video_path(record, "observation.images.image"))
        wrist_path = str(dataset._video_path(record, "observation.images.wrist_image"))
        for frame_idx in range(frames):
            frame_img = tensor_to_pil(_decode_video_frame(image_path, frame_idx)).resize((256, 256))
            wrist_img = tensor_to_pil(_decode_video_frame(wrist_path, frame_idx)).resize((256, 256))
            frame = Image.new("RGB", (560, 520), "white")
            frame.paste(frame_img, (10, 60))
            frame.paste(wrist_img, (286, 60))
            frame.paste(plot, (0, 320))
            draw = ImageDraw.Draw(frame)
            draw.text((10, 12), title, fill=(0, 0, 0), font=font)
            draw.text((10, 34), f"dataset future frame {frame_idx + 1}/{frames}; open-loop L1 mean={mean_l1:.4f}", fill=(0, 0, 0), font=font)
            bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

        rows.append(
            {
                "experiment": summary["experiment"],
                "environment": record.environment,
                "episode_index": record.episode_index,
                "task": record.task,
                "num_frames": record.num_frames,
                "open_loop_normalized_action_l1_mean": mean_l1,
                "figure_path": str(fig_path),
                "video_path": str(video_path),
            }
        )
    return rows


def validation_points(rows: list[dict], key: str) -> list[tuple[float, float]]:
    return [(row["step"], row[key]) for row in rows if row.get("phase") == "validation" and key in row]


def train_points(rows: list[dict], key: str) -> list[tuple[float, float]]:
    return [(row["step"], row[key]) for row in rows if row.get("phase") == "train" and key in row]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-per-env", action="store_true")
    parser.add_argument("--skip-qualitative", action="store_true")
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    QUAL_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_CASE_DIR.mkdir(parents=True, exist_ok=True)
    QUAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_settings = []
    training_rows = []
    checkpoint_rows = []
    verification_rows = []
    scalar_rows_by_exp: dict[str, list[dict]] = {}
    validation_rows_by_exp: dict[str, list[dict]] = {}

    for experiment, run_dir in RUNS.items():
        summary = read_json(run_dir / "metrics_summary.json")
        status = read_yaml(run_dir / "training_status.yaml")
        gate = read_yaml(run_dir / "gate_check.yaml")
        config = read_yaml(run_dir / "config.yaml")
        scalars = read_jsonl(Path(summary["scalars_jsonl"]))
        validations = read_jsonl(Path(summary["validation_jsonl"]))
        scalar_rows_by_exp[experiment] = scalars
        validation_rows_by_exp[experiment] = validations
        train_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="train")
        val_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="val")
        train_envs = sorted({record.environment for record in train_records})
        val_envs = sorted({record.environment for record in val_records})
        finite_ok, finite_detail = finite_scan(scalars)
        gate_ok = gate.get("status") == "passed"

        policy_actual = config["policy_actual"]
        train_cfg = config["train"]["train"]
        dataset_cfg = config["train"]["dataset"]
        all_settings.append(
            {
                "experiment": experiment,
                "split": summary["split"],
                "steps": summary["steps"],
                "batch_size": train_cfg["batch_size"],
                "train_num_workers": status.get("train_num_workers", train_cfg.get("num_workers")),
                "validation_num_workers": status.get("validation_num_workers", train_cfg.get("validation_num_workers", train_cfg.get("num_workers"))),
                "amp": summary["amp"],
                "device": train_cfg["device"],
                "cuda_visible_devices": summary["cuda_visible_devices"],
                "physical_gpu": summary["physical_gpu"],
                "validation_fraction": dataset_cfg["validation_fraction"],
                "image_keys": ";".join(dataset_cfg["image_keys"]),
                "vision_backbone": policy_actual["vision_backbone"],
                "pretrained_backbone_weights": policy_actual["pretrained_backbone_weights"],
                "chunk_size": policy_actual["chunk_size"],
                "n_action_steps": policy_actual["n_action_steps"],
                "dim_model": policy_actual["dim_model"],
                "n_heads": policy_actual["n_heads"],
                "dim_feedforward": policy_actual["dim_feedforward"],
                "n_encoder_layers": policy_actual["n_encoder_layers"],
                "n_decoder_layers": policy_actual["n_decoder_layers"],
                "n_vae_encoder_layers": policy_actual["n_vae_encoder_layers"],
                "optimizer_lr": policy_actual["optimizer_lr"],
                "optimizer_weight_decay": policy_actual["optimizer_weight_decay"],
                "grad_clip_norm": train_cfg["grad_clip_norm"],
            }
        )
        training_rows.append(
            {
                "experiment": experiment,
                "split": summary["split"],
                "status": summary["status"],
                "gate_status": gate.get("status"),
                "steps": summary["steps"],
                "runtime_seconds": summary["runtime_seconds"],
                "runtime_hours": summary["runtime_seconds"] / 3600,
                "mean_step_seconds": summary["mean_step_seconds"],
                "mean_samples_per_second": summary["mean_samples_per_second"],
                "final_loss": summary["final_loss"],
                "final_action_l1_loss": summary["final_action_l1_loss"],
                "final_val_loss": summary["final_val_loss"],
                "final_val_action_l1_loss": summary["final_val_action_l1_loss"],
                "best_val_action_l1_loss": summary["best_val_action_l1_loss"],
                "train_episodes": summary["train_summary"]["num_episodes"],
                "train_frames": summary["train_summary"]["num_frames"],
                "val_episodes": summary["val_summary"]["num_episodes"],
                "val_frames": summary["val_summary"]["num_frames"],
                "train_environments": ";".join(train_envs),
                "val_environments": ";".join(val_envs),
                "finite_scalars": finite_ok,
                "checkpoint_readable": summary["checkpoint_readable"],
                "single_gpu": "," not in str(summary["cuda_visible_devices"]),
            }
        )
        for name, ckpt_path, expected_step in [
            ("latest", Path(summary["latest_checkpoint"]), summary["steps"]),
            ("final", Path(summary["final_checkpoint"]), summary["steps"]),
            ("best_val_action_l1", Path(summary["best_val_action_l1_checkpoint"]), None),
        ]:
            ckpt = checkpoint_status(ckpt_path, expected_step)
            checkpoint_rows.append(
                {
                    "experiment": experiment,
                    "checkpoint_name": name,
                    **ckpt,
                }
            )
        verification_rows.extend(
            [
                {"experiment": experiment, "check": "training_completed", "passed": summary["status"] == "ok"},
                {"experiment": experiment, "check": "gate_check_passed", "passed": gate_ok},
                {"experiment": experiment, "check": "scalars_finite", "passed": finite_ok, "detail": finite_detail},
                {"experiment": experiment, "check": "checkpoint_readable", "passed": summary["checkpoint_readable"]},
                {"experiment": experiment, "check": "final_pt_exists", "passed": Path(summary["final_checkpoint"]).exists()},
                {"experiment": experiment, "check": "best_val_action_l1_pt_exists", "passed": Path(summary["best_val_action_l1_checkpoint"]).exists()},
                {"experiment": experiment, "check": "single_gpu", "passed": "," not in str(summary["cuda_visible_devices"]), "detail": summary["cuda_visible_devices"]},
                {"experiment": experiment, "check": "allowed_environments", "passed": (train_envs == ["B"] if summary["split"] == "B_only" else set(train_envs).issubset({"A", "B", "C"})), "detail": ";".join(train_envs)},
            ]
        )

    setting_compare_keys = [
        "batch_size",
        "amp",
        "image_keys",
        "vision_backbone",
        "pretrained_backbone_weights",
        "chunk_size",
        "n_action_steps",
        "dim_model",
        "n_heads",
        "dim_feedforward",
        "n_encoder_layers",
        "n_decoder_layers",
        "n_vae_encoder_layers",
        "optimizer_lr",
        "optimizer_weight_decay",
        "grad_clip_norm",
    ]
    consistent = all(all_settings[0][key] == all_settings[1][key] for key in setting_compare_keys)
    for row in all_settings:
        verification_rows.append({"experiment": row["experiment"], "check": "act_and_optimizer_hyperparams_match_between_runs", "passed": consistent})

    write_csv(TABLE_DIR / "experiment_settings.csv", all_settings)
    write_csv(TABLE_DIR / "training_summary.csv", training_rows)
    write_csv(TABLE_DIR / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(TABLE_DIR / "training_verification_checks.csv", verification_rows)

    b_rows = scalar_rows_by_exp["B_only_act_baseline"]
    abc_rows = scalar_rows_by_exp["ABC_mixed_act_baseline"]
    draw_line_chart(
        FIGURE_DIR / "action_l1_train_B_only_vs_ABC.png",
        "Training action L1: B-only vs ABC-mixed",
        [
            ("B-only train action L1", train_points(b_rows, "action_l1_loss"), color_for("B_only_act_baseline")),
            ("ABC-mixed train action L1", train_points(abc_rows, "action_l1_loss"), color_for("ABC_mixed_act_baseline")),
        ],
        "action L1",
    )
    draw_line_chart(
        FIGURE_DIR / "action_l1_val_B_only_vs_ABC.png",
        "Validation action L1: B-only vs ABC-mixed",
        [
            ("B-only validation action L1", validation_points(b_rows, "val_action_l1_loss"), color_for("B_only_act_baseline")),
            ("ABC-mixed validation action L1", validation_points(abc_rows, "val_action_l1_loss"), color_for("ABC_mixed_act_baseline")),
        ],
        "val action L1",
    )
    draw_line_chart(
        FIGURE_DIR / "total_loss_B_only_vs_ABC.png",
        "Training total loss: B-only vs ABC-mixed",
        [
            ("B-only train total loss", train_points(b_rows, "loss"), color_for("B_only_act_baseline")),
            ("ABC-mixed train total loss", train_points(abc_rows, "loss"), color_for("ABC_mixed_act_baseline")),
        ],
        "total loss",
    )
    draw_bar_chart(
        FIGURE_DIR / "best_val_action_l1_bar.png",
        "Best validation action L1",
        [(row["experiment"], row["best_val_action_l1_loss"], color_for(row["experiment"])) for row in training_rows],
        "best val action L1",
    )

    per_env_rows: list[dict] = []
    if not args.skip_per_env:
        for experiment, run_dir in RUNS.items():
            per_env_rows.extend(evaluate_per_env(run_dir, device))
        write_csv(TABLE_DIR / "per_env_validation_action_l1.csv", per_env_rows)
        draw_bar_chart(
            FIGURE_DIR / "per_env_validation_action_l1.png",
            "Per-environment validation action L1",
            [
                (f"{row['experiment'].replace('_act_baseline', '')}-{row['environment']}", row["val_action_l1_loss"], color_for(row["environment"]))
                for row in per_env_rows
            ],
            "val action L1",
        )

    qual_rows: list[dict] = []
    if not args.skip_qualitative:
        for experiment, run_dir in RUNS.items():
            qual_rows.extend(make_qualitative_for_run(run_dir, device))
        lines = [
            "# Qualitative Video Index",
            "",
            "Open-loop visualizations use `best_val_action_l1.pt` and validation episodes only. Videos are stored under `/data/dechao/cv_final_pj2/outputs/qualitative/`; small report figures are stored under `results/figures/qualitative/`.",
            "",
            "| experiment | env | episode | task | mean normalized action L1 | figure | video |",
            "|---|---:|---:|---|---:|---|---|",
        ]
        for row in qual_rows:
            lines.append(
                f"| {row['experiment']} | {row['environment']} | {row['episode_index']} | {row['task']} | "
                f"{row['open_loop_normalized_action_l1_mean']:.4f} | {row['figure_path']} | {row['video_path']} |"
            )
        (FAILURE_CASE_DIR / "qualitative_video_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "status": "ok",
        "device": str(device),
        "tables": {
            "experiment_settings": str(TABLE_DIR / "experiment_settings.csv"),
            "training_summary": str(TABLE_DIR / "training_summary.csv"),
            "checkpoint_summary": str(TABLE_DIR / "checkpoint_summary.csv"),
            "per_env_validation_action_l1": str(TABLE_DIR / "per_env_validation_action_l1.csv") if per_env_rows else None,
            "training_verification_checks": str(TABLE_DIR / "training_verification_checks.csv"),
        },
        "figures": {
            "action_l1_train": str(FIGURE_DIR / "action_l1_train_B_only_vs_ABC.png"),
            "action_l1_val": str(FIGURE_DIR / "action_l1_val_B_only_vs_ABC.png"),
            "total_loss": str(FIGURE_DIR / "total_loss_B_only_vs_ABC.png"),
            "best_val_bar": str(FIGURE_DIR / "best_val_action_l1_bar.png"),
            "per_env": str(FIGURE_DIR / "per_env_validation_action_l1.png") if per_env_rows else None,
        },
        "qualitative_count": len(qual_rows),
    }
    write_yaml(TABLE_DIR / "step3_training_verification_report.yaml", report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
