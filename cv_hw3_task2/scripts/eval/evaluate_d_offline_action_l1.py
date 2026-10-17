#!/usr/bin/env python3
"""Evaluate Step3 ACT checkpoints on CALVIN task_D_D LeRobot data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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

from src.data.calvin_d_lerobot_dataset import (  # noqa: E402
    DEFAULT_D_DATASET_ROOT,
    D_IMAGE_KEY_MAP,
    CalvinDDACTDataset,
    compute_d_dataset_stats,
    configure_episode_table_cache,
    episode_table_cache_status,
    load_d_episode_records,
    summarize_d_records,
)
from src.data.calvin_filtered_dataset import EpisodeRecord, compute_records_stats  # noqa: E402
from src.data.task_text_mapping import TaskTextMapper  # noqa: E402
from src.models.task_conditioned_act import TaskConditionedACTPolicy  # noqa: E402


DEFAULT_RUNS = {
    "B_only_act_baseline": Path("/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline"),
    "ABC_mixed_act_baseline": Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline"),
}
TASK_ID_FINETUNE_RUNS = {
    "B_only_task_id_finetune": Path(
        "/data/dechao/cv_final_pj2/outputs/task_conditioned/"
        "B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1"
    ),
    "ABC_mixed_task_id_finetune": Path(
        "/data/dechao/cv_final_pj2/outputs/task_conditioned/"
        "ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1"
    ),
}
TABLE_DIR = PROJECT_ROOT / "results/tables/step4"
FIGURE_DIR = PROJECT_ROOT / "results/figures/step4"
ACTION_DIM_NAMES = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def output_path(base_dir: Path, prefix: str, filename: str) -> Path:
    if not prefix:
        return base_dir / filename
    return base_dir / f"{prefix}_{filename}"


def read_split_csv(path: Path, split: str) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["split"] != split:
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


def build_policy_config(policy_actual: dict, device: str) -> ACTConfig:
    cfg = deepcopy(policy_actual)
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


def load_policy(run_dir: Path, checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, ACTConfig, dict, dict | None]:
    config_yaml = read_yaml(run_dir / "config.yaml")
    policy_config = build_policy_config(config_yaml["policy_actual"], str(device))
    task_conditioning = config_yaml.get("task_conditioning")
    if task_conditioning and bool(task_conditioning.get("enabled", False)):
        policy = TaskConditionedACTPolicy(
            config=policy_config,
            num_tasks=int(task_conditioning["num_tasks"]),
            task_embedding_dim=int(task_conditioning["embedding_dim"]),
            task_id_key=str(task_conditioning.get("task_id_key", "canonical_task_id")),
        ).to(device)
    else:
        policy = ACTPolicy(policy_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_result = policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    policy.eval()
    return policy, policy_config, {
        "checkpoint_step": checkpoint.get("step"),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "task_conditioning_enabled": bool(task_conditioning and task_conditioning.get("enabled", False)),
    }, task_conditioning


def tensor_shapes(batch: dict) -> dict:
    return {
        key: list(value.shape) if torch.is_tensor(value) else f"{type(value).__name__}[{len(value)}]"
        for key, value in batch.items()
    }


def normalize_for_policy(batch: dict, stats: dict, device: torch.device) -> tuple[dict, dict]:
    raw = {}
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            raw[key] = value.to(device, non_blocking=True)
            out[key] = raw[key]
        else:
            raw[key] = value
            out[key] = value

    for key in (OBS_STATE, ACTION, "observation.images.image", "observation.images.wrist_image"):
        if key not in out or key not in stats:
            continue
        mean = stats[key]["mean"].to(device)
        std = stats[key]["std"].to(device).clamp_min(1e-6)
        out[key] = (out[key] - mean) / std
    return raw, out


def _as_task_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def d_task_to_id(task_text: str, mapper: TaskTextMapper) -> int:
    """Map fywang D task labels such as ``close_drawer: close the drawer``."""

    raw_text = str(task_text)
    prefix, separator, suffix = raw_text.partition(":")
    candidates = []
    if separator:
        candidates.append(prefix.strip())
        candidates.append(suffix.strip())
    candidates.append(raw_text.strip())

    for candidate in candidates:
        if candidate in mapper.task_to_id:
            return mapper.task_to_id[candidate]
        if candidate in mapper.training_text_to_canonical:
            return mapper.task_to_id[mapper.training_text_to_canonical[candidate]]
        if candidate in mapper.validation_auto_lang_text_to_canonical:
            return mapper.task_to_id[mapper.validation_auto_lang_text_to_canonical[candidate]]
        if candidate in mapper.closed_loop_goal_text_to_canonical:
            return mapper.task_to_id[mapper.closed_loop_goal_text_to_canonical[candidate]]
    raise KeyError(f"D task text is not covered by canonical task vocab: {raw_text!r}")


def add_task_ids(batch: dict, mapper: TaskTextMapper, device: torch.device) -> dict:
    out = dict(batch)
    task_ids = [d_task_to_id(task, mapper) for task in _as_task_list(batch["task"])]
    out["canonical_task_id"] = torch.tensor(task_ids, dtype=torch.long, device=device)
    return out


def update_group(acc: dict, key, norm_abs: torch.Tensor, raw_abs: torch.Tensor, valid: torch.Tensor, samples: int) -> None:
    item = acc[key]
    item["sample_frames"] += samples
    item["valid_action_steps"] += int(valid.sum().detach().cpu().item())
    item["norm_sum"] += float((norm_abs * valid.unsqueeze(-1)).sum().detach().cpu().item())
    item["raw_sum"] += float((raw_abs * valid.unsqueeze(-1)).sum().detach().cpu().item())
    item["denom"] += int(valid.sum().detach().cpu().item()) * int(norm_abs.shape[-1])


def draw_bar_chart(path: Path, title: str, rows: list[tuple[str, float, tuple[int, int, int]]], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 680
    left, right, top, bottom = 90, 50, 70, 130
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max([value for _, value, _ in rows] + [1e-9]) * 1.15
    draw.text((left, 25), title, fill=(0, 0, 0), font=font)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(40, 40, 40), width=1)
    draw.line((left, top, left, top + plot_h), fill=(40, 40, 40), width=1)
    slot = plot_w / max(len(rows), 1)
    bar_w = slot * 0.55
    for idx, (name, value, color) in enumerate(rows):
        cx = left + slot * (idx + 0.5)
        bar_h = value / max_value * plot_h
        x0, x1 = int(cx - bar_w / 2), int(cx + bar_w / 2)
        y0, y1 = int(top + plot_h - bar_h), int(top + plot_h)
        draw.rectangle((x0, y0, x1, y1), fill=color)
        draw.text((x0, max(top, y0 - 18)), f"{value:.4f}", fill=(0, 0, 0), font=font)
        label = name if len(name) <= 24 else name[:21] + "..."
        draw.text((x0, y1 + 14), label, fill=(0, 0, 0), font=font)
    draw.text((8, height // 2), ylabel, fill=(40, 40, 40), font=font)
    image.save(path)


def run_colors(experiments: list[str]) -> dict[str, tuple[int, int, int]]:
    palette = [
        (41, 98, 255),
        (220, 88, 42),
        (42, 150, 90),
        (145, 80, 190),
        (45, 140, 170),
        (190, 70, 110),
    ]
    return {experiment: palette[idx % len(palette)] for idx, experiment in enumerate(experiments)}


def draw_grouped_action_dim_chart(path: Path, rows: list[dict], experiments: list[str], colors: dict[str, tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 720
    left, right, top, bottom = 90, 40, 70, 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    values = {
        (row["experiment"], int(row["action_dim"])): float(row["normalized_action_l1_valid_mean"])
        for row in rows
    }
    max_value = max(values.values()) * 1.15
    plot_w = width - left - right
    plot_h = height - top - bottom
    draw.text((left, 25), "D per-action-dimension open-loop Action L1", fill=(0, 0, 0), font=font)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(40, 40, 40), width=1)
    draw.line((left, top, left, top + plot_h), fill=(40, 40, 40), width=1)
    group_w = plot_w / len(ACTION_DIM_NAMES)
    bar_w = group_w / max(len(experiments) + 2, 4)
    for dim, dim_name in enumerate(ACTION_DIM_NAMES):
        for exp_idx, experiment in enumerate(experiments):
            value = values.get((experiment, dim), 0.0)
            offset = exp_idx - (len(experiments) - 1) / 2
            cx = left + group_w * (dim + 0.5) + offset * bar_w * 1.15
            bar_h = value / max(max_value, 1e-9) * plot_h
            x0, x1 = int(cx - bar_w / 2), int(cx + bar_w / 2)
            y0, y1 = int(top + plot_h - bar_h), int(top + plot_h)
            draw.rectangle((x0, y0, x1, y1), fill=colors[experiment])
        draw.text((int(left + group_w * dim + 8), top + plot_h + 14), dim_name, fill=(0, 0, 0), font=font)
    legend_x = width - 360
    for idx, experiment in enumerate(experiments):
        y = 35 + idx * 24
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=colors[experiment])
        draw.text((legend_x + 22, y), experiment.replace("_act_baseline", ""), fill=(0, 0, 0), font=font)
    draw.text((8, height // 2), "normalized action L1", fill=(40, 40, 40), font=font)
    image.save(path)


def draw_top_task_chart(
    path: Path,
    rows: list[dict],
    experiments: list[str],
    colors: dict[str, tuple[int, int, int]],
    top_n: int = 20,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    task_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        task = row["task"]
        by_task[task][row["experiment"]] = row
        task_counts[task] = max(task_counts[task], int(row["sample_frames"]))
    selected = sorted(task_counts, key=task_counts.get, reverse=True)[:top_n]

    width, height = 1500, max(720, 110 + len(selected) * 28)
    left, right, top, bottom = 430, 50, 60, 50
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    plot_w = width - left - right
    plot_h = height - top - bottom
    all_values = [
        float(by_task[task][exp]["normalized_action_l1_valid_mean"])
        for task in selected
        for exp in experiments
        if exp in by_task[task]
    ]
    max_value = max(all_values + [1e-9]) * 1.1
    draw.text((left, 25), f"Top {len(selected)} D tasks by evaluated frames", fill=(0, 0, 0), font=font)
    row_h = plot_h / max(len(selected), 1)
    for idx, task in enumerate(selected):
        y_mid = top + row_h * (idx + 0.5)
        label = task if len(task) <= 58 else task[:55] + "..."
        draw.text((20, int(y_mid - 7)), label, fill=(0, 0, 0), font=font)
        for exp_idx, experiment in enumerate(experiments):
            row = by_task[task].get(experiment)
            if row is None:
                continue
            value = float(row["normalized_action_l1_valid_mean"])
            bar_h = row_h / max(len(experiments) + 1, 3)
            offset = exp_idx - (len(experiments) - 1) / 2
            y0 = y_mid + offset * bar_h * 1.15 - bar_h / 2
            x1 = left + value / max(max_value, 1e-9) * plot_w
            draw.rectangle((left, int(y0), int(x1), int(y0 + bar_h)), fill=colors[experiment])
            draw.text((int(x1) + 6, int(y0)), f"{value:.3f}", fill=(0, 0, 0), font=font)
    legend_x = width - 380
    for idx, experiment in enumerate(experiments):
        y = 25 + idx * 24
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=colors[experiment])
        label = experiment.replace("_act_baseline", "")
        draw.text((legend_x + 22, y), label[:42], fill=(0, 0, 0), font=font)
    image.save(path)


def evaluate_run(
    experiment: str,
    run_dir: Path,
    dataset: CalvinDDACTDataset,
    dataloader: DataLoader,
    first_batch: dict,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict, list[dict], list[dict], list[dict], dict]:
    summary = read_json(run_dir / "metrics_summary.json")
    config = read_yaml(run_dir / "config.yaml")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = Path(summary["best_val_action_l1_checkpoint"])
        checkpoint_name = checkpoint_path.name

    train_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="train")
    stats = compute_records_stats(
        train_records,
        dataset_root=Path(config["train"]["dataset"]["dataset_root"]),
    )
    policy, policy_config, load_info, task_conditioning = load_policy(run_dir, checkpoint_path, device)
    task_mapper = None
    if task_conditioning and bool(task_conditioning.get("enabled", False)):
        task_mapper = TaskTextMapper.from_json(Path(task_conditioning["vocab_path"]))
    action_mean = stats[ACTION]["mean"].to(device)
    action_std = stats[ACTION]["std"].to(device).clamp_min(1e-6)

    raw_first, norm_first = normalize_for_policy(first_batch, stats, device)
    if task_mapper is not None:
        norm_first = add_task_ids(norm_first, task_mapper, device)
    with torch.no_grad():
        first_pred = policy.predict_action_chunk(norm_first)
    first_forward_ok = list(first_pred.shape) == list(raw_first[ACTION].shape)

    total_samples = 0
    total_batches = 0
    total_valid_steps = 0
    norm_sum = 0.0
    raw_sum = 0.0
    denom = 0
    dim_norm_sum = None
    dim_raw_sum = None
    dim_valid = 0
    per_task = defaultdict(lambda: {"sample_frames": 0, "valid_action_steps": 0, "norm_sum": 0.0, "raw_sum": 0.0, "denom": 0})
    per_episode = defaultdict(lambda: {"sample_frames": 0, "valid_action_steps": 0, "norm_sum": 0.0, "raw_sum": 0.0, "denom": 0, "task": ""})

    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            raw_batch, norm_batch = normalize_for_policy(batch, stats, device)
            if task_mapper is not None:
                norm_batch = add_task_ids(norm_batch, task_mapper, device)
            pred_norm = policy.predict_action_chunk(norm_batch)
            gt_norm = norm_batch[ACTION]
            gt_raw = raw_batch[ACTION]
            pred_raw = pred_norm * action_std + action_mean
            valid = ~raw_batch["action_is_pad"]
            norm_abs = torch.abs(pred_norm - gt_norm)
            raw_abs = torch.abs(pred_raw - gt_raw)

            batch_size = int(gt_raw.shape[0])
            action_dim = int(gt_raw.shape[-1])
            valid_steps = int(valid.sum().detach().cpu().item())
            batch_denom = valid_steps * action_dim
            total_samples += batch_size
            total_batches += 1
            total_valid_steps += valid_steps
            denom += batch_denom
            norm_sum += float((norm_abs * valid.unsqueeze(-1)).sum().detach().cpu().item())
            raw_sum += float((raw_abs * valid.unsqueeze(-1)).sum().detach().cpu().item())
            dim_norm = (norm_abs * valid.unsqueeze(-1)).sum(dim=(0, 1)).detach().cpu().numpy()
            dim_raw = (raw_abs * valid.unsqueeze(-1)).sum(dim=(0, 1)).detach().cpu().numpy()
            dim_norm_sum = dim_norm if dim_norm_sum is None else dim_norm_sum + dim_norm
            dim_raw_sum = dim_raw if dim_raw_sum is None else dim_raw_sum + dim_raw
            dim_valid += valid_steps

            tasks = _as_task_list(batch["task"])
            episodes = raw_batch["episode_index"].detach().cpu().numpy().tolist()
            for sample_idx, task in enumerate(tasks):
                sample_valid = valid[sample_idx]
                sample_norm = norm_abs[sample_idx : sample_idx + 1]
                sample_raw = raw_abs[sample_idx : sample_idx + 1]
                update_group(per_task, task, sample_norm, sample_raw, sample_valid.unsqueeze(0), 1)
                episode_key = int(episodes[sample_idx])
                per_episode[episode_key]["task"] = task
                update_group(per_episode, episode_key, sample_norm, sample_raw, sample_valid.unsqueeze(0), 1)

    runtime = time.perf_counter() - start
    if denom <= 0:
        raise RuntimeError(f"No valid D action steps evaluated for {experiment}")
    if not math.isfinite(norm_sum / denom) or not math.isfinite(raw_sum / denom):
        raise RuntimeError(f"Non-finite D Action L1 for {experiment}")

    eval_summary = {
        "experiment": experiment,
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset": "fywang/calvin-task-D-D-lerobot",
        "environment": "D",
        "protocol": "offline_open_loop_predict_action_chunk",
        "task_conditioning_enabled": bool(task_mapper is not None),
        "metric_primary": "normalized_action_l1_valid_mean",
        "normalization_stats_source": "source_training_split",
        "d_episodes": dataset.num_episodes,
        "d_frames": dataset.num_frames,
        "eval_batches": total_batches,
        "eval_sample_frames": total_samples,
        "valid_action_steps": total_valid_steps,
        "normalized_action_l1_valid_mean": norm_sum / denom,
        "raw_action_l1_valid_mean": raw_sum / denom,
        "runtime_seconds": runtime,
        "source_best_val_action_l1_loss_reference": summary["best_val_action_l1_loss"],
    }
    task_rows = []
    for task, values in per_task.items():
        if values["denom"] <= 0:
            continue
        task_rows.append(
            {
                "experiment": experiment,
                "task": task,
                "sample_frames": values["sample_frames"],
                "valid_action_steps": values["valid_action_steps"],
                "normalized_action_l1_valid_mean": values["norm_sum"] / values["denom"],
                "raw_action_l1_valid_mean": values["raw_sum"] / values["denom"],
            }
        )
    task_rows.sort(key=lambda row: (row["task"], row["experiment"]))

    episode_rows = []
    for episode_index, values in per_episode.items():
        if values["denom"] <= 0:
            continue
        episode_rows.append(
            {
                "experiment": experiment,
                "episode_index": episode_index,
                "task": values["task"],
                "sample_frames": values["sample_frames"],
                "valid_action_steps": values["valid_action_steps"],
                "normalized_action_l1_valid_mean": values["norm_sum"] / values["denom"],
                "raw_action_l1_valid_mean": values["raw_sum"] / values["denom"],
            }
        )
    episode_rows.sort(key=lambda row: (row["episode_index"], row["experiment"]))

    dim_rows = []
    for dim, (norm_value, raw_value) in enumerate(zip(dim_norm_sum, dim_raw_sum, strict=True)):
        dim_rows.append(
            {
                "experiment": experiment,
                "action_dim": dim,
                "action_dim_name": ACTION_DIM_NAMES[dim] if dim < len(ACTION_DIM_NAMES) else f"dim_{dim}",
                "valid_action_steps": dim_valid,
                "normalized_action_l1_valid_mean": float(norm_value / max(dim_valid, 1)),
                "raw_action_l1_valid_mean": float(raw_value / max(dim_valid, 1)),
            }
        )

    load_row = {
        "experiment": experiment,
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_size_mb": checkpoint_path.stat().st_size / 1024 / 1024 if checkpoint_path.exists() else 0.0,
        "checkpoint_step": load_info["checkpoint_step"],
        "missing_keys": len(load_info["missing_keys"]),
        "unexpected_keys": len(load_info["unexpected_keys"]),
        "task_conditioning_enabled": bool(load_info["task_conditioning_enabled"]),
        "policy_chunk_size": policy_config.chunk_size,
        "policy_image_features": ";".join(policy_config.image_features),
        "first_predict_shape": str(list(first_pred.shape)),
        "first_forward_shape_ok": first_forward_ok,
    }
    return eval_summary, task_rows, dim_rows, episode_rows, load_row


def write_sanity_tables(dataset_root: Path, dataset: CalvinDDACTDataset, first_batch: dict) -> None:
    info = read_json(dataset_root / "meta" / "info.json")
    records = load_d_episode_records(dataset_root)
    record_summary = summarize_d_records(records)
    d_stats = compute_d_dataset_stats(dataset_root)
    sample_shapes = tensor_shapes(first_batch)
    rows = [
        {
            "dataset": "fywang/calvin-task-D-D-lerobot",
            "local_root": str(dataset_root),
            "codebase_version": info.get("codebase_version"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "total_tasks": info.get("total_tasks"),
            "total_videos": info.get("total_videos"),
            "total_chunks": info.get("total_chunks"),
            "fps": info.get("fps"),
            "records_episodes": len(records),
            "records_frames": record_summary["num_frames"],
            "records_environment": "D",
            "records_task_count": len(record_summary["by_task"]),
            "state_dim": len(d_stats["observation.state"]["mean"]),
            "action_dim": len(d_stats["action"]["mean"]),
            "source_image_keys": ";".join(D_IMAGE_KEY_MAP.values()),
            "target_image_keys": ";".join(D_IMAGE_KEY_MAP.keys()),
            "adapter_image_size": "256x256",
            "sample_batch_shapes": json.dumps(sample_shapes, ensure_ascii=False),
            "status": "ok",
        }
    ]
    write_csv(TABLE_DIR / "d_dataset_sanity_summary.csv", rows)
    write_csv(
        TABLE_DIR / "d_adapter_check_summary.csv",
        [
            {
                "check": "state_dim_matches_policy",
                "passed": len(d_stats["observation.state"]["mean"]) == 15,
                "detail": len(d_stats["observation.state"]["mean"]),
            },
            {
                "check": "action_dim_matches_policy",
                "passed": len(d_stats["action"]["mean"]) == 7,
                "detail": len(d_stats["action"]["mean"]),
            },
            {
                "check": "image_key_mapping",
                "passed": True,
                "detail": json.dumps(D_IMAGE_KEY_MAP, ensure_ascii=False),
            },
            {
                "check": "image_resize_to_policy_shape",
                "passed": sample_shapes.get("observation.images.image") == [1, 3, 256, 256]
                and sample_shapes.get("observation.images.wrist_image") == [1, 3, 256, 256],
                "detail": json.dumps(sample_shapes, ensure_ascii=False),
            },
            {
                "check": "dataset_records",
                "passed": len(records) == int(info["total_episodes"]) and record_summary["num_frames"] == int(info["total_frames"]),
                "detail": f"episodes={len(records)}, frames={record_summary['num_frames']}",
            },
        ],
        fields=["check", "passed", "detail"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D_DATASET_ROOT)
    parser.add_argument("--checkpoint-name", default="best_val_action_l1.pt")
    parser.add_argument(
        "--include-task-id-finetunes",
        action="store_true",
        help="Also evaluate the formal B-only and ABC-mixed task-id fine-tune output directories.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Additional or replacement run spec. Can be repeated.",
    )
    parser.add_argument(
        "--only-custom-runs",
        action="store_true",
        help="Use only --run entries instead of the default visual baselines.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix table/figure/report filenames to avoid overwriting previous Track1 outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs. Disabled by default to release RAM promptly.",
    )
    parser.add_argument(
        "--table-cache-size",
        type=int,
        default=32,
        help="Per-process parquet episode-table LRU cache size. Use 0 to disable.",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if args.table_cache_size < 0:
        raise ValueError("--table-cache-size must be nonnegative")

    configure_episode_table_cache(args.table_cache_size)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    runs = {} if args.only_custom_runs else dict(DEFAULT_RUNS)
    if args.include_task_id_finetunes:
        runs.update(TASK_ID_FINETUNE_RUNS)
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"--run must have NAME=RUN_DIR format, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--run has empty NAME: {spec!r}")
        runs[name] = Path(path)
    if not runs:
        raise ValueError("No runs selected for D offline evaluation.")
    for name, run_dir in runs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory for {name!r} does not exist: {run_dir}")

    dataset = CalvinDDACTDataset(
        dataset_root=args.dataset_root,
        chunk_size=100,
        max_episodes=args.max_episodes,
    )
    sanity_dataset = CalvinDDACTDataset(
        dataset_root=args.dataset_root,
        chunk_size=100,
        max_episodes=1,
    )
    sanity_loader = DataLoader(sanity_dataset, batch_size=1, shuffle=False, num_workers=0)
    first_batch = next(iter(sanity_loader))
    write_sanity_tables(args.dataset_root, sanity_dataset if args.max_episodes else dataset, first_batch)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    eval_rows = []
    task_rows = []
    dim_rows = []
    episode_rows = []
    load_rows = []
    report = {
        "status": "running",
        "start_time": now_iso(),
        "dataset_root": str(args.dataset_root),
        "runs": {name: str(path) for name, path in runs.items()},
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
        "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
        "table_cache": episode_table_cache_status(),
        "max_episodes": args.max_episodes,
        "output_prefix": args.output_prefix,
    }
    report_path = output_path(TABLE_DIR, args.output_prefix, "d_evaluation_report.json")
    write_json(report_path, report)

    for experiment, run_dir in runs.items():
        eval_summary, per_task, per_dim, per_episode, load_row = evaluate_run(
            experiment=experiment,
            run_dir=run_dir,
            dataset=dataset,
            dataloader=dataloader,
            first_batch=first_batch,
            device=device,
            checkpoint_name=args.checkpoint_name,
        )
        eval_rows.append(eval_summary)
        task_rows.extend(per_task)
        dim_rows.extend(per_dim)
        episode_rows.extend(per_episode)
        load_rows.append(load_row)

    write_csv(output_path(TABLE_DIR, args.output_prefix, "d_checkpoint_load_summary.csv"), load_rows)
    write_csv(output_path(TABLE_DIR, args.output_prefix, "d_evaluation_summary.csv"), eval_rows)
    write_csv(output_path(TABLE_DIR, args.output_prefix, "d_per_task_action_l1.csv"), task_rows)
    write_csv(output_path(TABLE_DIR, args.output_prefix, "d_per_action_dim_error.csv"), dim_rows)
    write_csv(output_path(TABLE_DIR, args.output_prefix, "d_per_episode_action_l1.csv"), episode_rows)

    experiments = list(runs)
    colors = run_colors(experiments)
    draw_bar_chart(
        output_path(FIGURE_DIR, args.output_prefix, "d_action_l1_comparison.png"),
        "D offline open-loop Action L1",
        [
            (
                row["experiment"].replace("_act_baseline", "").replace("_task_id_finetune", "_tid_ft"),
                float(row["normalized_action_l1_valid_mean"]),
                colors[row["experiment"]],
            )
            for row in eval_rows
        ],
        "normalized action L1",
    )
    draw_grouped_action_dim_chart(
        output_path(FIGURE_DIR, args.output_prefix, "d_per_action_dim_error.png"),
        dim_rows,
        experiments,
        colors,
    )
    draw_top_task_chart(
        output_path(FIGURE_DIR, args.output_prefix, "d_per_task_action_l1.png"),
        task_rows,
        experiments,
        colors,
    )
    comparison_rows = []
    for row in eval_rows:
        comparison_rows.append(
            (
                row["experiment"].replace("_act_baseline", "").replace("_task_id_finetune", "_tid_ft")
                + " source-val ref",
                float(row["source_best_val_action_l1_loss_reference"]),
                (150, 150, 150),
            )
        )
        comparison_rows.append(
            (
                row["experiment"].replace("_act_baseline", "").replace("_task_id_finetune", "_tid_ft")
                + " D open-loop",
                float(row["normalized_action_l1_valid_mean"]),
                colors[row["experiment"]],
            )
        )
    draw_bar_chart(
        output_path(FIGURE_DIR, args.output_prefix, "source_vs_d_action_l1_comparison.png"),
        "Source validation reference vs D open-loop Action L1",
        comparison_rows,
        "normalized action L1",
    )

    report.update(
        {
            "status": "ok",
            "end_time": now_iso(),
            "tables": {
                "d_dataset_sanity_summary": str(TABLE_DIR / "d_dataset_sanity_summary.csv"),
                "d_adapter_check_summary": str(TABLE_DIR / "d_adapter_check_summary.csv"),
                "d_checkpoint_load_summary": str(
                    output_path(TABLE_DIR, args.output_prefix, "d_checkpoint_load_summary.csv")
                ),
                "d_evaluation_summary": str(output_path(TABLE_DIR, args.output_prefix, "d_evaluation_summary.csv")),
                "d_per_task_action_l1": str(output_path(TABLE_DIR, args.output_prefix, "d_per_task_action_l1.csv")),
                "d_per_action_dim_error": str(
                    output_path(TABLE_DIR, args.output_prefix, "d_per_action_dim_error.csv")
                ),
                "d_per_episode_action_l1": str(
                    output_path(TABLE_DIR, args.output_prefix, "d_per_episode_action_l1.csv")
                ),
            },
            "figures": {
                "d_action_l1": str(output_path(FIGURE_DIR, args.output_prefix, "d_action_l1_comparison.png")),
                "d_per_task": str(output_path(FIGURE_DIR, args.output_prefix, "d_per_task_action_l1.png")),
                "d_per_action_dim": str(output_path(FIGURE_DIR, args.output_prefix, "d_per_action_dim_error.png")),
                "source_vs_d": str(
                    output_path(FIGURE_DIR, args.output_prefix, "source_vs_d_action_l1_comparison.png")
                ),
            },
            "summary": eval_rows,
        }
    )
    write_json(report_path, report)
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
