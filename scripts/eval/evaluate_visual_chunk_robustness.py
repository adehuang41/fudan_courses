#!/usr/bin/env python3
"""Offline visual robustness diagnostics for CALVIN D ACT checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.utils.constants import ACTION  # noqa: E402

from scripts.eval.evaluate_d_offline_action_l1 import (  # noqa: E402
    ACTION_DIM_NAMES,
    DEFAULT_RUNS,
    load_policy,
    normalize_for_policy,
    now_iso,
    read_json,
    read_split_csv,
    read_yaml,
    write_csv,
    write_json,
)
from src.data.calvin_d_lerobot_dataset import (  # noqa: E402
    DEFAULT_D_DATASET_ROOT,
    CalvinDDACTDataset,
    configure_episode_table_cache,
    episode_table_cache_status,
)
from src.data.calvin_filtered_dataset import compute_records_stats  # noqa: E402


IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")
DEFAULT_CONDITIONS = (
    "clean",
    "brightness_dark",
    "brightness_bright",
    "gaussian_blur",
    "gaussian_noise",
    "static_grayout",
    "wrist_grayout",
    "both_camera_grayout",
)
CONDITION_LABELS = {
    "clean": "Clean",
    "brightness_dark": "Brightness 0.65x",
    "brightness_bright": "Brightness 1.25x",
    "gaussian_blur": "Gaussian blur",
    "gaussian_noise": "Gaussian noise",
    "static_grayout": "Static gray-out",
    "wrist_grayout": "Wrist gray-out",
    "both_camera_grayout": "Both cameras gray-out",
}


def parse_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    if "clean" not in conditions:
        conditions.insert(0, "clean")
    return conditions


def parse_run_specs(specs: list[str], only_custom_runs: bool) -> dict[str, Path]:
    runs = {} if only_custom_runs else dict(DEFAULT_RUNS)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--run must be NAME=RUN_DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"--run has empty NAME: {spec!r}")
        runs[name] = Path(path)
    if not runs:
        raise ValueError("No runs selected")
    for name, run_dir in runs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory for {name!r} does not exist: {run_dir}")
    return runs


def task_type(task: str) -> str:
    prefix, sep, _ = str(task).partition(":")
    return prefix.strip() if sep and prefix.strip() else str(task).strip()


def gaussian_blur(images: torch.Tensor) -> torch.Tensor:
    kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=images.dtype, device=images.device)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    channels = int(images.shape[1])
    kernel = kernel_2d.view(1, 1, 5, 5).repeat(channels, 1, 1, 1)
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    return F.conv2d(padded, kernel, groups=channels).clamp(0.0, 1.0)


def apply_condition(batch: dict, condition: str, batch_index: int, noise_seed: int, noise_std: float) -> dict:
    if condition == "clean":
        return batch

    out = dict(batch)
    if condition == "brightness_dark":
        for key in IMAGE_KEYS:
            out[key] = (batch[key].clone() * 0.65).clamp(0.0, 1.0)
        return out
    if condition == "brightness_bright":
        for key in IMAGE_KEYS:
            out[key] = (batch[key].clone() * 1.25 + 0.03).clamp(0.0, 1.0)
        return out
    if condition == "gaussian_blur":
        for key in IMAGE_KEYS:
            out[key] = gaussian_blur(batch[key].clone())
        return out
    if condition == "gaussian_noise":
        for offset, key in enumerate(IMAGE_KEYS):
            gen = torch.Generator(device=batch[key].device)
            gen.manual_seed(int(noise_seed) + int(batch_index) * 101 + offset)
            noise = torch.randn(batch[key].shape, generator=gen, dtype=batch[key].dtype, device=batch[key].device)
            out[key] = (batch[key].clone() + noise_std * noise).clamp(0.0, 1.0)
        return out
    if condition == "static_grayout":
        out["observation.images.image"] = torch.full_like(batch["observation.images.image"], 0.5)
        return out
    if condition == "wrist_grayout":
        out["observation.images.wrist_image"] = torch.full_like(batch["observation.images.wrist_image"], 0.5)
        return out
    if condition == "both_camera_grayout":
        out["observation.images.image"] = torch.full_like(batch["observation.images.image"], 0.5)
        out["observation.images.wrist_image"] = torch.full_like(batch["observation.images.wrist_image"], 0.5)
        return out

    raise ValueError(condition)


def _as_task_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def new_group() -> dict:
    return {
        "sample_frames": 0,
        "valid_action_steps": 0,
        "norm_sum": 0.0,
        "raw_sum": 0.0,
        "denom": 0,
        "episodes": set(),
    }


def update_group(acc: dict, key, norm_abs: torch.Tensor, raw_abs: torch.Tensor, valid: torch.Tensor, episode_index: int) -> None:
    item = acc[key]
    valid_steps = int(valid.sum().detach().cpu().item())
    action_dim = int(norm_abs.shape[-1])
    item["sample_frames"] += 1
    item["valid_action_steps"] += valid_steps
    item["denom"] += valid_steps * action_dim
    item["norm_sum"] += float((norm_abs * valid.unsqueeze(-1)).sum().detach().cpu().item())
    item["raw_sum"] += float((raw_abs * valid.unsqueeze(-1)).sum().detach().cpu().item())
    item["episodes"].add(int(episode_index))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def save_contact_sheet(path: Path, first_batch: dict, conditions: list[str], noise_seed: int, noise_std: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h = 256, 256
    label_h = 30
    pad = 12
    font = load_font(18)
    width = pad + len(conditions) * (cell_w + pad)
    height = pad + 2 * (label_h + cell_h + pad)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for col, condition in enumerate(conditions):
        shown = apply_condition(first_batch, condition, 0, noise_seed, noise_std)
        x = pad + col * (cell_w + pad)
        draw.text((x, pad), CONDITION_LABELS.get(condition, condition), fill=(35, 35, 35), font=font)
        for row, (key, name) in enumerate((("observation.images.image", "static"), ("observation.images.wrist_image", "wrist"))):
            y = pad + label_h + row * (cell_h + label_h + pad)
            image = tensor_to_pil(shown[key][0]).resize((cell_w, cell_h), Image.BILINEAR)
            canvas.paste(image, (x, y))
            draw.text((x + 4, y + 4), name, fill=(255, 255, 255), font=font)
    canvas.save(path)


def rows_from_group(acc: dict, key_names: list[str]) -> list[dict]:
    rows = []
    for key, values in acc.items():
        if values["denom"] <= 0:
            continue
        row = dict(zip(key_names, key, strict=True)) if isinstance(key, tuple) else {key_names[0]: key}
        row.update(
            {
                "sample_frames": values["sample_frames"],
                "episode_count": len(values["episodes"]),
                "valid_action_steps": values["valid_action_steps"],
                "normalized_action_l1_valid_mean": values["norm_sum"] / values["denom"],
                "raw_action_l1_valid_mean": values["raw_sum"] / values["denom"],
            }
        )
        rows.append(row)
    return rows


def evaluate_run(
    experiment: str,
    run_dir: Path,
    dataset: CalvinDDACTDataset,
    dataloader: DataLoader,
    first_batch: dict,
    device: torch.device,
    checkpoint_name: str,
    conditions: list[str],
    noise_seed: int,
    noise_std: float,
    log_every: int,
) -> dict[str, list[dict]]:
    summary = read_json(run_dir / "metrics_summary.json")
    config = read_yaml(run_dir / "config.yaml")
    checkpoint_path = run_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.exists():
        checkpoint_path = Path(summary["best_val_action_l1_checkpoint"])
        checkpoint_name = checkpoint_path.name

    train_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="train")
    stats = compute_records_stats(train_records, dataset_root=Path(config["train"]["dataset"]["dataset_root"]))
    policy, policy_config, load_info, task_conditioning = load_policy(run_dir, checkpoint_path, device)
    if task_conditioning and bool(task_conditioning.get("enabled", False)):
        raise RuntimeError(f"{experiment} is task-conditioned; this diagnostic is visual-only by design.")

    action_mean = stats[ACTION]["mean"].to(device)
    action_std = stats[ACTION]["std"].to(device).clamp_min(1e-6)

    raw_first, norm_first = normalize_for_policy(first_batch, stats, device)
    with torch.no_grad():
        first_pred = policy.predict_action_chunk(norm_first)
    if list(first_pred.shape) != list(raw_first[ACTION].shape):
        raise RuntimeError(f"Unexpected prediction shape for {experiment}: {list(first_pred.shape)}")

    summary_acc = defaultdict(new_group)
    episode_acc = defaultdict(new_group)
    task_acc = defaultdict(new_group)
    task_type_acc = defaultdict(new_group)
    pos_norm_sum = {condition: torch.zeros(policy_config.chunk_size, dtype=torch.float64) for condition in conditions}
    pos_raw_sum = {condition: torch.zeros(policy_config.chunk_size, dtype=torch.float64) for condition in conditions}
    pos_denom = {condition: torch.zeros(policy_config.chunk_size, dtype=torch.float64) for condition in conditions}
    dim_norm_sum = {condition: torch.zeros(len(ACTION_DIM_NAMES), dtype=torch.float64) for condition in conditions}
    dim_raw_sum = {condition: torch.zeros(len(ACTION_DIM_NAMES), dtype=torch.float64) for condition in conditions}
    dim_valid = {condition: 0 for condition in conditions}

    start = time.perf_counter()
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if log_every > 0 and (batch_index == 0 or (batch_index + 1) % log_every == 0):
                elapsed = time.perf_counter() - start
                print(
                    f"[{experiment}] batch {batch_index + 1}/{total_batches}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
            tasks = _as_task_list(batch["task"])
            episode_indices = batch["episode_index"].detach().cpu().numpy().tolist()
            for condition in conditions:
                conditioned = apply_condition(batch, condition, batch_index, noise_seed, noise_std)
                raw_batch, norm_batch = normalize_for_policy(conditioned, stats, device)
                pred_norm = policy.predict_action_chunk(norm_batch)
                gt_norm = norm_batch[ACTION]
                gt_raw = raw_batch[ACTION]
                pred_raw = pred_norm * action_std + action_mean
                valid = ~raw_batch["action_is_pad"]
                norm_abs = torch.abs(pred_norm - gt_norm)
                raw_abs = torch.abs(pred_raw - gt_raw)

                valid_float = valid.unsqueeze(-1)
                norm_sum_by_pos = (norm_abs * valid_float).sum(dim=(0, 2)).detach().cpu().double()
                raw_sum_by_pos = (raw_abs * valid_float).sum(dim=(0, 2)).detach().cpu().double()
                denom_by_pos = valid.sum(dim=0).detach().cpu().double() * int(norm_abs.shape[-1])
                pos_norm_sum[condition] += norm_sum_by_pos
                pos_raw_sum[condition] += raw_sum_by_pos
                pos_denom[condition] += denom_by_pos

                dim_norm_sum[condition] += (norm_abs * valid_float).sum(dim=(0, 1)).detach().cpu().double()
                dim_raw_sum[condition] += (raw_abs * valid_float).sum(dim=(0, 1)).detach().cpu().double()
                dim_valid[condition] += int(valid.sum().detach().cpu().item())

                for sample_idx, task in enumerate(tasks):
                    episode_index = int(episode_indices[sample_idx])
                    sample_norm = norm_abs[sample_idx]
                    sample_raw = raw_abs[sample_idx]
                    sample_valid = valid[sample_idx]
                    update_group(summary_acc, (experiment, condition), sample_norm, sample_raw, sample_valid, episode_index)
                    update_group(
                        episode_acc,
                        (experiment, condition, episode_index, task, task_type(task)),
                        sample_norm,
                        sample_raw,
                        sample_valid,
                        episode_index,
                    )
                    update_group(task_acc, (experiment, condition, task), sample_norm, sample_raw, sample_valid, episode_index)
                    update_group(task_type_acc, (experiment, condition, task_type(task)), sample_norm, sample_raw, sample_valid, episode_index)

    runtime = time.perf_counter() - start

    summary_rows = rows_from_group(summary_acc, ["experiment", "condition"])
    for row in summary_rows:
        row.update(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": load_info["checkpoint_step"],
                "policy_chunk_size": policy_config.chunk_size,
                "d_episodes": dataset.num_episodes,
                "d_frames": dataset.num_frames,
                "source_best_val_action_l1_loss_reference": summary["best_val_action_l1_loss"],
                "runtime_seconds_for_run": runtime,
            }
        )
        if not math.isfinite(float(row["normalized_action_l1_valid_mean"])):
            raise RuntimeError(f"Non-finite Action L1 for {experiment}/{row['condition']}")

    position_rows = []
    bin_rows = []
    tail_head_rows = []
    for condition in conditions:
        for position in range(policy_config.chunk_size):
            denom = float(pos_denom[condition][position].item())
            if denom <= 0:
                continue
            position_rows.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "chunk_position_1based": position + 1,
                    "valid_action_values": int(denom),
                    "normalized_action_l1_valid_mean": float(pos_norm_sum[condition][position].item() / denom),
                    "raw_action_l1_valid_mean": float(pos_raw_sum[condition][position].item() / denom),
                }
            )
        for bin_idx in range(10):
            start_pos = bin_idx * 10
            end_pos = min((bin_idx + 1) * 10, policy_config.chunk_size)
            denom = float(pos_denom[condition][start_pos:end_pos].sum().item())
            if denom <= 0:
                continue
            bin_rows.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "bin_index_1based": bin_idx + 1,
                    "position_start_1based": start_pos + 1,
                    "position_end_1based": end_pos,
                    "valid_action_values": int(denom),
                    "normalized_action_l1_valid_mean": float(
                        pos_norm_sum[condition][start_pos:end_pos].sum().item() / denom
                    ),
                    "raw_action_l1_valid_mean": float(pos_raw_sum[condition][start_pos:end_pos].sum().item() / denom),
                }
            )
        head_denom = float(pos_denom[condition][0:10].sum().item())
        tail_denom = float(pos_denom[condition][90:100].sum().item())
        if head_denom > 0 and tail_denom > 0:
            head = float(pos_norm_sum[condition][0:10].sum().item() / head_denom)
            tail = float(pos_norm_sum[condition][90:100].sum().item() / tail_denom)
            tail_head_rows.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "head_positions": "1-10",
                    "tail_positions": "91-100",
                    "head_normalized_action_l1": head,
                    "tail_normalized_action_l1": tail,
                    "tail_head_ratio": tail / max(head, 1e-12),
                    "head_valid_action_values": int(head_denom),
                    "tail_valid_action_values": int(tail_denom),
                }
            )

    dim_rows = []
    for condition in conditions:
        for dim, name in enumerate(ACTION_DIM_NAMES):
            denom = max(int(dim_valid[condition]), 1)
            dim_rows.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "action_dim": dim,
                    "action_dim_name": name,
                    "valid_action_steps": int(dim_valid[condition]),
                    "normalized_action_l1_valid_mean": float(dim_norm_sum[condition][dim].item() / denom),
                    "raw_action_l1_valid_mean": float(dim_raw_sum[condition][dim].item() / denom),
                }
            )

    return {
        "summary": summary_rows,
        "chunk_position": position_rows,
        "chunk_bin": bin_rows,
        "tail_head": tail_head_rows,
        "per_action_dim": dim_rows,
        "per_episode": rows_from_group(
            episode_acc,
            ["experiment", "condition", "episode_index", "task", "task_type"],
        ),
        "per_task": rows_from_group(task_acc, ["experiment", "condition", "task"]),
        "per_task_type": rows_from_group(task_type_acc, ["experiment", "condition", "task_type"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D_DATASET_ROOT)
    parser.add_argument("--checkpoint-name", default="best_val_action_l1.pt")
    parser.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR")
    parser.add_argument("--only-custom-runs", action="store_true")
    parser.add_argument("--output-prefix", default="visual_chunk_robustness")
    parser.add_argument("--table-dir", type=Path, default=PROJECT_ROOT / "results/tables/step4")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "results/figures/step4")
    parser.add_argument("--max-episodes", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--table-cache-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--noise-seed", type=int, default=20260617)
    parser.add_argument("--noise-std", type=float, default=0.06)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    conditions = parse_conditions(args.conditions)
    runs = parse_run_specs(args.run, args.only_custom_runs)
    configure_episode_table_cache(args.table_cache_size)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataset = CalvinDDACTDataset(dataset_root=args.dataset_root, chunk_size=100, max_episodes=args.max_episodes)
    sanity_dataset = CalvinDDACTDataset(dataset_root=args.dataset_root, chunk_size=100, max_episodes=1)
    first_batch = next(iter(DataLoader(sanity_dataset, batch_size=1, shuffle=False, num_workers=0)))
    save_contact_sheet(
        args.figure_dir / f"{args.output_prefix}_visual_shift_contact_sheet.png",
        first_batch,
        conditions,
        args.noise_seed,
        args.noise_std,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    report = {
        "status": "running",
        "start_time": now_iso(),
        "dataset_root": str(args.dataset_root),
        "runs": {name: str(path) for name, path in runs.items()},
        "device": str(device),
        "max_episodes": args.max_episodes,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "table_cache": episode_table_cache_status(),
        "conditions": conditions,
        "noise_seed": args.noise_seed,
        "noise_std": args.noise_std,
    }
    report_path = args.table_dir / f"{args.output_prefix}_visual_robustness_report.json"
    write_json(report_path, report)

    outputs = {
        "summary": [],
        "chunk_position": [],
        "chunk_bin": [],
        "tail_head": [],
        "per_action_dim": [],
        "per_episode": [],
        "per_task": [],
        "per_task_type": [],
    }
    for experiment, run_dir in runs.items():
        result = evaluate_run(
            experiment=experiment,
            run_dir=run_dir,
            dataset=dataset,
            dataloader=dataloader,
            first_batch=first_batch,
            device=device,
            checkpoint_name=args.checkpoint_name,
            conditions=conditions,
            noise_seed=args.noise_seed,
            noise_std=args.noise_std,
            log_every=args.log_every,
        )
        for key, rows in result.items():
            outputs[key].extend(rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    file_map = {
        "summary": "visual_robustness_summary.csv",
        "chunk_position": "chunk_position_action_l1.csv",
        "chunk_bin": "chunk_bin_action_l1.csv",
        "tail_head": "chunk_tail_head_ratio.csv",
        "per_action_dim": "visual_robustness_per_action_dim.csv",
        "per_episode": "visual_robustness_per_episode_action_l1.csv",
        "per_task": "visual_robustness_per_task_action_l1.csv",
        "per_task_type": "visual_robustness_per_task_type_action_l1.csv",
    }
    table_paths = {}
    for key, filename in file_map.items():
        path = args.table_dir / f"{args.output_prefix}_{filename}"
        if outputs[key]:
            write_csv(path, outputs[key])
        else:
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(["empty"])
        table_paths[key] = str(path)

    report.update(
        {
            "status": "ok",
            "end_time": now_iso(),
            "dataset_episodes": dataset.num_episodes,
            "dataset_frames": dataset.num_frames,
            "tables": table_paths,
            "contact_sheet": str(args.figure_dir / f"{args.output_prefix}_visual_shift_contact_sheet.png"),
        }
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
