#!/usr/bin/env python3
"""Run full ACT training with fixed episode-level validation split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml
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
    configure_video_frame_cache,
    compute_records_stats,
    select_episode_records,
    summarize_records,
    video_frame_cache_status,
)
from src.data.task_text_mapping import TaskTextMapper  # noqa: E402
from src.models.task_conditioned_act import (  # noqa: E402
    TaskConditionedACTPolicy,
    VisualCheckpointLoadReport,
    load_visual_act_checkpoint,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_stdout_log(path: Path, append: bool) -> object:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a" if append else "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)
    return handle


def trim_jsonl_to_step(path: Path, max_step: int) -> dict:
    if not path.exists():
        return {"path": str(path), "existed": False, "trimmed": False}
    rows = []
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rows.append(line)
            continue
        step = row.get("step")
        if isinstance(step, int) and step > max_step:
            removed += 1
            continue
        rows.append(json.dumps(row, ensure_ascii=False))
    if removed:
        backup = path.with_suffix(path.suffix + f".pre_resume_{now_iso().replace(':', '').replace('+', '_')}.bak")
        shutil.copy2(path, backup)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        return {
            "path": str(path),
            "existed": True,
            "trimmed": True,
            "removed_rows": removed,
            "backup": str(backup),
        }
    return {"path": str(path), "existed": True, "trimmed": False, "removed_rows": 0}


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def apply_resume_lr_override(optimizer: torch.optim.Optimizer, train_cfg: dict) -> dict | None:
    """Optionally override restored optimizer learning rates after resume."""

    if "resume_override_lr" not in train_cfg and "resume_override_backbone_lr" not in train_cfg:
        return None
    if "resume_override_lr" not in train_cfg:
        raise ValueError("resume_override_backbone_lr requires resume_override_lr")

    override_lr = float(train_cfg["resume_override_lr"])
    if override_lr <= 0:
        raise ValueError(f"resume_override_lr must be positive, got {override_lr}")
    override_backbone_lr_raw = train_cfg.get("resume_override_backbone_lr")
    override_backbone_lr = (
        float(override_backbone_lr_raw) if override_backbone_lr_raw is not None else override_lr
    )
    if override_backbone_lr <= 0:
        raise ValueError(f"resume_override_backbone_lr must be positive, got {override_backbone_lr}")

    old_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    new_lrs: list[float] = []
    for index, group in enumerate(optimizer.param_groups):
        new_lr = override_backbone_lr if index == 1 else override_lr
        group["lr"] = new_lr
        new_lrs.append(float(new_lr))
    return {
        "old_lrs": old_lrs,
        "new_lrs": new_lrs,
        "resume_override_lr": override_lr,
        "resume_override_backbone_lr": override_backbone_lr,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy_config(policy_yaml: dict, device: str, task_conditioning: dict | None = None) -> ACTConfig:
    policy_cfg = deepcopy(policy_yaml["policy"])
    task_conditioning = task_conditioning or {}
    if bool(task_conditioning.get("enabled", False)):
        state_cfg = policy_cfg["input_features"][OBS_STATE]
        base_shape = list(state_cfg["shape"])
        if len(base_shape) != 1:
            raise ValueError(f"Expected 1D robot state shape, got {base_shape}")
        task_embedding_dim = int(task_conditioning["embedding_dim"])
        state_cfg["shape"] = [int(base_shape[0]) + task_embedding_dim]
    policy_cfg["device"] = device
    policy_cfg["input_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in policy_cfg["input_features"].items()
    }
    policy_cfg["output_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in policy_cfg["output_features"].items()
    }
    policy_cfg.pop("type", None)
    return ACTConfig(**policy_cfg)


def task_conditioning_config(policy_yaml: dict) -> dict:
    cfg = deepcopy(policy_yaml.get("task_conditioning", {}))
    cfg.setdefault("enabled", False)
    return cfg


def serializable_policy_config(config: ACTConfig) -> dict:
    return {
        "type": "act",
        "input_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.input_features.items()
        },
        "output_features": {
            key: {"type": value.type.value, "shape": list(value.shape)}
            for key, value in config.output_features.items()
        },
        "vision_backbone": config.vision_backbone,
        "pretrained_backbone_weights": config.pretrained_backbone_weights,
        "use_vae": config.use_vae,
        "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps,
        "dim_model": config.dim_model,
        "n_heads": config.n_heads,
        "dim_feedforward": config.dim_feedforward,
        "n_encoder_layers": config.n_encoder_layers,
        "n_decoder_layers": config.n_decoder_layers,
        "n_vae_encoder_layers": config.n_vae_encoder_layers,
        "kl_weight": config.kl_weight,
        "dropout": config.dropout,
        "optimizer_lr": config.optimizer_lr,
        "optimizer_weight_decay": config.optimizer_weight_decay,
        "optimizer_lr_backbone": config.optimizer_lr_backbone,
        "device": config.device,
    }


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


def get_autocast(amp_mode: str):
    if amp_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def infinite_loader(dataloader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in dataloader:
            yield batch


def split_train_val(
    records: list[EpisodeRecord],
    validation_fraction: float,
    seed: int,
) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
    """Deterministic stratified episode split by environment."""

    rng = random.Random(seed)
    by_env: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_env[record.environment].append(record)

    train_records: list[EpisodeRecord] = []
    val_records: list[EpisodeRecord] = []
    for env in sorted(by_env):
        env_records = list(by_env[env])
        rng.shuffle(env_records)
        val_count = max(1, int(round(len(env_records) * validation_fraction)))
        val_records.extend(env_records[:val_count])
        train_records.extend(env_records[val_count:])

    train_records.sort(key=lambda record: (record.shard, record.episode_index))
    val_records.sort(key=lambda record: (record.shard, record.episode_index))
    return train_records, val_records


def write_split_manifest(path: Path, train_records: list[EpisodeRecord], val_records: list[EpisodeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "shard",
                "episode_index",
                "original_frame_idx_min",
                "original_frame_idx_max",
                "environment",
                "num_frames",
                "task",
            ],
        )
        writer.writeheader()
        for split_name, records in (("train", train_records), ("val", val_records)):
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "shard": record.shard,
                        "episode_index": record.episode_index,
                        "original_frame_idx_min": record.original_frame_idx_min,
                        "original_frame_idx_max": record.original_frame_idx_max,
                        "environment": record.environment,
                        "num_frames": record.num_frames,
                        "task": record.task,
                    }
                )


def make_dataloader(
    dataset: CalvinACTTrainingDataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": shuffle,
        "persistent_workers": persistent_workers and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def action_l1_from_loss_dict(output_dict: dict) -> float:
    return float(output_dict.get("l1_loss", math.nan))


def save_checkpoint(
    path: Path,
    step: int,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_config: ACTConfig,
    metrics: dict,
    task_conditioning: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": serializable_policy_config(policy_config),
            "task_conditioning": deepcopy(task_conditioning or {"enabled": False}),
            "metrics": metrics,
        },
        path,
    )


def checkpoint_readable(path: Path, expected_step: int | None = None) -> bool:
    if not path.exists():
        return False
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if "policy_state_dict" not in loaded:
        return False
    if expected_step is not None and loaded.get("step") != expected_step:
        return False
    return True


@torch.no_grad()
def run_validation(
    policy: torch.nn.Module,
    val_loader: DataLoader,
    stats: dict,
    device: torch.device,
    amp_mode: str,
    max_batches: int | None = None,
) -> dict:
    # LeRobot ACT's VAE loss path requires the ACT model to stay in train mode
    # because latent parameters are only produced when self.training is true.
    # Keep train mode for the ACT objective, but freeze BatchNorm running stats.
    bn_states = []
    for module in policy.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_states.append((module, module.training))
            module.eval()
    total_batches = 0
    total_samples = 0
    weighted_loss = 0.0
    weighted_action_l1 = 0.0
    start = time.perf_counter()

    try:
        for batch in val_loader:
            batch_size = int(batch["action"].shape[0])
            batch = normalize_batch(batch, stats, device)
            with get_autocast(amp_mode):
                loss, output_dict = policy.forward(batch)
            loss_value = float(loss.detach().cpu().item())
            action_l1 = action_l1_from_loss_dict(output_dict)
            if not math.isfinite(loss_value) or not math.isfinite(action_l1):
                raise RuntimeError(f"Non-finite validation metric: loss={loss_value}, action_l1={action_l1}")
            weighted_loss += loss_value * batch_size
            weighted_action_l1 += action_l1 * batch_size
            total_samples += batch_size
            total_batches += 1
            if max_batches is not None and total_batches >= max_batches:
                break
    finally:
        for module, training in bn_states:
            module.train(training)

    runtime = time.perf_counter() - start
    return {
        "val_loss": weighted_loss / total_samples,
        "val_action_l1_loss": weighted_action_l1 / total_samples,
        "val_batches": total_batches,
        "val_samples": total_samples,
        "val_runtime_seconds": runtime,
        "val_max_batches": max_batches,
        "validation_forward_mode": "train_mode_no_grad_batchnorm_eval",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", type=Path, default=PROJECT_ROOT / "configs/policy/act_baseline.yaml")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--stdout-log", type=Path, default=None)
    args = parser.parse_args()

    policy_yaml = read_yaml(args.policy_config)
    train_yaml = read_yaml(args.train_config)
    train_cfg = train_yaml["train"]
    dataset_cfg = train_yaml["dataset"]
    task_cfg = task_conditioning_config(policy_yaml)
    task_cfg.update(deepcopy(train_yaml.get("task_conditioning", {})))
    task_conditioning_enabled = bool(task_cfg.get("enabled", False))

    output_dir = Path(train_yaml["experiment"]["output_dir"])
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stdout_log_path = Path(args.stdout_log) if args.stdout_log else logs_dir / "train_stdout.log"
    stdout_log_handle = setup_stdout_log(stdout_log_path, append=args.resume_from is not None)

    status_path = output_dir / "training_status.yaml"
    scalars_path = logs_dir / "scalars.jsonl"
    validation_path = logs_dir / "validation.jsonl"
    resume_checkpoint = None
    resume_start_step = 0
    resume_log_actions = []
    if args.resume_from is not None:
        if not args.resume_from.exists():
            raise FileNotFoundError(f"--resume-from checkpoint does not exist: {args.resume_from}")
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if "policy_state_dict" not in resume_checkpoint or "optimizer_state_dict" not in resume_checkpoint:
            raise ValueError(f"Resume checkpoint is missing policy/optimizer state: {args.resume_from}")
        resume_start_step = int(resume_checkpoint.get("step", 0))
        if resume_start_step <= 0:
            raise ValueError(f"Resume checkpoint has invalid step: {resume_start_step}")
        resume_log_actions = [
            trim_jsonl_to_step(scalars_path, resume_start_step),
            trim_jsonl_to_step(validation_path, resume_start_step),
        ]
    else:
        for path in (scalars_path, validation_path):
            if path.exists():
                path.unlink()

    status = {
        "experiment": train_yaml["experiment"]["name"],
        "status": "starting",
        "start_time": now_iso(),
        "output_dir": str(output_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": os.environ.get("SELECTED_PHYSICAL_GPU"),
        "steps_target": int(train_cfg["steps"]),
        "resume_from": str(args.resume_from) if args.resume_from is not None else "",
        "resume_start_step": resume_start_step,
        "resume_log_actions": resume_log_actions,
        "stdout_log": str(stdout_log_path),
    }
    write_yaml(status_path, status)

    seed = int(train_cfg["seed"])
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full training.")
    device = torch.device(train_cfg["device"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    amp_raw = train_cfg.get("amp", "off")
    amp_mode = "off" if amp_raw is False else str(amp_raw)
    if amp_mode not in {"off", "bf16"}:
        raise ValueError(f"Unsupported amp mode: {amp_mode}")

    if task_conditioning_enabled:
        task_mapper = TaskTextMapper.from_json(Path(task_cfg["vocab_path"]))
        if int(task_cfg["num_tasks"]) != task_mapper.num_tasks:
            raise ValueError(
                f"task_conditioning.num_tasks={task_cfg['num_tasks']} does not match "
                f"vocab num_tasks={task_mapper.num_tasks}"
            )
    policy_config = build_policy_config(policy_yaml, str(device), task_cfg)
    split_name = dataset_cfg["split_name"]
    records = select_episode_records(split_name, Path(dataset_cfg["split_manifest"]))
    train_records, val_records = split_train_val(
        records=records,
        validation_fraction=float(dataset_cfg["validation_fraction"]),
        seed=seed,
    )
    split_manifest_path = output_dir / "train_val_episode_split.csv"
    write_split_manifest(split_manifest_path, train_records, val_records)

    stats = compute_records_stats(train_records, dataset_root=Path(dataset_cfg["dataset_root"]))

    video_frame_cache_size = int(
        train_cfg.get("video_frame_cache_size", os.environ.get("CALVIN_VIDEO_FRAME_CACHE_SIZE", 32))
    )
    configure_video_frame_cache(video_frame_cache_size)
    video_cache_status = video_frame_cache_status()

    train_dataset = CalvinACTTrainingDataset(
        split_name=split_name,
        dataset_root=Path(dataset_cfg["dataset_root"]),
        split_manifest=Path(dataset_cfg["split_manifest"]),
        chunk_size=policy_config.chunk_size,
        image_keys=dataset_cfg["image_keys"],
        episode_records=train_records,
        task_vocab_path=Path(task_cfg["vocab_path"]) if task_conditioning_enabled else None,
    )
    val_dataset = CalvinACTTrainingDataset(
        split_name=split_name,
        dataset_root=Path(dataset_cfg["dataset_root"]),
        split_manifest=Path(dataset_cfg["split_manifest"]),
        chunk_size=policy_config.chunk_size,
        image_keys=dataset_cfg["image_keys"],
        episode_records=val_records,
        task_vocab_path=Path(task_cfg["vocab_path"]) if task_conditioning_enabled else None,
    )

    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg["num_workers"])
    train_prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    val_prefetch_factor = int(train_cfg.get("validation_prefetch_factor", train_cfg.get("prefetch_factor", 2)))
    pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))
    train_loader = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        shuffle=True,
        prefetch_factor=train_prefetch_factor,
        persistent_workers=bool(train_cfg.get("persistent_workers", True)),
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=int(train_cfg.get("validation_num_workers", num_workers)),
        seed=seed + 1,
        shuffle=False,
        prefetch_factor=val_prefetch_factor,
        persistent_workers=bool(train_cfg.get("validation_persistent_workers", False)),
        pin_memory=pin_memory,
    )

    first_batch = next(iter(train_loader))
    first_batch_shapes = {
        key: list(value.shape) if torch.is_tensor(value) else f"{type(value).__name__}[{len(value)}]"
        for key, value in first_batch.items()
    }
    first_batch_envs = sorted(set(first_batch["environment"]))
    first_batch_task_ids = first_batch.get(str(task_cfg.get("task_id_key", "canonical_task_id")))
    first_batch_task_id_range = None
    if task_conditioning_enabled:
        if first_batch_task_ids is None:
            raise RuntimeError("Task conditioning enabled but first batch has no canonical task ids")
        first_batch_task_id_range = [
            int(first_batch_task_ids.min().item()),
            int(first_batch_task_ids.max().item()),
        ]
    if split_name == "B_only" and first_batch_envs != ["B"]:
        raise RuntimeError(f"B_only first batch contains non-B environments: {first_batch_envs}")
    if split_name == "ABC_mixed" and not set(first_batch_envs).issubset({"A", "B", "C"}):
        raise RuntimeError(f"ABC_mixed first batch contains invalid environments: {first_batch_envs}")

    print(f"Experiment: {train_yaml['experiment']['name']}", flush=True)
    print(f"Split: {split_name}", flush=True)
    print(f"Train summary: {summarize_records(train_records)}", flush=True)
    print(f"Val summary: {summarize_records(val_records)}", flush=True)
    print(f"First train batch shapes: {first_batch_shapes}", flush=True)
    print(f"First train batch envs: {first_batch_envs}", flush=True)
    if task_conditioning_enabled:
        print(f"First train batch task_id range: {first_batch_task_id_range}", flush=True)
        print("Creating task-id conditioned ACTPolicy wrapper", flush=True)
    else:
        print("Creating LeRobot ACTPolicy", flush=True)

    visual_init_report: VisualCheckpointLoadReport | None = None
    if task_conditioning_enabled:
        policy = TaskConditionedACTPolicy(
            config=policy_config,
            num_tasks=int(task_cfg["num_tasks"]),
            task_embedding_dim=int(task_cfg["embedding_dim"]),
            task_id_key=str(task_cfg.get("task_id_key", "canonical_task_id")),
        ).to(device)
        init_checkpoint = task_cfg.get("init_from_visual_checkpoint")
        if init_checkpoint:
            if resume_checkpoint is None:
                visual_init_report = load_visual_act_checkpoint(policy, Path(init_checkpoint))
                if not visual_init_report.clean:
                    print(
                        f"Visual checkpoint load was not clean; using scratch task-conditioned initialization: "
                        f"{visual_init_report.note}",
                        flush=True,
                    )
                    policy = TaskConditionedACTPolicy(
                        config=policy_config,
                        num_tasks=int(task_cfg["num_tasks"]),
                        task_embedding_dim=int(task_cfg["embedding_dim"]),
                        task_id_key=str(task_cfg.get("task_id_key", "canonical_task_id")),
                    ).to(device)
                else:
                    print(
                        f"Loaded compatible visual-only checkpoint weights: "
                        f"exact={visual_init_report.exact_loaded} expanded={visual_init_report.expanded_loaded}",
                        flush=True,
                    )
            else:
                print(
                    f"Skipping visual checkpoint init because --resume-from will restore full task-conditioned state: "
                    f"{args.resume_from}",
                    flush=True,
                )
    else:
        policy = ACTPolicy(policy_config).to(device)
    policy.train()
    optimizer = policy_config.get_optimizer_preset().build(policy.get_optim_params())
    grad_clip_norm = float(train_cfg["grad_clip_norm"])

    resume_load_info = None
    resume_lr_override_info = None
    if resume_checkpoint is not None:
        load_result = policy.load_state_dict(resume_checkpoint["policy_state_dict"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        resume_lr_override_info = apply_resume_lr_override(optimizer, train_cfg)
        resume_metrics = deepcopy(resume_checkpoint.get("metrics", {}))
        resume_load_info = {
            "checkpoint_path": str(args.resume_from),
            "checkpoint_step": resume_start_step,
            "policy_missing_keys": list(load_result.missing_keys),
            "policy_unexpected_keys": list(load_result.unexpected_keys),
            "optimizer_restored": True,
            "dataloader_position_restored": False,
            "dataloader_resume_note": (
                "Model and optimizer states are restored, but DataLoader iterator position is not bit-exactly "
                "restored; the shuffled loader restarts from its seeded stream."
            ),
            "lr_override": resume_lr_override_info,
            "checkpoint_metrics": resume_metrics,
        }
        print(
            f"Resumed checkpoint {args.resume_from} at step={resume_start_step}; "
            f"continuing to steps_target={int(train_cfg['steps'])}",
            flush=True,
        )
        if resume_lr_override_info is not None:
            print(
                "Applied resume LR override: "
                f"old_lrs={resume_lr_override_info['old_lrs']} "
                f"new_lrs={resume_lr_override_info['new_lrs']}",
                flush=True,
            )

    snapshot = {
        "policy_config_file": str(args.policy_config),
        "train_config_file": str(args.train_config),
        "policy_actual": serializable_policy_config(policy_config),
        "task_conditioning": task_cfg,
        "visual_checkpoint_init": visual_init_report.to_dict() if visual_init_report is not None else None,
        "train": train_yaml,
        "runtime": {
            "python": sys.executable,
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu": os.environ.get("SELECTED_PHYSICAL_GPU"),
            "torch_device": str(device),
            "torch_gpu_name": torch.cuda.get_device_name(0),
            "amp": amp_mode,
            "stdout_log": str(stdout_log_path),
            "video_frame_cache": video_cache_status,
            "pin_memory": pin_memory,
        },
        "optimizer_param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "resume": resume_load_info,
        "split": {
            "validation_fraction": float(dataset_cfg["validation_fraction"]),
            "train_summary": summarize_records(train_records),
            "val_summary": summarize_records(val_records),
            "episode_split_csv": str(split_manifest_path),
            "first_batch_shapes": first_batch_shapes,
            "first_batch_task_id_range": first_batch_task_id_range,
        },
    }
    write_yaml(output_dir / "config.yaml", snapshot)
    shutil.copy2(args.policy_config, output_dir / "source_policy_act_baseline.yaml")
    shutil.copy2(args.train_config, output_dir / f"source_{args.train_config.name}")

    status.update(
        {
            "status": "running",
            "policy_class": (
                "src.models.task_conditioned_act.TaskConditionedACTPolicy"
                if task_conditioning_enabled
                else "lerobot.policies.act.modeling_act.ACTPolicy"
            ),
            "optimizer": optimizer.__class__.__name__,
            "optimizer_param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
            "amp": amp_mode,
            "train_num_workers": num_workers,
            "validation_num_workers": int(train_cfg.get("validation_num_workers", num_workers)),
            "train_prefetch_factor": train_prefetch_factor,
            "validation_prefetch_factor": val_prefetch_factor,
            "pin_memory": pin_memory,
            "video_frame_cache": video_cache_status,
            "train_summary": summarize_records(train_records),
            "val_summary": summarize_records(val_records),
            "first_batch_shapes": first_batch_shapes,
            "first_batch_environments": first_batch_envs,
            "first_batch_task_id_range": first_batch_task_id_range,
            "task_conditioning": task_cfg,
            "visual_checkpoint_init": visual_init_report.to_dict() if visual_init_report is not None else None,
            "resume": resume_load_info,
            "checkpoints_dir": str(ckpt_dir),
            "scalars_jsonl": str(scalars_path),
            "validation_jsonl": str(validation_path),
            "stdout_log": str(stdout_log_path),
        }
    )
    write_yaml(status_path, status)

    total_steps = int(train_cfg["steps"])
    if resume_start_step >= total_steps:
        raise ValueError(
            f"Resume checkpoint step {resume_start_step} is >= train.steps {total_steps}. "
            "Increase train.steps to append more training."
        )
    log_interval = int(train_cfg["log_interval"])
    checkpoint_interval = int(train_cfg["checkpoint_interval"])
    validation_interval = int(train_cfg["validation_interval"])
    best_checkpoint = ckpt_dir / "best_val_action_l1.pt"
    latest_checkpoint = ckpt_dir / "latest.pt"
    final_checkpoint = ckpt_dir / "final.pt"
    train_step_times: list[float] = []
    resume_metrics_for_loop = deepcopy(resume_checkpoint.get("metrics", {})) if resume_checkpoint is not None else {}
    best_val_action_l1 = float(resume_metrics_for_loop.get("best_val_action_l1_loss", float("inf")))
    last_train_record: dict | None = resume_metrics_for_loop.get("train")
    last_val_record: dict | None = resume_metrics_for_loop.get("validation")
    start = time.perf_counter()

    iterator = infinite_loader(train_loader)

    for step in range(resume_start_step + 1, total_steps + 1):
        step_start = time.perf_counter()
        batch = next(iterator)
        batch = normalize_batch(batch, stats, device)

        with get_autocast(amp_mode):
            loss, output_dict = policy.forward(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}: {loss.item()}")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_seconds = time.perf_counter() - step_start
        train_step_times.append(step_seconds)
        loss_value = float(loss.detach().cpu().item())
        action_l1 = action_l1_from_loss_dict(output_dict)
        if not math.isfinite(action_l1):
            raise RuntimeError(f"Non-finite training action_l1 at step {step}: {action_l1}")

        last_train_record = {
            "phase": "train",
            "step": step,
            "loss": loss_value,
            "action_l1_loss": action_l1,
            "kld_loss": float(output_dict.get("kld_loss", math.nan)),
            "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
            "lr": optimizer.param_groups[0]["lr"],
            "step_seconds": step_seconds,
            "samples_per_second": batch_size / step_seconds,
            "time": now_iso(),
        }
        append_jsonl(scalars_path, last_train_record)

        if step % log_interval == 0 or step == 1:
            print(
                f"step={step}/{total_steps} train_loss={loss_value:.6f} "
                f"train_action_l1={action_l1:.6f} grad_norm={last_train_record['grad_norm']:.6f} "
                f"lr={last_train_record['lr']:.2e} step_s={step_seconds:.3f}",
                flush=True,
            )

        should_validate = step % validation_interval == 0 or step == total_steps
        should_checkpoint = step % checkpoint_interval == 0 or step == total_steps
        if should_validate:
            validation_start_record = {
                "phase": "validation_start",
                "step": step,
                "time": now_iso(),
                "validation_num_workers": int(train_cfg.get("validation_num_workers", num_workers)),
            }
            append_jsonl(scalars_path, validation_start_record)
            status.update(
                {
                    "status": "validating",
                    "last_step": step,
                    "updated_at": now_iso(),
                    "validation_num_workers": int(train_cfg.get("validation_num_workers", num_workers)),
                }
            )
            write_yaml(status_path, status)
            print(
                f"validation_start step={step}/{total_steps} "
                f"val_num_workers={int(train_cfg.get('validation_num_workers', num_workers))}",
                flush=True,
            )
            val_metrics = run_validation(
                policy,
                val_loader,
                stats,
                device,
                amp_mode,
                max_batches=train_cfg.get("validation_max_batches"),
            )
            last_val_record = {
                "phase": "validation",
                "step": step,
                "time": now_iso(),
                **val_metrics,
            }
            append_jsonl(scalars_path, last_val_record)
            append_jsonl(validation_path, last_val_record)
            print(
                f"validation step={step}/{total_steps} val_loss={val_metrics['val_loss']:.6f} "
                f"val_action_l1={val_metrics['val_action_l1_loss']:.6f} "
                f"val_batches={val_metrics['val_batches']} val_s={val_metrics['val_runtime_seconds']:.1f}",
                flush=True,
            )

        if should_checkpoint:
            if last_val_record is None:
                raise RuntimeError("Checkpoint requested before validation metrics are available.")
            metrics = {
                "train": last_train_record,
                "validation": last_val_record,
                "best_val_action_l1_loss": best_val_action_l1,
            }
            step_checkpoint = ckpt_dir / f"step_{step:06d}.pt"
            save_checkpoint(step_checkpoint, step, policy, optimizer, policy_config, metrics, task_cfg)
            save_checkpoint(latest_checkpoint, step, policy, optimizer, policy_config, metrics, task_cfg)

            val_action_l1 = float(last_val_record["val_action_l1_loss"])
            if val_action_l1 < best_val_action_l1:
                best_val_action_l1 = val_action_l1
                metrics["best_val_action_l1_loss"] = best_val_action_l1
                save_checkpoint(best_checkpoint, step, policy, optimizer, policy_config, metrics, task_cfg)

            status.update(
                {
                    "status": "running",
                    "last_step": step,
                    "latest_checkpoint": str(latest_checkpoint),
                    "best_val_action_l1_checkpoint": str(best_checkpoint),
                    "best_val_action_l1_loss": best_val_action_l1,
                    "last_train_loss": loss_value,
                    "last_train_action_l1_loss": action_l1,
                    "last_val_action_l1_loss": val_action_l1,
                    "updated_at": now_iso(),
                }
            )
            write_yaml(status_path, status)

    runtime = time.perf_counter() - start
    if last_train_record is None or last_val_record is None:
        raise RuntimeError("Training finished without train/validation records.")

    final_metrics = {
        "train": last_train_record,
        "validation": last_val_record,
        "best_val_action_l1_loss": best_val_action_l1,
    }
    save_checkpoint(final_checkpoint, total_steps, policy, optimizer, policy_config, final_metrics, task_cfg)
    save_checkpoint(latest_checkpoint, total_steps, policy, optimizer, policy_config, final_metrics, task_cfg)

    checkpoint_ok = all(
        [
            checkpoint_readable(final_checkpoint, total_steps),
            checkpoint_readable(latest_checkpoint, total_steps),
            checkpoint_readable(best_checkpoint),
        ]
    )
    summary = {
        "status": "ok",
        "experiment": train_yaml["experiment"]["name"],
        "split": split_name,
        "steps": total_steps,
        "runtime_seconds": runtime,
        "mean_step_seconds": float(np.mean(train_step_times)),
        "mean_samples_per_second": float(batch_size / np.mean(train_step_times)),
        "final_loss": last_train_record["loss"],
        "final_action_l1_loss": last_train_record["action_l1_loss"],
        "final_val_loss": last_val_record["val_loss"],
        "final_val_action_l1_loss": last_val_record["val_action_l1_loss"],
        "best_val_action_l1_loss": best_val_action_l1,
        "latest_checkpoint": str(latest_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "best_val_action_l1_checkpoint": str(best_checkpoint),
        "checkpoint_readable": checkpoint_ok,
        "scalars_jsonl": str(scalars_path),
        "validation_jsonl": str(validation_path),
        "config_yaml": str(output_dir / "config.yaml"),
        "train_val_episode_split_csv": str(split_manifest_path),
        "train_summary": summarize_records(train_records),
        "val_summary": summarize_records(val_records),
        "first_batch_shapes": first_batch_shapes,
        "first_batch_task_id_range": first_batch_task_id_range,
        "task_conditioning": task_cfg,
        "visual_checkpoint_init": visual_init_report.to_dict() if visual_init_report is not None else None,
        "optimizer_param_group_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": os.environ.get("SELECTED_PHYSICAL_GPU"),
        "amp": amp_mode,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    status.update(
        {
            "status": "completed",
            "end_time": now_iso(),
            "runtime_seconds": runtime,
            "final_step": total_steps,
            "final_loss": summary["final_loss"],
            "final_action_l1_loss": summary["final_action_l1_loss"],
            "final_val_loss": summary["final_val_loss"],
            "final_val_action_l1_loss": summary["final_val_action_l1_loss"],
            "best_val_action_l1_loss": best_val_action_l1,
            "latest_checkpoint": str(latest_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "best_val_action_l1_checkpoint": str(best_checkpoint),
            "checkpoint_readable": checkpoint_ok,
        }
    )
    write_yaml(status_path, status)

    print("Full training completed", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
