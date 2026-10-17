#!/usr/bin/env python3
"""Run a 100-step ACT debug smoke test with a thin CALVIN v2.1 adapter."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

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
    compute_split_stats,
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy_config(policy_yaml: dict, train_yaml: dict, device: str) -> ACTConfig:
    policy_cfg = deepcopy(policy_yaml["policy"])
    policy_cfg.update(train_yaml.get("debug_policy_overrides", {}))
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
        if not torch.is_tensor(value):
            out[key] = value
            continue
        out[key] = value.to(device, non_blocking=True)

    for key in (OBS_STATE, ACTION, "observation.images.image", "observation.images.wrist_image"):
        if key not in out or key not in stats:
            continue
        mean = stats[key]["mean"].to(device)
        std = stats[key]["std"].to(device).clamp_min(1e-6)
        out[key] = (out[key] - mean) / std
    return out


def action_l1_from_loss_dict(output_dict: dict) -> float:
    return float(output_dict.get("l1_loss", math.nan))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", type=Path, default=PROJECT_ROOT / "configs/policy/act_baseline.yaml")
    parser.add_argument("--train-config", type=Path, default=PROJECT_ROOT / "configs/train/debug.yaml")
    args = parser.parse_args()

    policy_yaml = read_yaml(args.policy_config)
    train_yaml = read_yaml(args.train_config)
    output_dir = Path(train_yaml["experiment"]["output_dir"])
    logs_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    status_path = output_dir / "training_status.yaml"
    status = {
        "experiment": train_yaml["experiment"]["name"],
        "status": "starting",
        "start_time": now_iso(),
        "output_dir": str(output_dir),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu": os.environ.get("SELECTED_PHYSICAL_GPU"),
        "steps_target": train_yaml["train"]["steps"],
    }
    write_yaml(status_path, status)

    seed = int(train_yaml["train"]["seed"])
    set_seed(seed)
    device = torch.device(train_yaml["train"]["device"] if torch.cuda.is_available() else "cpu")
    policy_config = build_policy_config(policy_yaml, train_yaml, str(device))

    snapshot = {
        "policy_config_file": str(args.policy_config),
        "train_config_file": str(args.train_config),
        "policy_actual": serializable_policy_config(policy_config),
        "train": train_yaml,
        "runtime": {
            "python": sys.executable,
            "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpu": os.environ.get("SELECTED_PHYSICAL_GPU"),
            "torch_device": str(device),
            "torch_gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_yaml(output_dir / "config.yaml", snapshot)

    dataset = CalvinACTTrainingDataset(
        split_name=train_yaml["dataset"]["split_name"],
        dataset_root=Path(train_yaml["dataset"]["dataset_root"]),
        split_manifest=Path(train_yaml["dataset"]["split_manifest"]),
        chunk_size=policy_config.chunk_size,
        image_keys=train_yaml["dataset"]["image_keys"],
        max_episodes=train_yaml["dataset"].get("max_episodes"),
    )
    split_summary = dataset.split_summary()
    stats = compute_split_stats(
        split_name=train_yaml["dataset"]["split_name"],
        dataset_root=Path(train_yaml["dataset"]["dataset_root"]),
        split_manifest=Path(train_yaml["dataset"]["split_manifest"]),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_yaml["train"]["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(train_yaml["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    first_batch = next(iter(dataloader))
    batch_shapes = {
        key: list(value.shape) if torch.is_tensor(value) else f"{type(value).__name__}[{len(value)}]"
        for key, value in first_batch.items()
    }
    observed_envs = sorted(set(first_batch["environment"]))

    print(f"Dataset split: {dataset.split_name}")
    print(f"Dataset summary: {split_summary}")
    print(f"First batch shapes: {batch_shapes}")
    print(f"First batch environments: {observed_envs}")
    if dataset.split_name == "B_only" and observed_envs != ["B"]:
        raise RuntimeError(f"B_only first batch contains non-B environments: {observed_envs}")

    print("Creating LeRobot ACTPolicy")
    policy = ACTPolicy(policy_config).to(device)
    policy.train()
    optimizer = policy_config.get_optimizer_preset().build(policy.get_optim_params())
    grad_clip_norm = float(train_yaml["train"]["grad_clip_norm"])

    scalars_path = logs_dir / "scalars.jsonl"
    scalar_file = scalars_path.open("w", encoding="utf-8")
    step_times: list[float] = []
    final_checkpoint = None
    best_loss = float("inf")
    best_checkpoint = None

    status.update(
        {
            "status": "running",
            "dataset_summary": split_summary,
            "first_batch_shapes": batch_shapes,
            "first_batch_environments": observed_envs,
            "policy_class": "lerobot.policies.act.modeling_act.ACTPolicy",
            "optimizer": optimizer.__class__.__name__,
        }
    )
    write_yaml(status_path, status)

    iterator = cycle(dataloader)
    total_steps = int(train_yaml["train"]["steps"])
    log_interval = int(train_yaml["train"]["log_interval"])
    checkpoint_interval = int(train_yaml["train"]["checkpoint_interval"])
    start = time.perf_counter()

    try:
        for step in range(1, total_steps + 1):
            step_start = time.perf_counter()
            batch = next(iterator)
            batch = normalize_batch(batch, stats, device)
            loss, output_dict = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            elapsed_step = time.perf_counter() - step_start
            step_times.append(elapsed_step)
            loss_value = float(loss.detach().cpu().item())
            action_l1 = action_l1_from_loss_dict(output_dict)
            record = {
                "step": step,
                "loss": loss_value,
                "action_l1_loss": action_l1,
                "kld_loss": float(output_dict.get("kld_loss", math.nan)),
                "grad_norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "step_seconds": elapsed_step,
                "time": now_iso(),
            }
            scalar_file.write(json.dumps(record) + "\n")
            scalar_file.flush()

            if loss_value < best_loss:
                best_loss = loss_value
                best_checkpoint = ckpt_dir / "best_policy.pt"
                torch.save(
                    {
                        "step": step,
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": serializable_policy_config(policy_config),
                    },
                    best_checkpoint,
                )

            if step % log_interval == 0 or step == 1:
                print(
                    f"step={step}/{total_steps} loss={loss_value:.6f} "
                    f"action_l1_loss={action_l1:.6f} grad_norm={float(record['grad_norm']):.6f} "
                    f"lr={record['lr']:.2e} step_s={elapsed_step:.3f}",
                    flush=True,
                )

            if step % checkpoint_interval == 0 or step == total_steps:
                final_checkpoint = ckpt_dir / f"step_{step:06d}_policy.pt"
                torch.save(
                    {
                        "step": step,
                        "policy_state_dict": policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": serializable_policy_config(policy_config),
                    },
                    final_checkpoint,
                )
    finally:
        scalar_file.close()

    runtime = time.perf_counter() - start
    if final_checkpoint is None:
        raise RuntimeError("No checkpoint was saved")
    loaded = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_readable = "policy_state_dict" in loaded and loaded.get("step") == total_steps

    metrics = []
    with scalars_path.open("r", encoding="utf-8") as f:
        for line in f:
            metrics.append(json.loads(line))
    summary = {
        "status": "ok",
        "experiment": train_yaml["experiment"]["name"],
        "split": dataset.split_name,
        "num_steps": total_steps,
        "runtime_seconds": runtime,
        "final_loss": metrics[-1]["loss"],
        "final_action_l1_loss": metrics[-1]["action_l1_loss"],
        "best_loss": best_loss,
        "final_checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "checkpoint_readable": checkpoint_readable,
        "scalars_jsonl": str(scalars_path),
        "dataset_summary": split_summary,
        "first_batch_shapes": batch_shapes,
        "first_batch_environments": observed_envs,
        "mean_step_seconds": float(np.mean(step_times)),
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
            "final_checkpoint": str(final_checkpoint),
            "checkpoint_readable": checkpoint_readable,
        }
    )
    write_yaml(status_path, status)
    shutil.copy2(args.train_config, output_dir / "source_train_debug.yaml")
    shutil.copy2(args.policy_config, output_dir / "source_policy_act_baseline.yaml")

    print("Debug smoke test completed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
