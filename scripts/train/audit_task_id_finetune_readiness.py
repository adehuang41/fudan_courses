#!/usr/bin/env python3
"""Preflight checks for task-id ACT fine-tuning from visual checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import OBS_STATE  # noqa: E402

from src.data.task_text_mapping import TaskTextMapper  # noqa: E402
from src.models.task_conditioned_act import TaskConditionedACTPolicy, load_visual_act_checkpoint  # noqa: E402


POLICY_CONFIG = PROJECT_ROOT / "configs/policy/act_task_id_conditioned.yaml"
RUNS = {
    "B_only": {
        "visual_run_dir": Path("/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline"),
        "train_config": PROJECT_ROOT
        / "configs/train/finetune_B_only_task_id_from_visual50k_extra50k_seed20260531_retry1.yaml",
    },
    "ABC_mixed": {
        "visual_run_dir": Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline"),
        "train_config": PROJECT_ROOT
        / "configs/train/finetune_ABC_mixed_task_id_from_visual50k_extra50k_seed20260531_retry1.yaml",
    },
}


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_policy_config(policy_yaml: dict, device: str, task_conditioning: dict) -> ACTConfig:
    policy_cfg = deepcopy(policy_yaml["policy"])
    if bool(task_conditioning.get("enabled", False)):
        state_cfg = policy_cfg["input_features"][OBS_STATE]
        state_cfg["shape"] = [int(state_cfg["shape"][0]) + int(task_conditioning["embedding_dim"])]
    policy_cfg["device"] = device
    policy_cfg["pretrained_backbone_weights"] = None
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


def visual_strict_load(run_dir: Path) -> dict:
    config_yaml = read_yaml(run_dir / "config.yaml")
    checkpoint_path = run_dir / "checkpoints" / "best_val_action_l1.pt"
    config = deepcopy(config_yaml["policy_actual"])
    config["device"] = "cpu"
    config["pretrained_backbone_weights"] = None
    config["input_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in config["input_features"].items()
    }
    config["output_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in config["output_features"].items()
    }
    config.pop("type", None)
    policy_config = ACTConfig(**config)
    policy = ACTPolicy(policy_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = policy.load_state_dict(checkpoint["policy_state_dict"], strict=True)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "checkpoint_step": checkpoint.get("step"),
        "strict_load_missing": len(result.missing_keys),
        "strict_load_unexpected": len(result.unexpected_keys),
        "strict_load_ok": len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0,
        "visual_best_val_action_l1_loss": config_yaml.get("train", {})
        .get("experiment", {})
        .get("best_val_action_l1_loss"),
    }


def optimizer_param_names(model: TaskConditionedACTPolicy, optimizer: torch.optim.Optimizer) -> set[str]:
    param_to_name = {id(param): name for name, param in model.named_parameters()}
    names = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            name = param_to_name.get(id(param))
            if name is not None:
                names.add(name)
    return names


def audit_run(name: str, device: torch.device) -> dict:
    policy_yaml = read_yaml(POLICY_CONFIG)
    train_config = RUNS[name]["train_config"]
    train_yaml = read_yaml(train_config)
    task_cfg = deepcopy(policy_yaml["task_conditioning"])
    task_cfg.update(deepcopy(train_yaml.get("task_conditioning", {})))
    mapper = TaskTextMapper.from_json(Path(task_cfg["vocab_path"]))
    policy_config = build_policy_config(policy_yaml, str(device), task_cfg)

    visual_report = visual_strict_load(RUNS[name]["visual_run_dir"])
    model = TaskConditionedACTPolicy(
        config=policy_config,
        num_tasks=int(task_cfg["num_tasks"]),
        task_embedding_dim=int(task_cfg["embedding_dim"]),
        task_id_key=str(task_cfg.get("task_id_key", "canonical_task_id")),
    ).to(device)
    init_report = load_visual_act_checkpoint(model, Path(task_cfg["init_from_visual_checkpoint"]))

    state_proj_keys = [
        "act.model.encoder_robot_state_input_proj.weight",
        "act.model.vae_encoder_robot_state_input_proj.weight",
    ]
    state_projection_checks = {}
    checkpoint = torch.load(task_cfg["init_from_visual_checkpoint"], map_location="cpu", weights_only=False)
    old_state = checkpoint["policy_state_dict"]
    for key in state_proj_keys:
        new_value = model.state_dict()[key].detach().cpu()
        old_key = key.removeprefix("act.")
        old_value = old_state[old_key]
        old_dim = old_value.shape[1]
        added = new_value[:, old_dim:]
        state_projection_checks[key] = {
            "old_shape": list(old_value.shape),
            "new_shape": list(new_value.shape),
            "old_columns_equal": bool(torch.allclose(new_value[:, :old_dim], old_value)),
            "added_columns_shape": list(added.shape),
            "added_columns_std": float(added.std().item()),
            "added_columns_abs_sum": float(added.abs().sum().item()),
            "added_columns_nonzero": bool(float(added.abs().sum().item()) > 0.0),
        }

    optimizer = policy_config.get_optimizer_preset().build(model.get_optim_params())
    opt_names = optimizer_param_names(model, optimizer)
    task_embedding_in_optimizer = "task_embedding.weight" in opt_names
    expanded_projection_in_optimizer = all(key in opt_names for key in state_proj_keys)

    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, 15, device=device),
        "canonical_task_id": torch.tensor([0, min(1, mapper.num_tasks - 1)], dtype=torch.long, device=device),
        "observation.images.image": torch.randn(batch_size, 3, 256, 256, device=device),
        "observation.images.wrist_image": torch.randn(batch_size, 3, 256, 256, device=device),
        "action": torch.randn(batch_size, policy_config.chunk_size, 7, device=device),
        "action_is_pad": torch.zeros(batch_size, policy_config.chunk_size, dtype=torch.bool, device=device),
    }
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, output = model.forward(batch)
    loss.backward()
    task_grad = model.task_embedding.weight.grad
    state_grad_checks = {}
    for key in state_proj_keys:
        param = dict(model.named_parameters())[key]
        state_grad_checks[key] = {
            "grad_exists": param.grad is not None,
            "grad_abs_sum": float(param.grad.detach().abs().sum().item()) if param.grad is not None else 0.0,
        }

    train = train_yaml["train"]
    dataset = train_yaml["dataset"]
    output_dir = Path(train_yaml["experiment"]["output_dir"])
    return {
        "name": name,
        "train_config": str(train_config),
        "output_dir": str(output_dir),
        "output_dir_exists": output_dir.exists(),
        "split_name": dataset["split_name"],
        "validation_fraction": float(dataset["validation_fraction"]),
        "steps": int(train["steps"]),
        "batch_size": int(train["batch_size"]),
        "num_workers": int(train["num_workers"]),
        "validation_num_workers": int(train["validation_num_workers"]),
        "prefetch_factor": int(train.get("prefetch_factor", 2)),
        "validation_prefetch_factor": int(train.get("validation_prefetch_factor", train.get("prefetch_factor", 2))),
        "persistent_workers": bool(train.get("persistent_workers", True)),
        "validation_persistent_workers": bool(train.get("validation_persistent_workers", False)),
        "video_frame_cache_size": int(train.get("video_frame_cache_size", 32)),
        "pin_memory": bool(train.get("pin_memory", True)),
        "seed": int(train["seed"]),
        "log_interval": int(train["log_interval"]),
        "checkpoint_interval": int(train["checkpoint_interval"]),
        "validation_interval": int(train["validation_interval"]),
        "grad_clip_norm": float(train["grad_clip_norm"]),
        "amp": str(train.get("amp", "off")),
        "optimizer_lr": float(policy_yaml["policy"]["optimizer_lr"]),
        "optimizer_lr_backbone": float(policy_yaml["policy"]["optimizer_lr_backbone"]),
        "optimizer_weight_decay": float(policy_yaml["policy"]["optimizer_weight_decay"]),
        "scheduler": "none_fixed_lr",
        "task_embedding_dim": int(task_cfg["embedding_dim"]),
        "num_tasks": int(task_cfg["num_tasks"]),
        "task_vocab_num_tasks": mapper.num_tasks,
        "visual_checkpoint": visual_report,
        "task_id_visual_init_report": init_report.to_dict(),
        "state_projection_checks": state_projection_checks,
        "optimizer_param_count": len(opt_names),
        "task_embedding_in_optimizer": task_embedding_in_optimizer,
        "expanded_projection_in_optimizer": expanded_projection_in_optimizer,
        "task_embedding_grad_exists": task_grad is not None,
        "task_embedding_grad_abs_sum": float(task_grad.detach().abs().sum().item()) if task_grad is not None else 0.0,
        "state_projection_grad_checks": state_grad_checks,
        "dry_forward_loss": float(loss.detach().cpu().item()),
        "dry_forward_action_l1": float(output.get("l1_loss", torch.tensor(float("nan")))),
        "audit_passed": bool(
            visual_report["strict_load_ok"]
            and init_report.clean
            and task_embedding_in_optimizer
            and expanded_projection_in_optimizer
            and task_grad is not None
            and float(task_grad.detach().abs().sum().item()) > 0.0
            and not output_dir.exists()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results/tables/step4/task_id_finetune_preflight.json")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "results/tables/step4/task_id_finetune_preflight.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    reports = [audit_run(name, device) for name in RUNS]
    write_json(args.output_json, {"runs": reports})
    flat_rows = []
    for report in reports:
        flat_rows.append(
            {
                "name": report["name"],
                "train_config": report["train_config"],
                "output_dir": report["output_dir"],
                "output_dir_exists": report["output_dir_exists"],
                "split_name": report["split_name"],
                "validation_fraction": report["validation_fraction"],
                "steps": report["steps"],
                "batch_size": report["batch_size"],
                "num_workers": report["num_workers"],
                "validation_num_workers": report["validation_num_workers"],
                "prefetch_factor": report["prefetch_factor"],
                "validation_prefetch_factor": report["validation_prefetch_factor"],
                "persistent_workers": report["persistent_workers"],
                "validation_persistent_workers": report["validation_persistent_workers"],
                "video_frame_cache_size": report["video_frame_cache_size"],
                "pin_memory": report["pin_memory"],
                "seed": report["seed"],
                "log_interval": report["log_interval"],
                "checkpoint_interval": report["checkpoint_interval"],
                "validation_interval": report["validation_interval"],
                "grad_clip_norm": report["grad_clip_norm"],
                "amp": report["amp"],
                "optimizer_lr": report["optimizer_lr"],
                "optimizer_lr_backbone": report["optimizer_lr_backbone"],
                "optimizer_weight_decay": report["optimizer_weight_decay"],
                "scheduler": report["scheduler"],
                "task_embedding_dim": report["task_embedding_dim"],
                "num_tasks": report["num_tasks"],
                "task_vocab_num_tasks": report["task_vocab_num_tasks"],
                "visual_strict_load_ok": report["visual_checkpoint"]["strict_load_ok"],
                "task_id_visual_init_clean": report["task_id_visual_init_report"]["clean"],
                "exact_loaded": report["task_id_visual_init_report"]["exact_loaded"],
                "expanded_loaded": report["task_id_visual_init_report"]["expanded_loaded"],
                "task_embedding_in_optimizer": report["task_embedding_in_optimizer"],
                "expanded_projection_in_optimizer": report["expanded_projection_in_optimizer"],
                "task_embedding_grad_abs_sum": report["task_embedding_grad_abs_sum"],
                "audit_passed": report["audit_passed"],
            }
        )
    write_csv(args.output_csv, flat_rows)
    print(json.dumps({"runs": flat_rows}, indent=2, ensure_ascii=False))
    if not all(report["audit_passed"] for report in reports):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
