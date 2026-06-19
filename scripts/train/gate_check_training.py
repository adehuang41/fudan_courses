#!/usr/bin/env python3
"""Gate check a completed ACT training run before starting the next run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml


PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
REAL_DATA_ROOT = Path("/data/dechao/cv_final_pj2")
SHARED_HF_ROOT = Path("/data/huggingface")


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def checkpoint_readable(path: Path, expected_step: int | None = None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, repr(exc)
    if "policy_state_dict" not in loaded:
        return False, "policy_state_dict missing"
    if expected_step is not None and loaded.get("step") != expected_step:
        return False, f"expected step {expected_step}, got {loaded.get('step')}"
    return True, "ok"


def under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def find_large_home_files() -> list[str]:
    large_files: list[str] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if d not in {".git"} and not (root_path / d).is_symlink()]
        for filename in files:
            path = root_path / filename
            if path.is_symlink():
                continue
            try:
                if path.stat().st_size > 20 * 1024 * 1024:
                    large_files.append(str(path))
            except FileNotFoundError:
                continue
    return large_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", type=Path, required=True)
    args = parser.parse_args()

    train_config = read_yaml(args.train_config)
    output_dir = Path(train_config["experiment"]["output_dir"])
    expected_steps = int(train_config["train"]["steps"])
    expected_gpu = str(train_config["train"]["physical_gpu"])

    status_path = output_dir / "training_status.yaml"
    summary_path = output_dir / "metrics_summary.json"
    report_path = output_dir / "gate_check.yaml"

    checks: list[dict] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if not status_path.exists() or not summary_path.exists():
        add("required_summary_files_exist", False, f"{status_path} or {summary_path} missing")
        report = {"status": "failed", "checks": checks}
        write_yaml(report_path, report)
        print(yaml.safe_dump(report, sort_keys=False))
        return 1

    status = read_yaml(status_path)
    summary = read_json(summary_path)

    add("training_completed", status.get("status") == "completed" and summary.get("status") == "ok")
    add("final_step_matches", int(status.get("final_step", -1)) == expected_steps)
    add(
        "metrics_finite",
        all(
            is_finite_number(summary.get(key))
            for key in [
                "final_loss",
                "final_action_l1_loss",
                "final_val_loss",
                "final_val_action_l1_loss",
                "best_val_action_l1_loss",
            ]
        ),
    )

    ckpt_specs = [
        ("latest_checkpoint", Path(summary.get("latest_checkpoint", "")), expected_steps),
        ("final_checkpoint", Path(summary.get("final_checkpoint", "")), expected_steps),
        ("best_val_action_l1_checkpoint", Path(summary.get("best_val_action_l1_checkpoint", "")), None),
    ]
    for name, path, expected_step in ckpt_specs:
        ok, detail = checkpoint_readable(path, expected_step)
        add(f"{name}_readable", ok, f"{path}: {detail}")

    cuda_visible = str(summary.get("cuda_visible_devices"))
    physical_gpu = str(summary.get("physical_gpu"))
    add("single_gpu_only", "," not in cuda_visible and cuda_visible == expected_gpu)
    add("physical_gpu_matches", physical_gpu == expected_gpu)

    output_paths = [
        output_dir,
        Path(summary.get("latest_checkpoint", "")),
        Path(summary.get("final_checkpoint", "")),
        Path(summary.get("best_val_action_l1_checkpoint", "")),
        Path(summary.get("scalars_jsonl", "")),
        Path(summary.get("validation_jsonl", "")),
    ]
    add("outputs_under_project_data_root", all(under(path, REAL_DATA_ROOT) for path in output_paths))
    add("outputs_not_under_shared_huggingface", not any(under(path, SHARED_HF_ROOT) for path in output_paths))
    add("no_large_files_written_to_home_project", len(find_large_home_files()) == 0)

    if summary.get("split") == "B_only":
        first_envs = status.get("first_batch_environments", [])
        add("b_only_first_batch_env", first_envs == ["B"], str(first_envs))
    elif summary.get("split") == "ABC_mixed":
        first_envs = set(status.get("first_batch_environments", []))
        add("abc_first_batch_envs_valid", first_envs.issubset({"A", "B", "C"}), str(sorted(first_envs)))

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "passed" if passed else "failed",
        "experiment": train_config["experiment"]["name"],
        "output_dir": str(output_dir),
        "checks": checks,
    }
    write_yaml(report_path, report)
    print(yaml.safe_dump(report, sort_keys=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
