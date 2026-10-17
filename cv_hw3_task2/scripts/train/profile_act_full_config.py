#!/usr/bin/env python3
"""Profile full ACT training throughput without starting full training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
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


DEFAULT_OUTPUT_DIR = Path("/data/dechao/cv_final_pj2/outputs/profiling/full_config_quick_profile")
DEFAULT_POLICY_CONFIG = PROJECT_ROOT / "configs/policy/act_baseline.yaml"
DEFAULT_SPLIT_MANIFEST = (
    Path("/data/dechao/cv_final_pj2/datasets/calvin_task_ABC_D")
    / "_download_logs"
    / "environment_episode_map.csv"
)
DEFAULT_DATASET_ROOT = Path("/data/dechao/cv_final_pj2/datasets/calvin_task_ABC_D")


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy_config(policy_yaml: dict, device: str) -> ACTConfig:
    policy_cfg = deepcopy(policy_yaml["policy"])
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


def action_l1_from_loss_dict(output_dict: dict) -> float:
    return float(output_dict.get("l1_loss", math.nan))


class GPUMonitor:
    def __init__(self, physical_gpu: str | None, interval_seconds: float = 0.5) -> None:
        self.physical_gpu = str(physical_gpu) if physical_gpu is not None else None
        self.interval_seconds = interval_seconds
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.physical_gpu:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
                for line in output.strip().splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 3 or parts[0] != self.physical_gpu:
                        continue
                    self.samples.append(
                        {
                            "time": now_iso(),
                            "utilization_gpu_percent": float(parts[1]),
                            "memory_used_mb": float(parts[2]),
                        }
                    )
                    break
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict:
        if not self.samples:
            return {
                "gpu_utilization_mean_percent": math.nan,
                "gpu_utilization_max_percent": math.nan,
                "gpu_memory_used_mean_mb": math.nan,
                "gpu_memory_used_max_mb": math.nan,
                "gpu_samples": 0,
            }
        utils = [sample["utilization_gpu_percent"] for sample in self.samples]
        mems = [sample["memory_used_mb"] for sample in self.samples]
        return {
            "gpu_utilization_mean_percent": float(statistics.mean(utils)),
            "gpu_utilization_max_percent": float(max(utils)),
            "gpu_memory_used_mean_mb": float(statistics.mean(mems)),
            "gpu_memory_used_max_mb": float(max(mems)),
            "gpu_samples": len(self.samples),
        }


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else math.nan


def stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def get_autocast(amp_mode: str):
    if amp_mode == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def run_single(args: argparse.Namespace) -> int:
    result_path = Path(args.result_path)
    output_dir = Path(args.output_dir)
    run_dir = output_dir / f"nw{args.num_workers:02d}_bs{args.batch_size:02d}_amp{args.amp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result_base = {
        "stage": args.stage,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "split": args.split_name,
        "physical_gpu": os.environ.get("SELECTED_PHYSICAL_GPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "start_time": now_iso(),
        "status": "starting",
    }
    write_json(result_path, result_base)

    try:
        set_seed(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; profiling requires a single GPU.")
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        policy_yaml = read_yaml(Path(args.policy_config))
        policy_config = build_policy_config(policy_yaml, str(device))

        dataset = CalvinACTTrainingDataset(
            split_name=args.split_name,
            dataset_root=Path(args.dataset_root),
            split_manifest=Path(args.split_manifest),
            chunk_size=policy_config.chunk_size,
            image_keys=list(policy_config.image_features),
            max_episodes=args.max_episodes,
        )
        stats = compute_split_stats(
            split_name=args.split_name,
            dataset_root=Path(args.dataset_root),
            split_manifest=Path(args.split_manifest),
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "shuffle": True,
            "generator": generator,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "drop_last": True,
            "persistent_workers": args.num_workers > 0,
        }
        if args.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = args.prefetch_factor
        dataloader = DataLoader(**dataloader_kwargs)

        policy = ACTPolicy(policy_config).to(device)
        policy.train()
        optimizer = policy_config.get_optimizer_preset().build(policy.get_optim_params())
        grad_clip_norm = float(args.grad_clip_norm)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        iterator = cycle(dataloader)
        total_steps = args.warmup_steps + args.measured_steps
        measured_records: list[dict] = []
        monitor = GPUMonitor(os.environ.get("SELECTED_PHYSICAL_GPU"), interval_seconds=args.gpu_sample_interval)
        first_batch_envs: list[str] | None = None
        first_batch_shapes: dict | None = None
        measured_start = None
        measured_end = None

        for step in range(1, total_steps + 1):
            measured = step > args.warmup_steps
            if measured and measured_start is None:
                torch.cuda.synchronize()
                measured_start = time.perf_counter()
                monitor.start()

            torch.cuda.synchronize()
            step_start = time.perf_counter()
            data_start = time.perf_counter()
            batch = next(iterator)
            data_loading_seconds = time.perf_counter() - data_start

            if first_batch_envs is None:
                first_batch_envs = sorted(set(batch["environment"]))
                first_batch_shapes = {
                    key: list(value.shape) if torch.is_tensor(value) else f"{type(value).__name__}[{len(value)}]"
                    for key, value in batch.items()
                }

            transfer_start = time.perf_counter()
            batch = normalize_batch(batch, stats, device)
            torch.cuda.synchronize()
            transfer_seconds = time.perf_counter() - transfer_start

            compute_start = time.perf_counter()
            with get_autocast(args.amp):
                loss, output_dict = policy.forward(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            forward_backward_seconds = time.perf_counter() - compute_start
            step_seconds = time.perf_counter() - step_start

            loss_value = float(loss.detach().cpu().item())
            action_l1 = action_l1_from_loss_dict(output_dict)
            finite = math.isfinite(loss_value) and math.isfinite(action_l1)
            if not finite:
                raise RuntimeError(f"Non-finite metric at step {step}: loss={loss_value}, action_l1={action_l1}")

            if measured:
                measured_records.append(
                    {
                        "step": step,
                        "loss": loss_value,
                        "action_l1_loss": action_l1,
                        "kld_loss": float(output_dict.get("kld_loss", math.nan)),
                        "grad_norm": float(
                            grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm
                        ),
                        "step_seconds": step_seconds,
                        "data_loading_seconds": data_loading_seconds,
                        "host_to_device_seconds": transfer_seconds,
                        "forward_backward_seconds": forward_backward_seconds,
                        "samples_per_second": args.batch_size / step_seconds,
                        "time": now_iso(),
                    }
                )

            if step == 1 or step == args.warmup_steps or measured:
                phase = "measure" if measured else "warmup"
                print(
                    f"[{phase}] step={step}/{total_steps} nw={args.num_workers} bs={args.batch_size} "
                    f"amp={args.amp} loss={loss_value:.6f} action_l1={action_l1:.6f} "
                    f"step_s={step_seconds:.3f} data_s={data_loading_seconds:.3f} "
                    f"fb_s={forward_backward_seconds:.3f}",
                    flush=True,
                )

        torch.cuda.synchronize()
        measured_end = time.perf_counter()
        monitor.stop()

        data_times = [record["data_loading_seconds"] for record in measured_records]
        transfer_times = [record["host_to_device_seconds"] for record in measured_records]
        fb_times = [record["forward_backward_seconds"] for record in measured_records]
        step_times = [record["step_seconds"] for record in measured_records]
        losses = [record["loss"] for record in measured_records]
        action_l1s = [record["action_l1_loss"] for record in measured_records]
        measured_wall = measured_end - measured_start if measured_start is not None else math.nan
        samples_per_second = args.batch_size * len(measured_records) / measured_wall

        rows_path = run_dir / "measured_steps.jsonl"
        for record in measured_records:
            append_jsonl(rows_path, record)

        result = {
            **result_base,
            "status": "ok",
            "end_time": now_iso(),
            "policy_actual": serializable_policy_config(policy_config),
            "dataset_summary": dataset.split_summary(),
            "first_batch_environments": first_batch_envs,
            "first_batch_shapes": first_batch_shapes,
            "measured_wall_seconds": measured_wall,
            "samples_per_second": float(samples_per_second),
            "seconds_per_step_mean": mean(step_times),
            "seconds_per_step_std": stdev(step_times),
            "seconds_per_step_p50": percentile(step_times, 50),
            "seconds_per_step_p95": percentile(step_times, 95),
            "data_loading_seconds_mean": mean(data_times),
            "host_to_device_seconds_mean": mean(transfer_times),
            "forward_backward_seconds_mean": mean(fb_times),
            "loss_final": losses[-1],
            "loss_mean": mean(losses),
            "action_l1_loss_final": action_l1s[-1],
            "action_l1_loss_mean": mean(action_l1s),
            "loss_finite": all(math.isfinite(value) for value in losses),
            "action_l1_loss_finite": all(math.isfinite(value) for value in action_l1s),
            "torch_peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024 / 1024),
            "torch_peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024 / 1024),
            "measured_steps_jsonl": str(rows_path),
            **monitor.summary(),
        }
        write_json(run_dir / "result.json", result)
        write_json(result_path, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        result = {
            **result_base,
            "status": "oom",
            "end_time": now_iso(),
            "error": str(exc),
        }
        write_json(result_path, result)
        write_json(run_dir / "result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result = {
            **result_base,
            "status": "failed",
            "end_time": now_iso(),
            "error": repr(exc),
        }
        write_json(result_path, result)
        write_json(run_dir / "result.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        return 1


def run_child(command: list[str], env: dict[str, str], stdout_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stdout_file.write(line)
            stdout_file.flush()
        return process.wait()


def is_stable_ok(result: dict) -> bool:
    return (
        result.get("status") == "ok"
        and bool(result.get("loss_finite"))
        and bool(result.get("action_l1_loss_finite"))
        and math.isfinite(float(result.get("samples_per_second", math.nan)))
    )


def select_best(results: list[dict]) -> dict:
    candidates = [result for result in results if is_stable_ok(result)]
    if not candidates:
        raise RuntimeError("No stable profiling configuration completed successfully.")
    return max(candidates, key=lambda result: float(result["samples_per_second"]))


def write_results_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "stage",
        "num_workers",
        "batch_size",
        "amp",
        "status",
        "samples_per_second",
        "seconds_per_step_mean",
        "seconds_per_step_std",
        "seconds_per_step_p95",
        "gpu_utilization_mean_percent",
        "gpu_utilization_max_percent",
        "gpu_memory_used_max_mb",
        "torch_peak_memory_allocated_mb",
        "data_loading_seconds_mean",
        "host_to_device_seconds_mean",
        "forward_backward_seconds_mean",
        "loss_final",
        "action_l1_loss_final",
        "loss_finite",
        "action_l1_loss_finite",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_orchestrator(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_jsonl = output_dir / "profile_results.jsonl"
    if results_jsonl.exists():
        results_jsonl.unlink()

    plan = {
        "created_at": now_iso(),
        "policy_config": str(args.policy_config),
        "dataset_root": str(args.dataset_root),
        "split_manifest": str(args.split_manifest),
        "split_name": args.split_name,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "physical_gpu": args.physical_gpu,
        "fixed_single_gpu": True,
        "stages": {
            "num_workers": [8, 16, 32, 48, 64],
            "batch_size": [4, 8, 16],
            "amp": ["off", "bf16"],
        },
        "notes": "Full ACT config only: no debug architecture overrides, no full training, no checkpoints.",
    }
    write_yaml(output_dir / "profiling_plan.yaml", plan)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    env["SELECTED_PHYSICAL_GPU"] = str(args.physical_gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"

    all_results: list[dict] = []

    def run_config(stage: str, num_workers: int, batch_size: int, amp: str) -> dict:
        run_name = f"nw{num_workers:02d}_bs{batch_size:02d}_amp{amp}"
        result_path = output_dir / run_name / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--stage",
            stage,
            "--policy-config",
            str(args.policy_config),
            "--dataset-root",
            str(args.dataset_root),
            "--split-manifest",
            str(args.split_manifest),
            "--split-name",
            args.split_name,
            "--output-dir",
            str(output_dir),
            "--result-path",
            str(result_path),
            "--num-workers",
            str(num_workers),
            "--batch-size",
            str(batch_size),
            "--amp",
            amp,
            "--warmup-steps",
            str(args.warmup_steps),
            "--measured-steps",
            str(args.measured_steps),
            "--seed",
            str(args.seed),
            "--grad-clip-norm",
            str(args.grad_clip_norm),
            "--prefetch-factor",
            str(args.prefetch_factor),
            "--gpu-sample-interval",
            str(args.gpu_sample_interval),
        ]
        if args.max_episodes is not None:
            command += ["--max-episodes", str(args.max_episodes)]

        print(f"\n=== Profiling {run_name} stage={stage} ===", flush=True)
        return_code = run_child(command, env, output_dir / run_name / "stdout.log")
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            result = {
                "stage": stage,
                "num_workers": num_workers,
                "batch_size": batch_size,
                "amp": amp,
                "status": "failed",
                "error": f"child process exited {return_code} without result.json",
            }
        result["child_return_code"] = return_code
        append_jsonl(results_jsonl, result)
        all_results.append(result)
        write_results_csv(output_dir / "profile_results.csv", all_results)
        return result

    stage1_results = [run_config("num_workers", workers, 4, "off") for workers in [8, 16, 32, 48, 64]]
    best_workers_result = select_best(stage1_results)
    best_workers = int(best_workers_result["num_workers"])

    stage2_results = [run_config("batch_size", best_workers, batch_size, "off") for batch_size in [4, 8, 16]]
    best_batch_result = select_best(stage2_results)
    best_batch = int(best_batch_result["batch_size"])

    stage3_results = [run_config("amp", best_workers, best_batch, amp) for amp in ["off", "bf16"]]
    best_amp_result = select_best(stage3_results)

    recommendation = {
        "completed_at": now_iso(),
        "recommended": {
            "num_workers": int(best_amp_result["num_workers"]),
            "batch_size": int(best_amp_result["batch_size"]),
            "amp": best_amp_result["amp"],
            "samples_per_second": best_amp_result["samples_per_second"],
            "seconds_per_step_mean": best_amp_result["seconds_per_step_mean"],
            "gpu_utilization_mean_percent": best_amp_result.get("gpu_utilization_mean_percent"),
            "gpu_memory_used_max_mb": best_amp_result.get("gpu_memory_used_max_mb"),
            "data_loading_seconds_mean": best_amp_result.get("data_loading_seconds_mean"),
            "forward_backward_seconds_mean": best_amp_result.get("forward_backward_seconds_mean"),
            "loss_final": best_amp_result.get("loss_final"),
            "action_l1_loss_final": best_amp_result.get("action_l1_loss_final"),
        },
        "stage_winners": {
            "num_workers": best_workers_result,
            "batch_size": best_batch_result,
            "amp": best_amp_result,
        },
        "all_results_csv": str(output_dir / "profile_results.csv"),
        "all_results_jsonl": str(results_jsonl),
    }
    write_json(output_dir / "profile_summary.json", recommendation)
    write_yaml(output_dir / "recommendation.yaml", recommendation)
    print("\n=== Profiling recommendation ===")
    print(json.dumps(recommendation["recommended"], indent=2, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--stage", default="manual")
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-name", default="B_only")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-path", type=Path, default=DEFAULT_OUTPUT_DIR / "single_result.json")
    parser.add_argument("--physical-gpu", default="5")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--amp", choices=["off", "bf16"], default="off")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.5)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.single:
        return run_single(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
