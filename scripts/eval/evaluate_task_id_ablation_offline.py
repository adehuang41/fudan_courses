#!/usr/bin/env python3
"""Offline task-id ablation for task-conditioned ACT on CALVIN D."""

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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.utils.constants import ACTION  # noqa: E402

from scripts.eval.evaluate_d_offline_action_l1 import (  # noqa: E402
    TASK_ID_FINETUNE_RUNS,
    d_task_to_id,
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
from src.data.task_text_mapping import TaskTextMapper  # noqa: E402


DEFAULT_VARIANTS = (
    "correct",
    "zero",
    "random",
    "same_family_wrong",
    "different_family_wrong",
)
COMPAT_VARIANTS = set(DEFAULT_VARIANTS) | {"wrong"}


def parse_run_specs(specs: list[str], only_custom_runs: bool) -> dict[str, Path]:
    runs = {} if only_custom_runs else dict(TASK_ID_FINETUNE_RUNS)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--run must be NAME=RUN_DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        runs[name.strip()] = Path(path)
    if not runs:
        raise ValueError("No task-id runs selected")
    for name, path in runs.items():
        if not path.exists():
            raise FileNotFoundError(f"Run dir missing for {name}: {path}")
    return runs


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(variants) - COMPAT_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return variants


def task_type(task: str) -> str:
    prefix, sep, _ = str(task).partition(":")
    return prefix.strip() if sep and prefix.strip() else str(task).strip()


def canonical_task_for_text(task_text: str, mapper: TaskTextMapper) -> str:
    task_id = d_task_to_id(task_text, mapper)
    return str(mapper.canonical_tasks[task_id])


def semantic_family(canonical_task: str) -> str:
    name = str(canonical_task)
    if name.startswith("lift_"):
        return "lift"
    if name.startswith("rotate_"):
        return "rotate"
    if name.startswith("push_"):
        return "push"
    if name.startswith("place_"):
        return "place"
    if name.startswith("turn_"):
        return "light"
    if name.startswith("move_slider"):
        return "slider"
    if name in {"open_drawer", "close_drawer"}:
        return "drawer"
    if name in {"stack_block", "unstack_block"}:
        return "stack"
    return name.split("_", 1)[0]


def family_index(mapper: TaskTextMapper) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for task_id, canonical in enumerate(mapper.canonical_tasks):
        groups[semantic_family(canonical)].append(int(task_id))
    return {key: sorted(values) for key, values in groups.items()}


def choose_same_family_wrong(correct: np.ndarray, mapper: TaskTextMapper, groups: dict[str, list[int]]) -> np.ndarray:
    ids = np.empty_like(correct)
    for idx, task_id in enumerate(correct.tolist()):
        family = semantic_family(mapper.canonical_tasks[int(task_id)])
        candidates = [candidate for candidate in groups.get(family, []) if candidate != int(task_id)]
        if not candidates:
            ids[idx] = (int(task_id) + 1) % mapper.num_tasks
            continue
        current_pos = groups[family].index(int(task_id))
        ids[idx] = candidates[current_pos % len(candidates)]
    return ids


def choose_different_family_wrong(correct: np.ndarray, mapper: TaskTextMapper) -> np.ndarray:
    ids = np.empty_like(correct)
    num_tasks = mapper.num_tasks
    for idx, task_id in enumerate(correct.tolist()):
        current_family = semantic_family(mapper.canonical_tasks[int(task_id)])
        candidate = (int(task_id) + max(num_tasks // 2, 1)) % num_tasks
        for _ in range(num_tasks):
            if candidate != int(task_id) and semantic_family(mapper.canonical_tasks[candidate]) != current_family:
                ids[idx] = candidate
                break
            candidate = (candidate + 1) % num_tasks
        else:
            ids[idx] = (int(task_id) + 1) % num_tasks
    return ids


def task_ids_for_variant(
    batch: dict,
    mapper: TaskTextMapper,
    variant: str,
    device: torch.device,
    num_tasks: int,
    seed: int,
    batch_index: int,
) -> torch.Tensor:
    correct = np.asarray([d_task_to_id(task, mapper) for task in _as_task_list(batch["task"])], dtype=np.int64)
    groups = family_index(mapper)
    if variant == "correct":
        ids = correct
    elif variant == "zero":
        ids = np.zeros_like(correct)
    elif variant == "same_family_wrong":
        ids = choose_same_family_wrong(correct, mapper, groups)
    elif variant in {"different_family_wrong", "wrong"}:
        ids = choose_different_family_wrong(correct, mapper)
    elif variant == "random":
        rng = np.random.default_rng(int(seed) + int(batch_index) * 1009)
        draws = rng.integers(0, max(num_tasks - 1, 1), size=correct.shape, dtype=np.int64)
        ids = draws + (draws >= correct)
        ids = ids % num_tasks
        ids = np.where(ids == correct, (ids + 1) % num_tasks, ids)
    else:
        raise ValueError(variant)
    return torch.as_tensor(ids, dtype=torch.long, device=device)


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
    device: torch.device,
    checkpoint_name: str,
    variants: list[str],
    random_seed: int,
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
    if not (task_conditioning and bool(task_conditioning.get("enabled", False))):
        raise RuntimeError(f"{experiment} is not task-conditioned")
    mapper = TaskTextMapper.from_json(Path(task_conditioning["vocab_path"]))
    num_tasks = int(task_conditioning["num_tasks"])
    action_mean = stats[ACTION]["mean"].to(device)
    action_std = stats[ACTION]["std"].to(device).clamp_min(1e-6)

    summary_acc = defaultdict(new_group)
    episode_acc = defaultdict(new_group)
    task_type_acc = defaultdict(new_group)
    start = time.perf_counter()
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if log_every > 0 and (batch_index == 0 or (batch_index + 1) % log_every == 0):
                elapsed = time.perf_counter() - start
                print(f"[{experiment}] batch {batch_index + 1}/{total_batches}, elapsed={elapsed:.1f}s", flush=True)
            raw_batch, norm_base = normalize_for_policy(batch, stats, device)
            tasks = _as_task_list(batch["task"])
            episode_indices = batch["episode_index"].detach().cpu().numpy().tolist()
            gt_norm = norm_base[ACTION]
            gt_raw = raw_batch[ACTION]
            valid = ~raw_batch["action_is_pad"]
            for variant in variants:
                norm_batch = dict(norm_base)
                norm_batch["canonical_task_id"] = task_ids_for_variant(
                    batch=batch,
                    mapper=mapper,
                    variant=variant,
                    device=device,
                    num_tasks=num_tasks,
                    seed=random_seed,
                    batch_index=batch_index,
                )
                pred_norm = policy.predict_action_chunk(norm_batch)
                pred_raw = pred_norm * action_std + action_mean
                norm_abs = torch.abs(pred_norm - gt_norm)
                raw_abs = torch.abs(pred_raw - gt_raw)
                for sample_idx, task in enumerate(tasks):
                    episode_index = int(episode_indices[sample_idx])
                    sample_norm = norm_abs[sample_idx]
                    sample_raw = raw_abs[sample_idx]
                    sample_valid = valid[sample_idx]
                    update_group(summary_acc, (experiment, variant), sample_norm, sample_raw, sample_valid, episode_index)
                    update_group(
                        episode_acc,
                        (experiment, variant, episode_index, task, task_type(task)),
                        sample_norm,
                        sample_raw,
                        sample_valid,
                        episode_index,
                    )
                    update_group(
                        task_type_acc,
                        (experiment, variant, task_type(task)),
                        sample_norm,
                        sample_raw,
                        sample_valid,
                        episode_index,
                    )
    runtime = time.perf_counter() - start
    summary_rows = rows_from_group(summary_acc, ["experiment", "task_id_variant"])
    for row in summary_rows:
        row.update(
            {
                "checkpoint": checkpoint_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": load_info["checkpoint_step"],
                "policy_chunk_size": policy_config.chunk_size,
                "d_episodes": dataset.num_episodes,
                "d_frames": dataset.num_frames,
                "runtime_seconds_for_run": runtime,
            }
        )
        if not math.isfinite(float(row["normalized_action_l1_valid_mean"])):
            raise RuntimeError(f"Non-finite L1 for {experiment}/{row['task_id_variant']}")
    return {
        "summary": summary_rows,
        "per_episode": rows_from_group(
            episode_acc,
            ["experiment", "task_id_variant", "episode_index", "task", "task_type"],
        ),
        "per_task_type": rows_from_group(task_type_acc, ["experiment", "task_id_variant", "task_type"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D_DATASET_ROOT)
    parser.add_argument("--checkpoint-name", default="best_val_action_l1.pt")
    parser.add_argument("--run", action="append", default=[], metavar="NAME=RUN_DIR")
    parser.add_argument("--only-custom-runs", action="store_true")
    parser.add_argument("--output-prefix", default="task_id_ablation")
    parser.add_argument("--table-dir", type=Path, default=PROJECT_ROOT / "results/tables/track2_extra")
    parser.add_argument("--max-episodes", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--table-cache-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--random-seed", type=int, default=20260617)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--torch-num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    configure_episode_table_cache(args.table_cache_size)
    runs = parse_run_specs(args.run, args.only_custom_runs)
    variants = parse_variants(args.variants)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    dataset = CalvinDDACTDataset(dataset_root=args.dataset_root, chunk_size=100, max_episodes=args.max_episodes)
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
        "variants": variants,
        "max_episodes": args.max_episodes,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "table_cache": episode_table_cache_status(),
    }
    report_path = args.table_dir / f"{args.output_prefix}_report.json"
    write_json(report_path, report)

    outputs = {"summary": [], "per_episode": [], "per_task_type": []}
    for experiment, run_dir in runs.items():
        result = evaluate_run(
            experiment=experiment,
            run_dir=run_dir,
            dataset=dataset,
            dataloader=dataloader,
            device=device,
            checkpoint_name=args.checkpoint_name,
            variants=variants,
            random_seed=args.random_seed,
            log_every=args.log_every,
        )
        for key, rows in result.items():
            outputs[key].extend(rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    file_map = {
        "summary": "summary.csv",
        "per_episode": "per_episode_action_l1.csv",
        "per_task_type": "per_task_type_action_l1.csv",
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
        }
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
