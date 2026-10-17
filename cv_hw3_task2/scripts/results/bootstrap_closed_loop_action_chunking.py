#!/usr/bin/env python3
"""Bootstrap closed-loop action-chunking rollout summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_family(task: str) -> str:
    if task in {"open_drawer", "close_drawer"}:
        return "drawer"
    if task.startswith("move_slider"):
        return "slider"
    if task.startswith("turn_on"):
        return "switch"
    if task.startswith("lift_"):
        return "lift"
    if task.startswith("place_"):
        return "place"
    if task.startswith("push_"):
        return "push"
    if task.startswith("rotate_"):
        return "rotate"
    if task.startswith("stack_"):
        return "stack"
    return task.split("_", 1)[0]


def wilson_ci(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = success / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * np.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_ci(values: np.ndarray, fn, rng: np.random.Generator, iters: int) -> tuple[float, float, float]:
    observed = float(fn(values))
    n = len(values)
    samples = np.empty(iters, dtype=np.float64)
    for idx in range(iters):
        draw = values[rng.integers(0, n, size=n)]
        samples[idx] = float(fn(draw))
    low, high = np.percentile(samples, [2.5, 97.5])
    return observed, float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "/data/dechao/cv_final_pj2/outputs/post_finetune_eval/"
            "track2_action_chunking/formal_50seq_ep360_20260613T2050Z"
        ),
    )
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables/step4"))
    parser.add_argument("--prefix", default="action_chunking_ablation_50seq_ep360")
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    ci_rows: list[dict] = []
    task_rows: list[dict] = []
    family_acc: dict[tuple[str, int, str], dict[str, int]] = {}

    json_paths = sorted(args.run_root.glob("*_50seq_ep360_n*/*_50seq_ep360_n*.json"))
    if not json_paths:
        raise SystemExit(f"No closed-loop JSON files under {args.run_root}")

    for json_path in json_paths:
        report = read_json(json_path)
        csv_path = json_path.with_suffix(".csv")
        rows = read_csv_rows(csv_path)
        successful = np.asarray([int(row["successful_subtasks"]) for row in rows], dtype=np.int64)
        policy = str(report["policy"])
        n_action_steps = int(report["policy_n_action_steps"])
        common = {
            "policy": policy,
            "n_action_steps": n_action_steps,
            "chunk_size": int(report["policy_chunk_size"]),
            "num_sequences": int(report["num_sequences"]),
            "ep_len": int(report["ep_len"]),
            "scale_label": str(report["scale_label"]),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
            "bootstrap_iters": args.bootstrap_iters,
            "seed": args.seed,
        }

        value, low, high = bootstrap_ci(successful, np.mean, rng, args.bootstrap_iters)
        ci_rows.append({**common, "metric": "avg_successful_subtasks", "value": value, "ci_low": low, "ci_high": high})
        for k in range(1, 6):
            value, low, high = bootstrap_ci(successful, lambda arr, kk=k: np.mean(arr >= kk), rng, args.bootstrap_iters)
            ci_rows.append({**common, "metric": f"SR@{k}", "value": value, "ci_low": low, "ci_high": high})

        for task, stats in sorted(report["task_success"].items()):
            success = int(stats["success"])
            attempted = int(stats["attempted"])
            low, high = wilson_ci(success, attempted)
            task_rows.append(
                {
                    "policy": policy,
                    "n_action_steps": n_action_steps,
                    "task": task,
                    "task_family": task_family(task),
                    "success": success,
                    "attempted": attempted,
                    "success_rate": success / attempted if attempted else 0.0,
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "json_path": str(json_path),
                }
            )
            key = (policy, n_action_steps, task_family(task))
            item = family_acc.setdefault(key, {"success": 0, "attempted": 0})
            item["success"] += success
            item["attempted"] += attempted

    family_rows = []
    for (policy, n_action_steps, family), item in sorted(family_acc.items()):
        success = item["success"]
        attempted = item["attempted"]
        low, high = wilson_ci(success, attempted)
        family_rows.append(
            {
                "policy": policy,
                "n_action_steps": n_action_steps,
                "task_family": family,
                "success": success,
                "attempted": attempted,
                "success_rate": success / attempted if attempted else 0.0,
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )

    write_csv(
        args.table_dir / f"{args.prefix}_bootstrap_ci.csv",
        ci_rows,
        [
            "policy",
            "n_action_steps",
            "chunk_size",
            "num_sequences",
            "ep_len",
            "metric",
            "value",
            "ci_low",
            "ci_high",
            "scale_label",
            "bootstrap_iters",
            "seed",
            "json_path",
            "csv_path",
        ],
    )
    write_csv(
        args.table_dir / f"{args.prefix}_task_success_with_ci.csv",
        task_rows,
        [
            "policy",
            "n_action_steps",
            "task",
            "task_family",
            "success",
            "attempted",
            "success_rate",
            "wilson95_low",
            "wilson95_high",
            "json_path",
        ],
    )
    write_csv(
        args.table_dir / f"{args.prefix}_task_family_success_with_ci.csv",
        family_rows,
        ["policy", "n_action_steps", "task_family", "success", "attempted", "success_rate", "wilson95_low", "wilson95_high"],
    )
    print(args.table_dir / f"{args.prefix}_bootstrap_ci.csv")
    print(args.table_dir / f"{args.prefix}_task_success_with_ci.csv")
    print(args.table_dir / f"{args.prefix}_task_family_success_with_ci.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
