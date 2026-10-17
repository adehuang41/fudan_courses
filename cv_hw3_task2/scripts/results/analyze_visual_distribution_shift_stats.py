#!/usr/bin/env python3
"""Compare source-train and D image statistics without loading image tensors."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.evaluate_d_offline_action_l1 import read_json, read_split_csv, read_yaml  # noqa: E402
from src.data.calvin_d_lerobot_dataset import DEFAULT_D_DATASET_ROOT, compute_d_dataset_stats  # noqa: E402
from src.data.calvin_filtered_dataset import compute_records_stats  # noqa: E402


RUNS = {
    "B_only_act_baseline": Path("/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline"),
    "ABC_mixed_act_baseline": Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline"),
}
IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().float().reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_D_DATASET_ROOT)
    parser.add_argument("--table-dir", type=Path, default=PROJECT_ROOT / "results/tables/step4")
    parser.add_argument("--prefix", default="visual_distribution_shift_stats")
    args = parser.parse_args()

    d_stats = compute_d_dataset_stats(args.dataset_root)
    rows: list[dict] = []
    channel_rows: list[dict] = []
    for experiment, run_dir in RUNS.items():
        summary = read_json(run_dir / "metrics_summary.json")
        config = read_yaml(run_dir / "config.yaml")
        train_records = read_split_csv(Path(summary["train_val_episode_split_csv"]), split="train")
        source_stats = compute_records_stats(
            train_records,
            dataset_root=Path(config["train"]["dataset"]["dataset_root"]),
            keys=IMAGE_KEYS,
        )
        for key in IMAGE_KEYS:
            source_mean = flat(source_stats[key]["mean"])
            source_std = flat(source_stats[key]["std"])
            d_mean = flat(d_stats[key]["mean"])
            d_std = flat(d_stats[key]["std"])
            mean_diff = source_mean - d_mean
            std_diff = source_std - d_std
            rows.append(
                {
                    "experiment": experiment,
                    "image_key": key,
                    "source_train_episodes": len(train_records),
                    "d_episodes": 6135,
                    "mean_l2_distance": float(torch.linalg.vector_norm(mean_diff).item()),
                    "mean_l1_distance": float(torch.mean(torch.abs(mean_diff)).item()),
                    "std_l2_distance": float(torch.linalg.vector_norm(std_diff).item()),
                    "std_l1_distance": float(torch.mean(torch.abs(std_diff)).item()),
                    "source_mean_avg": float(source_mean.mean().item()),
                    "d_mean_avg": float(d_mean.mean().item()),
                    "source_std_avg": float(source_std.mean().item()),
                    "d_std_avg": float(d_std.mean().item()),
                }
            )
            for channel, name in enumerate(["R", "G", "B"]):
                channel_rows.append(
                    {
                        "experiment": experiment,
                        "image_key": key,
                        "channel": name,
                        "source_mean": float(source_stats[key]["mean"].reshape(3, -1)[channel].mean().item()),
                        "d_mean": float(d_stats[key]["mean"].reshape(3, -1)[channel].mean().item()),
                        "source_std": float(source_stats[key]["std"].reshape(3, -1)[channel].mean().item()),
                        "d_std": float(d_stats[key]["std"].reshape(3, -1)[channel].mean().item()),
                    }
                )

    write_csv(
        args.table_dir / f"{args.prefix}.csv",
        rows,
        [
            "experiment",
            "image_key",
            "source_train_episodes",
            "d_episodes",
            "mean_l2_distance",
            "mean_l1_distance",
            "std_l2_distance",
            "std_l1_distance",
            "source_mean_avg",
            "d_mean_avg",
            "source_std_avg",
            "d_std_avg",
        ],
    )
    write_csv(
        args.table_dir / f"{args.prefix}_channels.csv",
        channel_rows,
        ["experiment", "image_key", "channel", "source_mean", "d_mean", "source_std", "d_std"],
    )
    print(args.table_dir / f"{args.prefix}.csv")
    print(args.table_dir / f"{args.prefix}_channels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
