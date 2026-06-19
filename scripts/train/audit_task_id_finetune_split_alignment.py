#!/usr/bin/env python3
"""Audit train/validation split alignment for task-id fine-tunes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_act_full import split_train_val  # noqa: E402
from src.data.calvin_filtered_dataset import EpisodeRecord, select_episode_records  # noqa: E402


PAIRS = [
    {
        "regime": "B_only",
        "visual_config": PROJECT_ROOT / "configs/train/full_B_only.yaml",
        "visual_split_csv": Path(
            "/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline/train_val_episode_split.csv"
        ),
        "finetune_config": PROJECT_ROOT
        / "configs/train/finetune_B_only_task_id_from_visual50k_extra50k_seed20260531_retry1.yaml",
    },
    {
        "regime": "ABC_mixed",
        "visual_config": PROJECT_ROOT / "configs/train/full_ABC_mixed.yaml",
        "visual_split_csv": Path(
            "/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline/train_val_episode_split.csv"
        ),
        "finetune_config": PROJECT_ROOT
        / "configs/train/finetune_ABC_mixed_task_id_from_visual50k_extra50k_seed20260531_retry1.yaml",
    },
]


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def record_key_from_episode(record: EpisodeRecord) -> tuple:
    return (
        record.shard,
        int(record.episode_index),
        int(record.original_frame_idx_min),
        int(record.original_frame_idx_max),
        record.environment,
        int(record.num_frames),
        record.task,
    )


def record_key_from_row(row: dict[str, str]) -> tuple:
    return (
        row["shard"],
        int(row["episode_index"]),
        int(row["original_frame_idx_min"]),
        int(row["original_frame_idx_max"]),
        row["environment"],
        int(row["num_frames"]),
        row["task"],
    )


def load_split_csv(path: Path) -> dict[str, set[tuple]]:
    splits = {"train": set(), "val": set()}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in splits:
                raise ValueError(f"Unknown split={split!r} in {path}")
            splits[split].add(record_key_from_row(row))
    return splits


def planned_split(config: dict) -> dict[str, set[tuple]]:
    dataset = config["dataset"]
    train = config["train"]
    records = select_episode_records(dataset["split_name"], Path(dataset["split_manifest"]))
    train_records, val_records = split_train_val(
        records=records,
        validation_fraction=float(dataset["validation_fraction"]),
        seed=int(train["seed"]),
    )
    return {
        "train": {record_key_from_episode(record) for record in train_records},
        "val": {record_key_from_episode(record) for record in val_records},
    }


def audit_pair(pair: dict) -> dict:
    visual_config = read_yaml(pair["visual_config"])
    finetune_config = read_yaml(pair["finetune_config"])
    visual_split = load_split_csv(pair["visual_split_csv"])
    finetune_split = planned_split(finetune_config)

    visual_train = visual_config["train"]
    visual_dataset = visual_config["dataset"]
    ft_train = finetune_config["train"]
    ft_dataset = finetune_config["dataset"]

    train_equal = visual_split["train"] == finetune_split["train"]
    val_equal = visual_split["val"] == finetune_split["val"]
    val_in_visual_train = len(finetune_split["val"] & visual_split["train"])
    train_in_visual_val = len(finetune_split["train"] & visual_split["val"])
    planned_train_val_overlap = len(finetune_split["train"] & finetune_split["val"])

    return {
        "regime": pair["regime"],
        "visual_config": str(pair["visual_config"]),
        "finetune_config": str(pair["finetune_config"]),
        "visual_split_csv": str(pair["visual_split_csv"]),
        "visual_seed": int(visual_train["seed"]),
        "finetune_seed": int(ft_train["seed"]),
        "seed_equal": int(visual_train["seed"]) == int(ft_train["seed"]),
        "visual_validation_fraction": float(visual_dataset["validation_fraction"]),
        "finetune_validation_fraction": float(ft_dataset["validation_fraction"]),
        "validation_fraction_equal": float(visual_dataset["validation_fraction"])
        == float(ft_dataset["validation_fraction"]),
        "visual_split_name": visual_dataset["split_name"],
        "finetune_split_name": ft_dataset["split_name"],
        "split_name_equal": visual_dataset["split_name"] == ft_dataset["split_name"],
        "visual_train_count": len(visual_split["train"]),
        "visual_val_count": len(visual_split["val"]),
        "finetune_train_count": len(finetune_split["train"]),
        "finetune_val_count": len(finetune_split["val"]),
        "train_split_exact_match": train_equal,
        "val_split_exact_match": val_equal,
        "finetune_val_in_visual_train_count": val_in_visual_train,
        "finetune_train_in_visual_val_count": train_in_visual_val,
        "planned_train_val_overlap_count": planned_train_val_overlap,
        "audit_passed": bool(
            train_equal
            and val_equal
            and val_in_visual_train == 0
            and train_in_visual_val == 0
            and planned_train_val_overlap == 0
            and int(visual_train["seed"]) == int(ft_train["seed"])
            and float(visual_dataset["validation_fraction"]) == float(ft_dataset["validation_fraction"])
            and visual_dataset["split_name"] == ft_dataset["split_name"]
        ),
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "results/tables/step4/task_id_finetune_split_alignment.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results/tables/step4/task_id_finetune_split_alignment.csv",
    )
    args = parser.parse_args()

    rows = [audit_pair(pair) for pair in PAIRS]
    write_json(args.output_json, {"runs": rows})
    write_csv(args.output_csv, rows)
    print(json.dumps({"runs": rows}, indent=2, ensure_ascii=False))
    return 0 if all(row["audit_passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
