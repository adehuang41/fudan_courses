#!/usr/bin/env python3
"""Verify B-only and ABC-mixed CALVIN episode filters before ACT training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.calvin_filtered_dataset import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    SPLIT_ENVIRONMENTS,
    CalvinFilteredFrameDataset,
    load_episode_manifest,
    make_calvin_dataloader,
    summarize_records,
)


def assert_equal(name: str, actual, expected, problems: list[str]) -> None:
    if actual != expected:
        problems.append(f"{name}: actual={actual!r}, expected={expected!r}")


def validate_split(
    split_name: str,
    dataset_root: Path,
    split_manifest: Path,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> dict:
    allowed_envs = set(SPLIT_ENVIRONMENTS[split_name])
    manifest_records = load_episode_manifest(split_manifest)
    expected_records = [record for record in manifest_records if record.environment in allowed_envs]
    expected_summary = summarize_records(expected_records)

    dataset = CalvinFilteredFrameDataset(
        split_name=split_name,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
    )
    dataset_summary = dataset.split_summary()

    problems: list[str] = []
    assert_equal("dataset.num_episodes", dataset.num_episodes, expected_summary["num_episodes"], problems)
    assert_equal("dataset.num_frames", dataset.num_frames, expected_summary["num_frames"], problems)
    assert_equal("dataset.by_environment", dataset_summary["by_environment"], expected_summary["by_environment"], problems)
    assert_equal("dataset.by_shard", dataset_summary["by_shard"], expected_summary["by_shard"], problems)

    observed_envs = set(dataset_summary["by_environment"])
    unexpected_envs = observed_envs.difference(allowed_envs)
    if unexpected_envs:
        problems.append(f"unexpected environments in dataset: {sorted(unexpected_envs)}")

    if split_name == "B_only":
        for env in ("A", "C"):
            count = dataset_summary["by_environment"].get(env, {}).get("num_episodes", 0)
            if count != 0:
                problems.append(f"B_only contains {count} {env} episodes")

    loader = make_calvin_dataloader(
        split_name=split_name,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=0,
    )
    sampled_envs: Counter[str] = Counter()
    sampled_pairs: Counter[tuple[str, int]] = Counter()
    sampled_frames = 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch_envs = list(batch["environment"])
        batch_shards = list(batch["shard"])
        batch_episode_indices = [int(x) for x in batch["episode_index"].tolist()]
        for env, shard, episode_index in zip(batch_envs, batch_shards, batch_episode_indices, strict=True):
            sampled_envs[env] += 1
            sampled_pairs[(shard, episode_index)] += 1
            sampled_frames += 1
            if env not in allowed_envs:
                problems.append(
                    f"dataloader sampled disallowed env={env} shard={shard} episode={episode_index}"
                )

    return {
        "split_name": split_name,
        "allowed_environments": sorted(allowed_envs),
        "status": "ok" if not problems else "failed",
        "problems": problems,
        "expected_summary": expected_summary,
        "dataset_summary": dataset_summary,
        "dataloader_sample": {
            "batch_size": batch_size,
            "max_batches": max_batches,
            "seed": seed,
            "sampled_frames": sampled_frames,
            "sampled_environment_counts": dict(sorted(sampled_envs.items())),
            "sampled_unique_episodes": len(sampled_pairs),
        },
    }


def write_summary_csv(path: Path, reports: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split_name",
            "status",
            "allowed_environments",
            "num_episodes",
            "num_frames",
            "by_environment",
            "dataloader_sampled_frames",
            "dataloader_sampled_environment_counts",
            "problems",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            summary = report["dataset_summary"]
            sample = report["dataloader_sample"]
            writer.writerow(
                {
                    "split_name": report["split_name"],
                    "status": report["status"],
                    "allowed_environments": ";".join(report["allowed_environments"]),
                    "num_episodes": summary["num_episodes"],
                    "num_frames": summary["num_frames"],
                    "by_environment": json.dumps(summary["by_environment"], sort_keys=True),
                    "dataloader_sampled_frames": sample["sampled_frames"],
                    "dataloader_sampled_environment_counts": json.dumps(
                        sample["sampled_environment_counts"], sort_keys=True
                    ),
                    "problems": "|".join(report["problems"]),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("/home/dechao/cv_final_pj2/results/tables/filtered_split_verification.json"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/home/dechao/cv_final_pj2/results/tables/filtered_split_verification.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    reports = [
        validate_split(
            split_name,
            args.dataset_root,
            args.split_manifest,
            args.batch_size,
            args.max_batches,
            args.seed,
        )
        for split_name in ("B_only", "ABC_mixed")
    ]

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary_csv(args.summary_csv, reports)

    for report in reports:
        summary = report["dataset_summary"]
        sample = report["dataloader_sample"]
        print(
            f"{report['split_name']}: status={report['status']} "
            f"episodes={summary['num_episodes']} frames={summary['num_frames']} "
            f"envs={summary['by_environment']} "
            f"dataloader_sample={sample['sampled_environment_counts']}"
        )
        for problem in report["problems"]:
            print(f"Problem: {problem}")

    print(f"Report JSON: {args.report_json}")
    print(f"Summary CSV: {args.summary_csv}")
    return 0 if all(report["status"] == "ok" for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
