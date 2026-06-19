#!/usr/bin/env python3
"""Audit CALVIN A/B/C environment labels for the local LeRobot conversion.

The Hugging Face repo used in this project stores CALVIN task_ABC_D as four
LeRobot upload shards. The shards are not environment labels. Environment
labels are recovered by aligning each episode's original_frame_idx with the
official CALVIN task_ABC_D training/scene_info.npy ranges.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_DATASET_ROOT = Path("/data/dechao/cv_final_pj2/datasets/calvin_task_ABC_D")
DEFAULT_PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
DEFAULT_SCENE_INFO_PATH = (
    DEFAULT_DATASET_ROOT
    / "_download_logs"
    / "scene_info"
    / "task_ABC_D"
    / "training"
    / "scene_info.npy"
)
SCENE_INFO_SOURCE_URL = "http://calvin.cs.uni-freiburg.de/scene_info_fix/task_ABC_D_scene_info.zip"
SCENE_INFO_ARCHIVE_SHA256 = "3c13651947dc4927b1df3b700143e1d47605db0958f296a0a691b911876d2a64"
SCENE_INFO_SHA256 = "cd254dd8c94abe8c9c360b079f8839eac1811c306a35248d8a5afe7a5820be67"

EXPECTED_SCENE_RANGES = {
    "B": {
        "scene": "calvin_scene_B",
        "min_original_frame_idx": 0,
        "max_original_frame_idx": 598909,
    },
    "C": {
        "scene": "calvin_scene_C",
        "min_original_frame_idx": 598910,
        "max_original_frame_idx": 1191338,
    },
    "A": {
        "scene": "calvin_scene_A",
        "min_original_frame_idx": 1191339,
        "max_original_frame_idx": 1795044,
    },
}

SPLIT_BASIS = (
    "official CALVIN task_ABC_D training/scene_info.npy ranges aligned with "
    "LeRobot original_frame_idx; 0_4..3_4 are upload shards, not environments"
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene_ranges(scene_info_path: Path) -> dict[str, dict[str, int | str]]:
    if not scene_info_path.exists():
        raise SystemExit(f"Missing scene_info.npy: {scene_info_path}")

    observed_sha256 = sha256_file(scene_info_path)
    if observed_sha256 != SCENE_INFO_SHA256:
        raise SystemExit(
            f"Unexpected scene_info.npy sha256: {observed_sha256}; expected {SCENE_INFO_SHA256}"
        )

    raw = np.load(scene_info_path, allow_pickle=True).item()
    scene_to_env = {
        "calvin_scene_A": "A",
        "calvin_scene_B": "B",
        "calvin_scene_C": "C",
    }
    ranges: dict[str, dict[str, int | str]] = {}
    for scene, bounds in raw.items():
        env = scene_to_env.get(scene)
        if env is None:
            continue
        ranges[env] = {
            "scene": scene,
            "min_original_frame_idx": int(bounds[0]),
            "max_original_frame_idx": int(bounds[1]),
        }

    if ranges != EXPECTED_SCENE_RANGES:
        raise SystemExit(
            "scene_info.npy does not match the expected corrected task_ABC_D/training ranges: "
            f"{ranges}"
        )
    return ranges


def label_from_original_frame_idx(lo: int, hi: int, scene_ranges: dict[str, dict[str, int | str]]) -> str:
    for env, meta in scene_ranges.items():
        if meta["min_original_frame_idx"] <= lo and hi <= meta["max_original_frame_idx"]:
            return env
    overlapping = [
        env
        for env, meta in scene_ranges.items()
        if not (hi < meta["min_original_frame_idx"] or lo > meta["max_original_frame_idx"])
    ]
    if overlapping:
        return "cross_or_unknown:" + "+".join(overlapping)
    return "unknown"


def feature_shape(info: dict, key: str) -> str:
    value = info["features"].get(key, {})
    shape = value.get("shape")
    return "x".join(str(v) for v in shape) if shape else ""


def camera_keys(info: dict) -> str:
    keys = [
        key
        for key, value in info["features"].items()
        if key.startswith("observation.images.") and value.get("dtype") == "video"
    ]
    return ";".join(sorted(keys))


def audit(dataset_root: Path, project_root: Path, scene_info_path: Path) -> None:
    shard_dirs = sorted(dataset_root.glob("calvin_task_ABC_D_lerobot_*_4"))
    if not shard_dirs:
        raise SystemExit(f"No LeRobot shard dirs found under {dataset_root}")

    scene_ranges = load_scene_ranges(scene_info_path)

    results_dir = project_root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = dataset_root / "_download_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_counts = defaultdict(lambda: {"episodes": 0, "frames": 0, "top_level_dirs": Counter()})
    env_tasks: dict[str, Counter] = defaultdict(Counter)
    env_episode_rows: list[dict[str, object]] = []
    semantics_rows: list[dict[str, object]] = []

    unknown_or_cross = []
    unknown_count = 0
    cross_boundary_count = 0
    reference_info = None
    total_episodes = 0
    total_frames = 0

    for shard in shard_dirs:
        info = read_json(shard / "meta" / "info.json")
        reference_info = reference_info or info
        episodes = {ep["episode_index"]: ep for ep in read_jsonl(shard / "meta" / "episodes.jsonl")}

        semantics_rows.append(
            {
                "item": shard.name,
                "type": "lerobot_upload_shard",
                "has_env_field": False,
                "splits": json.dumps(info.get("splits", {}), sort_keys=True),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "note": "contains mixed A/B/C episodes; not an A/B/C/D environment directory",
            }
        )

        for ep_stat in read_jsonl(shard / "meta" / "episodes_stats.jsonl"):
            ep_idx = ep_stat["episode_index"]
            stats = ep_stat["stats"]["original_frame_idx"]
            lo = int(stats["min"][0])
            hi = int(stats["max"][0])
            frame_count = int(stats["count"][0])
            env = label_from_original_frame_idx(lo, hi, scene_ranges)
            ep = episodes.get(ep_idx, {})
            tasks = ep.get("tasks", [])

            if env not in scene_ranges:
                unknown_or_cross.append((shard.name, ep_idx, lo, hi, env))
                if env == "unknown":
                    unknown_count += 1
                else:
                    cross_boundary_count += 1

            env_counts[env]["episodes"] += 1
            env_counts[env]["frames"] += frame_count
            env_counts[env]["top_level_dirs"][shard.name] += 1
            total_episodes += 1
            total_frames += frame_count
            for task in tasks:
                env_tasks[env][task] += 1

            env_episode_rows.append(
                {
                    "shard": shard.name,
                    "episode_index": ep_idx,
                    "original_frame_idx_min": lo,
                    "original_frame_idx_max": hi,
                    "environment": env,
                    "num_frames": frame_count,
                    "task": "|".join(tasks),
                }
            )

    if reference_info is None:
        raise SystemExit("No info.json files found")

    summary_path = results_dir / "environment_split_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "environment",
            "scene",
            "original_frame_idx_min",
            "original_frame_idx_max",
            "num_episodes",
            "num_frames",
            "num_tasks",
            "top_level_dirs",
            "cameras",
            "state_dim",
            "action_dim",
            "split_basis",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for env in ["A", "B", "C"]:
            meta = scene_ranges[env]
            counts = env_counts[env]
            writer.writerow(
                {
                    "environment": env,
                    "scene": meta["scene"],
                    "original_frame_idx_min": meta["min_original_frame_idx"],
                    "original_frame_idx_max": meta["max_original_frame_idx"],
                    "num_episodes": counts["episodes"],
                    "num_frames": counts["frames"],
                    "num_tasks": len(env_tasks[env]),
                    "top_level_dirs": ";".join(
                        f"{name}:{count}" for name, count in sorted(counts["top_level_dirs"].items())
                    ),
                    "cameras": camera_keys(reference_info),
                    "state_dim": feature_shape(reference_info, "observation.state"),
                    "action_dim": feature_shape(reference_info, "action"),
                    "split_basis": SPLIT_BASIS,
                }
            )
        writer.writerow(
            {
                "environment": "D",
                "scene": "calvin_scene_D",
                "original_frame_idx_min": "",
                "original_frame_idx_max": "",
                "num_episodes": 0,
                "num_frames": 0,
                "num_tasks": 0,
                "top_level_dirs": "",
                "cameras": camera_keys(reference_info),
                "state_dim": feature_shape(reference_info, "observation.state"),
                "action_dim": feature_shape(reference_info, "action"),
                "split_basis": "not present in huiwon/calvin_task_ABC_D LeRobot train-only repo",
            }
        )

    semantics_rows.append(
        {
            "item": "environment_mapping",
            "type": "recovered_from_original_frame_idx",
            "has_env_field": False,
            "splits": "train only",
            "total_episodes": sum(env_counts[e]["episodes"] for e in ["A", "B", "C"]),
            "total_frames": sum(env_counts[e]["frames"] for e in ["A", "B", "C"]),
            "note": SPLIT_BASIS,
        }
    )
    semantics_rows.append(
        {
            "item": "held_out_environment_D",
            "type": "absent_from_current_local_repo",
            "has_env_field": False,
            "splits": "not available",
            "total_episodes": 0,
            "total_frames": 0,
            "note": "D is the original CALVIN validation/evaluation environment, not included in this LeRobot train-only download",
        }
    )

    semantics_path = results_dir / "dataset_semantics_summary.csv"
    with semantics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "item",
            "type",
            "has_env_field",
            "splits",
            "total_episodes",
            "total_frames",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(semantics_rows)

    scene_info_source_path = results_dir / "scene_info_source.csv"
    with scene_info_source_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "item",
            "value",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [
                {"item": "local_scene_info_path", "value": str(scene_info_path)},
                {"item": "source_url", "value": SCENE_INFO_SOURCE_URL},
                {"item": "source_archive_sha256", "value": SCENE_INFO_ARCHIVE_SHA256},
                {"item": "scene_info_sha256", "value": sha256_file(scene_info_path)},
                {"item": "logical_path", "value": "task_ABC_D/training/scene_info.npy"},
                {"item": "corrected_scene_info_check", "value": "passed"},
                {
                    "item": "not_old_scene_info_reason",
                    "value": "downloaded from official scene_info_fix archive and hash/ranges match expected task_ABC_D/training corrected ranges",
                },
            ]
        )

    episode_map_path = logs_dir / "environment_episode_map.csv"
    with episode_map_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "shard",
            "episode_index",
            "original_frame_idx_min",
            "original_frame_idx_max",
            "environment",
            "num_frames",
            "task",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(env_episode_rows)

    abc_episodes = sum(env_counts[env]["episodes"] for env in ["A", "B", "C"])
    abc_frames = sum(env_counts[env]["frames"] for env in ["A", "B", "C"])
    sanity_rows = [
        {"check": "unknown_episodes", "expected": 0, "actual": unknown_count, "status": "ok" if unknown_count == 0 else "fail"},
        {
            "check": "cross_boundary_episodes",
            "expected": 0,
            "actual": cross_boundary_count,
            "status": "ok" if cross_boundary_count == 0 else "fail",
        },
        {
            "check": "abc_frame_total_equals_lerobot_total_frames",
            "expected": total_frames,
            "actual": abc_frames,
            "status": "ok" if abc_frames == total_frames else "fail",
        },
        {
            "check": "abc_episode_total_equals_lerobot_total_episodes",
            "expected": total_episodes,
            "actual": abc_episodes,
            "status": "ok" if abc_episodes == total_episodes else "fail",
        },
    ]
    sanity_path = results_dir / "environment_split_sanity_check.csv"
    with sanity_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["check", "expected", "actual", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sanity_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {semantics_path}")
    print(f"Wrote {scene_info_source_path}")
    print(f"Wrote {episode_map_path}")
    print(f"Wrote {sanity_path}")
    print(f"scene_info_path={scene_info_path}")
    print(f"scene_info_sha256={sha256_file(scene_info_path)}")
    for env in ["A", "B", "C"]:
        print(env, env_counts[env]["episodes"], "episodes", env_counts[env]["frames"], "frames")
    print("D 0 episodes 0 frames")
    for row in sanity_rows:
        print(f"{row['check']}: expected={row['expected']} actual={row['actual']} status={row['status']}")

    if unknown_or_cross:
        print("Unknown/cross-boundary episodes:")
        for row in unknown_or_cross[:20]:
            print(row)
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--scene-info-path", type=Path, default=DEFAULT_SCENE_INFO_PATH)
    args = parser.parse_args()
    audit(args.dataset_root, args.project_root, args.scene_info_path)


if __name__ == "__main__":
    main()
