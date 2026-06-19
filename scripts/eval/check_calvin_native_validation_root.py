#!/usr/bin/env python3
"""Check whether a directory looks like a native CALVIN validation root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


REQUIRED_RELATIVE_PATHS = [
    "validation/.hydra/merged_config.yaml",
    "validation/scene_info.npy",
    "validation/lang_annotations/auto_lang_ann.npy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a native CALVIN validation root.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for rel_path in REQUIRED_RELATIVE_PATHS:
        path = args.dataset_root / rel_path
        rows.append(
            {
                "check": rel_path,
                "status": "present" if path.exists() else "missing",
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )

    episode_paths = sorted((args.dataset_root / "validation").glob("episode_*.npz"))
    rows.append(
        {
            "check": "validation/episode_*.npz",
            "status": "present" if episode_paths else "missing",
            "path": str(args.dataset_root / "validation"),
            "size_bytes": len(episode_paths),
        }
    )

    summary = {
        "dataset_root": str(args.dataset_root),
        "source_note": "VyoJ/calvin-ABCD-D-subsets validation/subset_validation_000.zip; task_ABCD_D validation subset, not full task_ABC_D 517GB validation",
        "all_required_present": all(row["status"] == "present" for row in rows),
        "episode_count": len(episode_paths),
    }

    scene_info_path = args.dataset_root / "validation" / "scene_info.npy"
    if scene_info_path.exists():
        scene_info = np.load(scene_info_path, allow_pickle=True)
        summary["scene_info_type"] = type(scene_info.item()).__name__ if scene_info.shape == () else type(scene_info).__name__
        if scene_info.shape == ():
            scene_info_obj = scene_info.item()
            if isinstance(scene_info_obj, dict):
                summary["scene_info_keys"] = sorted(map(str, scene_info_obj.keys()))

    lang_ann_path = args.dataset_root / "validation" / "lang_annotations" / "auto_lang_ann.npy"
    if lang_ann_path.exists():
        lang_ann = np.load(lang_ann_path, allow_pickle=True).item()
        summary["lang_annotation_keys"] = sorted(map(str, lang_ann.keys()))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "status", "path", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
