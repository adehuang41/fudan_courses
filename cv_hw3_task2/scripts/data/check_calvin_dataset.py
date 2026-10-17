#!/usr/bin/env python3
"""Lightweight sanity check for the downloaded CALVIN LeRobot dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_SPLITS = [
    "calvin_task_ABC_D_lerobot_0_4",
    "calvin_task_ABC_D_lerobot_1_4",
    "calvin_task_ABC_D_lerobot_2_4",
    "calvin_task_ABC_D_lerobot_3_4",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_head(path: Path) -> dict[str, Any] | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    return None


def find_first(root: Path, suffixes: tuple[str, ...]) -> Path | None:
    for suffix in suffixes:
        found = next(root.rglob(f"*{suffix}"), None)
        if found is not None:
            return found
    return None


def decode_video_frame(video_path: Path, preview_dir: Path) -> tuple[bool, str | None]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "calvin_preview_frame.jpg"

    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            cv2.imwrite(str(preview_path), frame)
            return True, str(preview_path)
    except Exception:
        pass

    try:
        import imageio.v3 as iio  # type: ignore
        from PIL import Image  # type: ignore

        frame = iio.imread(video_path, index=0)
        Image.fromarray(frame).save(preview_path)
        return True, str(preview_path)
    except Exception:
        return False, None


def inspect_parquet(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        return {"readable": False, "error": f"pyarrow unavailable: {exc}"}

    try:
        table = pq.read_table(path)
    except Exception as exc:
        return {"readable": False, "error": str(exc)}

    columns = list(table.column_names)
    lowered = [name.lower() for name in columns]
    return {
        "readable": True,
        "path": str(path),
        "num_rows": table.num_rows,
        "columns": columns,
        "has_image_field": any("image" in name or "video" in name for name in lowered),
        "has_state_field": any("state" in name for name in lowered),
        "has_action_field": any("action" in name for name in lowered),
        "has_task_field": any("task" in name for name in lowered),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--preview-dir", required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    report_json = Path(args.report_json)
    summary_csv = Path(args.summary_csv)
    preview_dir = Path(args.preview_dir)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    if str(dataset_root).startswith("/home/"):
        raise SystemExit(f"Dataset root must not be under /home: {dataset_root}")
    if str(dataset_root).startswith("/data/huggingface"):
        raise SystemExit(f"Dataset root must not be under /data/huggingface: {dataset_root}")

    report: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "expected_splits": EXPECTED_SPLITS,
        "large_paths_ok": True,
        "splits": {},
        "metadata": {},
        "sample_table": {},
        "video_decode": {},
        "environment_paths": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
        },
    }

    problems: list[str] = []
    for split in EXPECTED_SPLITS:
        split_root = dataset_root / split
        split_report = {
            "exists": split_root.exists(),
            "data_exists": (split_root / "data").exists(),
            "meta_exists": (split_root / "meta").exists(),
            "videos_exists": (split_root / "videos").exists(),
            "parquet_count": len(list((split_root / "data").rglob("*.parquet"))) if (split_root / "data").exists() else 0,
            "video_count": len(list((split_root / "videos").rglob("*.mp4"))) if (split_root / "videos").exists() else 0,
        }
        report["splits"][split] = split_report
        if not split_report["exists"]:
            problems.append(f"missing split: {split}")
        for key in ("data_exists", "meta_exists", "videos_exists"):
            if not split_report[key]:
                problems.append(f"{split}: {key} is false")

    first_split_root = dataset_root / EXPECTED_SPLITS[0]
    meta_root = first_split_root / "meta"
    if meta_root.exists():
        for name in ("info.json", "stats.json"):
            path = meta_root / name
            if path.exists():
                try:
                    report["metadata"][name] = read_json(path)
                except Exception as exc:
                    report["metadata"][name] = {"error": str(exc)}
        for name in ("tasks.jsonl", "episodes.jsonl"):
            path = meta_root / name
            if path.exists():
                try:
                    report["metadata"][name] = read_jsonl_head(path)
                except Exception as exc:
                    report["metadata"][name] = {"error": str(exc)}

    parquet_path = find_first(first_split_root / "data", (".parquet",))
    if parquet_path is None:
        problems.append("no parquet data file found")
    else:
        report["sample_table"] = inspect_parquet(parquet_path)
        if not report["sample_table"].get("readable"):
            problems.append("sample parquet is not readable")
        else:
            for key in ("has_state_field", "has_action_field", "has_task_field"):
                if not report["sample_table"].get(key):
                    problems.append(f"sample parquet missing {key}")

    video_path = find_first(first_split_root / "videos", (".mp4", ".avi", ".mov"))
    if video_path is None:
        problems.append("no video file found")
        report["video_decode"] = {"ok": False, "video_path": None, "preview_path": None}
    else:
        ok, preview_path = decode_video_frame(video_path, preview_dir)
        report["video_decode"] = {
            "ok": ok,
            "video_path": str(video_path),
            "preview_path": preview_path,
        }
        if not ok:
            problems.append("failed to decode a video frame")

    unsafe_env = {
        key: value
        for key, value in report["environment_paths"].items()
        if value and (value.startswith("/home/") or value.startswith("/data/huggingface"))
    }
    if unsafe_env:
        report["large_paths_ok"] = False
        problems.append(f"unsafe cache environment paths: {unsafe_env}")

    report["status"] = "ok" if not problems else "failed"
    report["problems"] = problems

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item", "status", "detail"])
        writer.writeheader()
        writer.writerow({"item": "dataset_root", "status": "ok", "detail": str(dataset_root)})
        writer.writerow({"item": "large_paths_ok", "status": str(report["large_paths_ok"]), "detail": ""})
        writer.writerow({"item": "sample_table", "status": str(report["sample_table"].get("readable", False)), "detail": report["sample_table"].get("path", "")})
        writer.writerow({"item": "video_decode", "status": str(report["video_decode"].get("ok", False)), "detail": report["video_decode"].get("video_path", "")})
        for split, split_report in report["splits"].items():
            writer.writerow({"item": split, "status": str(split_report["exists"]), "detail": json.dumps(split_report, ensure_ascii=False)})

    print(f"Sanity status: {report['status']}")
    print(f"Report JSON: {report_json}")
    print(f"Summary CSV: {summary_csv}")
    if problems:
        for problem in problems:
            print(f"Problem: {problem}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
