#!/usr/bin/env python3
"""Validate a Hugging Face dataset repo before downloading it."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi


def sizeof_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def entry_path(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("path") or entry.get("rfilename") or "")
    return str(getattr(entry, "path", "") or getattr(entry, "rfilename", ""))


def entry_size(entry: object) -> int:
    if isinstance(entry, dict):
        value = entry.get("size", 0)
    else:
        value = getattr(entry, "size", 0)
    return int(value or 0)


def entry_type(entry: object) -> str:
    if isinstance(entry, dict):
        return str(entry.get("type") or "file")
    cls_name = entry.__class__.__name__.lower()
    if "folder" in cls_name:
        return "directory"
    return "file"


def attr_any(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def rows_from_dataset_info(api: HfApi, repo_id: str) -> tuple[list[dict[str, object]], int | None]:
    info = api.dataset_info(repo_id=repo_id)
    siblings = attr_any(info, "siblings") or []
    rows = []
    for sibling in siblings:
        path = attr_any(sibling, "rfilename", "path")
        if not path:
            continue
        rows.append(
            {
                "path": str(path),
                "type": "file",
                "size_bytes": int(attr_any(sibling, "size") or 0),
            }
        )
    used_storage = attr_any(info, "used_storage", "usedStorage")
    return rows, int(used_storage) if used_storage is not None else None


def rows_from_repo_tree(api: HfApi, repo_id: str, repo_type: str) -> list[dict[str, object]]:
    entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            repo_type=repo_type,
            recursive=True,
            expand=True,
        )
    )

    rows = []
    for item in entries:
        path = entry_path(item)
        if not path:
            continue
        rows.append(
            {
                "path": path,
                "type": entry_type(item),
                "size_bytes": entry_size(item),
            }
        )
    return rows


def has_prefix(paths: Iterable[str], prefix: str) -> bool:
    normalized = prefix.rstrip("/") + "/"
    return any(path == prefix.rstrip("/") or path.startswith(normalized) for path in paths)


def has_child(paths: Iterable[str], prefix: str, child: str) -> bool:
    normalized = prefix.rstrip("/") + "/" + child.strip("/") + "/"
    return any(path.startswith(normalized) for path in paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-gib", type=float, default=50.0)
    parser.add_argument("--required-prefix", action="append", default=[])
    args = parser.parse_args()

    output = Path(args.output)
    summary = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
    manifest_source = "dataset_info"
    used_storage_bytes: int | None = None
    if args.repo_type == "dataset":
        rows, used_storage_bytes = rows_from_dataset_info(api, args.repo_id)
    else:
        rows = []

    if not rows:
        manifest_source = "repo_tree"
        rows = rows_from_repo_tree(api, args.repo_id, args.repo_type)

    paths = [row["path"] for row in rows]
    summed_bytes = sum(int(row["size_bytes"]) for row in rows if row["type"] == "file")
    total_bytes = used_storage_bytes or summed_bytes
    missing_prefixes = [prefix for prefix in args.required_prefix if not has_prefix(paths, prefix)]
    missing_children: list[str] = []
    for prefix in args.required_prefix:
        for child in ("data", "meta", "videos"):
            if not has_child(paths, prefix, child):
                missing_children.append(f"{prefix.rstrip('/')}/{child}/")

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "type", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)

    status = "ok"
    problems: list[str] = []
    if not rows:
        status = "failed"
        problems.append("repository manifest is empty")
    if missing_prefixes:
        status = "failed"
        problems.append("missing required prefixes: " + ", ".join(missing_prefixes))
    if missing_children:
        status = "failed"
        problems.append("missing required child directories: " + ", ".join(missing_children))
    if total_bytes <= 0:
        status = "failed"
        problems.append("manifest total size is zero")
    if sizeof_gib(total_bytes) > args.max_gib:
        status = "failed"
        problems.append(
            f"manifest total size {sizeof_gib(total_bytes):.2f} GiB exceeds "
            f"MAX_DOWNLOAD_GIB={args.max_gib:.2f}"
        )

    with summary.open("w", encoding="utf-8") as handle:
        handle.write(f"repo_id: {args.repo_id}\n")
        handle.write(f"repo_type: {args.repo_type}\n")
        handle.write(f"hf_endpoint: {os.environ.get('HF_ENDPOINT', '')}\n")
        handle.write(f"manifest_source: {manifest_source}\n")
        handle.write(f"status: {status}\n")
        handle.write(f"file_or_directory_count: {len(rows)}\n")
        handle.write(f"summed_file_size_bytes: {summed_bytes}\n")
        if used_storage_bytes is not None:
            handle.write(f"used_storage_bytes: {used_storage_bytes}\n")
        handle.write(f"total_size_bytes: {total_bytes}\n")
        handle.write(f"total_size_gib: {sizeof_gib(total_bytes):.3f}\n")
        handle.write(f"max_download_gib: {args.max_gib:.3f}\n")
        if args.required_prefix:
            handle.write("required_prefixes:\n")
            for prefix in args.required_prefix:
                handle.write(f"  - {prefix}\n")
        if problems:
            handle.write("problems:\n")
            for problem in problems:
                handle.write(f"  - {problem}\n")

    print(f"Manifest status: {status}")
    print(f"Total size: {sizeof_gib(total_bytes):.3f} GiB")
    print(f"Manifest CSV: {output}")
    print(f"Summary: {summary}")

    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
