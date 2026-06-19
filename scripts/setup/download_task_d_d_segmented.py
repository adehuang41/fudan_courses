#!/usr/bin/env python3
"""Resumable segmented downloader for the official CALVIN task_D_D.zip."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
DEFAULT_ZIP = Path("/data/dechao/cv_final_pj2/datasets/task_D_D.zip")
EXPECTED_SIZE = 177_379_436_142
EXPECTED_SHA256 = "45efc2fb24a09a50ab3ed6cdc7637604ee857d3ba1bab23d63925c2d71e79d4f"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_log(path: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def part_bounds(total_size: int, num_parts: int) -> list[tuple[int, int]]:
    part_size = math.ceil(total_size / num_parts)
    bounds = []
    for index in range(num_parts):
        start = index * part_size
        end = min(total_size - 1, (index + 1) * part_size - 1)
        if start <= end:
            bounds.append((start, end))
    return bounds


def part_path(zip_path: Path, parts_dir: Path, index: int) -> Path:
    return parts_dir / f"{zip_path.name}.part{index:02d}"


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def curl_append(
    *,
    url: str,
    output_path: Path,
    byte_start: int,
    byte_end: int,
    rate_limit: str,
    stderr_path: Path,
) -> int:
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--speed-time",
        "600",
        "--speed-limit",
        "1048576",
        "--connect-timeout",
        "30",
        "--limit-rate",
        rate_limit,
        "-r",
        f"{byte_start}-{byte_end}",
        url,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("ab") as output, stderr_path.open("ab") as stderr:
        return subprocess.run(cmd, stdout=output, stderr=stderr, check=False).returncode


def download_part(
    *,
    index: int,
    bounds: tuple[int, int],
    url: str,
    zip_path: Path,
    parts_dir: Path,
    rate_limit: str,
    status_log: Path,
) -> None:
    start, end = bounds
    expected_len = end - start + 1
    output_path = part_path(zip_path, parts_dir, index)
    stderr_path = parts_dir / f"part{index:02d}.curl.log"
    while True:
        current_size = file_size(output_path)
        if current_size == expected_len:
            append_log(status_log, f"part={index:02d} complete bytes={current_size}")
            return
        if current_size > expected_len:
            raise RuntimeError(
                f"{output_path} is larger than expected part size: {current_size} > {expected_len}"
            )
        byte_start = start + current_size
        append_log(
            status_log,
            f"part={index:02d} resume range={byte_start}-{end} existing={current_size}/{expected_len}",
        )
        code = curl_append(
            url=url,
            output_path=output_path,
            byte_start=byte_start,
            byte_end=end,
            rate_limit=rate_limit,
            stderr_path=stderr_path,
        )
        if code == 0:
            continue
        append_log(status_log, f"part={index:02d} curl_exit={code}; retrying_after=20s")
        time.sleep(20)


def append_parts(zip_path: Path, parts_dir: Path, bounds: list[tuple[int, int]], status_log: Path) -> None:
    tmp_path = zip_path.with_name(f"{zip_path.name}.assembling")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("ab") as output:
        for index in range(len(bounds)):
            path = part_path(zip_path, parts_dir, index)
            expected_len = bounds[index][1] - bounds[index][0] + 1
            actual_len = file_size(path)
            if actual_len != expected_len:
                raise RuntimeError(f"{path} size {actual_len} != expected {expected_len}")
            append_log(status_log, f"assembling part={index:02d} bytes={actual_len}")
            with path.open("rb") as source:
                while True:
                    chunk = source.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    tmp_size = file_size(tmp_path)
    if tmp_size != EXPECTED_SIZE:
        raise RuntimeError(f"assembled temp size {tmp_size} != expected {EXPECTED_SIZE}")
    os.replace(tmp_path, zip_path)
    append_log(status_log, f"assembly_complete bytes={file_size(zip_path)}")


def sha256_file(path: Path, status_log: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(64 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    append_log(status_log, f"sha256={value}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--parts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-limit", default="8M")
    args = parser.parse_args()

    zip_path = args.zip
    log_dir = zip_path.parent / "_download_logs" / "task_D_D"
    parts_dir = log_dir / "segments"
    status_log = log_dir / "segmented_status.log"
    sha_log = log_dir / "sha256.log"

    if args.parts < 2:
        raise ValueError("--parts must be at least 2")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    bounds = part_bounds(EXPECTED_SIZE, args.parts)

    append_log(
        status_log,
        f"segmented_start url={args.url} zip={zip_path} parts={len(bounds)} workers={args.workers}",
    )
    append_log(status_log, f"expected_size={EXPECTED_SIZE} expected_sha256={EXPECTED_SHA256}")

    if file_size(zip_path) == EXPECTED_SIZE:
        append_log(status_log, "zip already has expected size; verifying sha256")
        actual = sha256_file(zip_path, status_log)
        sha_log.write_text(f"{actual}  {zip_path}\n", encoding="utf-8")
        return 0 if actual == EXPECTED_SHA256 else 2

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_part,
                index=index,
                bounds=bounds[index],
                url=args.url,
                zip_path=zip_path,
                parts_dir=parts_dir,
                rate_limit=args.rate_limit,
                status_log=status_log,
            )
            for index in range(len(bounds))
        ]
        for future in as_completed(futures):
            future.result()

    append_parts(zip_path, parts_dir, bounds, status_log)
    actual_size = file_size(zip_path)
    if actual_size != EXPECTED_SIZE:
        raise RuntimeError(f"assembled zip size {actual_size} != expected {EXPECTED_SIZE}")
    actual_sha = sha256_file(zip_path, status_log)
    sha_log.write_text(f"{actual_sha}  {zip_path}\n", encoding="utf-8")
    ok = actual_sha == EXPECTED_SHA256
    append_log(status_log, f"sha256_match={ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise
