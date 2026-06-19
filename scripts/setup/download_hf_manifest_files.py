#!/usr/bin/env python3
"""Download every file listed in a validated Hugging Face manifest."""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import httpx
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.file_download import parse_xet_file_data_from_response, refresh_xet_connection_info
from huggingface_hub.utils._xet import XetFileData


warnings.filterwarnings("ignore", category=DeprecationWarning)
_XET_TOKEN_LOCK = Lock()
_XET_LAST_TOKEN_REQUEST = 0.0


def load_manifest(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["path"] for row in reader if row.get("path") and row.get("type") == "file"]


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_line(path: Path, line: str, lock: Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def append_csv(path: Path, row: dict[str, object], lock: Lock) -> None:
    with lock:
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "path", "status", "attempt", "error"])
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def get_xet_connection(xet_file_data: XetFileData, headers: dict[str, str]):
    global _XET_LAST_TOKEN_REQUEST  # noqa: PLW0603 - process-local token rate limiter.
    with _XET_TOKEN_LOCK:
        min_interval = float(os.environ.get("XET_TOKEN_MIN_INTERVAL_SECONDS", "0.75"))
        backoff_seconds = int(os.environ.get("XET_TOKEN_BACKOFF_SECONDS", "90"))
        for attempt in range(1, 4):
            wait_seconds = (_XET_LAST_TOKEN_REQUEST + min_interval) - time.time()
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            try:
                connection_info = refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
                _XET_LAST_TOKEN_REQUEST = time.time()
                return connection_info
            except HfHubHTTPError as exc:
                _XET_LAST_TOKEN_REQUEST = time.time()
                response = getattr(exc, "response", None)
                if getattr(response, "status_code", None) == 429 and attempt < 3:
                    time.sleep(backoff_seconds)
                    continue
                raise
        raise RuntimeError("Failed to refresh Xet connection info")


def xet_download_to_file(incomplete: Path, xet_file_data: XetFileData, headers: dict[str, str], expected_size: int | None) -> None:
    from hf_xet import PyXetDownloadInfo, download_files  # type: ignore[import-not-found]

    connection_info = get_xet_connection(xet_file_data, headers)

    def token_refresher() -> tuple[str, int]:
        connection_info = get_xet_connection(xet_file_data, headers)
        return connection_info.access_token, connection_info.expiration_unix_epoch

    xet_headers = headers.copy()
    xet_headers.pop("authorization", None)

    download_files(
        [
            PyXetDownloadInfo(
                destination_path=str(incomplete.absolute()),
                hash=xet_file_data.file_hash,
                file_size=expected_size,
            )
        ],
        endpoint=connection_info.endpoint,
        token_info=(connection_info.access_token, connection_info.expiration_unix_epoch),
        token_refresher=token_refresher,
        progress_updater=[lambda _: None],
        request_headers=xet_headers,
    )


def endpoint_candidates() -> list[str | None]:
    primary = os.environ.get("HF_ENDPOINT") or None
    fallback_raw = os.environ.get("HF_DOWNLOAD_FALLBACK_ENDPOINTS", "https://huggingface.co")
    candidates: list[str | None] = []
    for endpoint in [primary, *[item.strip() or None for item in fallback_raw.split(",")]]:
        if endpoint not in candidates:
            candidates.append(endpoint)
    return candidates


def xet_download(repo_id: str, repo_type: str, local_dir: Path, filename: str, endpoint: str | None = None) -> None:
    target = local_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)

    url = hf_hub_url(repo_id=repo_id, repo_type=repo_type, filename=filename, endpoint=endpoint)
    headers = {"user-agent": "cv-final-pj2-calvin-downloader"}

    with httpx.Client(timeout=20, follow_redirects=False) as client:
        response = client.head(url, headers=headers)
    if response.status_code >= 400:
        response.raise_for_status()

    xet_file_data = parse_xet_file_data_from_response(response, endpoint=endpoint)
    if xet_file_data is None and "xet-reconstruction-info" in response.links and "xet-auth" in response.links:
        reconstruction_url = response.links["xet-reconstruction-info"]["url"]
        refresh_route = response.links["xet-auth"]["url"]
        file_hash = reconstruction_url.rstrip("/").split("/")[-1]
        xet_file_data = XetFileData(file_hash=file_hash, refresh_route=refresh_route)

    if xet_file_data is None:
        raise RuntimeError(f"No Xet metadata found for {filename}")

    expected_size_header = response.headers.get("x-linked-size") or response.headers.get("content-length")
    expected_size = int(expected_size_header) if expected_size_header else None
    incomplete = target.with_name(f"{target.name}.incomplete-xet")
    if incomplete.exists():
        incomplete.unlink()

    xet_download_to_file(
        incomplete=incomplete,
        xet_file_data=xet_file_data,
        headers=headers,
        expected_size=expected_size,
    )
    incomplete.replace(target)


def download_one(
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    filename: str,
    retries: int,
    completed_path: Path,
    failed_path: Path,
    status_csv: Path,
    lock: Lock,
) -> tuple[str, bool]:
    target = local_dir / filename
    if target.exists() and target.stat().st_size > 0:
        append_line(completed_path, filename, lock)
        append_csv(status_csv, {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": filename, "status": "skipped_existing", "attempt": 0, "error": ""}, lock)
        return filename, True

    last_error = ""
    prefer_xet = os.environ.get("PREFER_XET_FOR_VIDEOS", "0") == "1" and "/videos/" in filename
    endpoints = endpoint_candidates()
    for attempt in range(1, retries + 1):
        try:
            if prefer_xet:
                endpoint_errors = []
                for endpoint in endpoints:
                    try:
                        xet_download(repo_id, repo_type, local_dir, filename, endpoint=endpoint)
                        status = f"downloaded_xet:{endpoint or 'default'}"
                        break
                    except Exception as exc:  # noqa: BLE001 - try the next endpoint before failing.
                        endpoint_errors.append(f"{endpoint or 'default'}={exc!r}")
                else:
                    raise RuntimeError("; ".join(endpoint_errors))
            else:
                endpoint_errors = []
                downloaded = False
                status = ""
                for endpoint in endpoints:
                    try:
                        hf_hub_download(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            filename=filename,
                            local_dir=str(local_dir),
                            etag_timeout=5,
                            endpoint=endpoint,
                        )
                        status = f"downloaded:{endpoint or 'default'}"
                        downloaded = True
                        break
                    except Exception as exc:  # noqa: BLE001 - fall through to fallback endpoints.
                        endpoint_errors.append(f"{endpoint or 'default'}={exc!r}")

                if not downloaded:
                    for endpoint in endpoints:
                        try:
                            xet_download(repo_id, repo_type, local_dir, filename, endpoint=endpoint)
                            status = f"downloaded_xet_fallback:{endpoint or 'default'}"
                            downloaded = True
                            break
                        except Exception as exc:  # noqa: BLE001 - report all endpoint failures together.
                            endpoint_errors.append(f"xet:{endpoint or 'default'}={exc!r}")

                if not downloaded:
                    raise RuntimeError("; ".join(endpoint_errors))
            append_line(completed_path, filename, lock)
            append_csv(status_csv, {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": filename, "status": status, "attempt": attempt, "error": ""}, lock)
            return filename, True
        except Exception as exc:  # noqa: BLE001 - keep per-file failures resumable.
            last_error = repr(exc)
            append_csv(status_csv, {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": filename, "status": "failed_attempt", "attempt": attempt, "error": last_error}, lock)
            time.sleep(min(30, attempt * 5))

    append_line(failed_path, filename, lock)
    append_csv(status_csv, {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "path": filename, "status": "failed_final", "attempt": retries, "error": last_error}, lock)
    return filename, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    log_dir = Path(args.log_dir)
    completed_path = log_dir / "completed_files.txt"
    failed_path = log_dir / "failed_files.txt"
    status_csv = log_dir / "file_download_status.csv"
    log_dir.mkdir(parents=True, exist_ok=True)

    if str(local_dir).startswith("/home/"):
        raise SystemExit(f"Refusing to download under /home: {local_dir}")
    if str(local_dir).startswith("/data/huggingface"):
        raise SystemExit(f"Refusing to download under /data/huggingface: {local_dir}")

    files = load_manifest(Path(args.manifest))
    completed = load_completed(completed_path)
    pending = [path for path in files if path not in completed]
    if failed_path.exists():
        failed_path.unlink()

    print(f"manifest_files={len(files)}")
    print(f"already_completed={len(completed)}")
    print(f"pending={len(pending)}")
    print(f"workers={args.workers}")

    lock = Lock()
    ok_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                download_one,
                args.repo_id,
                args.repo_type,
                local_dir,
                filename,
                args.retries,
                completed_path,
                failed_path,
                status_csv,
                lock,
            )
            for filename in pending
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            filename, ok = future.result()
            if ok:
                ok_count += 1
            else:
                fail_count += 1
            if index % 100 == 0 or not ok:
                print(f"processed={index}/{len(pending)} ok={ok_count} failed={fail_count} last={filename}", flush=True)

    print(f"downloaded_or_existing={ok_count}")
    print(f"failed={fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
