#!/usr/bin/env python3
from __future__ import annotations

import argparse
import binascii
import json
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


TAIL_BYTES = 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ZipEntry:
    name: str
    compression_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one Mip-NeRF 360 scene from the official zip via HTTP Range "
            "without storing the full zip archive."
        )
    )
    parser.add_argument("--url", required=True, help="Remote zip URL.")
    parser.add_argument("--scene", required=True, help="Top-level scene directory to extract, e.g. counter.")
    parser.add_argument("--output_dir", required=True, help="Parent output directory. The scene directory is created inside it.")
    parser.add_argument("--summary_path", default=None, help="Optional JSON extraction summary path.")
    parser.add_argument("--dry_run", action="store_true", help="Only inspect zip metadata; do not extract files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing non-empty scene output directory.")
    parser.add_argument("--progress_every", type=int, default=25, help="Print progress every N extracted files.")
    parser.add_argument("--retries", type=int, default=3, help="HTTP retries per request.")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def request_bytes(url: str, *, start: int | None = None, end: int | None = None, method: str | None = None, retries: int = 3) -> tuple[bytes, dict[str, str]]:
    headers = {"User-Agent": "cv-final-mipnerf360-range-extractor"}
    expected_length = None
    if start is not None:
        if end is None or end < start:
            fail("Invalid HTTP range.")
        headers["Range"] = f"bytes={start}-{end}"
        expected_length = end - start + 1

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
                response_headers = {key: value for key, value in response.headers.items()}
            if expected_length is not None and len(data) != expected_length:
                raise RuntimeError(f"Range request returned {len(data)} bytes; expected {expected_length}.")
            return data, response_headers
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                time.sleep(1.5 * (attempt + 1))

    assert last_error is not None
    raise last_error


def content_length(url: str, retries: int) -> int:
    _, headers = request_bytes(url, method="HEAD", retries=retries)
    try:
        return int(headers["Content-Length"])
    except KeyError as exc:
        raise RuntimeError("HEAD response did not include Content-Length.") from exc


def parse_zip64_extra(extra: bytes, compressed_size: int, uncompressed_size: int, local_header_offset: int) -> tuple[int, int, int]:
    index = 0
    while index + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[index : index + 4])
        data = extra[index + 4 : index + 4 + data_size]
        if header_id == 0x0001:
            offset = 0
            if uncompressed_size == 0xFFFFFFFF and offset + 8 <= len(data):
                uncompressed_size = struct.unpack("<Q", data[offset : offset + 8])[0]
                offset += 8
            if compressed_size == 0xFFFFFFFF and offset + 8 <= len(data):
                compressed_size = struct.unpack("<Q", data[offset : offset + 8])[0]
                offset += 8
            if local_header_offset == 0xFFFFFFFF and offset + 8 <= len(data):
                local_header_offset = struct.unpack("<Q", data[offset : offset + 8])[0]
        index += 4 + data_size
    return compressed_size, uncompressed_size, local_header_offset


def central_directory_range(url: str, archive_size: int, retries: int) -> tuple[int, int]:
    tail_start = max(0, archive_size - TAIL_BYTES)
    tail, _ = request_bytes(url, start=tail_start, end=archive_size - 1, retries=retries)
    eocd_position = tail.rfind(b"PK\x05\x06")
    if eocd_position < 0:
        raise RuntimeError("Could not find ZIP end-of-central-directory record.")

    eocd = tail[eocd_position : eocd_position + 22]
    if len(eocd) != 22:
        raise RuntimeError("Incomplete ZIP end-of-central-directory record.")

    fields = struct.unpack("<4s4H2LH", eocd)
    central_size = fields[5]
    central_offset = fields[6]
    central_records = fields[4]

    needs_zip64 = (
        central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or central_records == 0xFFFF
    )
    if not needs_zip64:
        return central_offset, central_size

    locator_position = tail.rfind(b"PK\x06\x07", 0, eocd_position)
    if locator_position < 0:
        raise RuntimeError("ZIP64 locator not found.")
    locator = tail[locator_position : locator_position + 20]
    if len(locator) != 20:
        raise RuntimeError("Incomplete ZIP64 locator.")
    _, _, zip64_eocd_offset, _ = struct.unpack("<4sLQL", locator)

    record, _ = request_bytes(url, start=zip64_eocd_offset, end=zip64_eocd_offset + 80, retries=retries)
    if record[:4] != b"PK\x06\x06":
        raise RuntimeError("ZIP64 end-of-central-directory signature mismatch.")
    fields64 = struct.unpack("<4sQ2H2L4Q", record[:56])
    central_size = fields64[8]
    central_offset = fields64[9]
    return central_offset, central_size


def read_entries(url: str, retries: int) -> list[ZipEntry]:
    archive_size = content_length(url, retries)
    central_offset, central_size = central_directory_range(url, archive_size, retries)
    central_data, _ = request_bytes(
        url,
        start=central_offset,
        end=central_offset + central_size - 1,
        retries=retries,
    )

    entries: list[ZipEntry] = []
    index = 0
    while index + 46 <= len(central_data):
        if central_data[index : index + 4] != b"PK\x01\x02":
            break
        fields = struct.unpack("<4s6H3L5H2L", central_data[index : index + 46])
        compression_method = fields[4]
        crc32_value = fields[7]
        compressed_size = fields[8]
        uncompressed_size = fields[9]
        filename_length = fields[10]
        extra_length = fields[11]
        comment_length = fields[12]
        local_header_offset = fields[16]

        filename_start = index + 46
        filename_end = filename_start + filename_length
        extra_end = filename_end + extra_length
        name = central_data[filename_start:filename_end].decode("utf-8", "replace")
        extra = central_data[filename_end:extra_end]
        compressed_size, uncompressed_size, local_header_offset = parse_zip64_extra(
            extra,
            compressed_size,
            uncompressed_size,
            local_header_offset,
        )
        entries.append(
            ZipEntry(
                name=name,
                compression_method=compression_method,
                crc32=crc32_value,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
        index = extra_end + comment_length

    return entries


def local_data_offset(url: str, entry: ZipEntry, retries: int) -> int:
    local_header, _ = request_bytes(
        url,
        start=entry.local_header_offset,
        end=entry.local_header_offset + 29,
        retries=retries,
    )
    if local_header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Local header signature mismatch for {entry.name}.")
    fields = struct.unpack("<4s5H3L2H", local_header)
    flag_bits = fields[2]
    filename_length = fields[9]
    extra_length = fields[10]
    if flag_bits & 0x1:
        raise RuntimeError(f"Encrypted zip entry is not supported: {entry.name}")
    return entry.local_header_offset + 30 + filename_length + extra_length


def safe_output_path(output_dir: Path, entry_name: str) -> Path:
    if entry_name.startswith("/") or ".." in Path(entry_name).parts:
        raise RuntimeError(f"Unsafe zip entry path: {entry_name}")
    output_path = output_dir / entry_name
    output_root = output_dir.resolve()
    resolved = output_path.resolve()
    if output_root != resolved and output_root not in resolved.parents:
        raise RuntimeError(f"Zip entry escapes output directory: {entry_name}")
    return output_path


def extract_entry(url: str, entry: ZipEntry, output_dir: Path, retries: int) -> None:
    output_path = safe_output_path(output_dir, entry.name)
    if entry.name.endswith("/"):
        output_path.mkdir(parents=True, exist_ok=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_offset = local_data_offset(url, entry, retries)
    compressed_data, _ = request_bytes(
        url,
        start=data_offset,
        end=data_offset + entry.compressed_size - 1,
        retries=retries,
    )

    if entry.compression_method == 0:
        data = compressed_data
    elif entry.compression_method == 8:
        data = zlib.decompress(compressed_data, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported compression method {entry.compression_method} for {entry.name}")

    if len(data) != entry.uncompressed_size:
        raise RuntimeError(
            f"Uncompressed size mismatch for {entry.name}: got {len(data)}, expected {entry.uncompressed_size}"
        )
    crc32_value = binascii.crc32(data) & 0xFFFFFFFF
    if crc32_value != entry.crc32:
        raise RuntimeError(f"CRC mismatch for {entry.name}")

    output_path.write_bytes(data)


def summarize_entries(scene: str, entries: list[ZipEntry]) -> dict[str, object]:
    scene_entries = [entry for entry in entries if entry.name.startswith(f"{scene}/")]
    files = [entry for entry in scene_entries if not entry.name.endswith("/")]
    directories = [entry for entry in scene_entries if entry.name.endswith("/")]
    image_count = sum(1 for entry in files if Path(entry.name).suffix.lower() in IMAGE_SUFFIXES and "/images/" in entry.name)
    sparse_entry_count = sum(1 for entry in scene_entries if "/sparse/" in entry.name)
    method_counts: dict[str, int] = {}
    for entry in files:
        method_counts[str(entry.compression_method)] = method_counts.get(str(entry.compression_method), 0) + 1
    return {
        "scene": scene,
        "entry_count": len(scene_entries),
        "file_count": len(files),
        "directory_count": len(directories),
        "image_count": image_count,
        "sparse_entry_count": sparse_entry_count,
        "compressed_bytes": sum(entry.compressed_size for entry in files),
        "uncompressed_bytes": sum(entry.uncompressed_size for entry in files),
        "compression_method_counts": method_counts,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    scene_output_dir = output_dir / args.scene
    summary_path = Path(args.summary_path) if args.summary_path else scene_output_dir / "extraction_summary.json"

    if args.progress_every <= 0:
        fail("--progress_every must be positive.")

    entries = read_entries(args.url, args.retries)
    roots = sorted({entry.name.split("/")[0] for entry in entries if "/" in entry.name})
    scene_entries = [entry for entry in entries if entry.name.startswith(f"{args.scene}/")]
    if not scene_entries:
        fail(f"Scene {args.scene!r} not found. Available top-level directories: {', '.join(roots)}")

    summary = {
        "url": args.url,
        "available_top_level_directories": roots,
        "output_dir": str(output_dir),
        "output_scene_dir": str(scene_output_dir),
        **summarize_entries(args.scene, entries),
    }

    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    if scene_output_dir.exists() and any(scene_output_dir.iterdir()):
        if not args.overwrite:
            fail(f"{scene_output_dir} is not empty. Use --overwrite to replace it.")
        shutil.rmtree(scene_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [entry for entry in scene_entries if not entry.name.endswith("/")]
    for entry in scene_entries:
        if entry.name.endswith("/"):
            safe_output_path(output_dir, entry.name).mkdir(parents=True, exist_ok=True)

    for index, entry in enumerate(files, start=1):
        extract_entry(args.url, entry, output_dir, args.retries)
        if index == 1 or index == len(files) or index % args.progress_every == 0:
            print(f"[{index}/{len(files)}] extracted {entry.name}", flush=True)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote extraction summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
