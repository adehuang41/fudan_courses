#!/usr/bin/env python3
"""Select the highest-index idle CUDA device from nvidia-smi output."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-index", type=int, default=6)
    parser.add_argument("--min-index", type=int, default=0)
    parser.add_argument("--max-used-mib", type=int, default=512)
    parser.add_argument("--max-util", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)

    candidates: list[tuple[int, int, int]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        index, used_mib, util = map(int, parts)
        if args.min_index <= index <= args.max_index:
            candidates.append((index, used_mib, util))

    for index, used_mib, util in sorted(candidates, reverse=True):
        if used_mib <= args.max_used_mib and util <= args.max_util:
            print(index)
            return 0

    print("No idle CUDA device found", file=sys.stderr)
    for index, used_mib, util in sorted(candidates, reverse=True):
        print(f"cuda:{index} used_mib={used_mib} util={util}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
