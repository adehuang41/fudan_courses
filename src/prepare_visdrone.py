"""Prepare VisDrone-DET in YOLO format.

We delegate the heavy lifting (download + label conversion) to ultralytics's
built-in `VisDrone.yaml` script and just run a sanity check, since rewriting
the same logic adds nothing. Use `--check` after the download finishes to
confirm sample counts.

Usage:
    python -m src.prepare_visdrone           # trigger download + conversion
    python -m src.prepare_visdrone --check   # report sample counts only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.data.utils import check_det_dataset

DATA_ROOT = Path("/home/dechao/cv/hw2_task2/data/VisDrone")
EXPECTED = {"train": 6471, "val": 548, "test": 1610}


def count_split(split: str) -> int:
    p = DATA_ROOT / "images" / split
    if not p.exists():
        return 0
    return sum(1 for _ in p.glob("*.jpg"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Skip download, just report counts.")
    args = ap.parse_args()

    if not args.check:
        # Sets datasets_dir-relative VisDrone path under data/, runs download.
        check_det_dataset("VisDrone.yaml")

    print("=== VisDrone sample counts ===")
    ok = True
    for split, expected in EXPECTED.items():
        got = count_split(split)
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok = False
        print(f"  {split:5s} expected={expected:>5d}  found={got:>5d}  [{status}]")
    if not ok:
        raise SystemExit("Sample count mismatch — re-run download or check disk.")


if __name__ == "__main__":
    main()
