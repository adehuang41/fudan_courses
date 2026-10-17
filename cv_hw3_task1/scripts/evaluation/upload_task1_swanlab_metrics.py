#!/usr/bin/env python3
"""Replay selected Task 1 metrics to SwanLab."""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import swanlab


ROOT = Path(__file__).resolve().parents[2]

ITER_RE = re.compile(
    r"\[ITER (?P<iter>\d+)\] Loss=(?P<loss>[0-9.]+) "
    r"distort=(?P<distort>[0-9.]+) normal=(?P<normal>[0-9.]+) "
    r"Points=(?P<points>\d+)"
)
EVAL_RE = re.compile(
    r"\[ITER (?P<iter>\d+)\] Evaluating train: "
    r"L1 (?P<l1>[0-9.eE+-]+) PSNR (?P<psnr>[0-9.eE+-]+)"
)


RUNS_2DGS = {
    "object_A_book": ROOT / "outputs/reconstruction_2dgs/object_A_book/2dgs_final_r4_30k/train.log",
    "garden_D_background": ROOT
    / "outputs/reconstruction_2dgs/background_garden/2dgs_final_r4_30k_attempt002_20260610-0232/logs/train.log",
}

RUNS_AIGC = {
    "B_apple_final": ROOT
    / "outputs/aigc_assets/object_B_text_to_3d/final/magic3d-coarse-if-apple/"
    / "A1bappleStem_s7@20260614-A1bappleStem-farcam-s7/csv_logs/version_0/metrics.csv",
    "C_rubik_try16_final": ROOT
    / "outputs/aigc_assets/object_C_image_to_3d/rubiks_cube/final/coarse/magic123-coarse-sd/"
    / "rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/"
    / "csv_logs/version_0/metrics.csv",
}

REPORT_FIGURES = [
    ROOT / "report/assets/curves_2dgs_reconstruction.png",
    ROOT / "report/assets/curves_aigc_optimization.png",
    ROOT / "report/assets/training_curves.png",
]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except ValueError:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def replay_2dgs_logs() -> int:
    count = 0
    for run_name, log_path in RUNS_2DGS.items():
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        seen_iters: set[int] = set()
        with log_path.open(errors="ignore") as f:
            for line in f:
                m = ITER_RE.search(line)
                if m:
                    iteration = int(m.group("iter"))
                    if iteration in seen_iters:
                        continue
                    seen_iters.add(iteration)
                    swanlab.log(
                        {
                            f"2dgs/{run_name}/loss": float(m.group("loss")),
                            f"2dgs/{run_name}/normal": float(m.group("normal")),
                            f"2dgs/{run_name}/points": int(m.group("points")),
                        },
                        step=iteration,
                    )
                    count += 1
                    continue

                ev = EVAL_RE.search(line)
                if ev:
                    iteration = int(ev.group("iter"))
                    swanlab.log(
                        {
                            f"2dgs/{run_name}/eval_l1": float(ev.group("l1")),
                            f"2dgs/{run_name}/eval_psnr": float(ev.group("psnr")),
                        },
                        step=iteration,
                    )
                    count += 1
    return count


def replay_aigc_csv() -> int:
    keep = {
        "train/loss_sds",
        "train/loss_sparsity",
        "train/loss_opaque",
        "train/loss_rgb",
        "train/loss_mask",
        "train/loss_sd",
        "train/loss_sd_3d",
        "train/loss_z_variance",
    }
    count = 0
    for run_name, csv_path in RUNS_AIGC.items():
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step_f = parse_float(row.get("step"))
                if step_f is None:
                    continue
                step = int(step_f)
                payload = {}
                for key in keep:
                    value = parse_float(row.get(key))
                    if value is not None:
                        payload[f"aigc/{run_name}/{key.split('/')[-1]}"] = value
                if payload:
                    swanlab.log(payload, step=step)
                    count += 1
    return count


def upload_report_figures() -> int:
    count = 0
    for fig in REPORT_FIGURES:
        if not fig.exists():
            raise FileNotFoundError(fig)
        swanlab.log({f"report_figures/{fig.stem}": swanlab.Image(str(fig))})
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="cv_hw3_task1")
    parser.add_argument("--workspace", default="ade")
    parser.add_argument("--experiment-name", default="task1_final_metrics_replay")
    parser.add_argument("--mode", default="cloud", choices=["cloud", "local", "offline", "disabled"])
    args = parser.parse_args()

    swanlab.init(
        project=args.project,
        workspace=args.workspace,
        experiment_name=args.experiment_name,
        description="Task 1 training curves replayed from local logs and CSV scalars.",
        config={
            "source": "local logs and CSV scalars",
            "2dgs_runs": {k: str(v.relative_to(ROOT)) for k, v in RUNS_2DGS.items()},
            "aigc_runs": {k: str(v.relative_to(ROOT)) for k, v in RUNS_AIGC.items()},
        },
        mode=args.mode,
    )
    n_2dgs = replay_2dgs_logs()
    n_aigc = replay_aigc_csv()
    n_fig = upload_report_figures()
    swanlab.finish()
    print(f"uploaded: 2dgs={n_2dgs}, aigc={n_aigc}, figures={n_fig}")


if __name__ == "__main__":
    main()
