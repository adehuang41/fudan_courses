#!/usr/bin/env python3
"""Plot training curves used by the Task2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "B visual": Path("/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline"),
    "ABC visual": Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline"),
    "B task-id": Path(
        "/data/dechao/cv_final_pj2/outputs/task_conditioned/"
        "B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1"
    ),
    "ABC task-id": Path(
        "/data/dechao/cv_final_pj2/outputs/task_conditioned/"
        "ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1"
    ),
}
COLORS = {
    "B visual": "#4E79A7",
    "ABC visual": "#F28E2B",
    "B task-id": "#59A14F",
    "ABC task-id": "#E15759",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window or window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def downsample(x: np.ndarray, y: np.ndarray, max_points: int = 1600) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx], y[idx]


def load_series(run_dir: Path) -> dict[str, np.ndarray]:
    rows = read_jsonl(run_dir / "logs" / "scalars.jsonl")
    train = [row for row in rows if row.get("phase") == "train" and "action_l1_loss" in row]
    val = [row for row in rows if row.get("phase") == "validation" and "val_action_l1_loss" in row]
    return {
        "train_step": np.asarray([row["step"] for row in train], dtype=np.float64),
        "train_action": np.asarray([row["action_l1_loss"] for row in train], dtype=np.float64),
        "val_step": np.asarray([row["step"] for row in val], dtype=np.float64),
        "val_action": np.asarray([row["val_action_l1_loss"] for row in val], dtype=np.float64),
    }


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_validation_curves(series: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=False)
    panels = [
        ("visual", ["B visual", "ABC visual"]),
        ("task-id", ["B task-id", "ABC task-id"]),
    ]
    for ax, (title, names) in zip(axes, panels, strict=True):
        for name in names:
            data = series[name]
            ax.plot(
                data["val_step"],
                data["val_action"],
                marker="o",
                markersize=3.0,
                linewidth=1.8,
                color=COLORS[name],
                label=name,
            )
            if len(data["val_action"]) > 0:
                best_idx = int(np.argmin(data["val_action"]))
                ax.scatter(
                    [data["val_step"][best_idx]],
                    [data["val_action"][best_idx]],
                    s=30,
                    color=COLORS[name],
                    edgecolor="white",
                    zorder=3,
                )
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Validation Action L1")
        ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "training_validation_action_l1_curves")


def plot_taskid_train_val(series: dict[str, dict[str, np.ndarray]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for ax, name in zip(axes, ["B task-id", "ABC task-id"], strict=True):
        data = series[name]
        train_step = data["train_step"]
        train_action = data["train_action"]
        window = max(1, min(500, len(train_action) // 120))
        smooth = moving_average(train_action, window)
        smooth_step = train_step[window - 1 :] if window > 1 else train_step
        smooth_step, smooth = downsample(smooth_step, smooth)
        ax.plot(smooth_step, smooth, linewidth=1.5, color=COLORS[name], alpha=0.9, label="train L1 (smoothed)")
        ax.plot(
            data["val_step"],
            data["val_action"],
            marker="o",
            markersize=3.4,
            linewidth=1.6,
            color="#111827",
            label="validation L1",
        )
        ax.set_title(name)
        ax.set_xlabel("Fine-tuning step")
        ax.set_ylabel("Action L1")
        ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "training_taskid_train_val_action_l1")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures/step4"))
    args = parser.parse_args()
    setup_style()
    series = {name: load_series(path) for name, path in RUNS.items()}
    plot_validation_curves(series, args.figure_dir)
    plot_taskid_train_val(series, args.figure_dir)
    print(args.figure_dir / "training_validation_action_l1_curves.png")
    print(args.figure_dir / "training_taskid_train_val_action_l1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
