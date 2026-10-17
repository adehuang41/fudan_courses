#!/usr/bin/env python3
"""Upload historical training curves to SwanLab for the HW3 Task 2 report."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import swanlab


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGDIR = PROJECT_ROOT / "swanlog" / "cv_hw3_task2"
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "results" / "figures" / "swanlab"


@dataclass(frozen=True)
class RunSpec:
    key: str
    name: str
    group: str
    run_dir: Path
    color: str
    description: str


RUNS = [
    RunSpec(
        key="b_visual",
        name="B_only_visual_ACT",
        group="visual-act",
        run_dir=Path("/data/dechao/cv_final_pj2/outputs/baseline/B_only_act_baseline"),
        color="#4E79A7",
        description="B-only visual ACT baseline.",
    ),
    RunSpec(
        key="abc_visual",
        name="ABC_mixed_visual_ACT",
        group="visual-act",
        run_dir=Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline"),
        color="#F28E2B",
        description="ABC-mixed visual ACT baseline.",
    ),
    RunSpec(
        key="b_taskid",
        name="B_only_task_id_ACT",
        group="task-id-act",
        run_dir=Path(
            "/data/dechao/cv_final_pj2/outputs/task_conditioned/"
            "B_only_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1"
        ),
        color="#59A14F",
        description="B-only task-id ACT fine-tuned from the visual checkpoint.",
    ),
    RunSpec(
        key="abc_taskid",
        name="ABC_mixed_task_id_ACT",
        group="task-id-act",
        run_dir=Path(
            "/data/dechao/cv_final_pj2/outputs/task_conditioned/"
            "ABC_mixed_task_id_finetune_from_visual50k_extra50k_seed20260531_retry1"
        ),
        color="#E15759",
        description="ABC-mixed task-id ACT fine-tuned from the visual checkpoint.",
    ),
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def finite_metrics(row: dict, keys: Iterable[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def load_run_rows(run: RunSpec) -> tuple[list[dict], list[dict]]:
    scalars = read_jsonl(run.run_dir / "logs" / "scalars.jsonl")
    validation_path = run.run_dir / "logs" / "validation.jsonl"
    validation = read_jsonl(validation_path)
    if not validation:
        validation = [row for row in scalars if row.get("phase") == "validation"]
    train = [row for row in scalars if row.get("phase") == "train"]
    validation = [row for row in validation if row.get("phase") == "validation"]
    return train, validation


def downsample_train(rows: list[dict], stride: int) -> list[dict]:
    if stride <= 1:
        return rows
    kept: list[dict] = []
    for row in rows:
        step = int(row.get("step", 0))
        if step == 1 or step % stride == 0:
            kept.append(row)
    if rows and kept[-1].get("step") != rows[-1].get("step"):
        kept.append(rows[-1])
    return kept


def plot_curves(export_dir: Path, train_stride: int) -> tuple[Path, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "legend.frameon": False,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
        }
    )

    loss_png = export_dir / "swanlab_training_loss_curves.png"
    val_png = export_dir / "swanlab_validation_action_l1_curves.png"

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for run in RUNS:
        train, _ = load_run_rows(run)
        train = downsample_train(train, train_stride)
        ax.plot(
            [row["step"] for row in train],
            [row["loss"] for row in train],
            linewidth=1.25,
            color=run.color,
            label=run.name.replace("_", " "),
            alpha=0.9,
        )
    ax.set_title("Training Loss Curves Uploaded to SwanLab")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Total loss")
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(loss_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for run in RUNS:
        _, validation = load_run_rows(run)
        ax.plot(
            [row["step"] for row in validation],
            [row["val_action_l1_loss"] for row in validation],
            marker="o",
            markersize=3.0,
            linewidth=1.65,
            color=run.color,
            label=run.name.replace("_", " "),
        )
    ax.set_title("Validation Action L1 Curves Uploaded to SwanLab")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation Action L1")
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(val_png)
    plt.close(fig)
    return loss_png, val_png


def upload_run(
    run: RunSpec,
    project: str,
    workspace: str,
    logdir: Path,
    train_stride: int,
    image_paths: tuple[Path, Path],
    dry_run: bool,
) -> dict:
    train, validation = load_run_rows(run)
    sampled_train = downsample_train(train, train_stride)
    summary = {
        "run": run.name,
        "train_rows": len(train),
        "uploaded_train_rows": len(sampled_train),
        "validation_rows": len(validation),
        "last_train_step": int(train[-1]["step"]) if train else None,
        "last_validation_step": int(validation[-1]["step"]) if validation else None,
    }
    if dry_run:
        return summary

    swanlab.init(
        project=project,
        workspace=workspace,
        experiment_name=run.name,
        description=run.description,
        group=run.group,
        tags=["hw3-task2", "calvin", "act", run.group],
        id=f"cvhw3-task2-{run.key}",
        resume="allow",
        reinit=True,
        color=run.color,
        logdir=str(logdir),
        config={
            "source_run_dir": str(run.run_dir),
            "train_stride_for_upload": train_stride,
            "train_rows": len(train),
            "validation_rows": len(validation),
        },
    )

    for row in sampled_train:
        metrics = finite_metrics(
            row,
            [
                "loss",
                "action_l1_loss",
                "kld_loss",
                "grad_norm",
                "lr",
                "samples_per_second",
                "step_seconds",
            ],
        )
        if metrics:
            swanlab.log({f"train/{key}": value for key, value in metrics.items()}, step=int(row["step"]))

    for row in validation:
        metrics = finite_metrics(
            row,
            [
                "val_loss",
                "val_action_l1_loss",
                "val_runtime_seconds",
                "val_samples",
                "val_batches",
            ],
        )
        if metrics:
            swanlab.log({f"validation/{key}": value for key, value in metrics.items()}, step=int(row["step"]))

    image_step = summary["last_train_step"] or summary["last_validation_step"] or 0
    loss_png, val_png = image_paths
    swanlab.log(
        {
            "exported_charts/training_loss": swanlab.Image(str(loss_png), caption="Training loss curves"),
            "exported_charts/validation_action_l1": swanlab.Image(
                str(val_png), caption="Validation Action L1 curves"
            ),
        },
        step=int(image_step),
    )
    swanlab.finish()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="cv_hw3_task2")
    parser.add_argument("--workspace", default="ade")
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    image_paths = plot_curves(args.export_dir, args.train_stride)
    for run in RUNS:
        train, validation = load_run_rows(run)
        if not train and not validation:
            raise FileNotFoundError(f"No scalar or validation rows found for {run.name}: {run.run_dir}")

    summaries = [
        upload_run(
            run=run,
            project=args.project,
            workspace=args.workspace,
            logdir=args.logdir,
            train_stride=args.train_stride,
            image_paths=image_paths,
            dry_run=args.dry_run,
        )
        for run in RUNS
    ]
    print(json.dumps({"project": args.project, "workspace": args.workspace, "runs": summaries}, indent=2))
    print(f"SwanLab project URL: https://swanlab.cn/@{args.workspace}/{args.project}")
    print(f"Exported charts: {image_paths[0]} ; {image_paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
