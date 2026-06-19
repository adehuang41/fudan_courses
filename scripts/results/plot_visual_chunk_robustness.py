#!/usr/bin/env python3
"""Merge and plot visual chunk robustness diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "B_only_act_baseline": "#4E79A7",
    "ABC_mixed_act_baseline": "#F28E2B",
}
LABELS = {
    "B_only_act_baseline": "B-only visual",
    "ABC_mixed_act_baseline": "ABC-mixed visual",
}
CONDITION_LABELS = {
    "clean": "Clean",
    "brightness_dark": "Dark",
    "brightness_bright": "Bright",
    "gaussian_blur": "Blur",
    "gaussian_noise": "Noise",
    "wrist_grayout": "Wrist gray",
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
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        raise FileNotFoundError(f"No input CSV exists: {paths}")
    return pd.concat(frames, ignore_index=True)


def save_fig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def merge_tables(table_dir: Path, input_prefixes: list[str], output_prefix: str) -> dict[str, Path]:
    suffixes = {
        "summary": "visual_robustness_summary.csv",
        "chunk_bin": "chunk_bin_action_l1.csv",
        "tail_head": "chunk_tail_head_ratio.csv",
        "per_action_dim": "visual_robustness_per_action_dim.csv",
        "per_task": "visual_robustness_per_task_action_l1.csv",
        "per_task_type": "visual_robustness_per_task_type_action_l1.csv",
        "chunk_position": "chunk_position_action_l1.csv",
    }
    out_paths = {}
    for key, suffix in suffixes.items():
        df = read_concat([table_dir / f"{prefix}_{suffix}" for prefix in input_prefixes])
        out = table_dir / f"{output_prefix}_{suffix}"
        df.to_csv(out, index=False)
        out_paths[key] = out
    return out_paths


def plot_chunk_bins(paths: dict[str, Path], figure_dir: Path, output_prefix: str) -> None:
    df = pd.read_csv(paths["chunk_bin"])
    clean = df[df["condition"] == "clean"].copy()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for experiment, group in clean.groupby("experiment"):
        group = group.sort_values("bin_index_1based")
        ax.plot(
            group["bin_index_1based"],
            group["normalized_action_l1_valid_mean"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=COLORS.get(experiment, "#333333"),
            label=LABELS.get(experiment, experiment),
        )
    ax.set_xticks(sorted(clean["bin_index_1based"].unique()))
    ax.set_xlabel("ACT chunk bin (10 future steps per bin)")
    ax.set_ylabel("Normalized Action L1")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / f"{output_prefix}_chunk_10bin_action_l1")


def plot_visual_shift(paths: dict[str, Path], figure_dir: Path, output_prefix: str) -> None:
    df = pd.read_csv(paths["summary"])
    clean = df[df["condition"] == "clean"][["experiment", "normalized_action_l1_valid_mean"]].rename(
        columns={"normalized_action_l1_valid_mean": "clean_l1"}
    )
    merged = df.merge(clean, on="experiment", how="left")
    merged["delta_l1"] = merged["normalized_action_l1_valid_mean"] - merged["clean_l1"]
    merged["relative_delta_pct"] = 100.0 * merged["delta_l1"] / merged["clean_l1"].clip(lower=1e-12)
    merged.to_csv(paths["summary"].with_name(f"{output_prefix}_visual_shift_degradation_summary.csv"), index=False)

    conditions = [c for c in CONDITION_LABELS if c != "clean"]
    experiments = ["B_only_act_baseline", "ABC_mixed_act_baseline"]
    x = np.arange(len(conditions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for idx, experiment in enumerate(experiments):
        values = [
            float(
                merged[(merged["experiment"] == experiment) & (merged["condition"] == condition)]["delta_l1"].iloc[0]
            )
            for condition in conditions
        ]
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[experiment],
            edgecolor="white",
            linewidth=0.8,
            label=LABELS[experiment],
        )
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions])
    ax.set_ylabel("L1 increase over clean")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / f"{output_prefix}_visual_shift_degradation")


def plot_task_type(paths: dict[str, Path], figure_dir: Path, output_prefix: str, top_n: int) -> None:
    df = pd.read_csv(paths["per_task_type"])
    clean = df[df["condition"] == "clean"].copy()
    counts = clean.groupby("task_type")["sample_frames"].max().sort_values(ascending=False)
    tasks = list(counts.head(top_n).index)
    clean = clean[clean["task_type"].isin(tasks)]
    task_order = list(reversed(tasks))
    y = np.arange(len(task_order))
    height = 0.34

    fig_h = max(4.2, 0.29 * len(task_order) + 1.4)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    for idx, experiment in enumerate(["B_only_act_baseline", "ABC_mixed_act_baseline"]):
        values = []
        for task in task_order:
            row = clean[(clean["experiment"] == experiment) & (clean["task_type"] == task)]
            values.append(float(row["normalized_action_l1_valid_mean"].iloc[0]) if not row.empty else np.nan)
        offset = (idx - 0.5) * height
        ax.barh(
            y + offset,
            values,
            height=height,
            color=COLORS[experiment],
            edgecolor="white",
            linewidth=0.6,
            label=LABELS[experiment],
        )
    labels = [f"{task} (n={int(counts[task])})" for task in task_order]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlabel("Normalized Action L1")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, borderaxespad=0.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.98))
    save_fig(fig, figure_dir / f"{output_prefix}_per_task_type_clean_l1_counts")


def plot_action_dim_heatmap(paths: dict[str, Path], figure_dir: Path, output_prefix: str) -> None:
    df = pd.read_csv(paths["per_action_dim"])
    clean = df[df["condition"] == "clean"][["experiment", "action_dim_name", "normalized_action_l1_valid_mean"]].rename(
        columns={"normalized_action_l1_valid_mean": "clean_l1"}
    )
    merged = df.merge(clean, on=["experiment", "action_dim_name"], how="left")
    merged["delta_l1"] = merged["normalized_action_l1_valid_mean"] - merged["clean_l1"]
    conditions = [c for c in CONDITION_LABELS if c != "clean"]
    dims = list(df.sort_values("action_dim")["action_dim_name"].drop_duplicates())
    experiments = ["B_only_act_baseline", "ABC_mixed_act_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True, constrained_layout=True)
    vmax = float(merged[merged["condition"].isin(conditions)]["delta_l1"].abs().max())
    vmax = max(vmax, 1e-6)
    for ax, experiment in zip(axes, experiments, strict=True):
        matrix = []
        for condition in conditions:
            row_vals = []
            for dim in dims:
                row = merged[
                    (merged["experiment"] == experiment)
                    & (merged["condition"] == condition)
                    & (merged["action_dim_name"] == dim)
                ]
                row_vals.append(float(row["delta_l1"].iloc[0]))
            matrix.append(row_vals)
        im = ax.imshow(np.asarray(matrix), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(LABELS[experiment])
        ax.set_xticks(np.arange(len(dims)))
        ax.set_xticklabels(dims, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels([CONDITION_LABELS[c] for c in conditions])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, fraction=0.035, pad=0.035)
    cbar.set_label("L1 increase over clean")
    save_fig(fig, figure_dir / f"{output_prefix}_action_dim_visual_shift_heatmap")


def plot_tail_head(paths: dict[str, Path], figure_dir: Path, output_prefix: str) -> None:
    df = pd.read_csv(paths["chunk_bin"])
    conditions = list(CONDITION_LABELS)
    experiments = ["B_only_act_baseline", "ABC_mixed_act_baseline"]
    x = np.arange(len(conditions))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for idx, experiment in enumerate(experiments):
        values = []
        for condition in conditions:
            rows = df[(df["experiment"] == experiment) & (df["condition"] == condition)].sort_values("bin_index_1based")
            if rows.empty:
                values.append(np.nan)
                continue
            head = rows.iloc[0]
            min_valid = float(head["valid_action_values"]) * 0.05
            valid_rows = rows[rows["valid_action_values"] >= min_valid]
            tail = valid_rows.iloc[-1]
            ratio = float(tail["normalized_action_l1_valid_mean"]) / max(
                float(head["normalized_action_l1_valid_mean"]), 1e-12
            )
            values.append(ratio)
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            color=COLORS[experiment],
            edgecolor="white",
            linewidth=0.7,
            label=LABELS[experiment],
        )
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], rotation=20, ha="right")
    ax.set_ylabel("Last valid bin / first bin L1")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / f"{output_prefix}_tail_head_ratio")


def plot_chunk_condition_heatmap(paths: dict[str, Path], figure_dir: Path, output_prefix: str) -> None:
    df = pd.read_csv(paths["chunk_bin"])
    conditions = list(CONDITION_LABELS)
    bins = sorted(int(v) for v in df["bin_index_1based"].drop_duplicates())
    experiments = ["B_only_act_baseline", "ABC_mixed_act_baseline"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True, constrained_layout=True)
    clean = df[df["condition"] == "clean"][
        ["experiment", "bin_index_1based", "normalized_action_l1_valid_mean"]
    ].rename(columns={"normalized_action_l1_valid_mean": "clean_l1"})
    merged = df.merge(clean, on=["experiment", "bin_index_1based"], how="left")
    merged["delta_l1"] = merged["normalized_action_l1_valid_mean"] - merged["clean_l1"]
    vmax = float(merged["delta_l1"].abs().max())
    vmax = max(vmax, 1e-6)
    for ax, experiment in zip(axes, experiments, strict=True):
        matrix = []
        for condition in conditions:
            vals = []
            for bin_idx in bins:
                row = merged[
                    (merged["experiment"] == experiment)
                    & (merged["condition"] == condition)
                    & (merged["bin_index_1based"] == bin_idx)
                ]
                vals.append(float(row["delta_l1"].iloc[0]) if not row.empty else np.nan)
            matrix.append(vals)
        im = ax.imshow(np.asarray(matrix), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(LABELS[experiment])
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels([str(b) for b in bins])
        ax.set_xlabel("Chunk bin")
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels([CONDITION_LABELS[c] for c in conditions])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, fraction=0.035, pad=0.035)
    cbar.set_label("L1 increase over clean")
    save_fig(fig, figure_dir / f"{output_prefix}_chunk_condition_heatmap")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables/step4"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures/step4"))
    parser.add_argument(
        "--input-prefix",
        action="append",
        default=[],
        help="Input prefix. Repeat twice, e.g. formal1024_B... and formal1024_ABC...",
    )
    parser.add_argument("--output-prefix", default="formal1024_visual_chunk_robustness")
    parser.add_argument("--top-task-types", type=int, default=18)
    args = parser.parse_args()

    setup_style()
    input_prefixes = args.input_prefix or [
        "formal1024_B_visual_chunk_robustness",
        "formal1024_ABC_visual_chunk_robustness",
    ]
    paths = merge_tables(args.table_dir, input_prefixes, args.output_prefix)
    plot_chunk_bins(paths, args.figure_dir, args.output_prefix)
    plot_visual_shift(paths, args.figure_dir, args.output_prefix)
    plot_task_type(paths, args.figure_dir, args.output_prefix, args.top_task_types)
    plot_action_dim_heatmap(paths, args.figure_dir, args.output_prefix)
    plot_tail_head(paths, args.figure_dir, args.output_prefix)
    plot_chunk_condition_heatmap(paths, args.figure_dir, args.output_prefix)
    print(paths["summary"])
    print(args.figure_dir / f"{args.output_prefix}_chunk_10bin_action_l1.png")
    print(args.figure_dir / f"{args.output_prefix}_visual_shift_degradation.png")
    print(args.figure_dir / f"{args.output_prefix}_per_task_type_clean_l1_counts.png")
    print(args.figure_dir / f"{args.output_prefix}_action_dim_visual_shift_heatmap.png")
    print(args.figure_dir / f"{args.output_prefix}_tail_head_ratio.png")
    print(args.figure_dir / f"{args.output_prefix}_chunk_condition_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
