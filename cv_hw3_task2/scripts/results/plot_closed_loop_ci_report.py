#!/usr/bin/env python3
"""Plot closed-loop bootstrap CI and task-family success figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "B_only_task_id_finetune": "#4E79A7",
    "ABC_mixed_task_id_finetune": "#F28E2B",
}
LABELS = {
    "B_only_task_id_finetune": "B-only task-id",
    "ABC_mixed_task_id_finetune": "ABC-mixed task-id",
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


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_avg_ci(ci: pd.DataFrame, figure_dir: Path, prefix: str) -> None:
    df = ci[ci["metric"] == "avg_successful_subtasks"].copy()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for policy in ["B_only_task_id_finetune", "ABC_mixed_task_id_finetune"]:
        group = df[df["policy"] == policy].sort_values("n_action_steps")
        y = group["value"].to_numpy()
        low = group["ci_low"].to_numpy()
        high = group["ci_high"].to_numpy()
        x = group["n_action_steps"].to_numpy()
        ax.plot(x, y, marker="o", linewidth=2.0, markersize=4.8, color=COLORS[policy], label=LABELS[policy])
        ax.fill_between(x, low, high, color=COLORS[policy], alpha=0.18, linewidth=0)
        for xx, yy in zip(x, y, strict=True):
            ax.text(xx, yy + 0.05, f"{yy:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$n_\mathrm{action\ steps}$")
    ax.set_ylabel("Avg. successful subtasks")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save(fig, figure_dir / f"{prefix}_avg_success_ci")


def plot_task_family(family: pd.DataFrame, figure_dir: Path, prefix: str) -> None:
    df = family[family["n_action_steps"] == 20].copy()
    counts = df.groupby("task_family")["attempted"].max().sort_values(ascending=False)
    counts = counts[counts >= 5]
    families = list(counts.index)
    y = np.arange(len(families))
    height = 0.34
    fig, ax = plt.subplots(figsize=(6.9, max(3.8, 0.33 * len(families) + 1.4)))
    for idx, policy in enumerate(["B_only_task_id_finetune", "ABC_mixed_task_id_finetune"]):
        vals, xerr_low, xerr_high = [], [], []
        for fam in families:
            row = df[(df["policy"] == policy) & (df["task_family"] == fam)]
            if row.empty:
                vals.append(np.nan)
                xerr_low.append(0.0)
                xerr_high.append(0.0)
                continue
            val = float(row["success_rate"].iloc[0])
            vals.append(val)
            xerr_low.append(max(0.0, val - float(row["wilson95_low"].iloc[0])))
            xerr_high.append(max(0.0, float(row["wilson95_high"].iloc[0]) - val))
        ax.barh(
            y + (idx - 0.5) * height,
            vals,
            height=height,
            xerr=np.vstack([xerr_low, xerr_high]),
            color=COLORS[policy],
            edgecolor="white",
            linewidth=0.7,
            capsize=2,
            label=LABELS[policy],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{fam} (n={int(counts[fam])})" for fam in families])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Success / attempted subtasks")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    save(fig, figure_dir / f"{prefix}_task_family_success_n20")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci",
        type=Path,
        default=Path("results/tables/step4/action_chunking_ablation_50seq_ep360_bootstrap_ci.csv"),
    )
    parser.add_argument(
        "--family",
        type=Path,
        default=Path("results/tables/step4/action_chunking_ablation_50seq_ep360_task_family_success_with_ci.csv"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures/step4"))
    parser.add_argument("--prefix", default="action_chunking_ablation_50seq_ep360")
    args = parser.parse_args()
    setup_style()
    plot_avg_ci(pd.read_csv(args.ci), args.figure_dir, args.prefix)
    plot_task_family(pd.read_csv(args.family), args.figure_dir, args.prefix)
    print(args.figure_dir / f"{args.prefix}_avg_success_ci.png")
    print(args.figure_dir / f"{args.prefix}_task_family_success_n20.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
