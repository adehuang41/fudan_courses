#!/usr/bin/env python3
"""Plot source-vs-D image-statistic distances."""

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
    "B_only_act_baseline": "B-only source",
    "ABC_mixed_act_baseline": "ABC-mixed source",
}
IMAGE_LABELS = {
    "observation.images.image": "Static camera",
    "observation.images.wrist_image": "Wrist camera",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=Path("results/tables/step4/visual_distribution_shift_stats.csv"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures/step4"))
    parser.add_argument("--prefix", default="visual_distribution_shift_stats")
    args = parser.parse_args()
    setup_style()
    df = pd.read_csv(args.table)

    metrics = [("mean_l2_distance", "Image mean L2"), ("std_l2_distance", "Image std L2")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.35), sharey=False)
    x = np.arange(len(IMAGE_LABELS))
    width = 0.34
    for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
        for idx, experiment in enumerate(["B_only_act_baseline", "ABC_mixed_act_baseline"]):
            values = []
            for key in IMAGE_LABELS:
                row = df[(df["experiment"] == experiment) & (df["image_key"] == key)]
                values.append(float(row[metric].iloc[0]))
            ax.bar(
                x + (idx - 0.5) * width,
                values,
                width=width,
                color=COLORS[experiment],
                edgecolor="white",
                linewidth=0.8,
                label=LABELS[experiment],
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(IMAGE_LABELS.values()))
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.5,
        handlelength=1.4,
    )
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.90))
    save(fig, args.figure_dir / args.prefix)
    print(args.figure_dir / f"{args.prefix}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
