#!/usr/bin/env python3
"""Create CPU-only insight figures for Track 2 report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
STEP4_TABLE_DIR = PROJECT_ROOT / "results/tables/step4"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "results/tables/track2_extra"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures/track2_extra"

MODEL_LABELS = {
    "B_only_act_baseline": "B-only visual",
    "ABC_mixed_act_baseline": "ABC-mixed visual",
    "B_only_task_id_finetune": "B-only task-id",
    "ABC_mixed_task_id_finetune": "ABC-mixed task-id",
}
MODEL_COLORS = {
    "B_only_act_baseline": "#4E79A7",
    "ABC_mixed_act_baseline": "#F28E2B",
    "B_only_task_id_finetune": "#59A14F",
    "ABC_mixed_task_id_finetune": "#E15759",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 450,
        }
    )


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def semantic_family(task: str) -> str:
    name = str(task)
    if ":" in name:
        name = name.split(":", 1)[0].strip()
    if name.startswith("lift_"):
        return "lift"
    if name.startswith("rotate_"):
        return "rotate"
    if name.startswith("push_"):
        return "push"
    if name.startswith("place_"):
        return "place"
    if name.startswith("turn_"):
        return "switch"
    if name.startswith("move_slider"):
        return "slider"
    if name in {"open_drawer", "close_drawer"}:
        return "drawer"
    if name in {"stack_block", "unstack_block"}:
        return "stack"
    return name.split("_", 1)[0]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    weights = weights.astype(float)
    values = values.astype(float)
    return float((values * weights).sum() / max(weights.sum(), 1e-12))


def make_metric_bridge(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    offline = pd.read_csv(STEP4_TABLE_DIR / "post_finetune_best_95k_d_evaluation_summary.csv")
    offline = offline[
        ["experiment", "normalized_action_l1_valid_mean", "raw_action_l1_valid_mean", "d_episodes", "d_frames"]
    ].copy()
    closed_loop = pd.DataFrame(
        [
            {"experiment": "B_only_act_baseline", "avg_successful_subtasks": 0.02, "SR@1": 0.02},
            {"experiment": "ABC_mixed_act_baseline", "avg_successful_subtasks": 0.02, "SR@1": 0.02},
            {"experiment": "B_only_task_id_finetune", "avg_successful_subtasks": 0.28, "SR@1": 0.20},
            {"experiment": "ABC_mixed_task_id_finetune", "avg_successful_subtasks": 0.68, "SR@1": 0.42},
        ]
    )
    out = offline.merge(closed_loop, on="experiment", how="inner")
    out["model_label"] = out["experiment"].map(MODEL_LABELS)
    out.to_csv(table_dir / "extra1024_metric_bridge_source_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for _, row in out.iterrows():
        exp = row["experiment"]
        ax.scatter(
            row["normalized_action_l1_valid_mean"],
            row["avg_successful_subtasks"],
            s=92,
            color=MODEL_COLORS[exp],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        dx = -0.015 if "ABC" in exp else 0.008
        dy = 0.045 if "task" in exp else 0.035
        ax.text(
            row["normalized_action_l1_valid_mean"] + dx,
            row["avg_successful_subtasks"] + dy,
            row["model_label"],
            fontsize=7.2,
            ha="right" if "ABC" in exp else "left",
            va="bottom",
        )
    ax.invert_xaxis()
    ax.set_xlabel("D offline normalized Action L1 (lower is better)")
    ax.set_ylabel("Closed-loop avg successful subtasks")
    ax.set_ylim(-0.04, 0.82)
    ax.set_xlim(out["normalized_action_l1_valid_mean"].max() + 0.03, out["normalized_action_l1_valid_mean"].min() - 0.03)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_metric_bridge_offline_to_closed_loop")
    return out


def make_episode_improvement(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    per_ep = pd.read_csv(STEP4_TABLE_DIR / "post_finetune_best_95k_d_per_episode_action_l1.csv")
    wide = per_ep.pivot_table(
        index="episode_index",
        columns="experiment",
        values="normalized_action_l1_valid_mean",
        aggfunc="mean",
    )
    comparisons = [
        ("ABC visual vs B visual", "B_only_act_baseline", "ABC_mixed_act_baseline"),
        ("B task-id vs B visual", "B_only_act_baseline", "B_only_task_id_finetune"),
        ("ABC task-id vs ABC visual", "ABC_mixed_act_baseline", "ABC_mixed_task_id_finetune"),
        ("ABC task-id vs B task-id", "B_only_task_id_finetune", "ABC_mixed_task_id_finetune"),
        ("ABC task-id vs B visual", "B_only_act_baseline", "ABC_mixed_task_id_finetune"),
    ]
    rows = []
    plot_data = []
    for label, base, improved in comparisons:
        sub = wide[[base, improved]].dropna()
        gain = sub[base] - sub[improved]
        rows.append(
            {
                "comparison": label,
                "paired_episodes": int(gain.shape[0]),
                "mean_l1_reduction": float(gain.mean()),
                "median_l1_reduction": float(gain.median()),
                "p10_l1_reduction": float(np.percentile(gain, 10)),
                "p90_l1_reduction": float(np.percentile(gain, 90)),
                "win_rate": float((gain > 0).mean()),
            }
        )
        for value in gain.to_numpy(dtype=float):
            plot_data.append({"comparison": label, "gain": value})
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "extra1024_episode_paired_improvement_summary.csv", index=False)
    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    order = [row[0] for row in comparisons]
    positions = np.arange(len(order))
    parts = ax.violinplot(
        [plot_df[plot_df["comparison"] == item]["gain"].to_numpy() for item in order],
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(["#B7CDE8", "#BEDDB4", "#F3C18B", "#F4A6A1", "#CDB9E8"][idx])
        body.set_edgecolor("white")
        body.set_alpha(0.95)
    parts["cmedians"].set_color("#333333")
    parts["cmedians"].set_linewidth(1.0)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=18, ha="right")
    ax.set_ylabel("Episode-level L1 reduction")
    ax.set_xlim(-0.6, len(order) - 0.4)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_episode_paired_improvement_violin")
    return out


def make_action_dim_delta(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    dims = pd.read_csv(STEP4_TABLE_DIR / "post_finetune_best_95k_d_per_action_dim_error.csv")
    wide = dims.pivot_table(
        index="action_dim_name",
        columns="experiment",
        values="normalized_action_l1_valid_mean",
        aggfunc="mean",
    )
    order = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    rows = []
    comparisons = [
        ("ABC visual gain", "B_only_act_baseline", "ABC_mixed_act_baseline"),
        ("B task-id gain", "B_only_act_baseline", "B_only_task_id_finetune"),
        ("ABC task-id gain", "ABC_mixed_act_baseline", "ABC_mixed_task_id_finetune"),
        ("Full gain", "B_only_act_baseline", "ABC_mixed_task_id_finetune"),
    ]
    for label, base, improved in comparisons:
        for dim in order:
            rows.append(
                {
                    "comparison": label,
                    "action_dim": dim,
                    "l1_reduction": float(wide.loc[dim, base] - wide.loc[dim, improved]),
                    "base_l1": float(wide.loc[dim, base]),
                    "improved_l1": float(wide.loc[dim, improved]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "extra1024_action_dim_gain_table.csv", index=False)

    matrix = out.pivot(index="comparison", columns="action_dim", values="l1_reduction").loc[
        [item[0] for item in comparisons], order
    ]
    vmax = max(float(np.abs(matrix.to_numpy()).max()), 1e-6)
    fig, ax = plt.subplots(figsize=(6.6, 2.95))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=6.7, color="#111111")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, fraction=0.045, pad=0.025)
    cbar.set_label("L1 reduction")
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_action_dim_gain_heatmap")
    return out


def make_visual_family_l1(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(table_dir / "extra1024_visual_full_robustness_visual_robustness_per_task_type_action_l1.csv")
    clean = df[df["condition"] == "clean"].copy()
    clean["task_family"] = clean["task_type"].map(semantic_family)
    rows = []
    for (experiment, family), group in clean.groupby(["experiment", "task_family"]):
        rows.append(
            {
                "experiment": experiment,
                "task_family": family,
                "episode_count": int(group["episode_count"].sum()),
                "valid_action_steps": int(group["valid_action_steps"].sum()),
                "normalized_action_l1_valid_mean": weighted_mean(
                    group["normalized_action_l1_valid_mean"], group["valid_action_steps"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    wide = out.pivot(index="task_family", columns="experiment", values="normalized_action_l1_valid_mean")
    counts = out.groupby("task_family")["episode_count"].max()
    families = sorted(wide.index)
    result = wide.reset_index()
    result["episode_count"] = result["task_family"].map(counts).astype(int)
    result["abc_minus_b_l1"] = result["ABC_mixed_act_baseline"] - result["B_only_act_baseline"]
    result.to_csv(table_dir / "extra1024_visual_task_family_clean_l1.csv", index=False)

    y = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(5.75, 3.55))
    for idx, exp in enumerate(["B_only_act_baseline", "ABC_mixed_act_baseline"]):
        vals = [wide.loc[fam, exp] for fam in families]
        ax.barh(
            y + (idx - 0.5) * 0.34,
            vals,
            height=0.32,
            color=MODEL_COLORS[exp],
            edgecolor="white",
            linewidth=0.6,
            label=MODEL_LABELS[exp],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([f"{fam} (n={int(counts.loc[fam])})" for fam in families])
    ax.invert_yaxis()
    ax.set_xlabel("D-1024 clean normalized Action L1")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_visual_task_family_clean_l1")
    return result


def make_chunk_ci_curve(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    ci = pd.read_csv(STEP4_TABLE_DIR / "action_chunking_ablation_50seq_ep360_bootstrap_ci.csv")
    avg = ci[ci["metric"] == "avg_successful_subtasks"].copy()
    avg.to_csv(table_dir / "extra1024_action_chunking_avg_success_ci.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.5, 3.1))
    for policy, color in [
        ("B_only_task_id_finetune", "#59A14F"),
        ("ABC_mixed_task_id_finetune", "#E15759"),
    ]:
        sub = avg[avg["policy"] == policy].sort_values("n_action_steps")
        x = sub["n_action_steps"].to_numpy(dtype=float)
        y = sub["value"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=color, linewidth=1.8, label=MODEL_LABELS[policy])
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100])
    ax.set_xticklabels(["10", "20", "50", "100"])
    ax.set_xlabel(r"$n_\mathrm{action\ steps}$")
    ax.set_ylabel("Avg successful subtasks")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_action_chunking_avg_success_ci")
    return avg


def make_sr_attrition(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(STEP4_TABLE_DIR / "action_chunking_ablation_50seq_ep360_summary.csv")
    keep = summary[summary["n_action_steps"].isin([20, 100])].copy()
    rows = []
    for _, row in keep.iterrows():
        for k in range(1, 6):
            rows.append(
                {
                    "policy": row["policy"],
                    "n_action_steps": int(row["n_action_steps"]),
                    "k": k,
                    "success_rate": float(row[f"SR@{k}"]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "extra1024_closed_loop_sr_attrition.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.55, 3.15))
    styles = {
        ("B_only_task_id_finetune", 20): ("#59A14F", "-"),
        ("B_only_task_id_finetune", 100): ("#59A14F", "--"),
        ("ABC_mixed_task_id_finetune", 20): ("#E15759", "-"),
        ("ABC_mixed_task_id_finetune", 100): ("#E15759", "--"),
    }
    for (policy, n_action), group in out.groupby(["policy", "n_action_steps"]):
        color, linestyle = styles[(policy, n_action)]
        label = f"{MODEL_LABELS[policy]}, n={n_action}"
        ax.plot(
            group["k"],
            group["success_rate"],
            marker="o",
            linewidth=1.7,
            linestyle=linestyle,
            color=color,
            label=label,
        )
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("k in SR@k")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.02, 0.72)
    ax.legend(loc="upper right", fontsize=7.1)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_closed_loop_sr_attrition")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    setup_style()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    make_metric_bridge(args.table_dir, args.figure_dir)
    make_episode_improvement(args.table_dir, args.figure_dir)
    make_action_dim_delta(args.table_dir, args.figure_dir)
    make_visual_family_l1(args.table_dir, args.figure_dir)
    make_chunk_ci_curve(args.table_dir, args.figure_dir)
    make_sr_attrition(args.table_dir, args.figure_dir)

    print(args.figure_dir / "extra1024_metric_bridge_offline_to_closed_loop.png")
    print(args.figure_dir / "extra1024_episode_paired_improvement_violin.png")
    print(args.figure_dir / "extra1024_action_dim_gain_heatmap.png")
    print(args.figure_dir / "extra1024_visual_task_family_clean_l1.png")
    print(args.figure_dir / "extra1024_action_chunking_avg_success_ci.png")
    print(args.figure_dir / "extra1024_closed_loop_sr_attrition.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
