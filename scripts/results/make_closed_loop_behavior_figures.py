#!/usr/bin/env python3
"""Create closed-loop behavior figures from existing CALVIN rollout logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
CLOSED_LOOP_ROOT = Path("/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_closed_loop")
CHUNK_ROOT = Path(
    "/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_action_chunking/"
    "formal_50seq_ep360_20260613T2050Z"
)

POLICY_SPECS = [
    (
        "B visual",
        CLOSED_LOOP_ROOT / "B_only_visual_baseline_50seq_ep360" / "B_only_visual_baseline_50seq_ep360.json",
        "#4E79A7",
    ),
    (
        "ABC visual",
        CLOSED_LOOP_ROOT / "ABC_mixed_visual_baseline_50seq_ep360" / "ABC_mixed_visual_baseline_50seq_ep360.json",
        "#F28E2B",
    ),
    (
        "B task-id n=100",
        CLOSED_LOOP_ROOT / "B_only_task_id_finetune_50seq_ep360" / "B_only_task_id_finetune_50seq_ep360.json",
        "#59A14F",
    ),
    (
        "ABC task-id n=100",
        CHUNK_ROOT
        / "ABC_mixed_task_id_finetune_50seq_ep360_n100"
        / "ABC_mixed_task_id_finetune_50seq_ep360_n100.json",
        "#9C755F",
    ),
    (
        "ABC task-id n=20",
        CHUNK_ROOT
        / "ABC_mixed_task_id_finetune_50seq_ep360_n20"
        / "ABC_mixed_task_id_finetune_50seq_ep360_n20.json",
        "#E15759",
    ),
]

CHUNK_COMPARE = {
    20: CHUNK_ROOT
    / "ABC_mixed_task_id_finetune_50seq_ep360_n20"
    / "ABC_mixed_task_id_finetune_50seq_ep360_n20.json",
    100: CHUNK_ROOT
    / "ABC_mixed_task_id_finetune_50seq_ep360_n100"
    / "ABC_mixed_task_id_finetune_50seq_ep360_n100.json",
}

FAMILY_ORDER = ["drawer", "slider", "switch", "push", "place", "lift", "rotate", "stack"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
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


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def report_cases(path: Path) -> list[dict]:
    report = load_report(path)
    cases = report.get("cases") or report.get("results") or []
    return sorted(cases, key=lambda row: int(row.get("sequence_index", 0)))


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


def collect_outcomes(table_dir: Path) -> pd.DataFrame:
    rows = []
    for label, path, _ in POLICY_SPECS:
        report = load_report(path)
        n_action = report.get("policy_n_action_steps", 100 if "visual" not in label else None)
        for case in report_cases(path):
            rows.append(
                {
                    "policy": label,
                    "n_action_steps": n_action,
                    "sequence_index": int(case["sequence_index"]),
                    "successful_subtasks": int(case.get("successful_subtasks", 0)),
                    "eval_sequence": "|".join(str(x) for x in case.get("eval_sequence", [])),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "extra1024_closed_loop_outcome_by_sequence.csv", index=False)
    return out


def plot_outcome_heatmap(outcomes: pd.DataFrame, figure_dir: Path) -> None:
    policies = [label for label, _, _ in POLICY_SPECS]
    seqs = sorted(outcomes["sequence_index"].unique())
    matrix = np.zeros((len(policies), len(seqs)), dtype=float)
    for i, policy in enumerate(policies):
        sub = outcomes[outcomes["policy"] == policy].set_index("sequence_index")
        for j, seq in enumerate(seqs):
            matrix[i, j] = float(sub.loc[seq, "successful_subtasks"])

    colors = ["#F2F4F7", "#C8D9ED", "#8DB3D8", "#4E8BBE", "#1F6FA6", "#073B66"]
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=5, cmap=cmap)
    ax.set_yticks(np.arange(len(policies)))
    ax.set_yticklabels(policies)
    ax.set_xticks(np.arange(0, len(seqs), 5))
    ax.set_xticklabels([f"{seq:02d}" for seq in seqs[::5]])
    ax.set_xlabel("Rollout sequence index")
    ax.set_title("Completed subtasks per fixed D-environment sequence")
    ax.set_xticks(np.arange(-0.5, len(seqs), 5), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            if val >= 3:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=5.2, color="white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.018, ticks=range(6))
    cbar.set_label("Completed / 5")
    fig.tight_layout(pad=0.25)
    save_fig(fig, figure_dir / "extra1024_closed_loop_outcome_heatmap")


def make_rescue(outcomes: pd.DataFrame, table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    wide = outcomes.pivot(index="sequence_index", columns="policy", values="successful_subtasks")
    rescue = pd.DataFrame(
        {
            "sequence_index": wide.index.astype(int),
            "abc_n20_completed": wide["ABC task-id n=20"].to_numpy(dtype=int),
            "abc_n100_completed": wide["ABC task-id n=100"].to_numpy(dtype=int),
        }
    )
    rescue["delta_n20_minus_n100"] = rescue["abc_n20_completed"] - rescue["abc_n100_completed"]
    rescue = rescue.sort_values(["delta_n20_minus_n100", "sequence_index"], ascending=[False, True]).reset_index(drop=True)
    rescue.to_csv(table_dir / "extra1024_action_chunking_rescue_by_sequence.csv", index=False)

    x = np.arange(len(rescue))
    delta = rescue["delta_n20_minus_n100"].to_numpy()
    colors = np.where(delta > 0, "#59A14F", np.where(delta < 0, "#C44E52", "#C9CED6"))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.bar(x, delta, color=colors, width=0.86, edgecolor="white", linewidth=0.35)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("n=20 minus n=100")
    ax.set_xlabel("Sequences sorted by rescue effect")
    ax.set_title("Per-sequence action-chunking rescue effect")
    ax.set_ylim(min(delta.min() - 0.5, -1.2), max(delta.max() + 0.5, 1.2))
    ax.grid(axis="y", alpha=0.22)
    n_pos = int((delta > 0).sum())
    n_zero = int((delta == 0).sum())
    n_neg = int((delta < 0).sum())
    ax.text(0.01, 0.96, f"rescued {n_pos} | unchanged {n_zero} | harmed {n_neg}", transform=ax.transAxes, va="top")
    for idx in np.where(np.abs(delta) >= 3)[0]:
        seq = int(rescue.iloc[idx]["sequence_index"])
        ax.text(idx, delta[idx] + (0.12 if delta[idx] >= 0 else -0.12), f"{seq:02d}", ha="center", va="bottom" if delta[idx] >= 0 else "top", fontsize=5.8)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_action_chunking_rescue_waterfall")
    return rescue


def make_survival(outcomes: pd.DataFrame, table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    rows = []
    for label, _, color in POLICY_SPECS:
        values = outcomes[outcomes["policy"] == label]["successful_subtasks"].to_numpy(dtype=int)
        for depth in range(0, 6):
            rows.append(
                {
                    "policy": label,
                    "depth": depth,
                    "surviving_sequences": int((values >= depth).sum()),
                    "survival_rate": float((values >= depth).mean()),
                }
            )
    surv = pd.DataFrame(rows)
    surv.to_csv(table_dir / "extra1024_closed_loop_survival_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x_offsets = {
        "B visual": -0.045,
        "ABC visual": 0.045,
        "B task-id n=100": -0.015,
        "ABC task-id n=100": 0.015,
        "ABC task-id n=20": 0.0,
    }
    line_styles = {
        "B visual": (0, (3, 2)),
        "ABC visual": (0, (1.2, 1.8)),
        "B task-id n=100": "-",
        "ABC task-id n=100": "-",
        "ABC task-id n=20": "-",
    }
    markers = {
        "B visual": "o",
        "ABC visual": "s",
        "B task-id n=100": "o",
        "ABC task-id n=100": "o",
        "ABC task-id n=20": "o",
    }
    for label, _, color in POLICY_SPECS:
        sub = surv[surv["policy"] == label]
        lw = 2.2 if label in {"ABC task-id n=20", "ABC task-id n=100"} else 1.4
        x = sub["depth"].to_numpy(dtype=float) + x_offsets.get(label, 0.0)
        ax.plot(
            x,
            sub["surviving_sequences"],
            marker=markers.get(label, "o"),
            linestyle=line_styles.get(label, "-"),
            linewidth=lw,
            markersize=4.2,
            color=color,
            label=label,
            zorder=3 if "visual" not in label else 2,
        )
    ax.set_xticks(range(0, 6))
    ax.set_xlabel("Completed prefix length k")
    ax.set_ylabel("Sequences with at least k successes")
    ax.set_ylim(-1, 52)
    ax.set_xlim(-0.15, 5.15)
    ax.set_title("Closed-loop task-chain survival")
    ax.grid(axis="y", alpha=0.22)
    ax.text(0.02, 0.88, "k=0: all policies start from 50 sequences", transform=ax.transAxes, fontsize=7, color="#4B5563")
    ax.legend(loc="upper right", ncol=1, fontsize=7)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_closed_loop_survival_curve")
    return surv


def collect_success_steps(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    step_specs = [
        ("B task-id n=100", POLICY_SPECS[2][1], POLICY_SPECS[2][2]),
        ("ABC task-id n=100", CHUNK_COMPARE[100], "#9C755F"),
        ("ABC task-id n=20", CHUNK_COMPARE[20], "#E15759"),
    ]
    rows = []
    for label, path, _ in step_specs:
        for case in report_cases(path):
            for subtask in case.get("subtasks", []):
                if not subtask.get("success", False):
                    continue
                rows.append(
                    {
                        "policy": label,
                        "sequence_index": int(case["sequence_index"]),
                        "subtask_index": int(subtask.get("subtask_index", 0)) + 1,
                        "task": str(subtask.get("canonical_task") or subtask.get("task")),
                        "task_family": semantic_family(str(subtask.get("canonical_task") or subtask.get("task"))),
                        "success_step": int(subtask["success_step"]),
                    }
                )
    steps = pd.DataFrame(rows)
    steps.to_csv(table_dir / "extra1024_closed_loop_success_steps.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    for label, _, color in step_specs:
        vals = np.sort(steps[steps["policy"] == label]["success_step"].to_numpy(dtype=float))
        if vals.size == 0:
            continue
        y = np.arange(1, vals.size + 1) / vals.size
        after_120 = int((vals > 120).sum())
        ax.step(vals, y, where="post", color=color, linewidth=1.9, label=f"{label} (n={vals.size}, >120: {after_120})")
        ax.scatter(vals, np.zeros_like(vals) - 0.025, color=color, s=5, alpha=0.45, clip_on=False)
    ax.axvline(120, color="#333333", linestyle="--", linewidth=0.9)
    ax.text(122, 0.08, "120-step\nold horizon", fontsize=7, va="bottom", color="#333333")
    ax.set_xlim(0, 365)
    ax.set_ylim(-0.04, 1.03)
    ax.set_xlabel("Success step within one subtask")
    ax.set_ylabel("Cumulative fraction of successful subtasks")
    ax.set_title("When successful subtasks finish")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right", fontsize=6.7)
    fig.tight_layout(pad=0.35)
    save_fig(fig, figure_dir / "extra1024_closed_loop_success_step_distribution")
    return steps


def make_family_position_matrix(table_dir: Path, figure_dir: Path) -> pd.DataFrame:
    rows = []
    for n_action, path in CHUNK_COMPARE.items():
        for case in report_cases(path):
            for subtask in case.get("subtasks", []):
                task = str(subtask.get("canonical_task") or subtask.get("task"))
                rows.append(
                    {
                        "n_action_steps": n_action,
                        "sequence_index": int(case["sequence_index"]),
                        "subtask_position": int(subtask.get("subtask_index", 0)) + 1,
                        "task": task,
                        "task_family": semantic_family(task),
                        "success": int(bool(subtask.get("success", False))),
                    }
                )
    attempts = pd.DataFrame(rows)
    grouped = (
        attempts.groupby(["n_action_steps", "task_family", "subtask_position"], as_index=False)
        .agg(success=("success", "sum"), attempted=("success", "size"))
    )
    grouped["success_rate"] = grouped["success"] / grouped["attempted"].clip(lower=1)
    grouped.to_csv(table_dir / "extra1024_action_chunking_task_family_position_success.csv", index=False)

    positions = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85), sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#ECEFF3")
    last_im = None
    for ax, n_action in zip(axes, [20, 100], strict=True):
        matrix = np.full((len(FAMILY_ORDER), len(positions)), np.nan)
        labels = [["" for _ in positions] for _ in FAMILY_ORDER]
        sub = grouped[grouped["n_action_steps"] == n_action]
        for i, family in enumerate(FAMILY_ORDER):
            for j, pos in enumerate(positions):
                row = sub[(sub["task_family"] == family) & (sub["subtask_position"] == pos)]
                if row.empty:
                    continue
                success = int(row["success"].iloc[0])
                attempted = int(row["attempted"].iloc[0])
                matrix[i, j] = success / attempted if attempted else np.nan
                labels[i][j] = f"{success}/{attempted}"
        last_im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_title(f"ABC task-id n={n_action}")
        ax.set_xticks(np.arange(len(positions)))
        ax.set_xticklabels([str(x) for x in positions])
        ax.set_xlabel("Subtask position in chain")
        ax.set_yticks(np.arange(len(FAMILY_ORDER)))
        ax.set_yticklabels(FAMILY_ORDER)
        for i in range(len(FAMILY_ORDER)):
            for j in range(len(positions)):
                text = labels[i][j] or "N/A"
                color = "white" if np.isfinite(matrix[i, j]) and matrix[i, j] >= 0.55 else "#202020"
                if text == "N/A":
                    color = "#6B7280"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.0, color=color)
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.84, fraction=0.035, pad=0.025)
    cbar.set_label("Success rate")
    save_fig(fig, figure_dir / "extra1024_action_chunking_task_family_position_matrix")
    return grouped


def write_summary(table_dir: Path, outcomes: pd.DataFrame, rescue: pd.DataFrame, steps: pd.DataFrame) -> None:
    rows = []
    for policy, group in outcomes.groupby("policy"):
        values = group["successful_subtasks"].to_numpy(dtype=int)
        rows.append(
            {
                "metric_group": "outcome",
                "policy": policy,
                "avg_completed": float(values.mean()),
                "num_sequences": int(values.size),
                "num_zero_completed": int((values == 0).sum()),
                "num_full_5": int((values == 5).sum()),
            }
        )
    delta = rescue["delta_n20_minus_n100"].to_numpy(dtype=int)
    rows.append(
        {
            "metric_group": "rescue",
            "policy": "ABC task-id n20_minus_n100",
            "avg_completed": float(delta.mean()),
            "num_sequences": int(delta.size),
            "num_zero_completed": int((delta == 0).sum()),
            "num_full_5": int((delta > 0).sum()),
        }
    )
    for policy, group in steps.groupby("policy"):
        vals = group["success_step"].to_numpy(dtype=int)
        rows.append(
            {
                "metric_group": "success_step",
                "policy": policy,
                "avg_completed": float(vals.mean()),
                "num_sequences": int(vals.size),
                "num_zero_completed": int((vals > 120).sum()),
                "num_full_5": int((vals <= 120).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(table_dir / "extra1024_closed_loop_behavior_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=PROJECT_ROOT / "results/tables/track2_extra")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "figures/track2_extra")
    args = parser.parse_args()

    setup_style()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    outcomes = collect_outcomes(args.table_dir)
    plot_outcome_heatmap(outcomes, args.figure_dir)
    rescue = make_rescue(outcomes, args.table_dir, args.figure_dir)
    make_survival(outcomes, args.table_dir, args.figure_dir)
    steps = collect_success_steps(args.table_dir, args.figure_dir)
    make_family_position_matrix(args.table_dir, args.figure_dir)
    write_summary(args.table_dir, outcomes, rescue, steps)

    for name in [
        "extra1024_closed_loop_outcome_heatmap.png",
        "extra1024_action_chunking_rescue_waterfall.png",
        "extra1024_closed_loop_survival_curve.png",
        "extra1024_closed_loop_success_step_distribution.png",
        "extra1024_action_chunking_task_family_position_matrix.png",
    ]:
        print(args.figure_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
