#!/usr/bin/env python3
"""Summarize Track2 extra offline diagnostics and draw report figures."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
STEP4_TABLE_DIR = PROJECT_ROOT / "results/tables/step4"
CONDITIONS = [
    "clean",
    "brightness_dark",
    "brightness_bright",
    "gaussian_blur",
    "gaussian_noise",
    "static_grayout",
    "wrist_grayout",
    "both_camera_grayout",
]
COND_LABELS = {
    "clean": "Clean",
    "brightness_dark": "Dark",
    "brightness_bright": "Bright",
    "gaussian_blur": "Blur",
    "gaussian_noise": "Noise",
    "static_grayout": "Static gray",
    "wrist_grayout": "Wrist gray",
    "both_camera_grayout": "Both gray",
}
MODEL_ORDER = ["B_only_act_baseline", "ABC_mixed_act_baseline"]
MODEL_LABELS = {
    "B_only_act_baseline": "B-only visual",
    "ABC_mixed_act_baseline": "ABC-mixed visual",
}
MODEL_COLORS = {
    "B_only_act_baseline": "#4E79A7",
    "ABC_mixed_act_baseline": "#F28E2B",
}
TASK_ID_LABELS = {
    "B_only_task_id_finetune": "B-only task-id",
    "ABC_mixed_task_id_finetune": "ABC-mixed task-id",
}
TASK_ID_COLORS = {
    "B_only_task_id_finetune": "#4E79A7",
    "ABC_mixed_task_id_finetune": "#F28E2B",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
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


def read_concat(table_dir: Path, prefixes: list[str], suffix: str) -> pd.DataFrame:
    paths = [table_dir / f"{prefix}_{suffix}" for prefix in prefixes]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input tables: " + "; ".join(missing))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def bootstrap(values: np.ndarray, rng: np.random.Generator, iters: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, values.size, size=(iters, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_ratio(cond: np.ndarray, clean: np.ndarray, rng: np.random.Generator, iters: int = 10000) -> tuple[float, float, float]:
    cond = np.asarray(cond, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    valid = np.isfinite(cond) & np.isfinite(clean)
    cond = cond[valid]
    clean = clean[valid]
    if cond.size == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, cond.size, size=(iters, cond.size))
    vals = 100.0 * (cond[idx].mean(axis=1) - clean[idx].mean(axis=1)) / np.clip(clean[idx].mean(axis=1), 1e-12, None)
    point = 100.0 * (cond.mean() - clean.mean()) / max(clean.mean(), 1e-12)
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, iters: int = 10000) -> tuple[float, float, float]:
    return bootstrap(np.asarray(values, dtype=np.float64), rng, iters)


def paired_values(per_episode: pd.DataFrame, experiment: str, condition: str) -> pd.DataFrame:
    rows = per_episode[(per_episode["experiment"] == experiment) & (per_episode["condition"] == condition)]
    return rows[["episode_index", "normalized_action_l1_valid_mean"]].rename(
        columns={"normalized_action_l1_valid_mean": condition}
    )


def compute_visual_ci(per_episode: pd.DataFrame, rng: np.random.Generator, iters: int) -> pd.DataFrame:
    rows = []
    for experiment in MODEL_ORDER:
        clean = paired_values(per_episode, experiment, "clean")
        for condition in CONDITIONS[1:]:
            cond = paired_values(per_episode, experiment, condition)
            paired = clean.merge(cond, on="episode_index", how="inner")
            delta = paired[condition].to_numpy() - paired["clean"].to_numpy()
            mean, low, high = bootstrap(delta, rng, iters)
            pct, pct_low, pct_high = bootstrap_ratio(paired[condition].to_numpy(), paired["clean"].to_numpy(), rng, iters)
            rows.append(
                {
                    "analysis": "degradation_vs_clean",
                    "experiment": experiment,
                    "condition": condition,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "relative_pct": pct,
                    "relative_pct_ci_low": pct_low,
                    "relative_pct_ci_high": pct_high,
                    "n_episodes": int(paired["episode_index"].nunique()),
                }
            )
    for condition in CONDITIONS:
        b = paired_values(per_episode, "B_only_act_baseline", condition)
        abc = paired_values(per_episode, "ABC_mixed_act_baseline", condition)
        paired = b.merge(abc, on="episode_index", how="inner", suffixes=("_b", "_abc"))
        diff = paired[f"{condition}_abc"].to_numpy() - paired[f"{condition}_b"].to_numpy()
        mean, low, high = bootstrap(diff, rng, iters)
        rows.append(
            {
                "analysis": "abc_minus_b",
                "experiment": "ABC_mixed_minus_B_only",
                "condition": condition,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "relative_pct": np.nan,
                "relative_pct_ci_low": np.nan,
                "relative_pct_ci_high": np.nan,
                "n_episodes": int(paired["episode_index"].nunique()),
            }
        )
    for condition in CONDITIONS[1:]:
        b_clean = paired_values(per_episode, "B_only_act_baseline", "clean").rename(columns={"clean": "b_clean"})
        b_cond = paired_values(per_episode, "B_only_act_baseline", condition).rename(columns={condition: "b_cond"})
        abc_clean = paired_values(per_episode, "ABC_mixed_act_baseline", "clean").rename(
            columns={"clean": "abc_clean"}
        )
        abc_cond = paired_values(per_episode, "ABC_mixed_act_baseline", condition).rename(
            columns={condition: "abc_cond"}
        )
        paired = b_clean.merge(b_cond, on="episode_index", how="inner")
        paired = paired.merge(abc_clean, on="episode_index", how="inner")
        paired = paired.merge(abc_cond, on="episode_index", how="inner")
        b_delta = paired["b_cond"].to_numpy() - paired["b_clean"].to_numpy()
        abc_delta = paired["abc_cond"].to_numpy() - paired["abc_clean"].to_numpy()
        mean, low, high = bootstrap(abc_delta - b_delta, rng, iters)
        rows.append(
            {
                "analysis": "abc_minus_b_degradation",
                "experiment": "ABC_mixed_minus_B_only",
                "condition": condition,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "relative_pct": np.nan,
                "relative_pct_ci_low": np.nan,
                "relative_pct_ci_high": np.nan,
                "n_episodes": int(paired["episode_index"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def make_condition_distribution_summary(
    per_episode: pd.DataFrame,
    ci: pd.DataFrame,
    table_dir: Path,
    out_prefix: str,
    rng: np.random.Generator,
    iters: int,
) -> pd.DataFrame:
    rows = []
    diff_ci = ci[ci["analysis"] == "abc_minus_b"].set_index("condition")
    for condition in CONDITIONS:
        b = paired_values(per_episode, "B_only_act_baseline", condition)
        abc = paired_values(per_episode, "ABC_mixed_act_baseline", condition)
        paired = b.merge(abc, on="episode_index", how="inner", suffixes=("_b", "_abc"))
        b_vals = paired[f"{condition}_b"].to_numpy(dtype=np.float64)
        abc_vals = paired[f"{condition}_abc"].to_numpy(dtype=np.float64)
        wins = (abc_vals < b_vals).astype(np.float64)
        win_mean, win_low, win_high = bootstrap_mean_ci(wins, rng, iters)
        row = {
            "condition": condition,
            "condition_label": COND_LABELS[condition],
            "paired_episode_count": int(paired["episode_index"].nunique()),
            "b_mean": float(np.mean(b_vals)),
            "b_median": float(np.median(b_vals)),
            "b_p90": float(np.percentile(b_vals, 90)),
            "b_p95": float(np.percentile(b_vals, 95)),
            "abc_mean": float(np.mean(abc_vals)),
            "abc_median": float(np.median(abc_vals)),
            "abc_p90": float(np.percentile(abc_vals, 90)),
            "abc_p95": float(np.percentile(abc_vals, 95)),
            "abc_win_rate": win_mean,
            "abc_win_rate_ci_low": win_low,
            "abc_win_rate_ci_high": win_high,
        }
        if condition in diff_ci.index:
            row.update(
                {
                    "abc_minus_b_mean": float(diff_ci.loc[condition, "mean"]),
                    "abc_minus_b_ci_low": float(diff_ci.loc[condition, "ci_low"]),
                    "abc_minus_b_ci_high": float(diff_ci.loc[condition, "ci_high"]),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / f"{out_prefix}_condition_distribution_summary.csv", index=False)
    return out


def make_visual_tables(summary: pd.DataFrame, ci: pd.DataFrame, table_dir: Path, out_prefix: str) -> pd.DataFrame:
    clean = summary[summary["condition"] == "clean"][
        ["experiment", "normalized_action_l1_valid_mean", "raw_action_l1_valid_mean"]
    ].rename(
        columns={
            "normalized_action_l1_valid_mean": "clean_normalized_action_l1",
            "raw_action_l1_valid_mean": "clean_raw_action_l1",
        }
    )
    out = summary.merge(clean, on="experiment", how="left")
    out["absolute_delta_vs_clean"] = out["normalized_action_l1_valid_mean"] - out["clean_normalized_action_l1"]
    out["relative_degradation_pct"] = (
        100.0 * out["absolute_delta_vs_clean"] / out["clean_normalized_action_l1"].clip(lower=1e-12)
    )
    deg_ci = ci[ci["analysis"] == "degradation_vs_clean"][
        [
            "experiment",
            "condition",
            "ci_low",
            "ci_high",
            "relative_pct_ci_low",
            "relative_pct_ci_high",
            "n_episodes",
        ]
    ].rename(
        columns={
            "ci_low": "absolute_delta_ci_low",
            "ci_high": "absolute_delta_ci_high",
            "relative_pct_ci_low": "relative_degradation_pct_ci_low",
            "relative_pct_ci_high": "relative_degradation_pct_ci_high",
        }
    )
    out = out.merge(deg_ci, on=["experiment", "condition"], how="left")
    out.loc[out["condition"] == "clean", ["absolute_delta_ci_low", "absolute_delta_ci_high"]] = 0.0
    out.loc[
        out["condition"] == "clean",
        ["relative_degradation_pct_ci_low", "relative_degradation_pct_ci_high"],
    ] = 0.0
    out.to_csv(table_dir / f"{out_prefix}_perturbation_full_table.csv", index=False)
    return out


def make_chunk_prefix_drift_summary(
    chunk_position: pd.DataFrame,
    chunk_bin: pd.DataFrame,
    tail_head: pd.DataFrame,
    table_dir: Path,
    out_prefix: str,
) -> pd.DataFrame:
    prefix_rows = []
    for (experiment, condition), group in chunk_position.groupby(["experiment", "condition"]):
        group = group.sort_values("chunk_position_1based")
        x = group["chunk_position_1based"].to_numpy(dtype=np.float64)
        y = group["normalized_action_l1_valid_mean"].to_numpy(dtype=np.float64)
        w = group["valid_action_values"].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        if valid.sum() >= 2:
            slope = float(np.polyfit(x[valid], y[valid], deg=1, w=np.sqrt(w[valid]))[0])
        else:
            slope = np.nan
        th = (
            tail_head[(tail_head["experiment"] == experiment) & (tail_head["condition"] == condition)]
            if "experiment" in tail_head.columns
            else pd.DataFrame()
        )
        if not th.empty:
            tail_head_ratio = float(th["tail_head_ratio"].iloc[0])
            head_l1 = float(th["head_normalized_action_l1"].iloc[0])
            tail_l1 = float(th["tail_normalized_action_l1"].iloc[0])
            tail_position_label = str(th["tail_positions"].iloc[0])
        else:
            head_sub = group.head(10)
            tail_sub = group.tail(10)
            head_denom = float(head_sub["valid_action_values"].sum())
            tail_denom = float(tail_sub["valid_action_values"].sum())
            head_l1 = float(
                (head_sub["normalized_action_l1_valid_mean"] * head_sub["valid_action_values"]).sum()
                / max(head_denom, 1e-12)
            )
            tail_l1 = float(
                (tail_sub["normalized_action_l1_valid_mean"] * tail_sub["valid_action_values"]).sum()
                / max(tail_denom, 1e-12)
            )
            tail_head_ratio = tail_l1 / max(head_l1, 1e-12)
            tail_position_label = f"{int(tail_sub['chunk_position_1based'].min())}-{int(tail_sub['chunk_position_1based'].max())}"
        max_valid_position = int(group["chunk_position_1based"].max())
        for prefix_n in (10, 20, 50, 100):
            sub = group[group["chunk_position_1based"] <= prefix_n]
            denom = sub["valid_action_values"].sum()
            if denom <= 0:
                continue
            norm = float(
                (sub["normalized_action_l1_valid_mean"] * sub["valid_action_values"]).sum() / denom
            )
            raw = float((sub["raw_action_l1_valid_mean"] * sub["valid_action_values"]).sum() / denom)
            prefix_rows.append(
                {
                    "experiment": experiment,
                    "condition": condition,
                    "condition_label": COND_LABELS.get(condition, condition),
                    "prefix_n": prefix_n,
                    "max_valid_position": max_valid_position,
                    "valid_action_values": int(denom),
                    "prefix_normalized_action_l1": norm,
                    "prefix_raw_action_l1": raw,
                    "drift_slope_per_chunk_step": slope,
                    "tail_position_label": tail_position_label,
                    "tail_head_ratio_last10_vs_1_10": tail_head_ratio,
                    "head_positions_1_10_l1": head_l1,
                    "tail_positions_last10_l1": tail_l1,
                }
            )
    out = pd.DataFrame(prefix_rows)
    out.to_csv(table_dir / f"{out_prefix}_chunk_prefix_drift_summary.csv", index=False)

    counts = chunk_bin[
        ["experiment", "condition", "bin_index_1based", "position_start_1based", "position_end_1based", "valid_action_values"]
    ].copy()
    counts.to_csv(table_dir / f"{out_prefix}_chunk_bin_effective_counts.csv", index=False)
    return out


def plot_degradation_ci(ci: pd.DataFrame, figure_dir: Path, out_prefix: str) -> None:
    deg = ci[ci["analysis"] == "degradation_vs_clean"].copy()
    conditions = CONDITIONS[1:]
    x = np.arange(len(conditions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    for idx, experiment in enumerate(MODEL_ORDER):
        sub = deg[deg["experiment"] == experiment].set_index("condition").loc[conditions]
        values = sub["mean"].to_numpy()
        lows = values - sub["ci_low"].to_numpy()
        highs = sub["ci_high"].to_numpy() - values
        ax.bar(
            x + (idx - 0.5) * width,
            values,
            width=width,
            yerr=np.vstack([lows, highs]),
            capsize=2.2,
            color=MODEL_COLORS[experiment],
            edgecolor="white",
            linewidth=0.6,
            label=MODEL_LABELS[experiment],
        )
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in conditions], rotation=22, ha="right")
    ax.set_ylabel("Normalized L1 increase")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=1.6,
    )
    fig.tight_layout(pad=0.4, rect=(0.0, 0.0, 1.0, 0.92))
    save_fig(fig, figure_dir / f"{out_prefix}_visual_degradation_ci")


def plot_abc_minus_b_ci(ci: pd.DataFrame, figure_dir: Path, out_prefix: str) -> None:
    diff = ci[ci["analysis"] == "abc_minus_b"].set_index("condition").loc[CONDITIONS].copy()
    values = diff["mean"].to_numpy()
    lows = values - diff["ci_low"].to_numpy()
    highs = diff["ci_high"].to_numpy() - values
    fig, ax = plt.subplots(figsize=(6.8, 3.1))
    ax.errorbar(
        np.arange(len(CONDITIONS)),
        values,
        yerr=np.vstack([lows, highs]),
        fmt="o-",
        color="#3A3A3A",
        markerfacecolor="#F28E2B",
        markeredgecolor="white",
        linewidth=1.6,
        capsize=2.4,
    )
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels([COND_LABELS[c] for c in CONDITIONS], rotation=24, ha="right")
    ax.set_ylabel("ABC-mixed minus B-only L1")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / f"{out_prefix}_abc_minus_b_ci")


def plot_action_dim_heatmap(per_dim: pd.DataFrame, figure_dir: Path, out_prefix: str) -> None:
    clean = per_dim[per_dim["condition"] == "clean"][
        ["experiment", "action_dim_name", "normalized_action_l1_valid_mean"]
    ].rename(columns={"normalized_action_l1_valid_mean": "clean_l1"})
    merged = per_dim.merge(clean, on=["experiment", "action_dim_name"], how="left")
    merged["delta_l1"] = merged["normalized_action_l1_valid_mean"] - merged["clean_l1"]
    dims = list(per_dim.sort_values("action_dim")["action_dim_name"].drop_duplicates())
    conditions = CONDITIONS[1:]
    vmax = max(float(merged[merged["condition"].isin(conditions)]["delta_l1"].abs().max()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), sharey=True, constrained_layout=True)
    for ax, experiment in zip(axes, MODEL_ORDER, strict=True):
        matrix = []
        for condition in conditions:
            row = []
            for dim in dims:
                val = merged[
                    (merged["experiment"] == experiment)
                    & (merged["condition"] == condition)
                    & (merged["action_dim_name"] == dim)
                ]["delta_l1"].iloc[0]
                row.append(float(val))
            matrix.append(row)
        im = ax.imshow(np.asarray(matrix), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(MODEL_LABELS[experiment])
        ax.set_xticks(np.arange(len(dims)))
        ax.set_xticklabels(dims, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_yticklabels([COND_LABELS[c] for c in conditions])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, fraction=0.036, pad=0.035)
    cbar.set_label("L1 increase")
    save_fig(fig, figure_dir / f"{out_prefix}_action_dim_heatmap")


def plot_chunk_bins(chunk_bin: pd.DataFrame, figure_dir: Path, out_prefix: str) -> None:
    clean = chunk_bin[chunk_bin["condition"] == "clean"].copy()
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    for experiment in MODEL_ORDER:
        sub = clean[clean["experiment"] == experiment].sort_values("bin_index_1based")
        ax.plot(
            sub["bin_index_1based"],
            sub["normalized_action_l1_valid_mean"],
            marker="o",
            linewidth=1.8,
            markersize=4.0,
            color=MODEL_COLORS[experiment],
            label=MODEL_LABELS[experiment],
        )
    ax.set_xticks(sorted(clean["bin_index_1based"].unique()))
    ax.set_xlabel("ACT chunk bin")
    ax.set_ylabel("Normalized Action L1")
    ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / f"{out_prefix}_chunk_10bin_action_l1")


def plot_chunk_condition_heatmap(chunk_bin: pd.DataFrame, figure_dir: Path, out_prefix: str) -> None:
    clean = chunk_bin[chunk_bin["condition"] == "clean"][
        ["experiment", "bin_index_1based", "normalized_action_l1_valid_mean"]
    ].rename(columns={"normalized_action_l1_valid_mean": "clean_l1"})
    merged = chunk_bin.merge(clean, on=["experiment", "bin_index_1based"], how="left")
    merged["delta_l1"] = merged["normalized_action_l1_valid_mean"] - merged["clean_l1"]
    bins = sorted(int(v) for v in merged["bin_index_1based"].drop_duplicates())
    vmax = max(float(merged["delta_l1"].abs().max()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), sharey=True, constrained_layout=True)
    for ax, experiment in zip(axes, MODEL_ORDER, strict=True):
        matrix = []
        for condition in CONDITIONS:
            row = []
            for bin_idx in bins:
                vals = merged[
                    (merged["experiment"] == experiment)
                    & (merged["condition"] == condition)
                    & (merged["bin_index_1based"] == bin_idx)
                ]["delta_l1"]
                row.append(float(vals.iloc[0]) if not vals.empty else np.nan)
            matrix.append(row)
        im = ax.imshow(np.asarray(matrix), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(MODEL_LABELS[experiment])
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels([str(b) for b in bins])
        ax.set_xlabel("Chunk bin")
        ax.set_yticks(np.arange(len(CONDITIONS)))
        ax.set_yticklabels([COND_LABELS[c] for c in CONDITIONS])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, fraction=0.036, pad=0.035)
    cbar.set_label("L1 increase")
    save_fig(fig, figure_dir / f"{out_prefix}_chunk_condition_heatmap")


def plot_prefix_l1(prefix: pd.DataFrame, figure_dir: Path, out_prefix: str) -> None:
    clean = prefix[prefix["condition"] == "clean"].copy()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for experiment in MODEL_ORDER:
        sub = clean[clean["experiment"] == experiment].sort_values("prefix_n")
        ax.plot(
            sub["prefix_n"],
            sub["prefix_normalized_action_l1"],
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=MODEL_COLORS[experiment],
            label=MODEL_LABELS[experiment],
        )
    ax.set_xlabel("Prefix length n")
    ax.set_ylabel("Prefix Action L1")
    ax.set_xticks([10, 20, 50, 100])
    ax.legend(loc="best")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / f"{out_prefix}_chunk_prefix_l1")


def taskid_prefixes_default(table_dir: Path) -> list[str]:
    family_prefix = "extra1024_taskid_family_ablation"
    if (table_dir / f"{family_prefix}_summary.csv").exists():
        return [family_prefix]
    family_split = [
        "extra1024_B_taskid_family_ablation",
        "extra1024_ABC_taskid_family_ablation",
    ]
    if any((table_dir / f"{prefix}_summary.csv").exists() for prefix in family_split):
        return [prefix for prefix in family_split if (table_dir / f"{prefix}_summary.csv").exists()]
    return ["extra1024_B_taskid_ablation", "extra1024_ABC_taskid_ablation"]


def plot_taskid_ablation(table_dir: Path, figure_dir: Path, taskid_prefixes: list[str]) -> None:
    frames = []
    for prefix in taskid_prefixes:
        path = table_dir / f"{prefix}_summary.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["result_prefix"] = prefix
            frames.append(frame)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(table_dir / "extra1024_taskid_ablation_summary_merged.csv", index=False)
    clean = df[df["task_id_variant"] == "correct"][
        ["experiment", "normalized_action_l1_valid_mean", "raw_action_l1_valid_mean"]
    ].rename(
        columns={
            "normalized_action_l1_valid_mean": "correct_normalized_action_l1",
            "raw_action_l1_valid_mean": "correct_raw_action_l1",
        }
    )
    delta = df.merge(clean, on="experiment", how="left")
    delta["absolute_delta_vs_correct"] = (
        delta["normalized_action_l1_valid_mean"] - delta["correct_normalized_action_l1"]
    )
    delta["relative_delta_vs_correct_pct"] = (
        100.0 * delta["absolute_delta_vs_correct"] / delta["correct_normalized_action_l1"].clip(lower=1e-12)
    )
    delta.to_csv(table_dir / "extra1024_taskid_ablation_delta_table.csv", index=False)
    order = [
        item
        for item in ["correct", "zero", "random", "same_family_wrong", "different_family_wrong", "wrong"]
        if item in set(df["task_id_variant"])
    ]
    experiments = [exp for exp in ["B_only_task_id_finetune", "ABC_mixed_task_id_finetune"] if exp in set(df["experiment"])]
    x = np.arange(len(order))
    width = 0.34 if len(experiments) > 1 else 0.48
    fig, ax = plt.subplots(figsize=(7.25, 3.2))
    for idx, experiment in enumerate(experiments):
        sub = df[df["experiment"] == experiment].set_index("task_id_variant").loc[order]
        values = sub["normalized_action_l1_valid_mean"].to_numpy()
        offset = (idx - (len(experiments) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=TASK_ID_COLORS.get(experiment, "#777777"),
            edgecolor="white",
            linewidth=0.7,
            label=TASK_ID_LABELS.get(experiment, experiment),
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(
        [
            {
                "correct": "Correct",
                "zero": "Zero",
                "random": "Random",
                "same_family_wrong": "Same-family\nwrong",
                "different_family_wrong": "Different-family\nwrong",
                "wrong": "Wrong",
            }.get(item, item)
            for item in order
        ],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel("Normalized Action L1")
    ax.set_ylim(0, df["normalized_action_l1_valid_mean"].max() * 1.18)
    if len(experiments) > 1:
        ax.legend(loc="upper left", ncol=2)
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / "extra1024_taskid_ablation_summary")


def plot_rollout_family_heatmap(table_dir: Path, figure_dir: Path) -> None:
    src = STEP4_TABLE_DIR / "action_chunking_ablation_50seq_ep360_task_family_success_with_ci.csv"
    if not src.exists():
        return
    df = pd.read_csv(src)
    df.to_csv(table_dir / "extra1024_action_chunking_task_family_success_heatmap_table.csv", index=False)
    policies = [p for p in ["B_only_task_id_finetune", "ABC_mixed_task_id_finetune"] if p in set(df["policy"])]
    families = sorted(df["task_family"].dropna().unique())
    n_values = [10, 20, 50, 100]
    fig, axes = plt.subplots(1, len(policies), figsize=(7.2, 3.35), sharey=True, constrained_layout=True)
    if len(policies) == 1:
        axes = [axes]
    for ax, policy in zip(axes, policies, strict=True):
        matrix = np.full((len(n_values), len(families)), np.nan)
        labels = [["" for _ in families] for _ in n_values]
        sub = df[df["policy"] == policy]
        for i, n_action in enumerate(n_values):
            for j, family in enumerate(families):
                row = sub[(sub["n_action_steps"] == n_action) & (sub["task_family"] == family)]
                if row.empty:
                    continue
                matrix[i, j] = float(row["success_rate"].iloc[0])
                labels[i][j] = f"{int(row['success'].iloc[0])}/{int(row['attempted'].iloc[0])}"
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad("#ECEFF3")
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_title(TASK_ID_LABELS.get(policy, policy))
        ax.set_xticks(np.arange(len(families)))
        ax.set_xticklabels(families, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(n_values)))
        ax.set_yticklabels([f"n={n}" for n in n_values])
        for i in range(len(n_values)):
            for j in range(len(families)):
                if labels[i][j]:
                    color = "white" if np.isfinite(matrix[i, j]) and matrix[i, j] >= 0.55 else "#202020"
                    ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=5.6, color=color)
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=5.4, color="#6B7280")
    cbar = fig.colorbar(im, ax=axes, shrink=0.82, fraction=0.036, pad=0.035)
    cbar.set_label("Success rate")
    save_fig(fig, figure_dir / "extra1024_action_chunking_task_family_heatmap")


def rollout_completed_hist(table_dir: Path, figure_dir: Path) -> None:
    src = STEP4_TABLE_DIR / "action_chunking_ablation_50seq_ep360_summary.csv"
    if not src.exists():
        return
    summary = pd.read_csv(src)
    rows = []
    for _, summary_row in summary.iterrows():
        json_path = Path(str(summary_row["json_path"]))
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        counts = {i: 0 for i in range(6)}
        for case in data.get("cases", []):
            completed = int(case.get("successful_subtasks", 0))
            counts[min(max(completed, 0), 5)] += 1
        total = sum(counts.values())
        for completed, count in counts.items():
            rows.append(
                {
                    "policy": str(summary_row["policy"]),
                    "n_action_steps": int(summary_row["n_action_steps"]),
                    "completed_subtasks": int(completed),
                    "count": int(count),
                    "total_sequences": int(total),
                    "fraction": count / max(total, 1),
                }
            )
    hist = pd.DataFrame(rows)
    hist.to_csv(table_dir / "extra1024_action_chunking_completed_subtasks_hist.csv", index=False)
    abc = hist[
        (hist["policy"] == "ABC_mixed_task_id_finetune")
        & (hist["n_action_steps"].isin([20, 100]))
    ].copy()
    if abc.empty:
        return
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    width = 0.34
    for idx, n_action in enumerate([20, 100]):
        sub = abc[abc["n_action_steps"] == n_action].set_index("completed_subtasks").loc[range(6)]
        offset = (idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            sub["count"].to_numpy(),
            width=width,
            color=["#F28E2B", "#9C755F"][idx],
            edgecolor="white",
            linewidth=0.6,
            label=f"ABC n={n_action}",
        )
        for bar, count in zip(bars, sub["count"].to_numpy(), strict=True):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, count + 0.4, str(int(count)), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xlabel("Completed subtasks in one 5-task sequence")
    ax.set_ylabel("Sequence count")
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    save_fig(fig, figure_dir / "extra1024_action_chunking_completed_hist_abc_n20_n100")


def copy_contact_sheet(table_dir: Path, figure_dir: Path, prefixes: list[str], out_prefix: str) -> None:
    del table_dir
    src = figure_dir / f"{prefixes[0]}_visual_shift_contact_sheet.png"
    if src.exists():
        shutil.copyfile(src, figure_dir / f"{out_prefix}_visual_shift_contact_sheet.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=PROJECT_ROOT / "results/tables/track2_extra")
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "figures/track2_extra")
    parser.add_argument("--visual-prefix", action="append", default=[])
    parser.add_argument("--output-prefix", default="extra1024_visual_full_robustness")
    parser.add_argument("--taskid-prefix", action="append", default=[])
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    setup_style()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    prefixes = args.visual_prefix or ["extra1024_B_visual_full_robustness", "extra1024_ABC_visual_full_robustness"]
    suffixes = {
        "summary": "visual_robustness_summary.csv",
        "per_episode": "visual_robustness_per_episode_action_l1.csv",
        "per_action_dim": "visual_robustness_per_action_dim.csv",
        "chunk_position": "chunk_position_action_l1.csv",
        "chunk_bin": "chunk_bin_action_l1.csv",
        "tail_head": "chunk_tail_head_ratio.csv",
        "per_task_type": "visual_robustness_per_task_type_action_l1.csv",
    }
    merged = {}
    for key, suffix in suffixes.items():
        df = read_concat(args.table_dir, prefixes, suffix)
        out = args.table_dir / f"{args.output_prefix}_{suffix}"
        df.to_csv(out, index=False)
        merged[key] = df

    rng = np.random.default_rng(args.seed)
    ci = compute_visual_ci(merged["per_episode"], rng, args.bootstrap_iters)
    ci.to_csv(args.table_dir / f"{args.output_prefix}_paired_bootstrap_ci.csv", index=False)
    perturb = make_visual_tables(merged["summary"], ci, args.table_dir, args.output_prefix)
    del perturb
    condition_summary = make_condition_distribution_summary(
        merged["per_episode"],
        ci,
        args.table_dir,
        args.output_prefix,
        rng,
        args.bootstrap_iters,
    )
    del condition_summary
    prefix = make_chunk_prefix_drift_summary(
        merged["chunk_position"],
        merged["chunk_bin"],
        merged["tail_head"],
        args.table_dir,
        args.output_prefix,
    )

    plot_degradation_ci(ci, args.figure_dir, args.output_prefix)
    plot_abc_minus_b_ci(ci, args.figure_dir, args.output_prefix)
    plot_action_dim_heatmap(merged["per_action_dim"], args.figure_dir, args.output_prefix)
    plot_chunk_bins(merged["chunk_bin"], args.figure_dir, args.output_prefix)
    plot_chunk_condition_heatmap(merged["chunk_bin"], args.figure_dir, args.output_prefix)
    plot_prefix_l1(prefix, args.figure_dir, args.output_prefix)
    plot_rollout_family_heatmap(args.table_dir, args.figure_dir)
    rollout_completed_hist(args.table_dir, args.figure_dir)
    taskid_prefixes = args.taskid_prefix or taskid_prefixes_default(args.table_dir)
    plot_taskid_ablation(args.table_dir, args.figure_dir, taskid_prefixes)
    copy_contact_sheet(args.table_dir, args.figure_dir, prefixes, args.output_prefix)

    print(args.table_dir / f"{args.output_prefix}_paired_bootstrap_ci.csv")
    print(args.table_dir / f"{args.output_prefix}_perturbation_full_table.csv")
    print(args.figure_dir / f"{args.output_prefix}_visual_degradation_ci.png")
    print(args.figure_dir / f"{args.output_prefix}_abc_minus_b_ci.png")
    print(args.figure_dir / f"{args.output_prefix}_action_dim_heatmap.png")
    print(args.figure_dir / f"{args.output_prefix}_chunk_10bin_action_l1.png")
    print(args.figure_dir / f"{args.output_prefix}_chunk_condition_heatmap.png")
    print(args.figure_dir / f"{args.output_prefix}_chunk_prefix_l1.png")
    print(args.figure_dir / "extra1024_action_chunking_task_family_heatmap.png")
    print(args.figure_dir / "extra1024_action_chunking_completed_hist_abc_n20_n100.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
