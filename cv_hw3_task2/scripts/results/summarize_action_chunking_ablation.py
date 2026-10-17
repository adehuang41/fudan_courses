#!/usr/bin/env python3
"""Summarize task_D_D action-chunking closed-loop ablation results."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


def read_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                *body,
                "</svg>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#222">'
        f"{html.escape(text)}</text>"
    )


def write_svg_figures(rows: list[dict], figure_dir: Path, prefix: str) -> tuple[Path, Path]:
    policies = sorted({row["policy"] for row in rows})
    n_values = sorted({row["n_action_steps"] for row in rows})
    colors = {
        "B_only_task_id_finetune": "#7b3294",
        "ABC_mixed_task_id_finetune": "#008837",
    }
    labels = {
        "B_only_task_id_finetune": "B-only task-id",
        "ABC_mixed_task_id_finetune": "ABC-mixed task-id",
    }

    width, height = 760, 430
    left, right, top, bottom = 74, 34, 58, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    max_avg = max(row["avg_successful_subtasks"] for row in rows)
    y_max = max(1.5, round((max_avg + 0.2) * 2) / 2)
    x_pos = {n: left + idx * plot_w / (len(n_values) - 1) for idx, n in enumerate(n_values)}

    def y_pos(value: float, y_scale: float) -> float:
        return top + plot_h - (value / y_scale) * plot_h

    body = [
        svg_text(width / 2, 28, "Action Chunking Ablation on task_D_D Closed-Loop Rollouts", 17, "middle", "700"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.5, 1.0, 1.5]:
        if tick > y_max:
            continue
        y = y_pos(tick, y_max)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(svg_text(left - 10, y + 4, f"{tick:.1f}", 11, "end"))
    for n in n_values:
        x = x_pos[n]
        body.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        body.append(svg_text(x, top + plot_h + 23, str(n), 12, "middle"))
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        points = [(x_pos[row["n_action_steps"]], y_pos(row["avg_successful_subtasks"], y_max)) for row in policy_rows]
        point_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        body.append(
            f'<polyline points="{point_str}" fill="none" stroke="{colors.get(policy, "#444")}" '
            'stroke-width="3" stroke-linejoin="round"/>'
        )
        for row, (x, y) in zip(policy_rows, points, strict=True):
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colors.get(policy, "#444")}"/>')
            body.append(svg_text(x, y - 10, f'{row["avg_successful_subtasks"]:.2f}', 11, "middle"))
    body.append(svg_text(left + plot_w / 2, height - 22, "n_action_steps executed per ACT prediction", 13, "middle"))
    body.append(svg_text(14, top + plot_h / 2, "Avg. successful subtasks", 13, "start"))
    for idx, policy in enumerate(policies):
        x = left + 18 + idx * 190
        y = top - 18
        body.append(f'<rect x="{x}" y="{y - 10}" width="18" height="4" fill="{colors.get(policy, "#444")}"/>')
        body.append(svg_text(x + 25, y - 5, labels.get(policy, policy), 12))
    avg_fig = figure_dir / f"{prefix}_avg_successful_subtasks.svg"
    write_svg(avg_fig, width, height, body)

    body = [
        svg_text(width / 2, 28, "Closed-Loop Success Rate vs ACT Replanning Frequency", 17, "middle", "700"),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick, 1.0)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        body.append(svg_text(left - 10, y + 4, f"{tick:.2f}", 11, "end"))
    bar_group_w = plot_w / len(n_values)
    bar_w = 18
    metrics = ["SR@1", "SR@2", "SR@3"]
    for n_idx, n in enumerate(n_values):
        group_x = left + n_idx * bar_group_w + bar_group_w / 2
        body.append(svg_text(group_x, top + plot_h + 23, str(n), 12, "middle"))
        for p_idx, policy in enumerate(policies):
            row = next(row for row in rows if row["policy"] == policy and row["n_action_steps"] == n)
            for m_idx, metric in enumerate(metrics):
                x = group_x + (p_idx - 0.5) * 70 + (m_idx - 1) * (bar_w + 2)
                value = row[metric]
                y = y_pos(value, 1.0)
                bar_h = top + plot_h - y
                opacity = 0.45 + 0.22 * m_idx
                body.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" '
                    f'fill="{colors.get(policy, "#444")}" opacity="{opacity:.2f}"/>'
                )
    body.append(svg_text(left + plot_w / 2, height - 22, "n_action_steps", 13, "middle"))
    body.append(svg_text(14, top + plot_h / 2, "Chain Success Rate", 13, "start"))
    body.append(svg_text(left + 18, top - 18, "Opacity: SR@1, SR@2, SR@3", 12))
    for idx, policy in enumerate(policies):
        x = left + 250 + idx * 190
        y = top - 18
        body.append(f'<rect x="{x}" y="{y - 12}" width="18" height="8" fill="{colors.get(policy, "#444")}"/>')
        body.append(svg_text(x + 25, y - 5, labels.get(policy, policy), 12))
    sr_fig = figure_dir / f"{prefix}_chain_success_rate.svg"
    write_svg(sr_fig, width, height, body)
    return avg_fig, sr_fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables/step4"))
    parser.add_argument("--figure-dir", type=Path, default=Path("results/figures/step4"))
    parser.add_argument("--prefix", default="action_chunking_ablation_50seq_ep360")
    args = parser.parse_args()

    rows = []
    task_rows = []
    for path in sorted(args.run_root.glob("*_50seq_ep360_n*/*_50seq_ep360_n*.json")):
        report = read_report(path)
        chain_sr = report["chain_success_rate"]
        row = {
            "policy": report["policy"],
            "n_action_steps": int(report["policy_n_action_steps"]),
            "chunk_size": int(report["policy_chunk_size"]),
            "num_sequences": int(report["num_sequences"]),
            "ep_len": int(report["ep_len"]),
            "avg_successful_subtasks": float(report["avg_successful_subtasks"]),
            "SR@1": float(chain_sr["1"]),
            "SR@2": float(chain_sr["2"]),
            "SR@3": float(chain_sr["3"]),
            "SR@4": float(chain_sr["4"]),
            "SR@5": float(chain_sr["5"]),
            "server_error_count_total": int(report["server_error_count_total"]),
            "elapsed_sec": float(report["elapsed_sec"]),
            "scale_label": report["scale_label"],
            "json_path": str(path),
        }
        rows.append(row)
        for task, stats in sorted(report["task_success"].items()):
            task_rows.append(
                {
                    "policy": row["policy"],
                    "n_action_steps": row["n_action_steps"],
                    "task": task,
                    "success": int(stats["success"]),
                    "attempted": int(stats["attempted"]),
                    "success_rate": float(stats["success_rate"]),
                }
            )

    if not rows:
        raise SystemExit(f"No action-chunking reports found under {args.run_root}")

    rows.sort(key=lambda row: (row["policy"], row["n_action_steps"]))
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = args.table_dir / f"{args.prefix}_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    task_csv = args.table_dir / f"{args.prefix}_per_task_success.csv"
    with task_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0].keys()))
        writer.writeheader()
        writer.writerows(task_rows)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        avg_fig, sr_fig = write_svg_figures(rows, args.figure_dir, args.prefix)
        print(f"matplotlib unavailable; wrote SVG figures: {exc}")
        print(summary_csv)
        print(task_csv)
        print(avg_fig)
        print(sr_fig)
        return

    policies = sorted({row["policy"] for row in rows})
    n_values = sorted({row["n_action_steps"] for row in rows})
    color = {
        "B_only_task_id_finetune": "#7b3294",
        "ABC_mixed_task_id_finetune": "#008837",
    }
    label = {
        "B_only_task_id_finetune": "B-only task-id",
        "ABC_mixed_task_id_finetune": "ABC-mixed task-id",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        xs = [row["n_action_steps"] for row in policy_rows]
        ys = [row["avg_successful_subtasks"] for row in policy_rows]
        ax.plot(xs, ys, marker="o", linewidth=2.2, color=color.get(policy), label=label.get(policy, policy))
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y + 0.035, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(n_values)
    ax.set_xlabel("n_action_steps executed per ACT prediction")
    ax.set_ylabel("Average successful subtasks per 5-step chain")
    ax.set_title("Action Chunking Ablation on task_D_D Closed-Loop Rollouts")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    avg_fig = args.figure_dir / f"{args.prefix}_avg_successful_subtasks.png"
    fig.savefig(avg_fig, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for policy in policies:
        policy_rows = [row for row in rows if row["policy"] == policy]
        base = [row["n_action_steps"] for row in policy_rows]
        for idx, metric in enumerate(["SR@1", "SR@2", "SR@3", "SR@4"]):
            xs = [x + offsets[idx] for x in base]
            ys = [row[metric] for row in policy_rows]
            alpha = 0.35 + 0.15 * idx
            ax.bar(
                xs,
                ys,
                width=width,
                color=color.get(policy),
                alpha=alpha,
                label=f"{label.get(policy, policy)} {metric}" if idx == 0 else None,
            )
    ax.set_xticks(n_values)
    ax.set_xlabel("n_action_steps")
    ax.set_ylabel("Chain Success Rate")
    ax.set_title("Closed-Loop Success Rate vs ACT Replanning Frequency")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    sr_fig = args.figure_dir / f"{args.prefix}_chain_success_rate.png"
    fig.savefig(sr_fig, bbox_inches="tight")
    plt.close(fig)

    print(summary_csv)
    print(task_csv)
    print(avg_fig)
    print(sr_fig)


if __name__ == "__main__":
    main()
