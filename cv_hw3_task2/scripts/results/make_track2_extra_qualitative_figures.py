#!/usr/bin/env python3
"""Build rollout keyframe plates from existing CALVIN videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image


PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
CLOSED_LOOP_ROOT = Path("/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_closed_loop")
CHUNK_ROOT = Path(
    "/data/dechao/cv_final_pj2/outputs/post_finetune_eval/track2_action_chunking/"
    "formal_50seq_ep360_20260613T2050Z"
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "svg.fonttype": "none",
            "savefig.dpi": 450,
        }
    )


def read_log(seq_dir: Path) -> dict:
    return json.loads((seq_dir / "rollout_log.json").read_text(encoding="utf-8"))


def select_keyframes(seq_dir: Path) -> list[Path]:
    frames = sorted(seq_dir.glob("keyframe_*.png"))
    if len(frames) < 3:
        raise FileNotFoundError(f"Need at least 3 keyframes in {seq_dir}")
    return [frames[0], frames[len(frames) // 2], frames[-1]]


def load_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    crop_top = min(int(height * 0.49), height - 1)
    cropped = image.crop((0, crop_top, width, height))
    return cropped.resize((width, height), Image.BILINEAR)


def score_color(success: int) -> str:
    if success >= 5:
        return "#2D7D46"
    if success >= 3:
        return "#2F6FA3"
    if success >= 1:
        return "#9A6A1E"
    return "#A43B35"


def save_plate(rows: list[dict], out_base: Path, column_labels: tuple[str, str, str]) -> list[dict]:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    fig_w = 7.2
    label_ratio = 0.48 if n_rows <= 2 else 0.68
    image_w = fig_w / (3.0 + label_ratio)
    fig_h = 0.42 + n_rows * image_w
    fig = plt.figure(figsize=(fig_w, fig_h))
    grid = GridSpec(
        n_rows,
        4,
        figure=fig,
        width_ratios=[label_ratio, 1.0, 1.0, 1.0],
        wspace=0.035,
        hspace=0.055,
    )
    manifest_rows = []
    for r_idx, row in enumerate(rows):
        seq_dir = Path(row["seq_dir"])
        log = read_log(seq_dir)
        frames = select_keyframes(seq_dir)
        success = int(log.get("successful_subtasks", 0))
        tasks = log.get("eval_sequence", [])
        label_ax = fig.add_subplot(grid[r_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.55,
            row["label"],
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#202020",
            transform=label_ax.transAxes,
        )
        label_ax.text(
            0.98,
            0.32,
            f"{success}/5",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=score_color(success),
            transform=label_ax.transAxes,
        )
        for c_idx, frame_path in enumerate(frames):
            ax = fig.add_subplot(grid[r_idx, c_idx + 1])
            ax.imshow(load_image(frame_path))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.7)
                spine.set_edgecolor("#C8C8C8")
            if r_idx == 0:
                ax.set_title(column_labels[c_idx], pad=6, fontsize=10, fontweight="bold")
            if c_idx == len(frames) - 1:
                ax.text(
                    0.93,
                    0.08,
                    f"{success}/5",
                    ha="right",
                    va="bottom",
                    fontsize=13,
                    fontweight="bold",
                    color="white",
                    transform=ax.transAxes,
                    bbox={
                        "boxstyle": "round,pad=0.26,rounding_size=0.08",
                        "facecolor": score_color(success),
                        "edgecolor": "white",
                        "linewidth": 0.9,
                        "alpha": 0.96,
                    },
                )
            manifest_rows.append(
                {
                    "figure": out_base.name,
                    "row_label": row["label"],
                    "seq_dir": str(seq_dir),
                    "frame": str(frame_path),
                    "successful_subtasks": success,
                    "eval_sequence": " | ".join(tasks),
                }
            )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.965, bottom=0.01)
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=PROJECT_ROOT / "figures/track2_extra")
    parser.add_argument("--table-dir", type=Path, default=PROJECT_ROOT / "results/tables/track2_extra")
    parser.add_argument("--closed-loop-seq", default="seq_02")
    parser.add_argument("--chunk-seq", default="seq_08")
    parser.add_argument("--chunk-seqs", default="seq_08,seq_44,seq_22")
    parser.add_argument("--write-multi-seq-appendix", action="store_true")
    args = parser.parse_args()

    setup_style()
    closed_rows = [
        {
            "label": "B visual",
            "seq_dir": CLOSED_LOOP_ROOT / "B_only_visual_baseline_50seq_ep360" / args.closed_loop_seq,
        },
        {
            "label": "ABC visual",
            "seq_dir": CLOSED_LOOP_ROOT / "ABC_mixed_visual_baseline_50seq_ep360" / args.closed_loop_seq,
        },
        {
            "label": "B task-id",
            "seq_dir": CLOSED_LOOP_ROOT / "B_only_task_id_finetune_50seq_ep360" / args.closed_loop_seq,
        },
        {
            "label": "ABC task-id",
            "seq_dir": CLOSED_LOOP_ROOT / "ABC_mixed_task_id_finetune_50seq_ep360" / args.closed_loop_seq,
        },
    ]
    chunk_rows = [
        {
            "label": "ABC n=20",
            "seq_dir": CHUNK_ROOT / "ABC_mixed_task_id_finetune_50seq_ep360_n20" / args.chunk_seq,
        },
        {
            "label": "ABC n=100",
            "seq_dir": CHUNK_ROOT / "ABC_mixed_task_id_finetune_50seq_ep360_n100" / args.chunk_seq,
        },
    ]
    chunk_multi_rows = []
    for seq in [item.strip() for item in args.chunk_seqs.split(",") if item.strip()]:
        chunk_multi_rows.extend(
            [
                {
                    "label": f"{seq} ABC n=20",
                    "seq_dir": CHUNK_ROOT / "ABC_mixed_task_id_finetune_50seq_ep360_n20" / seq,
                },
                {
                    "label": f"{seq} ABC n=100",
                    "seq_dir": CHUNK_ROOT / "ABC_mixed_task_id_finetune_50seq_ep360_n100" / seq,
                },
            ]
        )
    manifest = []
    closed_loop_base = args.figure_dir / f"rollout_{args.closed_loop_seq}_four_model_keyframes"
    manifest.extend(
        save_plate(
            closed_rows,
            closed_loop_base,
            ("start", "contact", "outcome"),
        )
    )
    manifest.extend(
        save_plate(
            chunk_rows,
            args.figure_dir / "rollout_seq08_action_chunking_keyframes",
            ("start", "contact", "outcome"),
        )
    )
    if args.write_multi_seq_appendix:
        manifest.extend(
            save_plate(
                chunk_multi_rows,
                args.figure_dir / "rollout_action_chunking_keyframes_seq08_44_22",
                ("start", "contact", "outcome"),
            )
        )

    args.table_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.table_dir / "rollout_qualitative_keyframes_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["figure", "row_label", "seq_dir", "frame", "successful_subtasks", "eval_sequence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    print(closed_loop_base.with_suffix(".png"))
    print(args.figure_dir / "rollout_seq08_action_chunking_keyframes.png")
    if args.write_multi_seq_appendix:
        print(args.figure_dir / "rollout_action_chunking_keyframes_seq08_44_22.png")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
