"""Bar chart of all 5 runs by val mIoU + boundary IoU.

Used for the report's loss-weight ablation figure.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    ("E1\nCE",         "checkpoints/E1_ce/metrics.json"),
    ("E2\nDice",       "checkpoints/E2_dice/metrics.json"),
    ("E3a\n1:0.5",     "checkpoints/E3a_ce_dice_2to1/metrics.json"),
    ("E3\n1:1",        "checkpoints/E3_ce_dice/metrics.json"),
    ("E3b\n0.5:1",     "checkpoints/E3b_ce_dice_1to2/metrics.json"),
]


def main(out="report/ablation_bars.png"):
    labels, val_miou, val_bd, test_miou, test_bd = [], [], [], [], []
    for label, path in RUNS:
        m = json.load(open(path))
        labels.append(label)
        val_miou.append(m["best_val"]["miou"])
        val_bd.append(m["best_val"]["iou_per_class"][2])
        test_miou.append(m["test"]["miou"])
        test_bd.append(m["test"]["iou_per_class"][2])

    x = np.arange(len(labels))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.bar(x - w / 2, val_miou, w, label="val mIoU", color="#3a72b8")
    ax.bar(x + w / 2, test_miou, w, label="test mIoU", color="#e08020")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.74, 0.795)
    ax.set_ylabel("mIoU")
    ax.set_title("Overall mIoU across loss configurations")
    for i, v in enumerate(val_miou):
        ax.text(i - w / 2, v + 0.0005, f"{v:.4f}", ha="center", fontsize=8)
    for i, v in enumerate(test_miou):
        ax.text(i + w / 2, v + 0.0005, f"{v:.4f}", ha="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x - w / 2, val_bd, w, label="val Boundary IoU", color="#3a72b8")
    ax.bar(x + w / 2, test_bd, w, label="test Boundary IoU", color="#e08020")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.54, 0.605)
    ax.set_ylabel("Boundary IoU")
    ax.set_title("Boundary class IoU (the minority class, 10.95%)")
    for i, v in enumerate(val_bd):
        ax.text(i - w / 2, v + 0.0005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(test_bd):
        ax.text(i + w / 2, v + 0.0005, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
