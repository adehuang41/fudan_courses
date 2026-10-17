"""Parse training logs and plot report-friendly training curves."""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt


LOG_SPECS = [
    ("E1 CE", "logs/E1_ce.log"),
    ("E2 Dice", "logs/E2_dice.log"),
    ("E3 CE+Dice", "logs/E3_ce_dice.log"),
    ("E4 Weighted CE", "logs/E4_weighted_ce.log"),
]

LINE_RE = re.compile(
    r"\[(?P<run>[^\]]+)\] ep (?P<epoch>\d+)/(?P<epochs>\d+) "
    r"train_loss (?P<train_loss>\d+\.\d+) \| "
    r"val_loss (?P<val_loss>\d+\.\d+) "
    r"miou (?P<miou>\d+\.\d+) pacc (?P<pacc>\d+\.\d+) "
    r"\(foreground=(?P<fg>\d+\.\d+) background=(?P<bg>\d+\.\d+) boundary=(?P<bd>\d+\.\d+)\)"
)


def parse_log(path: str) -> dict[str, list[float]]:
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "miou": [],
        "pacc": [],
        "fg_iou": [],
        "bg_iou": [],
        "bd_iou": [],
    }
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            metrics["epoch"].append(int(m.group("epoch")))
            metrics["train_loss"].append(float(m.group("train_loss")))
            metrics["val_loss"].append(float(m.group("val_loss")))
            metrics["miou"].append(float(m.group("miou")))
            metrics["pacc"].append(float(m.group("pacc")))
            metrics["fg_iou"].append(float(m.group("fg")))
            metrics["bg_iou"].append(float(m.group("bg")))
            metrics["bd_iou"].append(float(m.group("bd")))
    if not metrics["epoch"]:
        raise ValueError(f"No epoch metrics parsed from {path}")
    return metrics


def _style_axis(ax, title: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_loss_curves(series, out: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax_train, ax_val = axes

    for label, m in series:
        epochs = m["epoch"]
        ax_train.plot(epochs, m["train_loss"], linewidth=2.2, label=label)
        ax_val.plot(epochs, m["val_loss"], linewidth=2.2, label=label)

    _style_axis(ax_train, "Training Loss Curves", "Train Loss")
    _style_axis(ax_val, "Validation Loss Curves", "Validation Loss")
    ax_train.legend(fontsize=10, ncol=2)
    ax_val.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation_metrics(series, out: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax_acc, ax_miou = axes

    for label, m in series:
        epochs = m["epoch"]
        ax_acc.plot(epochs, m["pacc"], linewidth=2.2, label=label)
        ax_miou.plot(epochs, m["miou"], linewidth=2.2, label=label)

    _style_axis(ax_acc, "Validation Pixel Accuracy", "Pixel Accuracy")
    _style_axis(ax_miou, "Validation mIoU", "mIoU")
    ax_acc.legend(fontsize=10, ncol=2)
    ax_miou.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_boundary_metric(series, out: str):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    for label, m in series:
        epochs = m["epoch"]
        ax.plot(epochs, m["bd_iou"], linewidth=2.4, label=label)

    _style_axis(ax, "Validation Boundary IoU", "Boundary IoU")
    ax.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(series, out: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_loss, ax_miou, ax_boundary, ax_pacc = axes.ravel()

    for label, m in series:
        epochs = m["epoch"]
        ax_loss.plot(epochs, m["train_loss"], label=f"{label} train")
        ax_loss.plot(epochs, m["val_loss"], linestyle="--", label=f"{label} val")
        ax_miou.plot(epochs, m["miou"], label=label)
        ax_boundary.plot(epochs, m["bd_iou"], label=label)
        ax_pacc.plot(epochs, m["pacc"], label=label)

    _style_axis(ax_loss, "Loss Dynamics", "Loss")
    ax_loss.legend(fontsize=8, ncol=2)

    _style_axis(ax_miou, "Validation mIoU", "mIoU")
    ax_miou.legend(fontsize=8)

    _style_axis(ax_boundary, "Validation Boundary IoU", "Boundary IoU")
    ax_boundary.legend(fontsize=8)

    _style_axis(ax_pacc, "Validation Pixel Accuracy", "Pixel Accuracy")
    ax_pacc.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    series = []
    for label, path in LOG_SPECS:
        if os.path.exists(path):
            series.append((label, parse_log(path)))

    if not series:
        raise FileNotFoundError("No training logs found")

    os.makedirs("report", exist_ok=True)
    plot_loss_curves(series, "report/training_loss_curves.png")
    plot_validation_metrics(series, "report/validation_metrics.png")
    plot_boundary_metric(series, "report/validation_boundary_iou.png")
    plot_overview(series, "report/training_dynamics.png")
    print("saved -> report/training_loss_curves.png")
    print("saved -> report/validation_metrics.png")
    print("saved -> report/validation_boundary_iou.png")
    print("saved -> report/training_dynamics.png")


if __name__ == "__main__":
    main()
