import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "report" / "assets"
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"

EPOCH_RE = re.compile(
    r"\[(?P<run>[^\]]+)\] epoch (?P<epoch>\d+)/(?P<total>\d+) "
    r"train_loss (?P<train_loss>\d+\.\d+) train_acc (?P<train_acc>\d+\.\d+) "
    r"val_loss (?P<val_loss>\d+\.\d+) val_acc (?P<val_acc>\d+\.\d+)"
)

LOG_NAME_MAP = {
    "E6_scratch_r18": "E6_scratch.log",
}

RUN_DISPLAY = {
    "E1_baseline_r18": "E1 R18 baseline",
    "E2_baseline_r34": "E2 R34 baseline",
    "E3_hp_lr_high": "E3 high LR",
    "E4_hp_lr_low": "E4 low LR",
    "E5_hp_long": "E5 60 epochs",
    "E6_scratch_r18": "E6 scratch tuned",
    "E7_se_r18": "E7 naive SE",
    "E7b_se_r18_fixed": "E7b fixed SE",
    "E7c_seresnet18_timm": "E7c pretrained SE",
    "E8_cbam_r18": "E8 naive CBAM",
    "E8b_cbam_r18_fixed": "E8b fixed CBAM",
    "E8c_cbam_r18_long": "E8c fixed CBAM 60 epochs",
    "E9_vit_tiny": "E9 ViT-Tiny",
    "E10_hp_ratio_1x": "E10 ratio 1:1",
    "E11_hp_ratio_20x": "E11 ratio 20:1",
    "E12_hp_freeze3": "E12 freeze 3",
    "E13_hp_freeze5": "E13 freeze 5",
    "E14_scratch_samecfg_r18": "E14 scratch R18 same cfg",
    "E15_scratch_samecfg_r34": "E15 scratch R34 same cfg",
    "E16_se_id_lowlr": "E16 SE id+low",
    "E17_se_naive_highlr": "E17 SE naive+high",
    "E18_cbam_id_lowlr": "E18 CBAM id+low",
    "E19_cbam_naive_highlr": "E19 CBAM naive+high",
}

RUN_COLORS = {
    "E1_baseline_r18": "#2C7A7B",
    "E2_baseline_r34": "#185ADB",
    "E3_hp_lr_high": "#E07A5F",
    "E4_hp_lr_low": "#B56576",
    "E5_hp_long": "#8D99AE",
    "E6_scratch_r18": "#F2CC8F",
    "E7_se_r18": "#C1121F",
    "E7b_se_r18_fixed": "#588157",
    "E7c_seresnet18_timm": "#7F5539",
    "E8_cbam_r18": "#9D0208",
    "E8b_cbam_r18_fixed": "#2A9D8F",
    "E8c_cbam_r18_long": "#3A86FF",
    "E9_vit_tiny": "#6A4C93",
    "E10_hp_ratio_1x": "#0E9594",
    "E11_hp_ratio_20x": "#FF6B35",
    "E12_hp_freeze3": "#6C757D",
    "E13_hp_freeze5": "#198754",
    "E14_scratch_samecfg_r18": "#D62828",
    "E15_scratch_samecfg_r34": "#9B2226",
    "E16_se_id_lowlr": "#43AA8B",
    "E17_se_naive_highlr": "#F9844A",
    "E18_cbam_id_lowlr": "#4D908E",
    "E19_cbam_naive_highlr": "#F94144",
}


def pct(x: float) -> float:
    return x * 100.0


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F4EE",
            "axes.facecolor": "#FFFEFB",
            "axes.edgecolor": "#2E2A24",
            "axes.labelcolor": "#2E2A24",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "font.size": 11,
            "grid.color": "#D9D3C7",
            "grid.alpha": 0.45,
            "legend.frameon": False,
            "xtick.color": "#2E2A24",
            "ytick.color": "#2E2A24",
            "text.color": "#2E2A24",
        }
    )


def ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics() -> dict:
    metrics = {}
    for path in sorted(CKPT_DIR.glob("*/metrics.json")):
        run_name = path.parent.name
        if run_name.startswith("smoke_"):
            continue
        metrics[run_name] = json.loads(path.read_text())
    return metrics


def parse_log(run_name: str) -> list[dict]:
    log_name = LOG_NAME_MAP.get(run_name, f"{run_name}.log")
    path = LOG_DIR / log_name
    rows = []
    for line in path.read_text().splitlines():
        match = EPOCH_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "train_loss": float(match.group("train_loss")),
                "val_loss": float(match.group("val_loss")),
                "val_acc": float(match.group("val_acc")),
            }
        )
    return rows


def add_bar_labels(ax, values: list[float], decimals: int = 2) -> None:
    for idx, value in enumerate(values):
        ax.text(
            idx,
            value + 0.45,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#2E2A24",
        )


def plot_overview(metrics: dict) -> None:
    groups = [
        ("Baseline", ["E1_baseline_r18", "E2_baseline_r34"]),
        ("Hyperparam", ["E3_hp_lr_high", "E4_hp_lr_low", "E5_hp_long", "E10_hp_ratio_1x", "E11_hp_ratio_20x", "E12_hp_freeze3", "E13_hp_freeze5"]),
        ("Pretraining", ["E6_scratch_r18", "E14_scratch_samecfg_r18", "E15_scratch_samecfg_r34"]),
        ("Attention", ["E7_se_r18", "E7b_se_r18_fixed", "E7c_seresnet18_timm", "E8_cbam_r18", "E8b_cbam_r18_fixed", "E8c_cbam_r18_long", "E16_se_id_lowlr", "E17_se_naive_highlr", "E18_cbam_id_lowlr", "E19_cbam_naive_highlr"]),
        ("Transformer", ["E9_vit_tiny"]),
    ]

    labels = []
    values = []
    colors = []
    separators = []

    cursor = 0
    for group_name, run_names in groups:
        for run_name in run_names:
            labels.append(RUN_DISPLAY[run_name].replace(" baseline", ""))
            values.append(pct(metrics[run_name]["test_acc"]))
            colors.append(RUN_COLORS[run_name])
            cursor += 1
        separators.append((cursor - 0.5, group_name))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(13.2, 10.2))
    ax.barh(y, values, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test Accuracy (%)")
    ax.set_title("Oxford-IIIT Pet: Unified Result Overview Across All Required Experiment Families")
    ax.set_xlim(0, 96)
    ax.grid(axis="x")
    ax.invert_yaxis()

    for yi, value in zip(y, values):
        ax.text(value + 0.8, yi, f"{value:.2f}", va="center", fontsize=9)

    start = 0
    for sep, group_name in separators:
        if sep < len(labels) - 0.5:
            ax.axhline(sep, color="#C8C1B4", linewidth=1.2)
        center = (start + min(sep, len(labels) - 0.5)) / 2.0
        ax.text(
            95.5,
            center,
            group_name,
            ha="right",
            va="center",
            fontsize=9,
            color="#5C554A",
            bbox={"boxstyle": "round,pad=0.22", "fc": "#F0E9DD", "ec": "none"},
        )
        start = sep + 0.5

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "final_overview.png", dpi=240)
    plt.close(fig)


def plot_curve_triplet(filename: str, title: str, run_names: list[str], legend_ncol: int = 2) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))
    metric_specs = [
        ("train_loss", "Train Loss"),
        ("val_loss", "Val Loss"),
        ("val_acc", "Val Accuracy (%)"),
    ]

    for ax, (metric_key, metric_title) in zip(axes, metric_specs):
        for run_name in run_names:
            rows = parse_log(run_name)
            xs = [row["epoch"] for row in rows]
            ys = [row[metric_key] for row in rows]
            if metric_key == "val_acc":
                ys = [pct(v) for v in ys]
            ax.plot(
                xs,
                ys,
                label=RUN_DISPLAY[run_name],
                color=RUN_COLORS[run_name],
                linewidth=2.2,
            )
        ax.set_title(metric_title)
        ax.set_xlabel("Epoch")
        ax.grid(True)

    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Loss")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(ASSETS_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_hparam_deep_dive(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    ratio_names = ["E10_hp_ratio_1x", "E1_baseline_r18", "E11_hp_ratio_20x"]
    ratio_labels = ["1:1", "10:1", "20:1"]
    ratio_vals = [pct(metrics[name]["test_acc"]) for name in ratio_names]
    axes[0].bar(ratio_labels, ratio_vals, color=[RUN_COLORS[name] for name in ratio_names])
    axes[0].set_title("Head / Backbone LR Ratio")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(axis="y")
    add_bar_labels(axes[0], ratio_vals)

    freeze_names = ["E1_baseline_r18", "E12_hp_freeze3", "E13_hp_freeze5"]
    freeze_labels = ["0", "3", "5"]
    freeze_vals = [pct(metrics[name]["test_acc"]) for name in freeze_names]
    axes[1].bar(freeze_labels, freeze_vals, color=[RUN_COLORS[name] for name in freeze_names])
    axes[1].set_title("Freeze Backbone Epochs")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_xlabel("Frozen epochs")
    axes[1].grid(axis="y")
    add_bar_labels(axes[1], freeze_vals)

    fig.suptitle("Hyperparameter Deep-Dive: Fine-Tuning Strategy", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "final_hparam_deep_dive.png", dpi=240)
    plt.close(fig)


def plot_attention_factorial(metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    se_names = ["E7_se_r18", "E17_se_naive_highlr", "E16_se_id_lowlr", "E7b_se_r18_fixed"]
    se_labels = ["naive+low", "naive+high", "id+low", "id+high"]
    se_vals = [pct(metrics[name]["test_acc"]) for name in se_names]
    axes[0].bar(se_labels, se_vals, color=[RUN_COLORS[name] for name in se_names])
    axes[0].set_title("SE Factorial Ablation")
    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].grid(axis="y")
    add_bar_labels(axes[0], se_vals)
    axes[0].tick_params(axis="x", rotation=10)

    cbam_names = ["E8_cbam_r18", "E19_cbam_naive_highlr", "E18_cbam_id_lowlr", "E8b_cbam_r18_fixed"]
    cbam_labels = ["naive+low", "naive+high", "id+low", "id+high"]
    cbam_vals = [pct(metrics[name]["test_acc"]) for name in cbam_names]
    axes[1].bar(cbam_labels, cbam_vals, color=[RUN_COLORS[name] for name in cbam_names])
    axes[1].set_title("CBAM Factorial Ablation")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].grid(axis="y")
    add_bar_labels(axes[1], cbam_vals)
    axes[1].tick_params(axis="x", rotation=10)

    fig.suptitle("Attention Factorial Ablation: Compatibility vs Learning Rate", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "final_attention_factorial.png", dpi=240)
    plt.close(fig)


def main() -> None:
    ensure_assets_dir()
    apply_theme()
    metrics = load_metrics()

    plot_overview(metrics)
    plot_curve_triplet(
        "final_core_curves.png",
        "Core Baseline and Hyperparameter Curves (Teacher-Required Baseline Family)",
        ["E1_baseline_r18", "E3_hp_lr_high", "E4_hp_lr_low", "E5_hp_long"],
    )
    plot_curve_triplet(
        "final_pretraining_curves.png",
        "Pretraining Ablation Curves (Pretrained vs Scratch Convergence)",
        ["E1_baseline_r18", "E6_scratch_r18", "E14_scratch_samecfg_r18", "E2_baseline_r34", "E15_scratch_samecfg_r34"],
        legend_ncol=3,
    )
    plot_curve_triplet(
        "final_attention_curves.png",
        "Attention Module Curves (Naive Insertion vs Compatibility-Preserving Insertion)",
        ["E7_se_r18", "E7b_se_r18_fixed", "E8_cbam_r18", "E8b_cbam_r18_fixed"],
    )
    plot_hparam_deep_dive(metrics)
    plot_attention_factorial(metrics)


if __name__ == "__main__":
    main()
