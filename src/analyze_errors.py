"""Fine-grained error analysis for segmentation checkpoints."""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import NUM_CLASSES, build_loaders
from .visualize import PALETTE
from .unet import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="space-separated NAME=PATH entries")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--base_ch", type=int, default=64)
    p.add_argument("--csv_out", default="report/error_analysis.csv")
    p.add_argument("--fig_out", default="report/error_analysis.png")
    p.add_argument("--hard_cases_out", default="report/hard_cases.png")
    return p.parse_args()


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def per_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        out[cls] = inter / union if union > 0 else 0.0
    return out


def load_models(entries: list[str], device: torch.device, base_ch: int):
    models = []
    for entry in entries:
        name, path = entry.split("=", 1)
        model = UNet(in_channels=3, num_classes=NUM_CLASSES, base_ch=base_ch).to(device)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        models.append((name, model))
    return models


def build_loader(split: str, data_root: str, batch_size: int):
    _, val_loader, test_loader = build_loaders(
        root=data_root, batch_size=batch_size, num_workers=0, val_ratio=0.1, seed=42
    )
    return val_loader if split == "val" else test_loader


def write_csv(rows: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict], model_names: list[str], out: str):
    difficulty = {
        "low boundary (<8%)": lambda r: r["boundary_ratio"] < 0.08,
        "mid boundary (8-14%)": lambda r: 0.08 <= r["boundary_ratio"] < 0.14,
        "high boundary (>=14%)": lambda r: r["boundary_ratio"] >= 0.14,
    }
    fg_bins = {
        "small fg (<20%)": lambda r: r["foreground_ratio"] < 0.20,
        "mid fg (20-40%)": lambda r: 0.20 <= r["foreground_ratio"] < 0.40,
        "large fg (>=40%)": lambda r: r["foreground_ratio"] >= 0.40,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, groups, metric, title in [
        (axes[0], difficulty, "boundary_iou", "Boundary IoU by Boundary-Pixel Ratio"),
        (axes[1], fg_bins, "miou", "mIoU by Foreground Occupancy"),
    ]:
        x = np.arange(len(groups))
        width = 0.18
        for idx, name in enumerate(model_names):
            vals = []
            for rule in groups.values():
                subset = [r[f"{name}_{metric}"] for r in rows if rule(r)]
                vals.append(float(np.mean(subset)) if subset else 0.0)
            ax.bar(x + (idx - (len(model_names) - 1) / 2) * width, vals, width, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(list(groups.keys()), rotation=10)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")


def plot_hard_cases(hard_rows: list[dict], hard_cases: list[dict], model_names: list[str], out: str):
    if not hard_rows:
        return
    n_rows = len(hard_rows)
    n_cols = 2 + len(model_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    headers = ["Input", "GT"] + model_names
    for c, h in enumerate(headers):
        axes[0, c].set_title(h, fontsize=11)

    for r, row in enumerate(hard_rows):
        case = hard_cases[row["sample_id"]]
        axes[r, 0].imshow(case["image"])
        axes[r, 1].imshow(colorize(case["target"]))
        for c, name in enumerate(model_names):
            axes[r, 2 + c].imshow(colorize(case["preds"][name]))
            axes[r, 2 + c].set_xlabel(
                f"bIoU={row[f'{name}_boundary_iou']:.3f}",
                fontsize=8,
            )
        for c in range(n_cols):
            axes[r, c].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved -> {out}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(args.split, args.data_root, args.batch_size)
    models = load_models(args.ckpts, device=device, base_ch=args.base_ch)
    model_names = [name for name, _ in models]

    rows: list[dict] = []
    hard_cases: list[dict] = []
    sample_id = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = {}
            for name, model in models:
                preds[name] = model(x).argmax(dim=1).cpu()
            x_np = (x.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
            y_cpu = y.cpu()
            for i in range(y.size(0)):
                row = {
                    "sample_id": sample_id,
                    "foreground_ratio": float((y_cpu[i] == 0).float().mean().item()),
                    "background_ratio": float((y_cpu[i] == 1).float().mean().item()),
                    "boundary_ratio": float((y_cpu[i] == 2).float().mean().item()),
                }
                case = {
                    "image": x_np[i],
                    "target": y_cpu[i].numpy(),
                    "preds": {},
                }
                for name in model_names:
                    pred_i = preds[name][i]
                    ious = per_class_iou(pred_i, y_cpu[i], NUM_CLASSES)
                    row[f"{name}_miou"] = float(ious.mean().item())
                    row[f"{name}_foreground_iou"] = float(ious[0].item())
                    row[f"{name}_background_iou"] = float(ious[1].item())
                    row[f"{name}_boundary_iou"] = float(ious[2].item())
                    case["preds"][name] = pred_i.numpy()
                rows.append(row)
                hard_cases.append(case)
                sample_id += 1

    write_csv(rows, args.csv_out)
    plot_summary(rows, model_names, args.fig_out)

    informative = [
        r for r in rows
        if r["boundary_ratio"] > 0.0 and r["foreground_ratio"] > 0.0
    ]
    # Focus on samples where boundary structure exists and the CE baseline fails badly.
    hardest = sorted(
        informative,
        key=lambda r: (r[f"{model_names[0]}_boundary_iou"], -r["boundary_ratio"]),
    )[:4]
    plot_hard_cases(hardest, hard_cases, model_names, args.hard_cases_out)


if __name__ == "__main__":
    main()
