"""Qualitative comparison: original | GT | CE pred | Dice pred | CE+Dice pred.

Usage:
    python -m src.visualize \
        --ckpts E1_ce=checkpoints/E1_ce/best.pt \
                E2_dice=checkpoints/E2_dice/best.pt \
                E3_ce_dice=checkpoints/E3_ce_dice/best.pt \
        --out report/qualitative.png --n 5
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.backends.cudnn.enabled = False

from .data import CLASS_NAMES, NUM_CLASSES, build_loaders
from .unet import UNet


# Foreground=red, background=light gray, boundary=green
PALETTE = np.array([
    [220,  50,  50],
    [200, 200, 200],
    [ 60, 200,  90],
], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="space-separated NAME=PATH entries, in display order")
    p.add_argument("--out", default="report/qualitative.png")
    p.add_argument("--n", type=int, default=5, help="num samples")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_ch", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    _, val_loader, _ = build_loaders(root=args.data_root, batch_size=args.n,
                                     num_workers=0, val_ratio=0.1, seed=42)
    x, y = next(iter(val_loader))
    x = x.to(device)
    y_np = y.cpu().numpy()

    models = []
    for entry in args.ckpts:
        name, path = entry.split("=", 1)
        m = UNet(in_channels=3, num_classes=NUM_CLASSES, base_ch=args.base_ch).to(device)
        state = torch.load(path, map_location=device)
        m.load_state_dict(state["model"])
        m.eval()
        models.append((name, m))

    preds = []
    with torch.no_grad():
        for name, m in models:
            preds.append((name, m(x).argmax(dim=1).cpu().numpy()))

    n_cols = 2 + len(models)  # image + GT + each prediction
    n_rows = args.n
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    headers = ["Input", "GT"] + [name for name, _ in models]
    for c, h in enumerate(headers):
        axes[0, c].set_title(h, fontsize=11)

    img_np = (x.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    for r in range(n_rows):
        axes[r, 0].imshow(img_np[r])
        axes[r, 1].imshow(colorize(y_np[r]))
        for c, (_, p) in enumerate(preds):
            axes[r, 2 + c].imshow(colorize(p[r]))
        for c in range(n_cols):
            axes[r, c].axis("off")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=PALETTE[i] / 255.0,
                                    label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)]
    fig.legend(handles=legend_handles, loc="lower center", ncol=NUM_CLASSES,
               bbox_to_anchor=(0.5, -0.01))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
