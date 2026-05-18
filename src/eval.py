"""Evaluate a trained YOLO model on VisDrone val (and optionally test).

Generates:
  - terminal table with mAP50, mAP50-95, per-class AP
  - report/figures/per_class_ap.png  (matplotlib bar chart)
  - report/figures/{pr_curve.png, confusion_matrix.png, ...} from ultralytics
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
torch.backends.cudnn.enabled = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ultralytics import YOLO  # noqa: E402

from src.utils import VISDRONE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "report/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _eval_split(model: YOLO, data_yaml: str, split: str, imgsz: int, device: str, batch: int):
    print(f"\n=== Evaluating {split} ===")
    res = model.val(
        data=data_yaml, split=split, imgsz=imgsz, device=device,
        batch=batch, plots=True, save_json=False,
        project=str(REPO_ROOT / "outputs/eval"), name=split, exist_ok=True,
    )
    print(f"mAP50      = {res.box.map50:.4f}")
    print(f"mAP50-95   = {res.box.map:.4f}")
    print("Per-class AP50 / AP50-95:")
    for i, name in enumerate(VISDRONE_NAMES):
        ap50 = float(res.box.ap50[i]) if i < len(res.box.ap50) else float("nan")
        ap = float(res.box.ap[i].mean()) if i < len(res.box.ap) else float("nan")
        print(f"  {i} {name:18s}  AP50={ap50:.4f}  AP={ap:.4f}")
    return res


def _plot_per_class(res, out_path: Path, title: str):
    ap50 = np.array([float(x) for x in res.box.ap50])
    order = np.argsort(ap50)[::-1]
    names = [VISDRONE_NAMES[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(names)), ap50[order], color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("AP@0.5")
    ax.set_ylim(0, max(0.05, ap50.max() * 1.15))
    ax.set_title(title)
    for b, v in zip(bars, ap50[order]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/visdrone.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="7")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--also_test", action="store_true",
                    help="Also evaluate on test split (test-dev) if labels exist.")
    args = ap.parse_args()

    import os
    os.chdir(REPO_ROOT)

    model = YOLO(args.weights)

    res_val = _eval_split(model, args.data, "val", args.imgsz, args.device, args.batch)
    _plot_per_class(res_val, FIG_DIR / "per_class_ap_val.png",
                    "VisDrone val — per-class AP@0.5")
    # copy ultralytics-generated PR / confusion plots into report/figures
    src_val = REPO_ROOT / "outputs/eval/val"
    for name in ["PR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                 "F1_curve.png", "P_curve.png", "R_curve.png"]:
        s = src_val / name
        if s.exists():
            shutil.copy(s, FIG_DIR / f"val_{name}")

    if args.also_test:
        try:
            res_test = _eval_split(model, args.data, "test", args.imgsz, args.device, args.batch)
            _plot_per_class(res_test, FIG_DIR / "per_class_ap_test.png",
                            "VisDrone test-dev — per-class AP@0.5")
        except Exception as e:
            print(f"[also_test] skipped: {e}")


if __name__ == "__main__":
    main()
