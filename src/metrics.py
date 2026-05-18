"""Confusion-matrix based segmentation metrics: per-class IoU, mIoU, pixel acc."""
from __future__ import annotations

import torch


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def reset(self):
        self.mat.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred & target: (B, H, W) int64, both on same device
        k = (target >= 0) & (target < self.num_classes)
        idx = self.num_classes * target[k].to(torch.long) + pred[k].to(torch.long)
        bins = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += bins.view(self.num_classes, self.num_classes).to(self.mat.device)

    def compute(self) -> dict:
        m = self.mat.float()
        tp = m.diag()
        fp = m.sum(dim=0) - tp
        fn = m.sum(dim=1) - tp
        denom = tp + fp + fn
        iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
        valid = denom > 0
        miou = iou[valid].mean().item() if valid.any() else 0.0
        pixel_acc = (tp.sum() / m.sum().clamp(min=1)).item()
        return {
            "miou": miou,
            "pixel_acc": pixel_acc,
            "iou_per_class": iou.tolist(),
        }
