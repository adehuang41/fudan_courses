"""Hand-written multi-class Dice loss + combined CE/weighted-CE + Dice loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropy(nn.Module):
    def __init__(self, ce_weight: torch.Tensor | None = None):
        super().__init__()
        if ce_weight is not None:
            self.register_buffer("ce_weight", ce_weight)
        else:
            self.ce_weight = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.ce_weight)


class DiceLoss(nn.Module):
    """Multi-class soft-Dice loss.

    Args:
        num_classes: total class count (one-hot dimension).
        eps: numerical stabilizer for Dice ratio.
        ignore_index: class index to drop from numerator/denominator (e.g. void),
            default None (all classes participate).
    """

    def __init__(self, num_classes: int, eps: float = 1e-6, ignore_index: int | None = None):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W), target: (B, H, W) long
        prob = F.softmax(logits, dim=1)
        target_oh = F.one_hot(target.clamp(0, self.num_classes - 1),
                              num_classes=self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dims = (0, 2, 3)
        inter = (prob * target_oh).sum(dims)
        denom = prob.sum(dims) + target_oh.sum(dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)  # (C,)

        if self.ignore_index is not None:
            mask = torch.ones_like(dice, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice = dice[mask]
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """alpha * CE + beta * Dice."""

    def __init__(self, num_classes: int, alpha: float = 1.0, beta: float = 1.0,
                 ignore_index: int | None = None,
                 ce_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = WeightedCrossEntropy(ce_weight=ce_weight)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.alpha * self.ce(logits, target) + self.beta * self.dice(logits, target)


def _ce_weight_from_cfg(cfg: dict) -> torch.Tensor | None:
    weights = cfg.get("class_weights")
    if weights is None:
        return None
    t = torch.tensor(weights, dtype=torch.float32)
    if t.ndim != 1:
        raise ValueError("class_weights must be a 1D list of floats")
    return t


def build_loss(cfg: dict, num_classes: int) -> nn.Module:
    """Factory keyed by cfg['loss'] in {'ce','dice','combined'}."""
    name = cfg.get("loss", "ce").lower()
    ce_weight = _ce_weight_from_cfg(cfg)
    if name == "ce":
        return WeightedCrossEntropy(ce_weight=ce_weight)
    if name == "dice":
        return DiceLoss(num_classes=num_classes)
    if name == "combined":
        return CombinedLoss(num_classes=num_classes,
                            alpha=cfg.get("alpha", 1.0),
                            beta=cfg.get("beta", 1.0),
                            ce_weight=ce_weight)
    raise ValueError(f"Unknown loss: {name}")
