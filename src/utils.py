import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        out.append(correct[:k].reshape(-1).float().sum(0).item() / target.size(0))
    return out


def save_ckpt(path: str, model, optimizer, epoch: int, best_acc: float, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
