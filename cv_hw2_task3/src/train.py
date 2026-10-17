"""From-scratch U-Net training loop with AMP, cosine LR, and swanlab logging."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import yaml
from torch.cuda.amp import GradScaler, autocast

# cuDNN 9 errors with CUDNN_STATUS_NOT_INITIALIZED on this A800/driver 570 host;
# fall back to native CUDA kernels (mirrors hw2_task1).
torch.backends.cudnn.enabled = False

from .data import NUM_CLASSES, CLASS_NAMES, build_loaders
from .losses import build_loss
from .metrics import ConfusionMatrix
from .unet import UNet
from .utils import AverageMeter, save_ckpt, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no_log", action="store_true")
    return p.parse_args()


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes):
    model.eval()
    loss_m = AverageMeter()
    cm = ConfusionMatrix(num_classes)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        cm.update(pred.cpu(), y.cpu())
        loss_m.update(loss.item(), y.size(0))
    metrics = cm.compute()
    metrics["loss"] = loss_m.avg
    return metrics


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    run_name = args.run_name or cfg.get("run_name",
                                        os.path.splitext(os.path.basename(args.config))[0])

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders(
        root=cfg.get("data_root", "./data"),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 4),
        img_size=cfg.get("img_size", 256),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )

    model = UNet(in_channels=3, num_classes=NUM_CLASSES,
                 base_ch=cfg.get("base_ch", 64)).to(device)

    base_lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)
    optim_name = cfg.get("optimizer", "adamw").lower()
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)
    elif optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9,
                                    weight_decay=wd, nesterov=True)
    else:
        raise ValueError(optim_name)

    epochs = cfg["epochs"]
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * cfg.get("warmup_epochs", 2)

    criterion = build_loss(cfg, num_classes=NUM_CLASSES).to(device)
    scaler = GradScaler()

    use_log = not args.no_log
    if use_log:
        try:
            import swanlab
            swanlab.init(project=cfg.get("project", "hw2-task3"),
                         experiment_name=run_name, config=cfg)
        except Exception as e:
            print(f"[warn] swanlab init failed: {e}; continuing without logger")
            use_log = False

    ckpt_dir = os.path.join(cfg.get("ckpt_root", "./checkpoints"), run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_miou = 0.0
    best_state = None
    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        loss_m = AverageMeter()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for pg in optimizer.param_groups:
                pg["lr"] = cosine_lr(global_step, total_steps, warmup_steps, base_lr)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_m.update(loss.item(), y.size(0))
            global_step += 1
            if use_log and global_step % 20 == 0:
                import swanlab
                swanlab.log({
                    "train/loss_step": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

        val = evaluate(model, val_loader, device, criterion, NUM_CLASSES)
        elapsed = time.time() - t0
        per_cls = " ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, val["iou_per_class"]))
        print(f"[{run_name}] ep {epoch+1}/{epochs} "
              f"train_loss {loss_m.avg:.4f} | val_loss {val['loss']:.4f} "
              f"miou {val['miou']:.4f} pacc {val['pixel_acc']:.4f} "
              f"({per_cls}) ({elapsed:.0f}s)", flush=True)

        if use_log:
            import swanlab
            log_d = {
                "train/loss_epoch": loss_m.avg,
                "val/loss": val["loss"],
                "val/miou": val["miou"],
                "val/pixel_acc": val["pixel_acc"],
                "epoch": epoch + 1,
            }
            for n, v in zip(CLASS_NAMES, val["iou_per_class"]):
                log_d[f"val/iou_{n}"] = v
            swanlab.log(log_d, step=global_step)

        if val["miou"] > best_miou:
            best_miou = val["miou"]
            best_state = {"epoch": epoch + 1, **val}
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer,
                      epoch, best_miou, extra={"config": cfg, "val_metrics": val})

    save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, optimizer,
              epochs - 1, best_miou, extra={"config": cfg})

    # Final test eval on best checkpoint
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(state["model"])
    test = evaluate(model, test_loader, device, criterion, NUM_CLASSES)
    per_cls_test = " ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, test["iou_per_class"]))
    print(f"[{run_name}] TEST loss {test['loss']:.4f} miou {test['miou']:.4f} "
          f"pacc {test['pixel_acc']:.4f} ({per_cls_test}) | best_val_miou {best_miou:.4f}",
          flush=True)

    if use_log:
        import swanlab
        log_d = {
            "test/loss": test["loss"],
            "test/miou": test["miou"],
            "test/pixel_acc": test["pixel_acc"],
            "best_val_miou": best_miou,
        }
        for n, v in zip(CLASS_NAMES, test["iou_per_class"]):
            log_d[f"test/iou_{n}"] = v
        swanlab.log(log_d, step=global_step)
        swanlab.finish()

    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump({
            "run_name": run_name,
            "best_val": best_state,
            "test": test,
            "config": cfg,
        }, f, indent=2)


if __name__ == "__main__":
    main()
