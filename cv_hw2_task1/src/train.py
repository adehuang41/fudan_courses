import argparse
import math
import os
import time

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast

# cuDNN 9 in this env errors with CUDNN_STATUS_NOT_INITIALIZED on A800/driver 570;
# fall back to native CUDA kernels (ResNet-18 still ~4 min / 30 epochs on A800).
torch.backends.cudnn.enabled = False

from .data import build_loaders, NUM_CLASSES
from .models.build import build_model, split_params
from .utils import AverageMeter, accuracy, save_ckpt, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--no_log", action="store_true", help="disable swanlab")
    return p.parse_args()


def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_m, top1_m = AverageMeter(), AverageMeter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        top1, = accuracy(logits, y, topk=(1,))
        bs = y.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(top1, bs)
    return loss_m.avg, top1_m.avg


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    run_name = args.run_name or cfg.get("run_name", os.path.splitext(os.path.basename(args.config))[0])

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_loaders(
        root=cfg.get("data_root", "./data"),
        batch_size=cfg.get("batch_size", 64),
        num_workers=cfg.get("num_workers", 4),
        img_size=cfg.get("img_size", 224),
        val_ratio=cfg.get("val_ratio", 0.1),
        seed=cfg.get("seed", 42),
    )

    model = build_model(cfg["model"], pretrained=cfg.get("pretrained", True),
                        num_classes=NUM_CLASSES,
                        identity_init=cfg.get("identity_init", False)).to(device)

    head_params, bb_params, attn_params = split_params(model, cfg["model"])
    head_lr = cfg["head_lr"]
    bb_lr = cfg.get("backbone_lr", head_lr * 0.1)
    attn_lr = cfg.get("attn_lr", bb_lr)
    weight_decay = cfg.get("weight_decay", 5e-4)
    optim_name = cfg.get("optimizer", "sgd").lower()
    param_groups = [
        {"params": head_params, "lr": head_lr, "name": "head", "base_lr": head_lr},
        {"params": bb_params, "lr": bb_lr, "name": "backbone", "base_lr": bb_lr},
    ]
    if attn_params:
        param_groups.append({"params": attn_params, "lr": attn_lr,
                             "name": "attn", "base_lr": attn_lr})
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9,
                                    weight_decay=weight_decay, nesterov=True)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        raise ValueError(optim_name)

    epochs = cfg["epochs"]
    freeze_backbone_epochs = cfg.get("freeze_backbone_epochs", 0)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * cfg.get("warmup_epochs", 1)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    scaler = GradScaler()

    use_log = not args.no_log
    if use_log:
        try:
            import swanlab
            swanlab.init(project=cfg.get("project", "hw2-task1"),
                         experiment_name=run_name, config=cfg)
        except Exception as e:
            print(f"[warn] swanlab init failed: {e}; continuing without logger")
            use_log = False

    ckpt_dir = os.path.join(cfg.get("ckpt_root", "./checkpoints"), run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_acc = 0.0
    global_step = 0
    t0 = time.time()
    for epoch in range(epochs):
        backbone_frozen = epoch < freeze_backbone_epochs
        for p in bb_params:
            p.requires_grad = not backbone_frozen
        model.train()
        loss_m, top1_m = AverageMeter(), AverageMeter()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            for pg in optimizer.param_groups:
                pg["lr"] = cosine_lr(global_step, total_steps, warmup_steps, pg["base_lr"])
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            top1, = accuracy(logits, y, topk=(1,))
            bs = y.size(0)
            loss_m.update(loss.item(), bs)
            top1_m.update(top1, bs)
            global_step += 1
            if use_log and global_step % 20 == 0:
                import swanlab
                log_d = {
                    "train/loss_step": loss.item(),
                    "train/lr_head": optimizer.param_groups[0]["lr"],
                    "train/lr_backbone": optimizer.param_groups[1]["lr"],
                }
                if len(optimizer.param_groups) > 2:
                    log_d["train/lr_attn"] = optimizer.param_groups[2]["lr"]
                swanlab.log(log_d, step=global_step)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        msg = (f"[{run_name}] epoch {epoch+1}/{epochs} "
               f"train_loss {loss_m.avg:.4f} train_acc {top1_m.avg:.4f} "
               f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} "
               f"freeze_backbone {int(backbone_frozen)} "
               f"({elapsed:.0f}s)")
        print(msg, flush=True)
        if use_log:
            import swanlab
            swanlab.log({
                "train/loss_epoch": loss_m.avg,
                "train/acc_epoch": top1_m.avg,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "train/freeze_backbone": int(backbone_frozen),
                "epoch": epoch + 1,
            }, step=global_step)
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt(os.path.join(ckpt_dir, "best.pt"), model, optimizer, epoch, best_acc,
                      extra={"config": cfg})

    save_ckpt(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epochs - 1, best_acc,
              extra={"config": cfg})

    # Test on best
    state = torch.load(os.path.join(ckpt_dir, "best.pt"), map_location=device)
    model.load_state_dict(state["model"])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"[{run_name}] TEST  loss {test_loss:.4f}  acc {test_acc:.4f}  best_val {best_acc:.4f}",
          flush=True)
    if use_log:
        import swanlab
        swanlab.log({"test/loss": test_loss, "test/acc": test_acc,
                     "best_val_acc": best_acc}, step=global_step)
        swanlab.finish()

    import json
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump({"best_val_acc": best_acc, "test_acc": test_acc, "test_loss": test_loss,
                   "run_name": run_name, "config": cfg}, f, indent=2)


if __name__ == "__main__":
    main()
