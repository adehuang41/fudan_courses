"""Strip optimizer state from best.pt and bundle all runs into one tarball."""
import json
import os
import tarfile
import tempfile
import torch

CKPT_ROOT = "checkpoints"
OUT_DIR = "release"
RUNS = sorted(d for d in os.listdir(CKPT_ROOT)
              if d.startswith("E") and os.path.isfile(os.path.join(CKPT_ROOT, d, "best.pt")))

os.makedirs(OUT_DIR, exist_ok=True)
manifest = {}

with tempfile.TemporaryDirectory() as tmp:
    for run in RUNS:
        src = os.path.join(CKPT_ROOT, run, "best.pt")
        dst = os.path.join(tmp, f"{run}.pt")
        state = torch.load(src, map_location="cpu", weights_only=False)
        slim = {"model": state["model"], "config": state.get("config"),
                "best_val_acc": state.get("best_acc")}
        torch.save(slim, dst)
        manifest[run] = {
            "best_val_acc": state.get("best_acc"),
            "size_mb": round(os.path.getsize(dst) / 1024 / 1024, 1),
        }
        # also copy metrics.json
        m_src = os.path.join(CKPT_ROOT, run, "metrics.json")
        if os.path.isfile(m_src):
            with open(m_src) as f, open(os.path.join(tmp, f"{run}.metrics.json"), "w") as g:
                g.write(f.read())
    with open(os.path.join(tmp, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    tar_path = os.path.join(OUT_DIR, "hw2_task1_weights.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for fn in sorted(os.listdir(tmp)):
            tar.add(os.path.join(tmp, fn), arcname=fn)
    print(f"Wrote {tar_path} ({os.path.getsize(tar_path)/1024/1024:.1f} MB)")
print(json.dumps(manifest, indent=2))
