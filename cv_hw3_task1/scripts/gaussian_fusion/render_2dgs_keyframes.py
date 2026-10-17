#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected frames from a 2DGS render_path trajectory.")
    parser.add_argument("--repo_dir", default="third_party/2d-gaussian-splatting")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--n_frames", type=int, default=240)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 40, 80, 120, 160, 200, 239])
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    return parser.parse_args()


def add_2dgs_to_path(repo_dir: Path) -> None:
    repo = repo_dir.resolve()
    if not repo.is_dir():
        raise SystemExit(f"2DGS repo not found: {repo}")
    sys.path.insert(0, str(repo))


def make_dataset(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=args.sh_degree,
        source_path=str(Path(args.source_path).resolve()),
        model_path=str(Path(args.model_path).resolve()),
        images="images",
        resolution=args.resolution,
        white_background=args.white_background,
        data_device="cuda",
        eval=False,
        render_items=["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"],
    )


def make_pipe() -> SimpleNamespace:
    return SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        depth_ratio=0.0,
        debug=False,
    )


def make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_width: int = 260) -> None:
    font = ImageFont.load_default()
    loaded = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.width
        thumb_height = int(round(img.height * scale))
        img = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        loaded.append((path, img))

    label_height = 24
    sheet = Image.new("RGB", (thumb_width * len(loaded), loaded[0][1].height + label_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, (path, img) in enumerate(loaded):
        x = index * thumb_width
        sheet.paste(img, (x, 0))
        draw.rectangle((x, 0, x + thumb_width - 1, img.height + label_height - 1), outline=(210, 210, 210))
        label = path.stem
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (thumb_width - (bbox[2] - bbox[0])) // 2, img.height + 5), label, fill=(20, 20, 20), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    add_2dgs_to_path(Path(args.repo_dir))

    from gaussian_renderer import GaussianModel, render
    from scene import Scene
    from utils.render_utils import generate_path

    torch.set_grad_enabled(False)
    dataset = make_dataset(args)
    pipe = make_pipe()
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")
    trajectory = generate_path(scene.getTrainCameras(), n_frames=args.n_frames)

    frames = sorted(set(args.frames))
    for frame in frames:
        if frame < 0 or frame >= len(trajectory):
            raise SystemExit(f"Frame {frame} outside trajectory length {len(trajectory)}")

    output_dir = Path(args.output_dir).resolve()
    renders_dir = output_dir / "renders"
    if renders_dir.exists():
        shutil.rmtree(renders_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for frame in frames:
        print(f"Rendering keyframe {frame}")
        pkg = render(trajectory[frame], gaussians, pipe, background)
        image = torch.clamp(pkg["render"], 0.0, 1.0)
        out_path = renders_dir / f"{frame:05d}.png"
        torchvision.utils.save_image(image, out_path)
        image_paths.append(out_path)

    contact_sheet = output_dir / "keyframes_contact_sheet.png"
    make_contact_sheet(image_paths, contact_sheet)
    summary = {
        "model_path": dataset.model_path,
        "source_path": dataset.source_path,
        "iteration": scene.loaded_iter,
        "n_frames": args.n_frames,
        "frames": frames,
        "renders_dir": str(renders_dir),
        "contact_sheet": str(contact_sheet),
    }
    summary_path = output_dir / "keyframes_render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote contact sheet: {contact_sheet}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
