import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.render_utils import generate_path


def camera_record(index, camera):
    world_view = camera.world_view_transform.detach().cpu().numpy()
    full_proj = camera.full_proj_transform.detach().cpu().numpy()
    c2w = np.linalg.inv(world_view.T)
    return {
        "id": index,
        "image_width": int(camera.image_width),
        "image_height": int(camera.image_height),
        "FoVx": float(camera.FoVx),
        "FoVy": float(camera.FoVy),
        "camera_center": [float(v) for v in camera.camera_center.detach().cpu().numpy()],
        "world_view_transform": world_view.tolist(),
        "full_proj_transform": full_proj.tolist(),
        "camera_to_world": c2w.tolist(),
    }


def main() -> None:
    parser = ArgumentParser(description="Export the same 2DGS render_path camera trajectory used by render.py.")
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--n_frames", default=240, type=int)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    trajectory = generate_path(scene.getTrainCameras(), n_frames=args.n_frames)

    output = {
        "model_path": str(Path(args.model_path).resolve()),
        "source_path": str(Path(args.source_path).resolve()),
        "iteration": scene.loaded_iter,
        "n_frames": args.n_frames,
        "note": "Generated with utils.render_utils.generate_path(scene.getTrainCameras()). Use with 2DGS render_path background frames.",
        "cameras": [camera_record(index, camera) for index, camera in enumerate(trajectory)],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
