#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a bright synthetic 2DGS surfel cube to a trained 2DGS Gaussian PLY."
    )
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_ply", required=True)
    parser.add_argument("--placement_json", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--anchor", default="object_C_anchor")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--side", type=float, default=0.32)
    parser.add_argument("--grid", type=int, default=22)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--color", nargs=3, type=float, default=[1.0, 0.0, 1.0])
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 0:
        raise ValueError("zero-length vector")
    return v / norm


def logit(value: float) -> float:
    value = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def quat_for_axes(axis_u: np.ndarray, axis_v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    u = normalize(axis_u)
    v = normalize(axis_v)
    n = normalize(normal)
    matrix = np.stack([u, v, n], axis=1)
    if np.linalg.det(matrix) < 0:
        matrix[:, 1] *= -1.0
    return matrix_to_quat_wxyz(matrix)


def load_basis(placement_json: Path, anchor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(placement_json.read_text(encoding="utf-8"))
    placement = data["placements"][anchor]
    surface_point = np.asarray(placement["world_point"], dtype=np.float32)
    tangent_x = normalize(np.asarray(placement["surface_tangent_x"], dtype=np.float32))
    tangent_y = normalize(np.asarray(placement["surface_tangent_y"], dtype=np.float32))
    up = normalize(-np.asarray(placement["surface_normal"], dtype=np.float32))
    return surface_point, tangent_x, tangent_y, up


def make_cube_surfel_records(
    dtype: np.dtype,
    surface_point: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_z: np.ndarray,
    side: float,
    grid: int,
    color: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if grid < 2:
        raise ValueError("--grid must be >= 2")

    fields = dtype.names or ()
    records = np.zeros(grid * grid * 6, dtype=dtype)
    half = side * 0.5
    center = surface_point + axis_z * half
    coords = np.linspace(-half, half, grid, dtype=np.float32)
    spacing = side / float(grid - 1)
    scale = max(spacing * 0.68, 1e-5)
    sh = (color - 0.5) / C0

    faces = [
        (axis_z, axis_x, axis_y, axis_z * half),
        (-axis_z, axis_x, axis_y, -axis_z * half),
        (axis_x, axis_y, axis_z, axis_x * half),
        (-axis_x, axis_y, axis_z, -axis_x * half),
        (axis_y, axis_x, axis_z, axis_y * half),
        (-axis_y, axis_x, axis_z, -axis_y * half),
    ]

    out_index = 0
    for normal, u_axis, v_axis, offset in faces:
        quat = quat_for_axes(u_axis, v_axis, normal)
        for u in coords:
            for v in coords:
                p = center + offset + u_axis * u + v_axis * v
                rec = records[out_index]
                rec["x"], rec["y"], rec["z"] = p
                if "f_dc_0" in fields:
                    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = sh
                if "opacity" in fields:
                    rec["opacity"] = logit(alpha)
                if "scale_0" in fields:
                    rec["scale_0"] = math.log(scale)
                    rec["scale_1"] = math.log(scale)
                if "rot_0" in fields:
                    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = quat
                out_index += 1

    return records


def copy_model_scaffold(base_model_dir: Path, output_model_dir: Path) -> None:
    output_model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = base_model_dir / name
        if src.exists():
            shutil.copy2(src, output_model_dir / name)


def main() -> None:
    args = parse_args()
    base_model_dir = Path(args.base_model_dir).resolve()
    base_ply = Path(args.base_ply).resolve()
    output_model_dir = Path(args.output_model_dir).resolve()
    placement_json = Path(args.placement_json).resolve()

    copy_model_scaffold(base_model_dir, output_model_dir)
    surface_point, axis_x, axis_y, axis_z = load_basis(placement_json, args.anchor)

    print(f"Reading base Gaussian PLY: {base_ply}")
    ply = PlyData.read(base_ply)
    base_vertices = ply["vertex"].data
    synthetic = make_cube_surfel_records(
        base_vertices.dtype,
        surface_point=surface_point,
        axis_x=axis_x,
        axis_y=axis_y,
        axis_z=axis_z,
        side=args.side,
        grid=args.grid,
        color=np.asarray(args.color, dtype=np.float32),
        alpha=args.alpha,
    )

    merged = np.empty(len(base_vertices) + len(synthetic), dtype=base_vertices.dtype)
    merged[: len(base_vertices)] = base_vertices
    merged[len(base_vertices) :] = synthetic

    out_ply = output_model_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing merged Gaussian PLY: {out_ply}")
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(out_ply)

    metadata = {
        "base_model_dir": str(base_model_dir),
        "base_ply": str(base_ply),
        "output_ply": str(out_ply),
        "placement_json": str(placement_json),
        "anchor": args.anchor,
        "base_vertex_count": int(len(base_vertices)),
        "synthetic_vertex_count": int(len(synthetic)),
        "merged_vertex_count": int(len(merged)),
        "side": args.side,
        "grid": args.grid,
        "alpha": args.alpha,
        "color_rgb": args.color,
        "encoding": {
            "f_dc": "(rgb - 0.5) / 0.28209479177387814",
            "opacity": "logit(alpha), consumed through sigmoid",
            "scale": "log(world_radius), consumed through exp",
            "rotation": "WXYZ quaternion, normalized by renderer",
        },
        "surface_point": surface_point.tolist(),
        "basis": {
            "axis_x": axis_x.tolist(),
            "axis_y": axis_y.tolist(),
            "axis_z_up": axis_z.tolist(),
        },
    }
    metadata_path = output_model_dir / "metadata" / "synthetic_merge.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
