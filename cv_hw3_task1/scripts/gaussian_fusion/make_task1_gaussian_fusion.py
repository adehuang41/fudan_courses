#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBJECT_A_PATH = PROJECT_ROOT / (
    "outputs/reconstruction_2dgs/object_A_book/mesh_candidates_from_2dgs_res384/"
    "book_region_rank00_margin012/top_artifact_cleaning/targeted_cleanup_candidates/"
    "book_A_clip075_combo_top_z88_white_y18.ply"
)
OBJECT_B_PATH = PROJECT_ROOT / (
    "outputs/aigc_assets/final_candidates/object_B_apple_A1b_step1000_thr10/assets/"
    "object_B_apple_A1b_step1000_thr10.obj"
)
OBJECT_C_PATH = PROJECT_ROOT / (
    "object_C_rubiks_cube/"
    "rubiks_cube_try16_try13_resume1000to2000_s008_o004_fovy25@20260612-221639/"
    "save/it2000-export-cleaned/rubiks_cube_try16_step2000_main_component.obj"
)


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    asset_path: Path
    anchor: str
    count: int
    target_dims: tuple[float, float, float] | None
    target_height: float | None
    target_side: float | None
    alpha: float
    color_gain: float
    color_gamma: float
    offset_table: tuple[float, float, float]
    yaw_delta_degrees: float = 0.0
    front_flip: bool = False
    cover_rotate_180: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample final A/B/C meshes into 2DGS Gaussian records and merge with garden D.")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--base_ply", required=True)
    parser.add_argument("--placement_json", required=True)
    parser.add_argument("--output_model_dir", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--object_a_count", type=int, default=50000)
    parser.add_argument("--object_b_count", type=int, default=35000)
    parser.add_argument("--object_c_count", type=int, default=40000)
    parser.add_argument("--object_a_dims", nargs=3, type=float, default=[0.42, 0.70, 0.22])
    parser.add_argument("--object_a_flip_front", action="store_true")
    parser.add_argument("--object_a_cover_rotate_180", action="store_true")
    parser.add_argument("--object_b_height", type=float, default=0.36)
    parser.add_argument("--object_c_side", type=float, default=0.30)
    parser.add_argument("--object_c_yaw_delta_degrees", type=float, default=0.0)
    parser.add_argument("--scale_factor", type=float, default=1.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 1e-12:
        raise ValueError("zero-length vector")
    return v / norm


def logit(value: float) -> float:
    value = min(max(value, 1e-5), 1.0 - 1e-5)
    return math.log(value / (1.0 - value))


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    kx, ky, kz = axis
    k = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


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


def quat_for_tangent_normal(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    n = normalize(normal)
    u = tangent - n * float(np.dot(tangent, n))
    if np.linalg.norm(u) <= 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, n))) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = fallback - n * float(np.dot(fallback, n))
    u = normalize(u)
    v = normalize(np.cross(n, u))
    matrix = np.stack([u, v, n], axis=1)
    return matrix_to_quat_wxyz(matrix)


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    return (rgb - 0.5) / C0


def corrected_rgb(colors: np.ndarray, gain: float, gamma: float) -> np.ndarray:
    rgb = colors[:, :3].astype(np.float32) / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, gamma)
    rgb = np.clip(rgb * gain, 0.0, 1.0)
    return rgb


def copy_model_scaffold(base_model_dir: Path, output_model_dir: Path) -> None:
    output_model_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "input.ply"):
        src = base_model_dir / name
        if src.exists():
            shutil.copy2(src, output_model_dir / name)


def table_basis(entry: dict, yaw_delta_degrees: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # In this 2DGS coordinate frame the fitted surface normal points toward the
    # visible side of the table. Using the opposite sign puts inserted splats
    # behind the tabletop, where the Gaussian renderer correctly occludes them.
    up = normalize(np.asarray(entry["surface_normal"], dtype=np.float64))
    tangent_x = np.asarray(entry["surface_tangent_x"], dtype=np.float64)
    tangent_x = tangent_x - up * float(np.dot(tangent_x, up))
    tangent_x = normalize(tangent_x)
    yaw = float(entry.get("surface_yaw", 0.0)) + math.radians(yaw_delta_degrees)
    if yaw:
        tangent_x = rodrigues(up, yaw) @ tangent_x
    tangent_y = normalize(np.cross(up, tangent_x))
    return tangent_x, tangent_y, up


def placement_surface_point(entry: dict) -> np.ndarray:
    point = entry.get("location_original_surface", entry.get("world_point", entry["location"]))
    return np.asarray(point, dtype=np.float64)


def add_table_offset(point: np.ndarray, basis: tuple[np.ndarray, np.ndarray, np.ndarray], offset: tuple[float, float, float]) -> np.ndarray:
    tx, ty, up = basis
    return point + tx * offset[0] + ty * offset[1] + up * offset[2]


def fit_object_a_local(
    points: np.ndarray,
    bounds: np.ndarray,
    dims: tuple[float, float, float],
    cover_rotate_180: bool = False,
) -> np.ndarray:
    min_v, max_v = bounds
    extent = np.maximum(max_v - min_v, 1e-6)
    width, height, thickness = dims
    scale = np.asarray([width / extent[0], height / extent[1], thickness / extent[2]], dtype=np.float64)
    center = (min_v + max_v) * 0.5
    out = np.empty_like(points, dtype=np.float64)
    if cover_rotate_180:
        out[:, 0] = (center[0] - points[:, 0]) * scale[0]
        out[:, 1] = (max_v[1] - points[:, 1]) * scale[1]
    else:
        out[:, 0] = (points[:, 0] - center[0]) * scale[0]
        out[:, 1] = (points[:, 1] - min_v[1]) * scale[1]
    out[:, 2] = (points[:, 2] - center[2]) * scale[2]
    return out


def object_a_basis(entry: dict, base: np.ndarray, up: np.ndarray, front_flip: bool = False) -> np.ndarray:
    to_camera = np.asarray(entry["camera_center"], dtype=np.float64) - base
    cover_normal = to_camera - up * float(np.dot(to_camera, up))
    if np.linalg.norm(cover_normal) <= 1e-8:
        cover_normal = np.asarray(entry["surface_tangent_y"], dtype=np.float64)
    cover_normal = normalize(cover_normal)
    yaw = float(entry.get("surface_yaw", 0.0))
    if yaw:
        cover_normal = rodrigues(up, yaw) @ cover_normal
    if front_flip:
        cover_normal = -cover_normal
    local_y = up
    local_z = normalize(cover_normal)
    local_x = normalize(np.cross(local_y, local_z))
    return np.stack([local_x, local_y, local_z], axis=1)


def fit_upright_local(points: np.ndarray, bounds: np.ndarray, target_height: float | None, target_side: float | None) -> np.ndarray:
    min_v, max_v = bounds
    extents = np.maximum(max_v - min_v, 1e-6)
    if target_height is not None:
        scale = target_height / extents[2]
    elif target_side is not None:
        scale = target_side / float(np.max(extents))
    else:
        raise ValueError("Need target_height or target_side")
    center_xy = (min_v[:2] + max_v[:2]) * 0.5
    out = np.empty_like(points, dtype=np.float64)
    out[:, 0] = (points[:, 0] - center_xy[0]) * scale
    out[:, 1] = (points[:, 1] - center_xy[1]) * scale
    out[:, 2] = (points[:, 2] - min_v[2]) * scale
    return out


def transformed_triangles(mesh: trimesh.Trimesh, face_index: np.ndarray, transform_fn) -> np.ndarray:
    triangles = mesh.triangles[face_index].reshape(-1, 3)
    return transform_fn(triangles).reshape(-1, 3, 3)


def triangle_tangents_normals(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge0, edge1)
    normal_norm = np.linalg.norm(normals, axis=1)
    bad = normal_norm <= 1e-10
    normal_norm[bad] = 1.0
    normals = normals / normal_norm[:, None]
    tangent_norm = np.linalg.norm(edge0, axis=1)
    tangent_norm[tangent_norm <= 1e-10] = 1.0
    tangents = edge0 / tangent_norm[:, None]
    return tangents, normals


def total_area(triangles: np.ndarray) -> float:
    edge0 = triangles[:, 1] - triangles[:, 0]
    edge1 = triangles[:, 2] - triangles[:, 0]
    return float(np.linalg.norm(np.cross(edge0, edge1), axis=1).sum() * 0.5)


def records_from_samples(
    dtype: np.dtype,
    name: str,
    positions: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    rgb: np.ndarray,
    area: float,
    alpha: float,
    scale_factor: float,
) -> np.ndarray:
    fields = dtype.names or ()
    count = len(positions)
    records = np.zeros(count, dtype=dtype)
    radius = max(math.sqrt(max(area, 1e-8) / max(count, 1)) * scale_factor, 0.0025)
    sh = rgb_to_sh(rgb)
    opacity = logit(alpha)
    scale_log = math.log(radius)

    records["x"] = positions[:, 0].astype(np.float32)
    records["y"] = positions[:, 1].astype(np.float32)
    records["z"] = positions[:, 2].astype(np.float32)
    if "f_dc_0" in fields:
        records["f_dc_0"] = sh[:, 0].astype(np.float32)
        records["f_dc_1"] = sh[:, 1].astype(np.float32)
        records["f_dc_2"] = sh[:, 2].astype(np.float32)
    if "opacity" in fields:
        records["opacity"] = np.float32(opacity)
    if "scale_0" in fields:
        records["scale_0"] = np.float32(scale_log)
        records["scale_1"] = np.float32(scale_log)
    if "rot_0" in fields:
        quats = np.stack([quat_for_tangent_normal(t, n) for t, n in zip(tangents, normals)], axis=0)
        records["rot_0"] = quats[:, 0]
        records["rot_1"] = quats[:, 1]
        records["rot_2"] = quats[:, 2]
        records["rot_3"] = quats[:, 3]

    print(f"{name}: {count} splats, area={area:.4f}, radius={radius:.5f}")
    return records


def sample_object(spec: ObjectSpec, placement: dict, base_dtype: np.dtype, seed: int, scale_factor: float) -> tuple[np.ndarray, dict]:
    entry = placement["placements"][spec.anchor]
    mesh = trimesh.load(spec.asset_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{spec.name} did not load as Trimesh: {spec.asset_path}")

    points, face_index, colors = trimesh.sample.sample_surface(mesh, spec.count, sample_color=True, seed=seed)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)

    if spec.name == "object_A":
        basis_table = table_basis(entry)
        _, _, up = basis_table
        base = add_table_offset(placement_surface_point(entry), basis_table, spec.offset_table)
        basis = object_a_basis(entry, base, up, front_flip=spec.front_flip)

        def transform_fn(p: np.ndarray) -> np.ndarray:
            local = fit_object_a_local(
                p,
                bounds,
                spec.target_dims or (0.42, 0.70, 0.22),
                cover_rotate_180=spec.cover_rotate_180,
            )
            return base + (local @ basis.T)

    else:
        basis_table = table_basis(entry, spec.yaw_delta_degrees)
        base = add_table_offset(placement_surface_point(entry), basis_table, spec.offset_table)
        axis_x, axis_y, up = basis_table
        basis = np.stack([axis_x, axis_y, up], axis=1)

        def transform_fn(p: np.ndarray) -> np.ndarray:
            local = fit_upright_local(p, bounds, spec.target_height, spec.target_side)
            return base + (local @ basis.T)

    positions = transform_fn(points)
    triangles_world = transformed_triangles(mesh, face_index, transform_fn)
    tangents, normals = triangle_tangents_normals(triangles_world)
    area = total_area(transform_fn(mesh.triangles.reshape(-1, 3)).reshape(-1, 3, 3))
    rgb = corrected_rgb(colors, gain=spec.color_gain, gamma=spec.color_gamma)
    records = records_from_samples(base_dtype, spec.name, positions, tangents, normals, rgb, area, spec.alpha, scale_factor)
    summary = {
        "name": spec.name,
        "asset_path": str(spec.asset_path),
        "anchor": spec.anchor,
        "count": spec.count,
        "target_dims": spec.target_dims,
        "target_height": spec.target_height,
        "target_side": spec.target_side,
        "alpha": spec.alpha,
        "color_gain": spec.color_gain,
        "color_gamma": spec.color_gamma,
        "offset_table": spec.offset_table,
        "yaw_delta_degrees": spec.yaw_delta_degrees,
        "front_flip": spec.front_flip,
        "cover_rotate_180": spec.cover_rotate_180,
        "world_bbox": [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        "area": area,
    }
    return records, summary


def make_specs(args: argparse.Namespace) -> list[ObjectSpec]:
    return [
        ObjectSpec(
            name="object_A",
            asset_path=OBJECT_A_PATH,
            anchor="object_A",
            count=args.object_a_count,
            target_dims=tuple(float(v) for v in args.object_a_dims),
            target_height=None,
            target_side=None,
            alpha=0.90,
            color_gain=1.15,
            color_gamma=0.90,
            offset_table=(0.0, 0.0, 0.010),
            front_flip=bool(args.object_a_flip_front),
            cover_rotate_180=bool(args.object_a_cover_rotate_180),
        ),
        ObjectSpec(
            name="object_B",
            asset_path=OBJECT_B_PATH,
            anchor="object_B_anchor",
            count=args.object_b_count,
            target_dims=None,
            target_height=float(args.object_b_height),
            target_side=None,
            alpha=0.92,
            color_gain=2.10,
            color_gamma=0.78,
            offset_table=(0.0, 0.025, 0.010),
        ),
        ObjectSpec(
            name="object_C",
            asset_path=OBJECT_C_PATH,
            anchor="object_C_anchor",
            count=args.object_c_count,
            target_dims=None,
            target_height=None,
            target_side=float(args.object_c_side),
            alpha=0.92,
            color_gain=1.45,
            color_gamma=0.85,
            offset_table=(0.0, 0.045, 0.010),
            yaw_delta_degrees=float(args.object_c_yaw_delta_degrees),
        ),
    ]


def main() -> None:
    args = parse_args()
    base_model_dir = Path(args.base_model_dir).resolve()
    base_ply = Path(args.base_ply).resolve()
    output_model_dir = Path(args.output_model_dir).resolve()
    placement_json = Path(args.placement_json).resolve()
    out_ply = output_model_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if out_ply.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {out_ply}. Use --overwrite or new output dir.")

    placement = json.loads(placement_json.read_text(encoding="utf-8"))
    copy_model_scaffold(base_model_dir, output_model_dir)

    print(f"Reading base Gaussian PLY: {base_ply}")
    ply = PlyData.read(base_ply)
    base_vertices = ply["vertex"].data
    object_records = []
    object_summaries = []
    for index, spec in enumerate(make_specs(args)):
        records, summary = sample_object(spec, placement, base_vertices.dtype, seed=args.seed + index, scale_factor=args.scale_factor)
        object_records.append(records)
        object_summaries.append(summary)

    append_count = sum(len(records) for records in object_records)
    merged = np.empty(len(base_vertices) + append_count, dtype=base_vertices.dtype)
    merged[: len(base_vertices)] = base_vertices
    cursor = len(base_vertices)
    for records in object_records:
        merged[cursor : cursor + len(records)] = records
        cursor += len(records)

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing merged Gaussian PLY: {out_ply}")
    PlyData([PlyElement.describe(merged, "vertex")], text=False).write(out_ply)

    metadata = {
        "base_model_dir": str(base_model_dir),
        "base_ply": str(base_ply),
        "output_ply": str(out_ply),
        "placement_json": str(placement_json),
        "base_vertex_count": int(len(base_vertices)),
        "added_vertex_count": int(append_count),
        "merged_vertex_count": int(len(merged)),
        "seed": args.seed,
        "scale_factor": args.scale_factor,
        "encoding": {
            "f_dc": "(rgb - 0.5) / 0.28209479177387814",
            "opacity": "logit(alpha), consumed through sigmoid",
            "scale": "log(world_radius), consumed through exp",
            "rotation": "WXYZ quaternion, normalized by renderer",
            "f_rest": "zero, leaving only DC color for inserted assets",
        },
        "objects": object_summaries,
    }
    metadata_path = output_model_dir / "metadata" / "task1_gaussian_fusion.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
