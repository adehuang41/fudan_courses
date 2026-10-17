import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a colored triangle mesh by face color.")
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_mesh", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--keep_dark_red", action="store_true")
    return parser.parse_args()


def mesh_from_triangle_mask(mesh, triangle_mask: np.ndarray):
    candidate = copy.deepcopy(mesh)
    candidate.remove_triangles_by_mask(~triangle_mask)
    candidate.remove_unreferenced_vertices()
    candidate.remove_degenerate_triangles()
    candidate.remove_duplicated_triangles()
    candidate.remove_duplicated_vertices()
    return candidate


def mesh_stats(mesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    stats = {"vertices": int(len(vertices)), "triangles": int(len(faces))}
    if len(vertices):
        stats["bbox_min"] = vertices.min(axis=0).tolist()
        stats["bbox_max"] = vertices.max(axis=0).tolist()
        stats["extent"] = (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
    if len(colors):
        stats["mean_color"] = colors.mean(axis=0).tolist()
    return stats


def main() -> None:
    args = parse_args()
    input_mesh = Path(args.input_mesh)
    output_mesh = Path(args.output_mesh)
    summary_path = Path(args.summary)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(input_mesh))
    colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)
    face_colors = colors[faces].mean(axis=1)
    r, g, b = face_colors[:, 0], face_colors[:, 1], face_colors[:, 2]
    brightness = face_colors.mean(axis=1)
    saturation = face_colors.max(axis=1) - face_colors.min(axis=1)

    red = (r > 0.30) & (r > 1.20 * g) & (r > 1.08 * b)
    cream = (
        (r > 0.58)
        & (g > 0.48)
        & (b > 0.36)
        & (r < 1.35 * g)
        & (g < 1.35 * b)
    )
    dark_red = (
        (r > 0.18)
        & (r > 1.12 * g)
        & (r > 1.08 * b)
        & (brightness < 0.50)
        & (saturation > 0.06)
    )
    keep = red | cream
    if args.keep_dark_red:
        keep |= dark_red

    filtered = mesh_from_triangle_mask(mesh, keep)
    o3d.io.write_triangle_mesh(str(output_mesh), filtered)

    summary = {
        "input_mesh": str(input_mesh),
        "output_mesh": str(output_mesh),
        "keep_dark_red": bool(args.keep_dark_red),
        "input": mesh_stats(mesh),
        "output": mesh_stats(filtered),
        "face_counts": {
            "total": int(len(faces)),
            "red": int(red.sum()),
            "cream": int(cream.sum()),
            "dark_red": int(dark_red.sum()),
            "kept": int(keep.sum()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output_mesh}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
