import argparse
import json
import copy
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract red book-like connected mesh candidates from Object A TSDF mesh."
    )
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--red_min", type=float, default=0.30)
    parser.add_argument("--red_green_ratio", type=float, default=1.25)
    parser.add_argument("--red_blue_ratio", type=float, default=1.10)
    return parser.parse_args()


def mesh_from_triangles(mesh, triangle_indices: np.ndarray):
    candidate = copy.deepcopy(mesh)
    keep = np.zeros(len(candidate.triangles), dtype=bool)
    keep[triangle_indices] = True
    candidate.remove_triangles_by_mask(~keep)
    candidate.remove_unreferenced_vertices()
    candidate.remove_degenerate_triangles()
    candidate.remove_duplicated_triangles()
    candidate.remove_duplicated_vertices()
    return candidate


def mesh_stats(mesh) -> dict:
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    triangles = np.asarray(mesh.triangles)
    stats = {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
    }
    if len(vertices):
        stats.update(
            {
                "bbox_min": vertices.min(axis=0).tolist(),
                "bbox_max": vertices.max(axis=0).tolist(),
                "extent": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
            }
        )
    if len(colors):
        stats["mean_color"] = colors.mean(axis=0).tolist()
    return stats


def main() -> None:
    args = parse_args()
    input_mesh = Path(args.input_mesh)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(str(input_mesh))
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors)
    faces = np.asarray(mesh.triangles)
    if len(colors) != len(vertices):
        raise SystemExit("Input mesh has no per-vertex colors.")

    face_colors = colors[faces].mean(axis=1)
    r, g, b = face_colors[:, 0], face_colors[:, 1], face_colors[:, 2]
    red_face_mask = (
        (r > args.red_min)
        & (r > args.red_green_ratio * g)
        & (r > args.red_blue_ratio * b)
    )

    red_mesh = mesh_from_triangles(mesh, np.where(red_face_mask)[0])
    clusters, cluster_n_triangles, _cluster_area = red_mesh.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    order = np.argsort(-cluster_n_triangles)[: args.top_k]

    red_faces = np.asarray(red_mesh.triangles)
    summary = {
        "input_mesh": str(input_mesh),
        "red_face_count": int(red_face_mask.sum()),
        "red_mesh": mesh_stats(red_mesh),
        "candidates": [],
    }

    o3d.io.write_triangle_mesh(str(output_dir / "red_filtered_mesh.ply"), red_mesh)

    for rank, cluster_id in enumerate(order):
        triangle_indices = np.where(clusters == cluster_id)[0]
        candidate = mesh_from_triangles(red_mesh, triangle_indices)
        path = output_dir / f"red_cluster_{rank:02d}_tri{len(triangle_indices):06d}.ply"
        o3d.io.write_triangle_mesh(str(path), candidate)
        item = {
            "rank": int(rank),
            "cluster_id": int(cluster_id),
            "triangle_count_before_cleanup": int(len(triangle_indices)),
            "path": str(path),
            **mesh_stats(candidate),
        }
        if len(red_faces):
            original_vertex_ids = np.unique(red_faces[triangle_indices].reshape(-1))
            item["red_mesh_vertex_count_before_cleanup"] = int(len(original_vertex_ids))
        summary["candidates"].append(item)

    summary_path = output_dir / "mesh_candidates_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
