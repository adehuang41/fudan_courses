import argparse
import copy
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a full Object A TSDF mesh around a red book-shell candidate bbox."
    )
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--candidate_summary", required=True)
    parser.add_argument("--candidate_rank", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--margin", type=float, default=0.12)
    parser.add_argument("--top_components", type=int, default=6)
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
    stats = {"vertices": int(len(vertices)), "triangles": int(len(triangles))}
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.candidate_summary).read_text())
    candidate = next(
        item for item in summary["candidates"] if item["rank"] == args.candidate_rank
    )
    bbox_min = np.asarray(candidate["bbox_min"], dtype=np.float64) - args.margin
    bbox_max = np.asarray(candidate["bbox_max"], dtype=np.float64) + args.margin

    mesh = o3d.io.read_triangle_mesh(args.input_mesh)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    centroids = vertices[faces].mean(axis=1)
    inside = np.all((centroids >= bbox_min) & (centroids <= bbox_max), axis=1)
    crop = mesh_from_triangles(mesh, np.where(inside)[0])
    crop_path = output_dir / f"book_region_rank{args.candidate_rank:02d}_margin{args.margin:.2f}.ply"
    o3d.io.write_triangle_mesh(str(crop_path), crop)

    clusters, cluster_n_triangles, _cluster_area = crop.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    order = np.argsort(-cluster_n_triangles)[: args.top_components]
    crop_faces = np.asarray(crop.triangles)

    payload = {
        "input_mesh": args.input_mesh,
        "candidate_summary": args.candidate_summary,
        "candidate_rank": args.candidate_rank,
        "margin": args.margin,
        "expanded_bbox_min": bbox_min.tolist(),
        "expanded_bbox_max": bbox_max.tolist(),
        "crop_path": str(crop_path),
        "crop": mesh_stats(crop),
        "components": [],
    }

    for rank, cluster_id in enumerate(order):
        tri_idx = np.where(clusters == cluster_id)[0]
        component = mesh_from_triangles(crop, tri_idx)
        path = output_dir / f"book_region_component_{rank:02d}_tri{len(tri_idx):06d}.ply"
        o3d.io.write_triangle_mesh(str(path), component)
        payload["components"].append(
            {
                "rank": int(rank),
                "cluster_id": int(cluster_id),
                "triangle_count_before_cleanup": int(len(tri_idx)),
                "path": str(path),
                **mesh_stats(component),
                "crop_vertex_count_before_cleanup": int(
                    len(np.unique(crop_faces[tri_idx].reshape(-1)))
                ),
            }
        )

    summary_path = output_dir / "book_region_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
