#!/usr/bin/env python3
"""Remove dark top-face artifacts from a colored triangle mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input colored PLY mesh.")
    parser.add_argument("--output", required=True, help="Output cleaned PLY mesh.")
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="y",
        help="Local axis whose positive end contains the artifact.",
    )
    parser.add_argument(
        "--top-norm",
        type=float,
        default=0.9,
        help="Remove faces whose centroid normalized coordinate on --axis is >= this.",
    )
    parser.add_argument(
        "--max-brightness",
        type=float,
        default=0.45,
        help="Remove only faces whose mean RGB brightness is <= this.",
    )
    parser.add_argument(
        "--keep-largest-component",
        action="store_true",
        help="After filtering, keep the largest connected triangle component.",
    )
    return parser.parse_args()


def compact_mesh(mesh: o3d.geometry.TriangleMesh, face_mask: np.ndarray) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)

    kept_triangles = triangles[face_mask]
    used_vertices = np.unique(kept_triangles.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int64)

    cleaned = o3d.geometry.TriangleMesh()
    cleaned.vertices = o3d.utility.Vector3dVector(vertices[used_vertices])
    cleaned.triangles = o3d.utility.Vector3iVector(remap[kept_triangles])
    if len(colors) == len(vertices):
        cleaned.vertex_colors = o3d.utility.Vector3dVector(colors[used_vertices])
    cleaned.remove_duplicated_vertices()
    cleaned.remove_degenerate_triangles()
    cleaned.remove_duplicated_triangles()
    cleaned.remove_unreferenced_vertices()
    cleaned.compute_vertex_normals()
    return cleaned


def keep_largest_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters_np = np.asarray(clusters)
    counts_np = np.asarray(counts)
    if len(counts_np) == 0:
        return mesh
    largest = int(np.argmax(counts_np))
    return compact_mesh(mesh, clusters_np == largest)


def main() -> None:
    args = parse_args()
    mesh = o3d.io.read_triangle_mesh(args.input)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if len(vertices) == 0 or len(triangles) == 0:
        raise SystemExit("Input mesh has no triangles.")
    if len(colors) != len(vertices):
        raise SystemExit("Input mesh must contain vertex colors.")

    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    centroids = vertices[triangles].mean(axis=1)
    face_colors = colors[triangles].mean(axis=1)
    axis_values = vertices[:, axis_idx]
    coord_norm = (centroids[:, axis_idx] - axis_values.min()) / (np.ptp(axis_values) + 1e-12)
    brightness = face_colors.mean(axis=1)

    artifact = (coord_norm >= args.top_norm) & (brightness <= args.max_brightness)
    keep = ~artifact
    cleaned = compact_mesh(mesh, keep)
    if args.keep_largest_component:
        cleaned = keep_largest_component(cleaned)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), cleaned, write_ascii=False):
        raise SystemExit(f"Failed to write {output}")

    print(f"input_vertices={len(vertices)} input_faces={len(triangles)}")
    print(f"removed_faces={int(artifact.sum())} kept_faces={int(keep.sum())}")
    print(f"output_vertices={len(cleaned.vertices)} output_faces={len(cleaned.triangles)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
