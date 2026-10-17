#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODEL_TXT_DIR = Path("outputs/reconstruction_2dgs/object_A_book/colmap_full/sparse_txt")
DEFAULT_OUTPUT_DIR = Path("outputs/reconstruction_2dgs/object_A_book/colmap_full")


@dataclass
class RegisteredImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    frame_index: int | None
    center: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize camera trajectory from a COLMAP TXT sparse model.")
    parser.add_argument("--model_txt_dir", default=str(DEFAULT_MODEL_TXT_DIR), help="COLMAP TXT model directory.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for trajectory figures.")
    parser.add_argument(
        "--top_output",
        default=None,
        help="Top-view trajectory PNG path. Defaults to output_dir/camera_trajectory_top.png.",
    )
    parser.add_argument(
        "--view3d_output",
        default=None,
        help="3D trajectory PNG path. Defaults to output_dir/camera_trajectory_3d.png.",
    )
    parser.add_argument(
        "--summary_output",
        default=None,
        help="JSON summary path. Defaults to output_dir/camera_trajectory_summary.json.",
    )
    parser.add_argument("--max_points", type=int, default=50000, help="Maximum sparse points to draw.")
    parser.add_argument("--label_stride", type=int, default=10, help="Label every Nth camera center.")
    parser.add_argument(
        "--outlier_iqr_scale",
        type=float,
        default=5.0,
        help="IQR multiplier for conservative radial and step outlier detection.",
    )
    parser.add_argument(
        "--plot_point_percentile",
        type=float,
        default=98.0,
        help="Central sparse-point percentile used to set robust plot limits.",
    )
    parser.add_argument("--plot_margin", type=float, default=0.08, help="Fractional margin around robust plot limits.")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1.0 - 2.0 * qy * qy - 2.0 * qz * qz, 2.0 * qx * qy - 2.0 * qw * qz, 2.0 * qz * qx + 2.0 * qw * qy],
            [2.0 * qx * qy + 2.0 * qw * qz, 1.0 - 2.0 * qx * qx - 2.0 * qz * qz, 2.0 * qy * qz - 2.0 * qw * qx],
            [2.0 * qz * qx - 2.0 * qw * qy, 2.0 * qy * qz + 2.0 * qw * qx, 1.0 - 2.0 * qx * qx - 2.0 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation = qvec_to_rotmat(qvec)
    return -rotation.T @ tvec


def parse_frame_index(name: str) -> int | None:
    match = re.search(r"frame_(\d+)", name)
    if not match:
        return None
    return int(match.group(1))


def read_images(images_path: Path) -> list[RegisteredImage]:
    if not images_path.is_file():
        fail(f"Missing COLMAP images.txt: {images_path}")

    images: list[RegisteredImage] = []
    lines = images_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            fail(f"Unexpected images.txt image line: {line[:120]}")

        try:
            image_id = int(parts[0])
            qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float64)
            camera_id = int(parts[8])
        except ValueError as exc:
            fail(f"Could not parse images.txt image line: {line[:120]} ({exc})")

        name = " ".join(parts[9:])
        images.append(
            RegisteredImage(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                frame_index=parse_frame_index(name),
                center=camera_center(qvec, tvec),
            )
        )

        if index < len(lines):
            index += 1

    return sorted(images, key=lambda item: (item.frame_index if item.frame_index is not None else math.inf, item.name))


def read_points(points_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not points_path.is_file():
        fail(f"Missing COLMAP points3D.txt: {points_path}")

    coords: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    with points_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
                colors.append((int(parts[4]), int(parts[5]), int(parts[6])))
            except ValueError:
                continue

    if not coords:
        fail(f"No sparse points parsed from {points_path}")

    return np.asarray(coords, dtype=np.float64), np.asarray(colors, dtype=np.float64) / 255.0


def downsample_points(points: np.ndarray, colors: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(points) <= max_points:
        return points, colors
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
    return points[indices], colors[indices]


def pca_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return center, vt[:2].T


def project_top(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (points - center) @ basis


def iqr_threshold(values: np.ndarray, scale: float) -> float:
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-12:
        return float(q3)
    return float(q3 + scale * iqr)


def detect_outliers(centers: np.ndarray, iqr_scale: float) -> dict[str, object]:
    median_center = np.median(centers, axis=0)
    radial_distances = np.linalg.norm(centers - median_center, axis=1)
    radial_threshold = iqr_threshold(radial_distances, iqr_scale)
    radial_outliers = set(np.flatnonzero(radial_distances > radial_threshold).tolist())

    step_lengths = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    if len(step_lengths) > 0:
        step_threshold = iqr_threshold(step_lengths, iqr_scale)
        step_outliers = set((np.flatnonzero(step_lengths > step_threshold) + 1).tolist())
    else:
        step_threshold = 0.0
        step_outliers = set()

    outliers = sorted(radial_outliers | step_outliers)
    return {
        "method": "radial distance from median camera center and adjacent trajectory step length, both using Q3 + scale * IQR",
        "iqr_scale": iqr_scale,
        "radial_threshold": radial_threshold,
        "step_threshold": step_threshold,
        "radial_outlier_indices_0_based": sorted(radial_outliers),
        "step_outlier_indices_0_based": sorted(step_outliers),
        "outlier_indices_0_based": outliers,
        "radial_distances": radial_distances,
        "step_lengths": step_lengths,
    }


def robust_limits(
    points: np.ndarray,
    cameras: np.ndarray,
    percentile: float,
    margin_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    percentile = min(max(percentile, 50.0), 100.0)
    tail = (100.0 - percentile) / 2.0
    point_lower = np.percentile(points, tail, axis=0)
    point_upper = np.percentile(points, 100.0 - tail, axis=0)
    lower = np.minimum(point_lower, cameras.min(axis=0))
    upper = np.maximum(point_upper, cameras.max(axis=0))
    span = np.maximum(upper - lower, 1e-6)
    return lower - span * margin_fraction, upper + span * margin_fraction


def count_inside(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> int:
    mask = np.all((values >= lower) & (values <= upper), axis=1)
    return int(mask.sum())


def should_label(index: int, count: int, outlier_indices: set[int], stride: int) -> bool:
    if index in outlier_indices or index in {0, count - 1}:
        return True
    if stride <= 0:
        return False
    return index % stride == 0


def label_for(image: RegisteredImage, sequence_index: int) -> str:
    if image.frame_index is not None:
        return f"f{image.frame_index:03d}"
    return str(sequence_index + 1)


def set_equal_3d_axes(ax: plt.Axes, values: np.ndarray) -> None:
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def write_top_plot(
    output_path: Path,
    points_2d: np.ndarray,
    point_colors: np.ndarray,
    cameras_2d: np.ndarray,
    images: list[RegisteredImage],
    outlier_indices: set[int],
    label_stride: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    order = np.arange(len(images))
    outlier_text = "possible outlier cameras: none" if not outlier_indices else f"possible outlier cameras: {len(outlier_indices)}"
    fig, ax = plt.subplots(figsize=(14, 10), dpi=180)
    ax.scatter(points_2d[:, 0], points_2d[:, 1], c=point_colors, s=0.35, alpha=0.25, linewidths=0, label="sparse points")
    ax.plot(cameras_2d[:, 0], cameras_2d[:, 1], color="#2563eb", linewidth=1.4, alpha=0.9, label="camera trajectory")
    camera_scatter = ax.scatter(
        cameras_2d[:, 0],
        cameras_2d[:, 1],
        c=order,
        cmap="viridis",
        s=34,
        edgecolors="black",
        linewidths=0.4,
        label="camera centers",
        zorder=5,
    )

    if outlier_indices:
        outlier_list = sorted(outlier_indices)
        ax.scatter(
            cameras_2d[outlier_list, 0],
            cameras_2d[outlier_list, 1],
            marker="x",
            c="#dc2626",
            s=90,
            linewidths=2.0,
            label="possible outlier cameras",
            zorder=7,
        )

    for index, image in enumerate(images):
        if should_label(index, len(images), outlier_indices, label_stride):
            ax.annotate(
                label_for(image, index),
                (cameras_2d[index, 0], cameras_2d[index, 1]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#111827",
            )

    cbar = fig.colorbar(camera_scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("camera order in selected sequence")
    ax.set_title(f"COLMAP camera trajectory top view, sparse points + registered cameras ({outlier_text})")
    ax.set_xlabel("PCA axis 1")
    ax.set_ylabel("PCA axis 2")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.25)
    ax.legend(loc="best", markerscale=1)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_3d_plot(
    output_path: Path,
    points: np.ndarray,
    point_colors: np.ndarray,
    centers: np.ndarray,
    images: list[RegisteredImage],
    outlier_indices: set[int],
    label_stride: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    order = np.arange(len(images))
    outlier_text = "possible outlier cameras: none" if not outlier_indices else f"possible outlier cameras: {len(outlier_indices)}"
    fig = plt.figure(figsize=(14, 10), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=point_colors, s=0.35, alpha=0.18, linewidths=0)
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], color="#2563eb", linewidth=1.5, alpha=0.95)
    camera_scatter = ax.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        c=order,
        cmap="viridis",
        s=34,
        edgecolors="black",
        linewidths=0.35,
        depthshade=False,
    )

    if outlier_indices:
        outlier_list = sorted(outlier_indices)
        ax.scatter(
            centers[outlier_list, 0],
            centers[outlier_list, 1],
            centers[outlier_list, 2],
            marker="x",
            c="#dc2626",
            s=90,
            linewidths=2.0,
            depthshade=False,
        )

    for index, image in enumerate(images):
        if should_label(index, len(images), outlier_indices, label_stride):
            ax.text(
                centers[index, 0],
                centers[index, 1],
                centers[index, 2],
                label_for(image, index),
                fontsize=6,
                color="#111827",
            )

    set_equal_3d_axes(ax, np.vstack([np.asarray([lower, upper]), centers]))
    cbar = fig.colorbar(camera_scatter, ax=ax, fraction=0.025, pad=0.05)
    cbar.set_label("camera order in selected sequence")
    ax.set_title(f"COLMAP camera trajectory 3D view, sparse points + registered cameras ({outlier_text})")
    ax.set_xlabel("COLMAP X")
    ax.set_ylabel("COLMAP Y")
    ax.set_zlabel("COLMAP Z")
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_txt_dir = Path(args.model_txt_dir)
    output_dir = Path(args.output_dir)
    top_output = Path(args.top_output) if args.top_output else output_dir / "camera_trajectory_top.png"
    view3d_output = Path(args.view3d_output) if args.view3d_output else output_dir / "camera_trajectory_3d.png"
    summary_output = Path(args.summary_output) if args.summary_output else output_dir / "camera_trajectory_summary.json"

    if not model_txt_dir.is_dir():
        fail(f"COLMAP TXT model directory not found: {model_txt_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    top_output.parent.mkdir(parents=True, exist_ok=True)
    view3d_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    images = read_images(model_txt_dir / "images.txt")
    points, colors = read_points(model_txt_dir / "points3D.txt")
    plot_points, plot_colors = downsample_points(points, colors, args.max_points)

    if len(images) < 2:
        fail("Need at least two registered images to visualize a trajectory.")

    centers = np.vstack([image.center for image in images])
    outlier_info = detect_outliers(centers, args.outlier_iqr_scale)
    outlier_indices = set(outlier_info["outlier_indices_0_based"])

    pca_center, top_basis = pca_basis(plot_points)
    points_2d = project_top(plot_points, pca_center, top_basis)
    cameras_2d = project_top(centers, pca_center, top_basis)
    top_lower, top_upper = robust_limits(points_2d, cameras_2d, args.plot_point_percentile, args.plot_margin)
    view3d_lower, view3d_upper = robust_limits(plot_points, centers, args.plot_point_percentile, args.plot_margin)

    write_top_plot(
        top_output,
        points_2d,
        plot_colors,
        cameras_2d,
        images,
        outlier_indices,
        args.label_stride,
        (float(top_lower[0]), float(top_upper[0])),
        (float(top_lower[1]), float(top_upper[1])),
    )
    write_3d_plot(
        view3d_output,
        plot_points,
        plot_colors,
        centers,
        images,
        outlier_indices,
        args.label_stride,
        view3d_lower,
        view3d_upper,
    )

    outlier_records = []
    for index in sorted(outlier_indices):
        image = images[index]
        outlier_records.append(
            {
                "sequence_index_1_based": index + 1,
                "image_name": image.name,
                "frame_index": image.frame_index,
                "camera_center": image.center.tolist(),
            }
        )

    bounds = {
        axis: {"min": float(centers[:, axis_index].min()), "max": float(centers[:, axis_index].max())}
        for axis_index, axis in enumerate(["x", "y", "z"])
    }
    summary = {
        "model_txt_dir": str(model_txt_dir),
        "registered_image_count": len(images),
        "sparse_point_count": int(len(points)),
        "drawn_sparse_point_count": int(len(plot_points)),
        "top_output": str(top_output),
        "view3d_output": str(view3d_output),
        "camera_center_bounds": bounds,
        "plot_limits": {
            "note": "Figures use robust sparse-point percentiles plus all camera centers so distant sparse artifacts do not shrink the trajectory.",
            "plot_point_percentile": args.plot_point_percentile,
            "top_view": {
                "axis_1": {"min": float(top_lower[0]), "max": float(top_upper[0])},
                "axis_2": {"min": float(top_lower[1]), "max": float(top_upper[1])},
                "drawn_sparse_points_inside_limits": count_inside(points_2d, top_lower, top_upper),
            },
            "view3d": {
                "x": {"min": float(view3d_lower[0]), "max": float(view3d_upper[0])},
                "y": {"min": float(view3d_lower[1]), "max": float(view3d_upper[1])},
                "z": {"min": float(view3d_lower[2]), "max": float(view3d_upper[2])},
                "drawn_sparse_points_inside_limits": count_inside(plot_points, view3d_lower, view3d_upper),
            },
        },
        "labeling": {
            "label": "frame index when filename contains frame_XXXXXX; otherwise selected-sequence index",
            "label_stride": args.label_stride,
            "always_labeled": ["first camera", "last camera", "possible outlier cameras"],
        },
        "outlier_detection": {
            key: value
            for key, value in outlier_info.items()
            if key not in {"radial_distances", "step_lengths"}
        },
        "possible_outlier_camera_count": len(outlier_records),
        "possible_outlier_cameras": outlier_records,
        "geometry_check": "pass_no_obvious_camera_outliers" if not outlier_records else "review_possible_outlier_cameras",
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote top trajectory figure: {top_output}")
    print(f"Wrote 3D trajectory figure: {view3d_output}")
    print(f"Wrote trajectory summary: {summary_output}")
    print(f"Registered images: {len(images)}")
    print(f"Sparse points: {len(points)}")
    print(f"Possible outlier cameras: {len(outlier_records)}")
    print(f"Geometry check: {summary['geometry_check']}")


if __name__ == "__main__":
    main()
