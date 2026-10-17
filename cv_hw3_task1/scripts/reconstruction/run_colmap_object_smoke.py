#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small-scale COLMAP smoke test for Object A.")
    parser.add_argument("--image_dir", required=True, help="Input image directory. Expected: images_smoke.")
    parser.add_argument("--output_dir", required=True, help="COLMAP smoke output directory.")
    parser.add_argument("--single_camera", action="store_true", help="Use COLMAP single-camera assumption.")
    parser.add_argument("--matcher", choices=["exhaustive"], default="exhaustive", help="Matcher type for this smoke test.")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL", help="COLMAP camera model.")
    parser.add_argument("--colmap_cmd", default="colmap", help="COLMAP executable name or path.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU SIFT extraction and matching.")
    parser.add_argument("--max_images", type=int, default=20, help="Refuse to run if image count exceeds this limit.")
    parser.add_argument("--overwrite", action="store_true", help="Clear output_dir before running.")
    parser.add_argument("--clean_output", action="store_true", help="Alias for --overwrite.")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def list_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS)


def run_step(name: str, command: list[str], log_dir: Path) -> str:
    log_path = log_dir / f"{name}.log"
    printable = " ".join(command)
    print(f"Running {name}: {printable}")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(f"$ {printable}\n\n{completed.stdout}", encoding="utf-8")
    if completed.returncode != 0:
        print(completed.stdout)
        fail(f"COLMAP step failed: {name}. See {log_path}")
    return completed.stdout


def parse_model_analyzer(output: str) -> dict[str, int | None]:
    registered = None
    points = None

    registered_match = re.search(r"Registered images:\s*(\d+)", output)
    if registered_match:
        registered = int(registered_match.group(1))

    points_match = re.search(r"Points:\s*(\d+)", output)
    if points_match:
        points = int(points_match.group(1))

    return {"registered_images": registered, "sparse_points": points}


def numeric_viability(input_count: int, registered_images: int | None) -> str:
    if registered_images is None or input_count == 0:
        return "unknown_needs_manual_log_check"
    ratio = registered_images / input_count
    if ratio >= 0.80:
        return "ideal_for_entering_formal_colmap_if_manual_checks_pass"
    if ratio >= 0.70:
        return "can_enter_formal_colmap_if_manual_checks_pass"
    if ratio < 0.50:
        return "needs_frame_selection_adjustment"
    return "borderline_review_frames_before_formal_colmap"


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if shutil.which(args.colmap_cmd) is None:
        fail(f"COLMAP executable not found: {args.colmap_cmd}")
    if not image_dir.is_dir():
        fail(f"Image directory not found: {image_dir}")

    images = list_images(image_dir)
    if not images:
        fail(f"No input images found in {image_dir}")
    if len(images) > args.max_images:
        fail(f"Smoke test has {len(images)} images, above max_images={args.max_images}.")

    clean = args.overwrite or args.clean_output
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clean:
            fail(f"Output directory is not empty: {output_dir}. Use --overwrite or --clean_output to rerun.")
        shutil.rmtree(output_dir)

    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    feature_cmd = [
        args.colmap_cmd,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        args.camera_model,
        "--FeatureExtraction.use_gpu",
        "1" if args.use_gpu else "0",
    ]
    if args.single_camera:
        feature_cmd.extend(["--ImageReader.single_camera", "1"])

    matcher_cmd = [
        args.colmap_cmd,
        "exhaustive_matcher",
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        "1" if args.use_gpu else "0",
    ]

    mapper_cmd = [
        args.colmap_cmd,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]

    run_step("01_feature_extractor", feature_cmd, log_dir)
    run_step("02_exhaustive_matcher", matcher_cmd, log_dir)
    run_step("03_mapper", mapper_cmd, log_dir)

    sparse_models = sorted(path for path in sparse_dir.iterdir() if path.is_dir())
    if not sparse_models:
        summary = {
            "image_dir": str(image_dir),
            "output_dir": str(output_dir),
            "input_image_count": len(images),
            "sparse_model_found": False,
            "registered_images": None,
            "registered_ratio": None,
            "sparse_points": None,
            "numeric_viability": "failed_no_sparse_model",
            "manual_checks_required": [
                "reselect 15-20 frames with wider viewpoint coverage",
                "remove blurred or reflective frames",
                "check COLMAP logs",
            ],
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        fail("Mapper did not produce a sparse model.")

    model_path = sparse_models[0]
    analyzer_output = run_step("04_model_analyzer", [args.colmap_cmd, "model_analyzer", "--path", str(model_path)], log_dir)
    parsed = parse_model_analyzer(analyzer_output)
    registered = parsed["registered_images"]
    points = parsed["sparse_points"]
    ratio = registered / len(images) if isinstance(registered, int) else None

    summary = {
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "database_path": str(database_path),
        "sparse_model_path": str(model_path),
        "input_image_count": len(images),
        "input_filenames": [path.name for path in images],
        "matcher": args.matcher,
        "single_camera": args.single_camera,
        "camera_model": args.camera_model,
        "use_gpu": args.use_gpu,
        "sparse_model_found": True,
        "registered_images": registered,
        "registered_ratio": ratio,
        "sparse_points": points,
        "numeric_viability": numeric_viability(len(images), registered if isinstance(registered, int) else None),
        "manual_checks_required": [
            "camera trajectory roughly surrounds the object",
            "sparse points mainly cover the book or tabletop near the book",
            "no obviously flying cameras",
            "no severe wrong matches",
            "not only reconstructing the background",
        ],
        "next_step_boundary": "If numeric and manual checks pass, enter formal-scale COLMAP with 150-180 selected frames, not 2DGS yet.",
    }

    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_txt.write_text(
        "COLMAP Object A small-scale smoke test summary\n"
        f"input_image_count: {len(images)}\n"
        f"registered_images: {registered}\n"
        f"registered_ratio: {ratio}\n"
        f"sparse_points: {points}\n"
        f"numeric_viability: {summary['numeric_viability']}\n"
        "manual_checks_required:\n"
        + "\n".join(f"- {item}" for item in summary["manual_checks_required"])
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote summary to {summary_txt}")


if __name__ == "__main__":
    main()
