#!/usr/bin/env python3
"""Minimal CALVIN ground-truth action replay gate.

This is a protocol gate, not a policy evaluation. It replays native CALVIN
validation demonstration actions from a few annotated episodes and checks the
same task oracle used by closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

CALVIN_ROOT = Path("/data/dechao/cv_final_pj2/external/calvin")
CALVIN_MODELS = CALVIN_ROOT / "calvin_models"
CALVIN_ENV = CALVIN_ROOT / "calvin_env"
for path in (CALVIN_MODELS, CALVIN_ENV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calvin_env  # noqa: E402
from calvin_env.envs.play_table_env import get_env  # noqa: E402


TASKS_CFG = CALVIN_MODELS / "conf" / "callbacks" / "rollout" / "tasks" / "new_playtable_tasks.yaml"
OFFICIAL_EP_LEN = 360
DEFAULT_DATASET_LABEL = "task_D_D / D-environment GT-action replay gate"


def validation_folder(dataset_path: Path) -> Path:
    if (dataset_path / "validation").is_dir():
        return dataset_path / "validation"
    return dataset_path


def create_env(dataset_path: Path):
    if getattr(calvin_env, "__file__", None) is None:
        calvin_env.__file__ = str(CALVIN_ENV / "calvin_env" / "__init__.py")
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    return get_env(validation_folder(dataset_path), obs_space=obs_space, show_gui=False)


def load_task_oracle():
    task_cfg = OmegaConf.load(TASKS_CFG)
    return hydra.utils.instantiate(task_cfg)


def load_annotations(dataset_path: Path) -> dict:
    ann_path = validation_folder(dataset_path) / "lang_annotations" / "auto_lang_ann.npy"
    return np.load(ann_path, allow_pickle=True).item()


def episode_path(validation_dir: Path, frame_idx: int) -> Path:
    return validation_dir / f"episode_{frame_idx:07d}.npz"


def safe_frame(obs: dict) -> np.ndarray:
    image = np.asarray(obs["rgb_obs"]["rgb_static"])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def draw_overlay(frame_rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    frame_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    height = 24 + 22 * len(lines)
    cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], height), (0, 0, 0), thickness=-1)
    for idx, line in enumerate(lines):
        cv2.putText(
            frame_bgr,
            line[:92],
            (8, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def write_video(path: Path, frames_rgb: list[np.ndarray], fps: int) -> None:
    if not frames_rgb:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_keyframes(out_dir: Path, frames_rgb: list[np.ndarray]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frames_rgb:
        return []
    indices = sorted(set([0, len(frames_rgb) // 2, len(frames_rgb) - 1]))
    paths = []
    for idx in indices:
        path = out_dir / f"keyframe_{idx:03d}.png"
        Image.fromarray(frames_rgb[idx]).save(path)
        paths.append(str(path))
    return paths


def array_summary(value) -> dict:
    arr = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def action_summary(actions: list[np.ndarray]) -> dict:
    if not actions:
        return {"count": 0}
    arr = np.asarray(actions, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "first_action": arr[0].astype(float).tolist(),
        "last_action": arr[-1].astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "l2_mean": float(np.linalg.norm(arr[:, :6], axis=1).mean()),
        "l2_max": float(np.linalg.norm(arr[:, :6], axis=1).max()),
        "gripper_first": float(arr[0, -1]),
        "gripper_last": float(arr[-1, -1]),
    }


def replay_case(
    *,
    env,
    task_oracle,
    validation_dir: Path,
    annotation_index: int,
    task: str,
    language: str,
    frame_start: int,
    frame_end: int,
    action_key: str,
    ep_len: int,
    frame_stride: int,
    output_dir: Path,
) -> dict:
    start_episode = np.load(episode_path(validation_dir, frame_start), allow_pickle=True)
    robot_obs = np.asarray(start_episode["robot_obs"], dtype=np.float64)
    scene_obs = np.asarray(start_episode["scene_obs"], dtype=np.float64)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    obs = env.get_obs()
    start_info = env.get_info()
    current_info = start_info
    success = False
    success_step = None
    actions = []
    frames = [
        draw_overlay(
            safe_frame(obs),
            [
                f"GT replay ann={annotation_index} task={task}",
                f"goal: {language}",
                "step=0",
            ],
        )
    ]

    max_frame = min(int(frame_end), int(frame_start) + int(ep_len) - 1)
    for step, frame_idx in enumerate(range(int(frame_start), max_frame + 1), start=1):
        episode = np.load(episode_path(validation_dir, frame_idx), allow_pickle=True)
        if action_key not in episode.files:
            raise KeyError(f"{action_key!r} not found in {episode_path(validation_dir, frame_idx)}")
        action = np.asarray(episode[action_key], dtype=np.float32).reshape(-1)
        actions.append(action)
        obs, _, _, current_info = env.step(action)
        task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task})
        success = bool(task_info)
        if step % frame_stride == 0 or success or frame_idx == max_frame:
            frames.append(
                draw_overlay(
                    safe_frame(obs),
                    [
                        f"GT replay ann={annotation_index} task={task}",
                        f"goal: {language}",
                        f"step={step} success={success}",
                    ],
                )
            )
        if success:
            success_step = step
            break

    case_dir = output_dir / f"ann_{annotation_index:04d}_{task}"
    video_path = case_dir / "gt_replay.mp4"
    write_video(video_path, frames, fps=6)
    keyframe_paths = save_keyframes(case_dir, frames)
    return {
        "annotation_index": int(annotation_index),
        "task": str(task),
        "language": str(language),
        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "action_key": str(action_key),
        "horizon": int(ep_len),
        "steps": int(len(actions)),
        "success": bool(success),
        "success_step": int(success_step) if success_step is not None else None,
        "robot_obs_start_summary": array_summary(robot_obs),
        "scene_obs_start_summary": array_summary(scene_obs),
        "action_summary": action_summary(actions),
        "video_path": str(video_path),
        "keyframe_paths": keyframe_paths,
    }


def run(args: argparse.Namespace) -> dict:
    validation_dir = validation_folder(args.dataset_path)
    annotations = load_annotations(args.dataset_path)
    tasks = annotations["language"]["task"]
    anns = annotations["language"]["ann"]
    indices = annotations["info"]["indx"]

    env = create_env(args.dataset_path)
    task_oracle = load_task_oracle()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    cases = []
    try:
        for offset in range(args.num_cases):
            ann_idx = int(args.annotation_offset + offset)
            frame_start, frame_end = indices[ann_idx]
            cases.append(
                replay_case(
                    env=env,
                    task_oracle=task_oracle,
                    validation_dir=validation_dir,
                    annotation_index=ann_idx,
                    task=str(tasks[ann_idx]),
                    language=str(anns[ann_idx]),
                    frame_start=int(frame_start),
                    frame_end=int(frame_end),
                    action_key=args.action_key,
                    ep_len=args.ep_len,
                    frame_stride=args.frame_stride,
                    output_dir=args.output_dir,
                )
            )
    finally:
        if hasattr(env, "close"):
            env.close()

    success_counter = Counter(case["task"] for case in cases if case["success"])
    attempted_counter = Counter(case["task"] for case in cases)
    report = {
        "protocol": {
            "dataset_label": args.dataset_label,
            "protocol_tag": "gt_action_replay_gate",
            "dataset_path": str(args.dataset_path),
            "validation_folder": str(validation_dir),
            "action_key": args.action_key,
            "ep_len": int(args.ep_len),
            "official_ep_len": OFFICIAL_EP_LEN,
            "num_cases": len(cases),
            "annotation_offset": int(args.annotation_offset),
            "is_policy_evaluation": False,
            "is_official_benchmark": False,
            "source_note": args.source_note,
        },
        "num_cases": len(cases),
        "num_success": int(sum(1 for case in cases if case["success"])),
        "success_rate": float(sum(1 for case in cases if case["success"]) / len(cases)) if cases else 0.0,
        "task_success": {
            task: {
                "success": int(success_counter[task]),
                "attempted": int(attempted_counter[task]),
                "success_rate": float(success_counter[task] / attempted_counter[task]),
            }
            for task in sorted(attempted_counter)
        },
        "elapsed_sec": time.time() - started_at,
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = args.output_json.with_suffix(".csv")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "annotation_index",
            "task",
            "success",
            "success_step",
            "steps",
            "frame_start",
            "frame_end",
            "video_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case[key] for key in fieldnames})

    printable = {key: value for key, value in report.items() if key != "cases"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a few native CALVIN validation GT action demos.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-label", default=DEFAULT_DATASET_LABEL)
    parser.add_argument("--annotation-offset", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--action-key", default="rel_actions", choices=["rel_actions", "actions"])
    parser.add_argument("--ep-len", type=int, default=OFFICIAL_EP_LEN)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument(
        "--source-note",
        default="minimal GT-action replay gate for simulator/oracle/action convention validation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
