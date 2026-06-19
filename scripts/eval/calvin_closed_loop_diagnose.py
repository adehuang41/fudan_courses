#!/usr/bin/env python3
"""Diagnose CALVIN closed-loop zero-success rollouts.

This reruns only the selected small-scale first-subtask failures. It records
videos, sampled frames, actions, gripper values, goal text, and simulator state
changes. It is diagnostic instrumentation, not a benchmark runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import socket
import struct
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator
import zlib

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
ANNOTATIONS_CFG = CALVIN_MODELS / "conf" / "annotations" / "new_playtable_validation.yaml"


@contextmanager
def temp_seed(seed: int) -> Iterator[None]:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


utils_stub = ModuleType("calvin_agent.evaluation.utils")
utils_stub.temp_seed = temp_seed
sys.modules["calvin_agent.evaluation.utils"] = utils_stub

from calvin_agent.evaluation.multistep_sequences import get_sequences  # noqa: E402


def deterministic_seed(initial_condition: dict) -> int:
    values = tuple(initial_condition.values())
    return zlib.crc32(str(values).encode("utf-8")) & 0xFFFFFFFF


def get_env_state_for_initial_condition(initial_condition: dict) -> tuple[np.ndarray, np.ndarray]:
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ],
        dtype=np.float64,
    )
    block_rot_z_range = (np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    rng = np.random.RandomState(deterministic_seed(initial_condition))
    rng.shuffle(block_table)

    scene_obs = np.zeros(24, dtype=np.float64)
    if initial_condition["slider"] == "left":
        scene_obs[0] = 0.28
    if initial_condition["drawer"] == "open":
        scene_obs[1] = 0.22
    if initial_condition["lightbulb"] == 1:
        scene_obs[3] = 0.088
    scene_obs[4] = initial_condition["lightbulb"]
    scene_obs[5] = initial_condition["led"]

    if initial_condition["red_block"] == "slider_right":
        scene_obs[6:9] = block_slider_right
    elif initial_condition["red_block"] == "slider_left":
        scene_obs[6:9] = block_slider_left
    else:
        scene_obs[6:9] = block_table[0]
    scene_obs[11] = rng.uniform(*block_rot_z_range)

    if initial_condition["blue_block"] == "slider_right":
        scene_obs[12:15] = block_slider_right
    elif initial_condition["blue_block"] == "slider_left":
        scene_obs[12:15] = block_slider_left
    elif initial_condition["red_block"] == "table":
        scene_obs[12:15] = block_table[1]
    else:
        scene_obs[12:15] = block_table[0]
    scene_obs[17] = rng.uniform(*block_rot_z_range)

    if initial_condition["pink_block"] == "slider_right":
        scene_obs[18:21] = block_slider_right
    elif initial_condition["pink_block"] == "slider_left":
        scene_obs[18:21] = block_slider_left
    else:
        scene_obs[18:21] = block_table[1]
    scene_obs[23] = rng.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def recvall(sock: socket.socket, n_bytes: int) -> bytes:
    chunks = []
    remaining = n_bytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> object:
    header = recvall(sock, 8)
    length = struct.unpack("!Q", header)[0]
    payload = recvall(sock, length)
    return pickle.loads(payload)


def send_msg(sock: socket.socket, msg: object) -> None:
    payload = pickle.dumps(msg, protocol=4)
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


class RemoteDiagnosticPolicy:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=120)

    def reset(self) -> None:
        send_msg(self.sock, {"type": "reset"})
        response = recv_msg(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response)

    def step(self, obs: dict, goal: str) -> dict:
        send_msg(self.sock, {"type": "step", "obs": obs, "goal": goal})
        response = recv_msg(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response)
        return response

    def close(self) -> None:
        try:
            send_msg(self.sock, {"type": "close"})
            recv_msg(self.sock)
        finally:
            self.sock.close()


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


def load_annotations():
    return OmegaConf.load(ANNOTATIONS_CFG)


def arr(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def arr_summary(value) -> dict:
    x = arr(value)
    return {
        "shape": list(x.shape),
        "min": float(x.min()) if x.size else 0.0,
        "max": float(x.max()) if x.size else 0.0,
        "mean": float(x.mean()) if x.size else 0.0,
        "std": float(x.std()) if x.size else 0.0,
    }


def scene_key_values(scene_obs: np.ndarray) -> dict:
    x = arr(scene_obs)
    return {
        "slider": float(x[0]),
        "drawer": float(x[1]),
        "lightbulb_intensity": float(x[3]),
        "lightbulb_state": float(x[4]),
        "led_state": float(x[5]),
        "red_block_xyz": x[6:9].tolist(),
        "blue_block_xyz": x[12:15].tolist(),
        "pink_block_xyz": x[18:21].tolist(),
    }


def safe_frame(obs: dict) -> np.ndarray:
    image = np.asarray(obs["rgb_obs"]["rgb_static"])
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def draw_overlay(frame_rgb: np.ndarray, label: str) -> np.ndarray:
    frame = frame_rgb.copy()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 42), (0, 0, 0), thickness=-1)
    cv2.putText(frame_bgr, label[:80], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
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


def run_case(
    *,
    env,
    policy: RemoteDiagnosticPolicy,
    task_oracle,
    val_annotations,
    checkpoint_label: str,
    sequence_index: int,
    initial_state: dict,
    eval_sequence,
    ep_len: int,
    frame_stride: int,
    out_root: Path,
) -> dict:
    task_name = str(eval_sequence[0])
    lang_annotation = str(val_annotations[task_name][0])
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    policy.reset()
    start_info = env.get_info()
    start_robot = arr(obs["robot_obs"])
    start_scene = arr(obs["scene_obs"])
    prev_robot = start_robot.copy()
    prev_scene = start_scene.copy()
    frames = [draw_overlay(safe_frame(obs), f"{checkpoint_label} seq{sequence_index} step0 {task_name}")]
    step_rows = []
    success = False
    current_info = start_info

    for step in range(ep_len):
        response = policy.step(obs, lang_annotation)
        action = arr(response["action"])
        action_raw = arr(response["action_raw"])
        action_norm = arr(response["action_norm"])
        obs, _, _, current_info = env.step(action.astype(np.float32))
        current_robot = arr(obs["robot_obs"])
        current_scene = arr(obs["scene_obs"])
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task_name})
        success = bool(current_task_info)
        step_rows.append(
            {
                "checkpoint_label": checkpoint_label,
                "sequence_index": sequence_index,
                "task": task_name,
                "step": step + 1,
                "goal": lang_annotation,
                "success": success,
                "action": action.tolist(),
                "action_raw": action_raw.tolist(),
                "action_norm": action_norm.tolist(),
                "gripper_final": float(action[-1]),
                "gripper_raw": float(action_raw[-1]),
                "tcp_delta_from_start": float(np.linalg.norm(current_robot[:3] - start_robot[:3])),
                "tcp_delta_from_prev": float(np.linalg.norm(current_robot[:3] - prev_robot[:3])),
                "robot_delta_from_start": float(np.linalg.norm(current_robot - start_robot)),
                "scene_delta_from_start": float(np.linalg.norm(current_scene - start_scene)),
                "scene_delta_from_prev": float(np.linalg.norm(current_scene - prev_scene)),
                "robot_obs": current_robot.tolist(),
                "scene_obs_key_values": scene_key_values(current_scene),
                "task_oracle_info": sorted(map(str, current_task_info)),
            }
        )
        if (step + 1) % frame_stride == 0 or success or (step + 1) == ep_len:
            frames.append(
                draw_overlay(
                    safe_frame(obs),
                    f"{checkpoint_label} seq{sequence_index} step{step + 1} {task_name} ok={success}",
                )
            )
        prev_robot = current_robot.copy()
        prev_scene = current_scene.copy()
        if success:
            break

    case_dir = out_root / checkpoint_label / f"seq_{sequence_index:02d}_{task_name}"
    video_path = case_dir / "rollout.mp4"
    write_video(video_path, frames, fps=6)
    keyframe_paths = save_keyframes(case_dir, frames)
    csv_path = case_dir / "steps.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "checkpoint_label",
            "sequence_index",
            "task",
            "step",
            "goal",
            "success",
            "gripper_final",
            "gripper_raw",
            "tcp_delta_from_start",
            "tcp_delta_from_prev",
            "robot_delta_from_start",
            "scene_delta_from_start",
            "scene_delta_from_prev",
            "task_oracle_info",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    actions = np.asarray([row["action"] for row in step_rows], dtype=np.float64)
    action_raws = np.asarray([row["action_raw"] for row in step_rows], dtype=np.float64)
    action_norms = np.asarray([row["action_norm"] for row in step_rows], dtype=np.float64)
    grippers = np.asarray([row["gripper_final"] for row in step_rows], dtype=np.float64)
    final_robot = arr(obs["robot_obs"])
    final_scene = arr(obs["scene_obs"])
    case = {
        "checkpoint_label": checkpoint_label,
        "sequence_index": sequence_index,
        "task": task_name,
        "goal": lang_annotation,
        "eval_sequence": [str(task) for task in eval_sequence],
        "success": success,
        "steps": len(step_rows),
        "initial_state": initial_state,
        "start_robot_obs": start_robot.tolist(),
        "end_robot_obs": final_robot.tolist(),
        "start_scene_obs_key_values": scene_key_values(start_scene),
        "end_scene_obs_key_values": scene_key_values(final_scene),
        "tcp_delta_total": float(np.linalg.norm(final_robot[:3] - start_robot[:3])),
        "robot_delta_total": float(np.linalg.norm(final_robot - start_robot)),
        "scene_delta_total": float(np.linalg.norm(final_scene - start_scene)),
        "action_summary": arr_summary(actions),
        "action_raw_summary": arr_summary(action_raws),
        "action_norm_summary": arr_summary(action_norms),
        "action_dim_min": actions.min(axis=0).tolist() if actions.size else [],
        "action_dim_max": actions.max(axis=0).tolist() if actions.size else [],
        "action_dim_mean": actions.mean(axis=0).tolist() if actions.size else [],
        "action_raw_dim_mean": action_raws.mean(axis=0).tolist() if action_raws.size else [],
        "action_norm_dim_mean": action_norms.mean(axis=0).tolist() if action_norms.size else [],
        "gripper_values": sorted(map(float, np.unique(grippers))) if grippers.size else [],
        "gripper_positive_fraction": float((grippers > 0).mean()) if grippers.size else 0.0,
        "video_path": str(video_path),
        "keyframe_paths": keyframe_paths,
        "step_csv": str(csv_path),
        "step_rows": step_rows,
    }
    (case_dir / "failure_log.json").write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    return case


def run(args: argparse.Namespace) -> dict:
    env = create_env(args.dataset_path)
    policy = RemoteDiagnosticPolicy(args.host, args.port)
    task_oracle = load_task_oracle()
    val_annotations = load_annotations()
    sequences = get_sequences(args.num_sequences, num_workers=args.num_workers)
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    cases = []
    try:
        for sequence_index, (initial_state, eval_sequence) in enumerate(sequences):
            cases.append(
                run_case(
                    env=env,
                    policy=policy,
                    task_oracle=task_oracle,
                    val_annotations=val_annotations,
                    checkpoint_label=args.checkpoint_label,
                    sequence_index=sequence_index,
                    initial_state=initial_state,
                    eval_sequence=eval_sequence,
                    ep_len=args.ep_len,
                    frame_stride=args.frame_stride,
                    out_root=out_root,
                )
            )
    finally:
        policy.close()
        if hasattr(env, "close"):
            env.close()

    all_actions = np.asarray([row["action"] for case in cases for row in case["step_rows"]], dtype=np.float64)
    all_raw = np.asarray([row["action_raw"] for case in cases for row in case["step_rows"]], dtype=np.float64)
    attempted = Counter(case["task"] for case in cases)
    successes = Counter(case["task"] for case in cases if case["success"])
    report = {
        "dataset_path": str(args.dataset_path),
        "validation_folder": str(validation_folder(args.dataset_path)),
        "checkpoint_label": args.checkpoint_label,
        "num_sequences": len(cases),
        "ep_len": args.ep_len,
        "frame_stride": args.frame_stride,
        "success_count": int(sum(case["success"] for case in cases)),
        "success_rate_first_subtask": float(sum(case["success"] for case in cases) / len(cases)) if cases else 0.0,
        "attempted_tasks": dict(sorted(attempted.items())),
        "successful_tasks": dict(sorted(successes.items())),
        "all_action_summary": arr_summary(all_actions) if all_actions.size else {},
        "all_action_raw_summary": arr_summary(all_raw) if all_raw.size else {},
        "elapsed_sec": time.time() - started_at,
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = args.output_json.with_suffix(".csv")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "checkpoint_label",
            "sequence_index",
            "task",
            "goal",
            "success",
            "steps",
            "tcp_delta_total",
            "robot_delta_total",
            "scene_delta_total",
            "gripper_values",
            "gripper_positive_fraction",
            "video_path",
            "step_csv",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow({key: case[key] for key in fieldnames})
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CALVIN closed-loop failures.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ep-len", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
