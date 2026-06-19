#!/usr/bin/env python3
"""Small-scale CALVIN closed-loop evaluation with per-sequence videos."""

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
OFFICIAL_EP_LEN = 360
OFFICIAL_NUM_SEQUENCES = 1000
DEFAULT_DATASET_LABEL = "task_D_D / D-environment closed-loop evaluation"
INITIAL_STATE_METHOD = "local_official_compatible_fnv1_32_from_calvin_utils"


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


def count_success(results: list[int]) -> list[float]:
    count = Counter(results)
    return [sum(count[j] for j in reversed(range(i, 6))) / len(results) for i in range(1, 6)]


def fnv1_32(text: str) -> int:
    """Pure-Python FNV-1 32-bit hash matching CALVIN's pyhash.fnv1_32 use."""
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value = (value * 0x01000193) & 0xFFFFFFFF
        value ^= byte
    return value


def deterministic_seed(initial_condition: dict) -> int:
    return fnv1_32(str(initial_condition.values()))


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
    with temp_seed(deterministic_seed(initial_condition)):
        np.random.shuffle(block_table)

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
        scene_obs[11] = np.random.uniform(*block_rot_z_range)

        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)

        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


def array_summary(value) -> dict:
    arr = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def array_delta_summary(before, after) -> dict:
    before_arr = np.asarray(before, dtype=np.float64)
    after_arr = np.asarray(after, dtype=np.float64)
    if before_arr.shape != after_arr.shape:
        return {"before_shape": list(before_arr.shape), "after_shape": list(after_arr.shape), "shape_match": False}
    delta = after_arr - before_arr
    return {
        "shape": list(delta.shape),
        "shape_match": True,
        "l2": float(np.linalg.norm(delta.reshape(-1))) if delta.size else 0.0,
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
    }


def flatten_numeric(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_numeric(next_prefix, child, out)
        return
    if isinstance(value, (str, bytes)):
        return
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return
    if arr.size and np.all(np.isfinite(arr)):
        out[prefix] = arr


def numeric_delta_summary(before, after, max_changed: int = 20) -> dict:
    before_flat: dict = {}
    after_flat: dict = {}
    flatten_numeric("", before, before_flat)
    flatten_numeric("", after, after_flat)
    changed = []
    for key in sorted(set(before_flat) & set(after_flat)):
        before_arr = before_flat[key]
        after_arr = after_flat[key]
        if before_arr.shape != after_arr.shape:
            changed.append({"key": key, "before_shape": list(before_arr.shape), "after_shape": list(after_arr.shape)})
            continue
        delta = after_arr - before_arr
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        if max_abs > 1e-9:
            changed.append(
                {
                    "key": key,
                    "shape": list(delta.shape),
                    "l2": float(np.linalg.norm(delta.reshape(-1))) if delta.size else 0.0,
                    "max_abs": max_abs,
                }
            )
    return {
        "numeric_keys_before": len(before_flat),
        "numeric_keys_after": len(after_flat),
        "changed_numeric_keys": len(changed),
        "changed": changed[:max_changed],
        "truncated": len(changed) > max_changed,
    }


def actions_summary(actions: list[dict]) -> dict:
    if not actions:
        return {"count": 0}
    physical = np.asarray([row["action"] for row in actions], dtype=np.float64)
    raw_rows = [row.get("action_raw") for row in actions if row.get("action_raw") is not None]
    norm_rows = [row.get("action_norm") for row in actions if row.get("action_norm") is not None]
    summary = {
        "count": int(physical.shape[0]),
        "first_action": physical[0].astype(float).tolist(),
        "last_action": physical[-1].astype(float).tolist(),
        "mean": physical.mean(axis=0).astype(float).tolist(),
        "std": physical.std(axis=0).astype(float).tolist(),
        "min": physical.min(axis=0).astype(float).tolist(),
        "max": physical.max(axis=0).astype(float).tolist(),
        "l2_mean": float(np.linalg.norm(physical[:, :6], axis=1).mean()),
        "l2_max": float(np.linalg.norm(physical[:, :6], axis=1).max()),
        "gripper_first": float(physical[0, -1]),
        "gripper_last": float(physical[-1, -1]),
        "gripper_mean": float(physical[:, -1].mean()),
    }
    if raw_rows:
        raw = np.asarray(raw_rows, dtype=np.float64)
        summary["raw_first_action"] = raw[0].astype(float).tolist()
        summary["raw_last_action"] = raw[-1].astype(float).tolist()
    if norm_rows:
        norm = np.asarray(norm_rows, dtype=np.float64)
        summary["norm_first_action"] = norm[0].astype(float).tolist()
        summary["norm_last_action"] = norm[-1].astype(float).tolist()
    return summary


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


class RemoteACTPolicy:
    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=120)
        self.step_count = 0
        self.error_count = 0

    def reset(self) -> None:
        send_msg(self.sock, {"type": "reset"})
        response = recv_msg(self.sock)
        if not response.get("ok"):
            self.error_count += 1
            raise RuntimeError(response)

    def step(self, obs: dict, goal: str) -> dict:
        send_msg(self.sock, {"type": "step", "obs": obs, "goal": goal})
        response = recv_msg(self.sock)
        if not response.get("ok"):
            self.error_count += 1
            raise RuntimeError(response)
        self.step_count += 1
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


def run_sequence(
    *,
    env,
    policy: RemoteACTPolicy,
    task_oracle,
    val_annotations,
    initial_state: dict,
    eval_sequence,
    sequence_index: int,
    checkpoint_label: str,
    ep_len: int,
    frame_stride: int,
    out_dir: Path,
) -> dict:
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    subtask_rows = []
    frames = []
    state_seed = deterministic_seed(initial_state)

    for subtask_index, subtask in enumerate(eval_sequence):
        obs = env.get_obs()
        task_name = str(subtask)
        lang_annotation = str(val_annotations[task_name][0])
        policy.reset()
        start_info = env.get_info()
        current_info = start_info
        start_robot_obs = np.asarray(obs["robot_obs"], dtype=np.float64).copy()
        start_policy_steps = policy.step_count
        start_policy_errors = policy.error_count
        success = False
        steps_run = 0
        success_step = None
        action_rows = []
        last_response = {}
        frames.append(
            draw_overlay(
                safe_frame(obs),
                [
                    f"{checkpoint_label} seq={sequence_index} sub={subtask_index + 1}/5 {task_name}",
                    f"goal: {lang_annotation}",
                    f"step=0 success_count={success_counter}",
                ],
            )
        )

        for step in range(ep_len):
            response = policy.step(obs, lang_annotation)
            last_response = response
            action = np.asarray(response["action"], dtype=np.float32)
            action_rows.append(
                {
                    "action": action.astype(np.float32).tolist(),
                    "action_raw": response.get("action_raw"),
                    "action_norm": response.get("action_norm"),
                }
            )
            obs, _, _, current_info = env.step(action)
            steps_run = step + 1
            current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
            success = bool(current_task_info)
            if (step + 1) % frame_stride == 0 or success or (step + 1) == ep_len:
                frames.append(
                    draw_overlay(
                        safe_frame(obs),
                        [
                            f"{checkpoint_label} seq={sequence_index} sub={subtask_index + 1}/5 {task_name}",
                            f"goal: {lang_annotation}",
                            f"step={step + 1} success={success} success_count={success_counter}",
                        ],
                    )
                )
            if success:
                success_step = step + 1
                break

        end_robot_obs = np.asarray(obs["robot_obs"], dtype=np.float64).copy()
        scene_before = start_info.get("scene_info", start_info) if isinstance(start_info, dict) else start_info
        scene_after = current_info.get("scene_info", current_info) if isinstance(current_info, dict) else current_info

        subtask_rows.append(
            {
                "subtask_index": subtask_index,
                "task": task_name,
                "language": lang_annotation,
                "canonical_task": last_response.get("canonical_task"),
                "canonical_task_id": last_response.get("canonical_task_id"),
                "goal_used_by_model": last_response.get("goal_used_by_model"),
                "success": success,
                "success_step": success_step,
                "steps": steps_run,
                "horizon": ep_len,
                "action_summary": actions_summary(action_rows),
                "robot_obs_start_summary": array_summary(start_robot_obs),
                "robot_obs_end_summary": array_summary(end_robot_obs),
                "robot_obs_delta_summary": array_delta_summary(start_robot_obs, end_robot_obs),
                "scene_info_delta_summary": numeric_delta_summary(scene_before, scene_after),
                "server_step_count_delta": int(policy.step_count - start_policy_steps),
                "server_error_count_delta": int(policy.error_count - start_policy_errors),
            }
        )
        if success:
            success_counter += 1
        else:
            break

    case_dir = out_dir / f"seq_{sequence_index:02d}"
    video_path = case_dir / "rollout.mp4"
    write_video(video_path, frames, fps=6)
    keyframe_paths = save_keyframes(case_dir, frames)
    case = {
        "sequence_index": sequence_index,
        "initial_state": initial_state,
        "initial_state_seed": int(state_seed),
        "initial_state_method": INITIAL_STATE_METHOD,
        "eval_sequence": [str(task) for task in eval_sequence],
        "successful_subtasks": success_counter,
        "subtasks": subtask_rows,
        "video_path": str(video_path),
        "keyframe_paths": keyframe_paths,
        "server_step_count_total": int(policy.step_count),
        "server_error_count_total": int(policy.error_count),
    }
    (case_dir / "rollout_log.json").write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
    return case


def run(args: argparse.Namespace) -> dict:
    env = create_env(args.dataset_path)
    policy = RemoteACTPolicy(args.host, args.port)
    task_oracle = load_task_oracle()
    val_annotations = load_annotations()
    sequences = get_sequences(args.num_sequences, num_workers=args.num_workers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    cases = []
    try:
        for sequence_index, (initial_state, eval_sequence) in enumerate(sequences):
            cases.append(
                run_sequence(
                    env=env,
                    policy=policy,
                    task_oracle=task_oracle,
                    val_annotations=val_annotations,
                    initial_state=initial_state,
                    eval_sequence=eval_sequence,
                    sequence_index=sequence_index,
                    checkpoint_label=args.checkpoint_label,
                    ep_len=args.ep_len,
                    frame_stride=args.frame_stride,
                    out_dir=args.output_dir,
                )
            )
    finally:
        policy.close()
        if hasattr(env, "close"):
            env.close()

    successful_lengths = [int(case["successful_subtasks"]) for case in cases]
    chain_sr = {str(i + 1): float(sr) for i, sr in enumerate(count_success(successful_lengths))}
    attempted_counter = Counter(sub["task"] for case in cases for sub in case["subtasks"])
    success_counter = Counter(sub["task"] for case in cases for sub in case["subtasks"] if sub["success"])
    short_horizon_smoke = bool(args.ep_len < OFFICIAL_EP_LEN)
    official_scale = bool(args.num_sequences == OFFICIAL_NUM_SEQUENCES and args.ep_len == OFFICIAL_EP_LEN)
    if short_horizon_smoke:
        scale_label = "smoke_only_short_horizon"
    elif args.num_sequences < 20:
        scale_label = "small_scale_gate"
    elif args.num_sequences < OFFICIAL_NUM_SEQUENCES:
        scale_label = "medium_scale_non_official"
    else:
        scale_label = "official_sequence_count_non_abc_to_d"
    task_success = {
        task: {
            "success": int(success_counter[task]),
            "attempted": int(attempted_counter[task]),
            "success_rate": float(success_counter[task] / attempted_counter[task]) if attempted_counter[task] else 0.0,
        }
        for task in sorted(attempted_counter)
    }
    report = {
        "protocol": {
            "dataset_label": args.dataset_label,
            "protocol_tag": args.protocol_tag,
            "scale_label": scale_label,
            "policy_label": args.policy_label,
            "checkpoint_label": args.checkpoint_label,
            "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else "",
            "policy_chunk_size": args.policy_chunk_size,
            "policy_n_action_steps": args.policy_n_action_steps,
            "ep_len": args.ep_len,
            "official_ep_len": OFFICIAL_EP_LEN,
            "num_sequences": len(cases),
            "official_benchmark_num_sequences": OFFICIAL_NUM_SEQUENCES,
            "is_short_horizon_smoke": short_horizon_smoke,
            "is_official_benchmark": False,
            "uses_official_sequence_count": official_scale,
            "sequence_source": "calvin_agent.evaluation.multistep_sequences.get_sequences",
            "initial_state_method": INITIAL_STATE_METHOD,
            "notes": [
                "task_D_D is a D-environment closed-loop source, not the full task_ABC_D ABC-to-D benchmark.",
                "ep_len below 360 is smoke-only and must not be reported as formal Success Rate.",
            ],
        },
        "dataset_path": str(args.dataset_path),
        "validation_folder": str(validation_folder(args.dataset_path)),
        "dataset_label": args.dataset_label,
        "policy": args.policy_label,
        "checkpoint_label": args.checkpoint_label,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else "",
        "policy_chunk_size": args.policy_chunk_size,
        "policy_n_action_steps": args.policy_n_action_steps,
        "source_note": args.source_note,
        "num_sequences": len(cases),
        "ep_len": args.ep_len,
        "frame_stride": args.frame_stride,
        "official_ep_len": OFFICIAL_EP_LEN,
        "official_benchmark_num_sequences": OFFICIAL_NUM_SEQUENCES,
        "is_official_benchmark": False,
        "is_short_horizon_smoke": short_horizon_smoke,
        "scale_label": scale_label,
        "initial_state_method": INITIAL_STATE_METHOD,
        "server_step_count_total": int(policy.step_count),
        "server_error_count_total": int(policy.error_count),
        "avg_successful_subtasks": float(np.mean(successful_lengths)) if successful_lengths else 0.0,
        "chain_success_rate": chain_sr,
        "sequence_success_lengths": successful_lengths,
        "task_success": task_success,
        "elapsed_sec": time.time() - started_at,
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_csv = args.output_json.with_suffix(".csv")
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sequence_index",
            "successful_subtasks",
            "eval_sequence",
            "scale_label",
            "ep_len",
            "server_step_count_total",
            "server_error_count_total",
            "video_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "sequence_index": case["sequence_index"],
                    "successful_subtasks": case["successful_subtasks"],
                    "eval_sequence": " -> ".join(case["eval_sequence"]),
                    "scale_label": scale_label,
                    "ep_len": args.ep_len,
                    "server_step_count_total": case["server_step_count_total"],
                    "server_error_count_total": case["server_error_count_total"],
                    "video_path": case["video_path"],
                }
            )
    print(json.dumps({k: report[k] for k in report if k != "cases"}, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small-scale CALVIN closed-loop evaluation with videos.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--num-sequences", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--ep-len", type=int, default=OFFICIAL_EP_LEN)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--dataset-label", default=DEFAULT_DATASET_LABEL)
    parser.add_argument("--policy-label", default="remote_act")
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--policy-chunk-size", type=int, default=None)
    parser.add_argument("--policy-n-action-steps", type=int, default=None)
    parser.add_argument("--protocol-tag", default="task_D_D_small_scale_closed_loop_ep360")
    parser.add_argument(
        "--source-note",
        default="task_D_D validation small-scale closed-loop evaluation with videos; not official 1000-sequence benchmark",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
