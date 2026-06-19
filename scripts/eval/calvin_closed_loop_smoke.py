#!/usr/bin/env python3
"""Small CALVIN closed-loop smoke test for a native validation root.

Run this in the isolated CALVIN environment. It can either use a zero-action
policy or connect to scripts/eval/act_inference_server.py running in the
LeRobot environment.
"""

from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import sys
import zlib
from pathlib import Path
from typing import Iterable

import hydra
import numpy as np
from numpy import pi
from omegaconf import OmegaConf

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


SMOKE_CASES = [
    (
        "turn_on_led",
        {
            "led": 0,
            "lightbulb": 0,
            "slider": "right",
            "drawer": "closed",
            "red_block": "table",
            "blue_block": "table",
            "pink_block": "slider_right",
            "grasped": 0,
        },
    ),
    (
        "turn_on_lightbulb",
        {
            "led": 0,
            "lightbulb": 0,
            "slider": "right",
            "drawer": "closed",
            "red_block": "table",
            "blue_block": "table",
            "pink_block": "slider_right",
            "grasped": 0,
        },
    ),
    (
        "move_slider_left",
        {
            "led": 0,
            "lightbulb": 0,
            "slider": "right",
            "drawer": "closed",
            "red_block": "table",
            "blue_block": "table",
            "pink_block": "slider_right",
            "grasped": 0,
        },
    ),
]


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

    def reset(self) -> None:
        send_msg(self.sock, {"type": "reset"})
        response = recv_msg(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response)

    def step(self, obs: dict, goal: str) -> np.ndarray:
        send_msg(self.sock, {"type": "step", "obs": obs, "goal": goal})
        response = recv_msg(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response)
        return np.asarray(response["action"], dtype=np.float32)

    def close(self) -> None:
        try:
            send_msg(self.sock, {"type": "close"})
            recv_msg(self.sock)
        finally:
            self.sock.close()


class ZeroPolicy:
    def reset(self) -> None:
        return None

    def step(self, obs: dict, goal: str) -> np.ndarray:
        del obs, goal
        action = np.zeros(7, dtype=np.float32)
        action[-1] = 1.0
        return action

    def close(self) -> None:
        return None


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
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
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


def select_cases(max_tasks: int) -> Iterable[tuple[str, dict]]:
    return SMOKE_CASES[:max_tasks]


def run_smoke(args: argparse.Namespace) -> dict:
    env = create_env(args.dataset_path)
    task_oracle = load_task_oracle()
    val_annotations = load_annotations()
    if args.policy == "remote":
        policy = RemoteACTPolicy(args.host, args.port)
    else:
        policy = ZeroPolicy()

    results = []
    try:
        for task_name, initial_condition in select_cases(args.max_tasks):
            robot_obs, scene_obs = get_env_state_for_initial_condition(initial_condition)
            obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
            start_info = env.get_info()
            lang_annotation = val_annotations[task_name][0]
            policy.reset()
            success = False
            steps_run = 0
            for step in range(args.ep_len):
                action = policy.step(obs, lang_annotation)
                obs, _, _, current_info = env.step(action)
                steps_run = step + 1
                current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {task_name})
                if current_task_info:
                    success = True
                    break
            results.append(
                {
                    "task": task_name,
                    "success": success,
                    "steps": steps_run,
                    "language": str(lang_annotation),
                }
            )
    finally:
        policy.close()
        if hasattr(env, "close"):
            env.close()

    success_count = sum(int(row["success"]) for row in results)
    report = {
        "dataset_path": str(args.dataset_path),
        "validation_folder": str(validation_folder(args.dataset_path)),
        "data_source_note": args.source_note,
        "policy": args.policy,
        "ep_len": args.ep_len,
        "num_tasks": len(results),
        "success_count": success_count,
        "success_rate": success_count / len(results) if results else 0.0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny CALVIN closed-loop smoke test.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--policy", choices=["zero", "remote"], default="zero")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--ep-len", type=int, default=10)
    parser.add_argument(
        "--source-note",
        default="CALVIN native validation smoke source",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/step4/calvin_closed_loop_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
