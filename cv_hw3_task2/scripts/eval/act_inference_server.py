#!/usr/bin/env python3
"""ACT inference socket server for CALVIN closed-loop smoke tests.

Run this in the LeRobot environment. The CALVIN simulator process can run in a
separate Python 3.8 environment and send observations over localhost.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import socket
import struct
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path("/home/dechao/cv_final_pj2")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from src.data.calvin_filtered_dataset import EpisodeRecord, compute_records_stats  # noqa: E402


DEFAULT_RUN_DIR = Path("/data/dechao/cv_final_pj2/outputs/baseline/ABC_mixed_act_baseline")
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints" / "best_val_action_l1.pt"
IMAGE_KEYS = ("observation.images.image", "observation.images.wrist_image")


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_policy_config(policy_actual: dict, device: str, n_action_steps: int | None = None) -> ACTConfig:
    cfg = deepcopy(policy_actual)
    cfg["device"] = device
    cfg["pretrained_backbone_weights"] = None
    if n_action_steps is not None:
        cfg["n_action_steps"] = int(n_action_steps)
    cfg["input_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in cfg["input_features"].items()
    }
    cfg["output_features"] = {
        key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
        for key, value in cfg["output_features"].items()
    }
    cfg.pop("type", None)
    return ACTConfig(**cfg)


def read_split_records(split_csv: Path, split: str = "train") -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with split_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            records.append(
                EpisodeRecord(
                    shard=row["shard"],
                    episode_index=int(row["episode_index"]),
                    original_frame_idx_min=int(row["original_frame_idx_min"]),
                    original_frame_idx_max=int(row["original_frame_idx_max"]),
                    environment=row["environment"],
                    num_frames=int(row["num_frames"]),
                    task=row["task"],
                )
            )
    if not records:
        raise ValueError(f"No {split!r} records found in {split_csv}")
    return records


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


def image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {arr.shape}")
    tensor = torch.as_tensor(arr.copy(), dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )[0]
    return tensor


class ACTInferenceModel:
    def __init__(
        self,
        run_dir: Path,
        checkpoint: Path,
        device: torch.device,
        n_action_steps: int | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.device = device

        config_yaml = read_yaml(run_dir / "config.yaml")
        self.policy_config = build_policy_config(config_yaml["policy_actual"], str(device), n_action_steps)
        self.policy = ACTPolicy(self.policy_config).to(device)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.policy.eval()

        split_csv = Path(config_yaml["split"]["episode_split_csv"])
        dataset_root = Path(config_yaml["train"]["dataset"]["dataset_root"])
        records = read_split_records(split_csv, split="train")
        self.stats = compute_records_stats(records=records, dataset_root=dataset_root)
        self.stats = {
            key: {name: tensor.to(device) for name, tensor in value.items()}
            for key, value in self.stats.items()
        }

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _normalize(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        if key not in self.stats:
            return tensor
        mean = self.stats[key]["mean"]
        std = self.stats[key]["std"].clamp_min(1e-6)
        return (tensor - mean) / std

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        stats = self.stats[ACTION]
        return action * stats["std"] + stats["mean"]

    def step(self, obs: dict, goal: str | None = None) -> dict:
        del goal
        rgb_obs = obs["rgb_obs"]
        state = torch.as_tensor(np.asarray(obs["robot_obs"]).copy(), dtype=torch.float32, device=self.device)
        image = image_to_tensor(rgb_obs["rgb_static"], self.device)
        wrist_image = image_to_tensor(rgb_obs["rgb_gripper"], self.device)

        batch = {
            OBS_STATE: self._normalize(OBS_STATE, state).unsqueeze(0),
            "observation.images.image": self._normalize("observation.images.image", image).unsqueeze(0),
            "observation.images.wrist_image": self._normalize(
                "observation.images.wrist_image", wrist_image
            ).unsqueeze(0),
        }

        with torch.inference_mode():
            action_norm = self.policy.select_action(batch)
        action_physical = self._unnormalize_action(action_norm).detach().float().cpu().numpy().reshape(-1)
        action = action_physical.copy()
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        action[6] = 1.0 if action[6] >= 0.0 else -1.0
        return {
            "action": action.astype(np.float32).tolist(),
            "action_raw": action_physical.astype(np.float32).tolist(),
            "action_norm": action_norm.detach().float().cpu().numpy().reshape(-1).astype(np.float32).tolist(),
            "goal_used_by_model": False,
        }


def serve(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = ACTInferenceModel(args.run_dir, args.checkpoint, device, args.n_action_steps)

    request_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"ACT inference server ready on {args.host}:{args.port} "
            f"device={device} checkpoint={args.checkpoint} "
            f"chunk_size={model.policy_config.chunk_size} n_action_steps={model.policy_config.n_action_steps}",
            flush=True,
        )
        while True:
            conn, addr = server.accept()
            print(f"client connected from {addr}", flush=True)
            with conn:
                while True:
                    try:
                        request = recv_msg(conn)
                    except EOFError:
                        break
                    request_count += 1
                    if request["type"] == "reset":
                        model.reset()
                        send_msg(conn, {"ok": True})
                    elif request["type"] == "step":
                        response = model.step(request["obs"], request.get("goal"))
                        response["ok"] = True
                        send_msg(conn, response)
                    elif request["type"] == "close":
                        send_msg(conn, {"ok": True})
                        return
                    else:
                        send_msg(conn, {"ok": False, "error": f"unknown request type {request['type']!r}"})
                    if args.max_requests and request_count >= args.max_requests:
                        return
            print("client disconnected", flush=True)
            if args.one_shot:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve ACT checkpoint actions over a localhost socket.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--one-shot", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
