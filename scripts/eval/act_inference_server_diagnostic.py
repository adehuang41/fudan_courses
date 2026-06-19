#!/usr/bin/env python3
"""Diagnostic ACT inference server for CALVIN closed-loop failures.

This wraps the normal ACTInferenceModel but records per-request observations,
goals, and actions. It is instrumentation only; it does not change action
generation.
"""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from act_inference_server import ACTInferenceModel, recv_msg, send_msg
from lerobot.utils.constants import ACTION, OBS_STATE


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def arr_summary(value) -> dict:
    arr = np.asarray(value, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
    }


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().float().cpu().reshape(-1).tolist()


def jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_metadata(model: ACTInferenceModel, args: argparse.Namespace) -> dict:
    stats = {}
    for key, value in model.stats.items():
        stats[key] = {
            name: tensor_to_list(tensor)
            for name, tensor in value.items()
            if name in {"mean", "std", "min", "max"}
        }
    return {
        "type": "metadata",
        "time": now_utc(),
        "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(model.device),
        "goal_used_by_model": False,
        "goal_ignored_reason": "Current ACT checkpoint input_features contain state and two image streams only; no language feature is passed to ACTPolicy.",
        "policy_input_features": {
            key: {
                "type": value.type.value,
                "shape": list(value.shape),
            }
            for key, value in model.policy_config.input_features.items()
        },
        "policy_output_features": {
            key: {
                "type": value.type.value,
                "shape": list(value.shape),
            }
            for key, value in model.policy_config.output_features.items()
        },
        "normalization_stats": stats,
    }


def observation_summary(obs: dict) -> dict:
    rgb_obs = obs["rgb_obs"]
    return {
        "robot_obs": arr_summary(obs["robot_obs"]),
        "rgb_static": arr_summary(rgb_obs["rgb_static"]),
        "rgb_gripper": arr_summary(rgb_obs["rgb_gripper"]),
        "rgb_static_shape": list(np.asarray(rgb_obs["rgb_static"]).shape),
        "rgb_gripper_shape": list(np.asarray(rgb_obs["rgb_gripper"]).shape),
    }


def serve(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = ACTInferenceModel(args.run_dir, args.checkpoint, device)
    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.log_jsonl.exists():
        args.log_jsonl.unlink()
    jsonl_append(args.log_jsonl, model_metadata(model, args))

    request_count = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(
            f"ACT diagnostic server ready on {args.host}:{args.port} "
            f"device={device} checkpoint={args.checkpoint} log={args.log_jsonl}",
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
                        jsonl_append(
                            args.log_jsonl,
                            {
                                "type": "reset",
                                "time": now_utc(),
                                "request_count": request_count,
                            },
                        )
                        send_msg(conn, {"ok": True})
                    elif request["type"] == "step":
                        goal = request.get("goal")
                        response = model.step(request["obs"], goal)
                        response["ok"] = True
                        action = np.asarray(response["action"], dtype=np.float64)
                        action_raw = np.asarray(response["action_raw"], dtype=np.float64)
                        action_norm = np.asarray(response["action_norm"], dtype=np.float64)
                        jsonl_append(
                            args.log_jsonl,
                            {
                                "type": "step",
                                "time": now_utc(),
                                "request_count": request_count,
                                "goal_received": str(goal),
                                "goal_used_by_model": False,
                                "obs_summary": observation_summary(request["obs"]),
                                "action": action.tolist(),
                                "action_raw": action_raw.tolist(),
                                "action_norm": action_norm.tolist(),
                                "action_summary": arr_summary(action),
                                "action_raw_summary": arr_summary(action_raw),
                                "action_norm_summary": arr_summary(action_norm),
                                "gripper_final": float(action[-1]),
                                "gripper_raw": float(action_raw[-1]),
                                "gripper_norm": float(action_norm[-1]),
                            },
                        )
                        send_msg(conn, response)
                    elif request["type"] == "close":
                        jsonl_append(
                            args.log_jsonl,
                            {
                                "type": "close",
                                "time": now_utc(),
                                "request_count": request_count,
                            },
                        )
                        send_msg(conn, {"ok": True})
                        return
                    else:
                        send_msg(conn, {"ok": False, "error": f"unknown request type {request['type']!r}"})
            print("client disconnected", flush=True)
            if args.one_shot:
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve ACT actions and log diagnostics over localhost.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-jsonl", type=Path, required=True)
    parser.add_argument("--one-shot", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
