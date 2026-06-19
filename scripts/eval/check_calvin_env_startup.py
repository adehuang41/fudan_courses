#!/usr/bin/env python3
"""Check whether CALVIN can instantiate an env from a dataset root."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

CALVIN_ROOT = Path("/data/dechao/cv_final_pj2/external/calvin")
CALVIN_ENV = CALVIN_ROOT / "calvin_env"
if str(CALVIN_ENV) not in sys.path:
    sys.path.insert(0, str(CALVIN_ENV))

import calvin_env  # noqa: E402
from calvin_env.envs.play_table_env import get_env  # noqa: E402


def validation_folder(dataset_root: Path) -> Path:
    if (dataset_root / "validation").is_dir():
        return dataset_root / "validation"
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CALVIN env startup.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if getattr(calvin_env, "__file__", None) is None:
        calvin_env.__file__ = str(CALVIN_ENV / "calvin_env" / "__init__.py")

    val_folder = validation_folder(args.dataset_root)
    obs_space = {"rgb_obs": ["rgb_static", "rgb_gripper"], "depth_obs": []}
    report = {
        "dataset_root": str(args.dataset_root),
        "validation_folder": str(val_folder),
        "status": "unknown",
        "obs_space": obs_space,
    }
    env = None
    try:
        env = get_env(val_folder, obs_space=obs_space, show_gui=False)
        obs = env.reset()
        report["status"] = "started"
        report["obs_keys"] = sorted(obs.keys())
        report["rgb_keys"] = sorted(obs.get("rgb_obs", {}).keys())
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
    finally:
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
