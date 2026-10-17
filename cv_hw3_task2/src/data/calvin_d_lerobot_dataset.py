"""CALVIN task_D_D LeRobot dataset adapter for offline ACT evaluation."""

from __future__ import annotations

import bisect
import io
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_D_DATASET_ROOT = Path("/data/dechao/cv_final_pj2/datasets/calvin_D_D_lerobot")

D_IMAGE_KEY_MAP = {
    "observation.images.image": "observation.images.top",
    "observation.images.wrist_image": "observation.images.wrist",
}


def _read_nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return value


@dataclass(frozen=True)
class CalvinDEpisodeRecord:
    episode_index: int
    task: str
    task_index: int
    num_frames: int
    episode_chunk: int
    environment: str = "D"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_d_episode_records(dataset_root: Path = DEFAULT_D_DATASET_ROOT) -> list[CalvinDEpisodeRecord]:
    dataset_root = Path(dataset_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    chunks_size = int(info["chunks_size"])
    tasks_by_index = {
        int(row["task_index"]): row["task"]
        for row in _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    }

    records = []
    for row in _read_jsonl(dataset_root / "meta" / "episodes.jsonl"):
        episode_index = int(row["episode_index"])
        task = str(row["tasks"][0]) if row.get("tasks") else ""
        task_index = next(
            (idx for idx, name in tasks_by_index.items() if name == task),
            -1,
        )
        records.append(
            CalvinDEpisodeRecord(
                episode_index=episode_index,
                task=task,
                task_index=task_index,
                num_frames=int(row["length"]),
                episode_chunk=episode_index // chunks_size,
            )
        )
    return records


def summarize_d_records(records: Iterable[CalvinDEpisodeRecord]) -> dict:
    summary = {
        "num_episodes": 0,
        "num_frames": 0,
        "by_environment": {},
        "by_task": {},
    }
    for record in records:
        summary["num_episodes"] += 1
        summary["num_frames"] += record.num_frames
        env_summary = summary["by_environment"].setdefault(
            record.environment, {"num_episodes": 0, "num_frames": 0}
        )
        env_summary["num_episodes"] += 1
        env_summary["num_frames"] += record.num_frames
        task_summary = summary["by_task"].setdefault(record.task, {"num_episodes": 0, "num_frames": 0})
        task_summary["num_episodes"] += 1
        task_summary["num_frames"] += record.num_frames
    return summary


def _read_episode_table_uncached(path: str):
    return pq.read_table(path)


def _make_read_episode_table(cache_size: int):
    if cache_size <= 0:
        return _read_episode_table_uncached
    return lru_cache(maxsize=cache_size)(_read_episode_table_uncached)


_EPISODE_TABLE_CACHE_SIZE = _read_nonnegative_int_env("CALVIN_D_PARQUET_TABLE_CACHE_SIZE", 32)
_read_episode_table = _make_read_episode_table(_EPISODE_TABLE_CACHE_SIZE)


def configure_episode_table_cache(maxsize: int) -> None:
    """Set per-process parquet episode-table cache size before DataLoader workers fork."""

    global _EPISODE_TABLE_CACHE_SIZE, _read_episode_table
    if maxsize < 0:
        raise ValueError(f"episode table cache size must be nonnegative, got {maxsize}")
    cache_clear = getattr(_read_episode_table, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    _EPISODE_TABLE_CACHE_SIZE = int(maxsize)
    _read_episode_table = _make_read_episode_table(_EPISODE_TABLE_CACHE_SIZE)


def episode_table_cache_status() -> dict:
    cache_info = getattr(_read_episode_table, "cache_info", None)
    return {
        "enabled": _EPISODE_TABLE_CACHE_SIZE > 0,
        "maxsize": _EPISODE_TABLE_CACHE_SIZE,
        "cache_info": cache_info()._asdict() if cache_info is not None else None,
    }


def _image_struct_to_tensor(value: dict, size: tuple[int, int]) -> torch.Tensor:
    image_bytes = value.get("bytes")
    if image_bytes is None:
        raise ValueError("Expected parquet image struct with a bytes field")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if image.size != size:
        image = image.resize(size, Image.BILINEAR)
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0


class CalvinDDACTDataset(Dataset):
    """Frame-level D dataset that emits ACT-compatible observation/action chunks."""

    def __init__(
        self,
        dataset_root: Path = DEFAULT_D_DATASET_ROOT,
        chunk_size: int = 100,
        image_key_map: dict[str, str] | None = None,
        image_size: tuple[int, int] = (256, 256),
        episode_records: Sequence[CalvinDEpisodeRecord] | None = None,
        max_episodes: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.chunk_size = int(chunk_size)
        self.image_key_map = dict(image_key_map or D_IMAGE_KEY_MAP)
        self.image_size = image_size
        records = list(episode_records) if episode_records is not None else load_d_episode_records(self.dataset_root)
        if max_episodes is not None:
            records = records[:max_episodes]
        if not records:
            raise ValueError("No D episodes available")
        self.episode_records = records
        self._cumulative_frames: list[int] = []
        total = 0
        for record in self.episode_records:
            total += record.num_frames
            self._cumulative_frames.append(total)

    @property
    def num_episodes(self) -> int:
        return len(self.episode_records)

    @property
    def num_frames(self) -> int:
        return self._cumulative_frames[-1]

    def split_summary(self) -> dict:
        return summarize_d_records(self.episode_records)

    def __len__(self) -> int:
        return self.num_frames

    def _locate(self, global_frame_idx: int) -> tuple[CalvinDEpisodeRecord, int]:
        if global_frame_idx < 0 or global_frame_idx >= len(self):
            raise IndexError(global_frame_idx)
        episode_pos = bisect.bisect_right(self._cumulative_frames, global_frame_idx)
        episode_start = 0 if episode_pos == 0 else self._cumulative_frames[episode_pos - 1]
        return self.episode_records[episode_pos], global_frame_idx - episode_start

    def _parquet_path(self, record: CalvinDEpisodeRecord) -> Path:
        return (
            self.dataset_root
            / "data"
            / f"chunk-{record.episode_chunk:03d}"
            / f"episode_{record.episode_index:06d}.parquet"
        )

    def _read_frame(self, record: CalvinDEpisodeRecord, local_frame_idx: int) -> dict:
        table = _read_episode_table(str(self._parquet_path(record)))
        row = table.slice(local_frame_idx, 1).to_pydict()
        return {key: value[0] for key, value in row.items()}

    def _read_action_chunk(self, record: CalvinDEpisodeRecord, local_frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        table = _read_episode_table(str(self._parquet_path(record)))
        action_dim = len(table.column("action")[0].as_py())
        actions = torch.zeros((self.chunk_size, action_dim), dtype=torch.float32)
        is_pad = torch.ones((self.chunk_size,), dtype=torch.bool)
        for offset in range(self.chunk_size):
            row_idx = local_frame_idx + offset
            if row_idx >= record.num_frames:
                break
            actions[offset] = torch.tensor(table.column("action")[row_idx].as_py(), dtype=torch.float32)
            is_pad[offset] = False
        return actions, is_pad

    def __getitem__(self, idx: int) -> dict:
        record, local_frame_idx = self._locate(idx)
        frame = self._read_frame(record, local_frame_idx)
        actions, action_is_pad = self._read_action_chunk(record, local_frame_idx)
        item = {
            "observation.state": torch.tensor(frame["observation.state"], dtype=torch.float32),
            "action": actions,
            "action_is_pad": action_is_pad,
            "episode_index": torch.tensor(record.episode_index, dtype=torch.long),
            "frame_index": torch.tensor(int(frame["frame_index"]), dtype=torch.long),
            "index": torch.tensor(int(frame["index"]), dtype=torch.long),
            "task_index": torch.tensor(int(frame["task_index"]), dtype=torch.long),
            "environment": record.environment,
            "task": record.task,
        }
        for target_key, source_key in self.image_key_map.items():
            item[target_key] = _image_struct_to_tensor(frame[source_key], self.image_size)
        return item


def compute_d_dataset_stats(
    dataset_root: Path = DEFAULT_D_DATASET_ROOT,
    key_map: dict[str, str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Aggregate D metadata stats and expose them with ACT-compatible keys."""

    dataset_root = Path(dataset_root)
    key_map = dict(key_map or D_IMAGE_KEY_MAP)
    source_to_target = {source: target for target, source in key_map.items()}
    accum: dict[str, dict[str, np.ndarray | float]] = {}
    with (dataset_root / "meta" / "episodes_stats.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for source_key, item in row["stats"].items():
                target_key = source_to_target.get(source_key, source_key)
                if target_key not in {
                    "observation.state",
                    "action",
                    "observation.images.image",
                    "observation.images.wrist_image",
                }:
                    continue
                mean = np.asarray(item["mean"], dtype=np.float64).reshape(-1)
                std = np.asarray(item["std"], dtype=np.float64).reshape(-1)
                count = float(item["count"][0])
                if target_key not in accum:
                    accum[target_key] = {
                        "count": 0.0,
                        "sum": np.zeros_like(mean, dtype=np.float64),
                        "sum_sq": np.zeros_like(mean, dtype=np.float64),
                    }
                accum[target_key]["count"] = float(accum[target_key]["count"]) + count
                accum[target_key]["sum"] = accum[target_key]["sum"] + count * mean
                accum[target_key]["sum_sq"] = accum[target_key]["sum_sq"] + count * (std**2 + mean**2)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in accum.items():
        count = float(value["count"])
        mean = value["sum"] / count
        var = np.maximum(value["sum_sq"] / count - mean**2, 1e-12)
        std = np.sqrt(var)
        if key.startswith("observation.images."):
            mean = mean.reshape(3, 1, 1)
            std = std.reshape(3, 1, 1)
        out[key] = {
            "mean": torch.tensor(mean, dtype=torch.float32),
            "std": torch.tensor(std, dtype=torch.float32),
        }
    return out
