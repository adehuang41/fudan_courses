"""Episode-filtered CALVIN datasets built from the authoritative split manifest.

This module does not copy parquet or video files. It reads
environment_episode_map.csv, keeps only the requested episode records, and
loads frames lazily from the original LeRobot v2.1 shard directories.
"""

from __future__ import annotations

import bisect
import csv
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.task_text_mapping import DEFAULT_TASK_TEXT_VOCAB, TaskTextMapper


DEFAULT_DATASET_ROOT = Path("/data/dechao/cv_final_pj2/datasets/calvin_task_ABC_D")
DEFAULT_SPLIT_MANIFEST = DEFAULT_DATASET_ROOT / "_download_logs" / "environment_episode_map.csv"

SPLIT_ENVIRONMENTS = {
    "B_only": ("B",),
    "ABC_mixed": ("A", "B", "C"),
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
class EpisodeRecord:
    shard: str
    episode_index: int
    original_frame_idx_min: int
    original_frame_idx_max: int
    environment: str
    num_frames: int
    task: str


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_episode_manifest(manifest_path: Path = DEFAULT_SPLIT_MANIFEST) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected_columns = [
            "shard",
            "episode_index",
            "original_frame_idx_min",
            "original_frame_idx_max",
            "environment",
            "num_frames",
            "task",
        ]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"Unexpected split manifest columns: {reader.fieldnames}")
        for row in reader:
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
    return records


def select_episode_records(
    split_name: str,
    manifest_path: Path = DEFAULT_SPLIT_MANIFEST,
) -> list[EpisodeRecord]:
    if split_name not in SPLIT_ENVIRONMENTS:
        raise ValueError(f"Unknown split_name={split_name!r}; expected one of {sorted(SPLIT_ENVIRONMENTS)}")
    allowed_envs = set(SPLIT_ENVIRONMENTS[split_name])
    return [record for record in load_episode_manifest(manifest_path) if record.environment in allowed_envs]


def summarize_records(records: Iterable[EpisodeRecord]) -> dict:
    summary = {
        "num_episodes": 0,
        "num_frames": 0,
        "by_environment": {},
        "by_shard": {},
    }
    for record in records:
        summary["num_episodes"] += 1
        summary["num_frames"] += record.num_frames
        env_summary = summary["by_environment"].setdefault(
            record.environment, {"num_episodes": 0, "num_frames": 0}
        )
        env_summary["num_episodes"] += 1
        env_summary["num_frames"] += record.num_frames
        shard_summary = summary["by_shard"].setdefault(record.shard, {"num_episodes": 0, "num_frames": 0})
        shard_summary["num_episodes"] += 1
        shard_summary["num_frames"] += record.num_frames
    return summary


class CalvinFilteredFrameDataset(Dataset):
    """Frame-level dataset filtered by CALVIN environment at episode granularity."""

    def __init__(
        self,
        split_name: str,
        dataset_root: Path = DEFAULT_DATASET_ROOT,
        split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
        episode_records: Sequence[EpisodeRecord] | None = None,
    ) -> None:
        self.split_name = split_name
        self.dataset_root = Path(dataset_root)
        self.split_manifest = Path(split_manifest)
        self.allowed_environments = SPLIT_ENVIRONMENTS[split_name]
        self.episode_records = list(episode_records) if episode_records is not None else select_episode_records(split_name, self.split_manifest)
        if not self.episode_records:
            raise ValueError(f"No episodes selected for split {split_name}")

        invalid_envs = sorted({record.environment for record in self.episode_records} - set(self.allowed_environments))
        if invalid_envs:
            raise ValueError(f"Records for {split_name} contain invalid environments: {invalid_envs}")

        self._cumulative_frames: list[int] = []
        total = 0
        for record in self.episode_records:
            total += record.num_frames
            self._cumulative_frames.append(total)

        self._shard_info = {
            shard.name: _read_json(shard / "meta" / "info.json")
            for shard in sorted(self.dataset_root.glob("calvin_task_ABC_D_lerobot_*_4"))
        }

    @property
    def num_episodes(self) -> int:
        return len(self.episode_records)

    @property
    def num_frames(self) -> int:
        return self._cumulative_frames[-1]

    def split_summary(self) -> dict:
        return summarize_records(self.episode_records)

    def __len__(self) -> int:
        return self.num_frames

    def _locate(self, global_frame_idx: int) -> tuple[EpisodeRecord, int]:
        if global_frame_idx < 0 or global_frame_idx >= len(self):
            raise IndexError(global_frame_idx)
        episode_pos = bisect.bisect_right(self._cumulative_frames, global_frame_idx)
        episode_start = 0 if episode_pos == 0 else self._cumulative_frames[episode_pos - 1]
        return self.episode_records[episode_pos], global_frame_idx - episode_start

    def _episode_chunk(self, record: EpisodeRecord) -> int:
        info = self._shard_info[record.shard]
        return record.episode_index // int(info.get("chunks_size", 1000))

    def _parquet_path(self, record: EpisodeRecord) -> Path:
        chunk = self._episode_chunk(record)
        return (
            self.dataset_root
            / record.shard
            / "data"
            / f"chunk-{chunk:03d}"
            / f"episode_{record.episode_index:06d}.parquet"
        )

    @staticmethod
    @lru_cache(maxsize=256)
    def _read_episode_table(path: str):
        return pq.read_table(path)

    def _read_frame(self, record: EpisodeRecord, local_frame_idx: int) -> dict:
        table = self._read_episode_table(str(self._parquet_path(record)))
        row = table.slice(local_frame_idx, 1).to_pydict()
        return {key: value[0] for key, value in row.items()}

    def __getitem__(self, idx: int) -> dict:
        record, local_frame_idx = self._locate(idx)
        frame = self._read_frame(record, local_frame_idx)
        return {
            "observation.state": torch.tensor(frame["observation.state"], dtype=torch.float32),
            "action": torch.tensor(frame["action"], dtype=torch.float32),
            "episode_index": torch.tensor(record.episode_index, dtype=torch.long),
            "frame_index": torch.tensor(int(frame["frame_index"]), dtype=torch.long),
            "index": torch.tensor(int(frame["index"]), dtype=torch.long),
            "task_index": torch.tensor(int(frame["task_index"]), dtype=torch.long),
            "original_frame_idx": torch.tensor(int(frame["original_frame_idx"]), dtype=torch.long),
            "environment": record.environment,
            "shard": record.shard,
            "task": record.task,
        }


def _flatten_stats_value(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def compute_split_stats(
    split_name: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
    keys: Sequence[str] = (
        "observation.state",
        "action",
        "observation.images.image",
        "observation.images.wrist_image",
    ),
) -> dict[str, dict[str, torch.Tensor]]:
    """Aggregate selected split statistics from LeRobot v2.1 per-episode stats."""

    records = select_episode_records(split_name, split_manifest)
    return compute_records_stats(records=records, dataset_root=dataset_root, keys=keys)


def compute_records_stats(
    records: Sequence[EpisodeRecord],
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    keys: Sequence[str] = (
        "observation.state",
        "action",
        "observation.images.image",
        "observation.images.wrist_image",
    ),
) -> dict[str, dict[str, torch.Tensor]]:
    """Aggregate normalization statistics for an explicit episode subset."""

    selected = {(record.shard, record.episode_index) for record in records}
    accum: dict[str, dict[str, np.ndarray | float]] = {}

    shards = sorted({record.shard for record in records})
    for shard in shards:
        path = Path(dataset_root) / shard / "meta" / "episodes_stats.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                if (shard, episode_index) not in selected:
                    continue
                stats = row["stats"]
                for key in keys:
                    if key not in stats:
                        continue
                    item = stats[key]
                    mean = _flatten_stats_value(item["mean"])
                    std = _flatten_stats_value(item["std"])
                    count = float(item["count"][0])
                    if key not in accum:
                        accum[key] = {
                            "count": 0.0,
                            "sum": np.zeros_like(mean, dtype=np.float64),
                            "sum_sq": np.zeros_like(mean, dtype=np.float64),
                        }
                    accum[key]["count"] = float(accum[key]["count"]) + count
                    accum[key]["sum"] = accum[key]["sum"] + count * mean
                    accum[key]["sum_sq"] = accum[key]["sum_sq"] + count * (std**2 + mean**2)

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


def _decode_video_frame_uncached(video_path: str, frame_index: int) -> torch.Tensor:
    """Decode a single RGB frame with PyAV and return float CHW in [0, 1]."""

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx == frame_index:
                image = frame.to_ndarray(format="rgb24")
                tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
                return tensor
    raise IndexError(f"Frame {frame_index} not found in {video_path}")


VIDEO_FRAME_CACHE_SIZE = _read_nonnegative_int_env("CALVIN_VIDEO_FRAME_CACHE_SIZE", 32)


def _make_decode_video_frame(cache_size: int):
    if cache_size <= 0:
        return _decode_video_frame_uncached
    return lru_cache(maxsize=cache_size)(_decode_video_frame_uncached)


_decode_video_frame = _make_decode_video_frame(VIDEO_FRAME_CACHE_SIZE)


def configure_video_frame_cache(maxsize: int) -> None:
    """Set per-process decoded-frame cache size before DataLoader workers fork."""

    if maxsize < 0:
        raise ValueError(f"video frame cache size must be nonnegative, got {maxsize}")
    global VIDEO_FRAME_CACHE_SIZE, _decode_video_frame
    cache_clear = getattr(_decode_video_frame, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    VIDEO_FRAME_CACHE_SIZE = int(maxsize)
    _decode_video_frame = _make_decode_video_frame(VIDEO_FRAME_CACHE_SIZE)


def video_frame_cache_status() -> dict:
    cache_info = getattr(_decode_video_frame, "cache_info", None)
    return {
        "maxsize": VIDEO_FRAME_CACHE_SIZE,
        "enabled": VIDEO_FRAME_CACHE_SIZE > 0,
        "cache_info": cache_info()._asdict() if cache_info is not None else None,
    }


class CalvinACTTrainingDataset(CalvinFilteredFrameDataset):
    """ACT training dataset backed by filtered LeRobot v2.1 parquet/video files."""

    def __init__(
        self,
        split_name: str,
        dataset_root: Path = DEFAULT_DATASET_ROOT,
        split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
        chunk_size: int = 10,
        image_keys: Sequence[str] = ("observation.images.image", "observation.images.wrist_image"),
        max_episodes: int | None = None,
        episode_records: Sequence[EpisodeRecord] | None = None,
        task_vocab_path: Path | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.image_keys = tuple(image_keys)
        self.task_mapper = (
            TaskTextMapper.from_json(Path(task_vocab_path or DEFAULT_TASK_TEXT_VOCAB))
            if task_vocab_path is not None
            else None
        )
        super().__init__(
            split_name=split_name,
            dataset_root=dataset_root,
            split_manifest=split_manifest,
            episode_records=episode_records,
        )
        if max_episodes is not None:
            self.episode_records = self.episode_records[:max_episodes]
            self._cumulative_frames = []
            total = 0
            for record in self.episode_records:
                total += record.num_frames
                self._cumulative_frames.append(total)

    def _video_path(self, record: EpisodeRecord, image_key: str) -> Path:
        chunk = self._episode_chunk(record)
        return (
            self.dataset_root
            / record.shard
            / "videos"
            / f"chunk-{chunk:03d}"
            / image_key
            / f"episode_{record.episode_index:06d}.mp4"
        )

    def _read_action_chunk(self, record: EpisodeRecord, local_frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        table = self._read_episode_table(str(self._parquet_path(record)))
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
            "original_frame_idx": torch.tensor(int(frame["original_frame_idx"]), dtype=torch.long),
            "environment": record.environment,
            "shard": record.shard,
            "task": record.task,
        }
        if self.task_mapper is not None:
            canonical_task = self.task_mapper.canonical_for_training_text(record.task)
            item["canonical_task"] = canonical_task
            item["canonical_task_id"] = torch.tensor(
                self.task_mapper.task_to_id[canonical_task],
                dtype=torch.long,
            )
        for image_key in self.image_keys:
            item[image_key] = _decode_video_frame(str(self._video_path(record, image_key)), local_frame_idx)
        return item


def make_calvin_dataloader(
    split_name: str,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    split_manifest: Path = DEFAULT_SPLIT_MANIFEST,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CalvinFilteredFrameDataset(
        split_name=split_name,
        dataset_root=dataset_root,
        split_manifest=split_manifest,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )
