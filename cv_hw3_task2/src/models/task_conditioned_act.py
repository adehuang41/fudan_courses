"""Task-id conditioned ACT wrapper.

This keeps the LeRobot ACT vision stack, transformer, chunk size, and action
head unchanged. The only architectural change is to concatenate a learned
task embedding to the normalized robot state before the ACT state projection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_STATE


@dataclass(frozen=True)
class VisualCheckpointLoadReport:
    checkpoint_path: str
    checkpoint_exists: bool
    checkpoint_step: int | None
    attempted: bool
    used_for_initialization: bool
    clean: bool
    exact_loaded: int
    expanded_loaded: int
    skipped: int
    missing_after_load: int
    unexpected_after_load: int
    expanded_keys: list[str]
    skipped_examples: list[str]
    missing_examples: list[str]
    unexpected_examples: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskConditionedACTPolicy(nn.Module):
    """Wrap ACTPolicy and append a learned task embedding to observation.state."""

    def __init__(
        self,
        config: ACTConfig,
        num_tasks: int,
        task_embedding_dim: int,
        task_id_key: str = "canonical_task_id",
    ) -> None:
        super().__init__()
        if task_embedding_dim <= 0:
            raise ValueError(f"task_embedding_dim must be positive, got {task_embedding_dim}")
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        self.config = config
        self.num_tasks = int(num_tasks)
        self.task_embedding_dim = int(task_embedding_dim)
        self.task_id_key = task_id_key
        self.task_embedding = nn.Embedding(self.num_tasks, self.task_embedding_dim)
        self.act = ACTPolicy(config)

    def reset(self) -> None:
        self.act.reset()

    def get_optim_params(self) -> list[dict]:
        non_backbone_params = [
            param
            for name, param in self.named_parameters()
            if not name.startswith("act.model.backbone") and param.requires_grad
        ]
        backbone_params = [
            param
            for name, param in self.named_parameters()
            if name.startswith("act.model.backbone") and param.requires_grad
        ]
        return [
            {"params": non_backbone_params},
            {"params": backbone_params, "lr": self.config.optimizer_lr_backbone},
        ]

    def _task_ids(self, batch: dict[str, Tensor]) -> Tensor:
        if self.task_id_key not in batch:
            raise KeyError(f"Task-conditioned ACT requires batch[{self.task_id_key!r}]")
        task_ids = batch[self.task_id_key]
        if not torch.is_tensor(task_ids):
            task_ids = torch.as_tensor(task_ids, device=batch[OBS_STATE].device)
        task_ids = task_ids.to(device=batch[OBS_STATE].device, dtype=torch.long)
        if task_ids.ndim == 2 and task_ids.shape[-1] == 1:
            task_ids = task_ids.squeeze(-1)
        if task_ids.ndim != 1:
            raise ValueError(f"Expected task ids with shape [B], got {list(task_ids.shape)}")
        if torch.any(task_ids < 0) or torch.any(task_ids >= self.num_tasks):
            raise ValueError(
                f"Task ids out of range [0, {self.num_tasks}): "
                f"min={int(task_ids.min())}, max={int(task_ids.max())}"
            )
        return task_ids

    def _with_task_embedding(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if OBS_STATE not in batch:
            raise KeyError(f"Task-conditioned ACT requires batch[{OBS_STATE!r}]")
        task_embedding = self.task_embedding(self._task_ids(batch))
        state = batch[OBS_STATE]
        if state.ndim != 2:
            raise ValueError(f"Expected state with shape [B, state_dim], got {list(state.shape)}")
        if state.shape[0] != task_embedding.shape[0]:
            raise ValueError(
                f"State batch size {state.shape[0]} does not match task ids {task_embedding.shape[0]}"
            )
        out = dict(batch)
        out[OBS_STATE] = torch.cat([state, task_embedding.to(dtype=state.dtype)], dim=-1)
        return out

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        return self.act.predict_action_chunk(self._with_task_embedding(batch))

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        return self.act.select_action(self._with_task_embedding(batch))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.act.forward(self._with_task_embedding(batch))


def load_visual_act_checkpoint(
    policy: TaskConditionedACTPolicy,
    checkpoint_path: Path,
) -> VisualCheckpointLoadReport:
    """Load compatible visual-only ACT weights into a task-conditioned wrapper.

    Old ACT keys are mapped under the wrapper's ``act.`` prefix. State projection
    weights are expanded by copying the original robot-state columns and leaving
    the newly added task-embedding columns at their random initialization.
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return VisualCheckpointLoadReport(
            checkpoint_path=str(checkpoint_path),
            checkpoint_exists=False,
            checkpoint_step=None,
            attempted=True,
            used_for_initialization=False,
            clean=False,
            exact_loaded=0,
            expanded_loaded=0,
            skipped=0,
            missing_after_load=0,
            unexpected_after_load=0,
            expanded_keys=[],
            skipped_examples=[],
            missing_examples=[],
            unexpected_examples=[],
            note="checkpoint path does not exist; scratch initialization required",
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_state = checkpoint.get("policy_state_dict")
    if not isinstance(old_state, dict):
        return VisualCheckpointLoadReport(
            checkpoint_path=str(checkpoint_path),
            checkpoint_exists=True,
            checkpoint_step=checkpoint.get("step"),
            attempted=True,
            used_for_initialization=False,
            clean=False,
            exact_loaded=0,
            expanded_loaded=0,
            skipped=0,
            missing_after_load=0,
            unexpected_after_load=0,
            expanded_keys=[],
            skipped_examples=[],
            missing_examples=[],
            unexpected_examples=[],
            note="checkpoint has no policy_state_dict; scratch initialization required",
        )

    target_state = policy.state_dict()
    mapped_state: dict[str, Tensor] = {}
    skipped: list[str] = []
    expanded_keys: list[str] = []
    exact_loaded = 0
    expanded_loaded = 0

    expandable_suffixes = (
        "model.encoder_robot_state_input_proj.weight",
        "model.vae_encoder_robot_state_input_proj.weight",
    )

    for old_key, value in old_state.items():
        target_key = f"act.{old_key}"
        if target_key not in target_state:
            skipped.append(f"{old_key}:missing_target")
            continue
        target_value = target_state[target_key]
        if tuple(value.shape) == tuple(target_value.shape):
            mapped_state[target_key] = value
            exact_loaded += 1
            continue
        if old_key.endswith(expandable_suffixes) and value.ndim == 2 and target_value.ndim == 2:
            if value.shape[0] == target_value.shape[0] and value.shape[1] < target_value.shape[1]:
                expanded = target_value.clone()
                expanded[:, : value.shape[1]] = value
                mapped_state[target_key] = expanded
                expanded_loaded += 1
                expanded_keys.append(target_key)
                continue
        skipped.append(f"{old_key}:shape {list(value.shape)} -> {list(target_value.shape)}")

    load_result = policy.load_state_dict(mapped_state, strict=False)
    missing = list(load_result.missing_keys)
    unexpected = list(load_result.unexpected_keys)
    allowed_missing = {"task_embedding.weight"}
    clean = not skipped and not unexpected and set(missing).issubset(allowed_missing)

    return VisualCheckpointLoadReport(
        checkpoint_path=str(checkpoint_path),
        checkpoint_exists=True,
        checkpoint_step=checkpoint.get("step"),
        attempted=True,
        used_for_initialization=clean,
        clean=clean,
        exact_loaded=exact_loaded,
        expanded_loaded=expanded_loaded,
        skipped=len(skipped),
        missing_after_load=len(missing),
        unexpected_after_load=len(unexpected),
        expanded_keys=expanded_keys,
        skipped_examples=skipped[:20],
        missing_examples=missing[:20],
        unexpected_examples=unexpected[:20],
        note="clean partial load" if clean else "unclean partial load; scratch initialization recommended",
    )
