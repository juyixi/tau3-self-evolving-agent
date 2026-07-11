from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


SCHEMA_VERSION = 1


class EventWriter(Protocol):
    def append(self, event: dict[str, Any]) -> None:
        """Durably append one canonical rollout event."""


@dataclass(frozen=True, slots=True)
class RunContext:
    """Stable provenance and output dependencies for a no-memory baseline run."""

    run_id: str
    iteration: int
    split: str
    model_revision: str
    adapter_revision: None
    memory_snapshot_id: None
    seed: int
    event_writer: EventWriter
    task_groups: Mapping[str, str] = field(default_factory=dict)
    temperature: float = 1.0
    top_p: float = 0.95

    def task_group_for(self, task_id: str) -> str:
        return self.task_groups.get(task_id, "baseline")

    def event(self, event_type: str, task_id: str, **payload: Any) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "split": self.split,
            "task_id": task_id,
            "task_group": self.task_group_for(task_id),
            "model_revision": self.model_revision,
            "adapter_revision": self.adapter_revision,
            "memory_snapshot_id": self.memory_snapshot_id,
            "seed": self.seed,
            **payload,
        }
