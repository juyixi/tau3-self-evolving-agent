from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


SCHEMA_VERSION = 3


class EventWriter(Protocol):
    def append(self, event: dict[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    run_id: str
    benchmark: str
    mode: str
    split: str
    model_revision: str
    checkpoint: str | None
    memory_source_namespace: str | None
    memory_snapshot_id: str | None
    cross_domain_memory: bool
    memory_generation: int
    seed: int
    event_writer: EventWriter
    task_groups: Mapping[str, str] = field(default_factory=dict)
    default_task_group: str = "default"

    def task_group_for(self, task_id: str) -> str:
        return self.task_groups.get(task_id, self.default_task_group)

    def event(self, event_type: str, task_id: str, **payload: Any) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type,
            "run_id": self.run_id,
            "benchmark": self.benchmark,
            "mode": self.mode,
            "split": self.split,
            "task_id": task_id,
            "task_group": self.task_group_for(task_id),
            "model_revision": self.model_revision,
            "checkpoint": self.checkpoint,
            "memory_source_namespace": self.memory_source_namespace,
            "memory_snapshot_id": self.memory_snapshot_id,
            "cross_domain_memory": self.cross_domain_memory,
            "memory_generation": self.memory_generation,
            "seed": self.seed,
            **payload,
        }


@dataclass(slots=True)
class BufferedEventWriter:
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def flush_to(self, writer: EventWriter) -> None:
        for event in self.events:
            writer.append(event)
