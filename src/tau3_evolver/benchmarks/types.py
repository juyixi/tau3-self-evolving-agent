from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from tau3_evolver.config import ProjectConfig
from tau3_evolver.execution.request import ExecutionMode


class BenchmarkDefinition(Protocol):
    """Immutable registration data that can prepare one benchmark runtime."""

    name: str
    default_memory_namespace: str

    def prepare(
        self, config: ProjectConfig, mode: ExecutionMode
    ) -> "PreparedBenchmark": ...


@dataclass(frozen=True, slots=True)
class RuntimeOrigin:
    source_root: Path
    package_version: str | None
    git_commit: str | None


@dataclass(frozen=True, slots=True)
class PreparedBenchmark:
    """Runtime-loaded benchmark data consumed by the generic executor."""

    name: str
    task_type: type[Any]
    task_catalog: tuple[Any, ...]
    task_ids: tuple[str, ...]
    split_name: str
    split_hash: str
    environment_factory: Callable[..., Any]
    runtime: Any
    run_domain: Callable[[Any], Any]
    text_run_config_type: type[Any]
    registry: Any
    runtime_origin: RuntimeOrigin
    default_memory_namespace: str
    task_group: str
    evaluator_binding: Callable[..., Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_catalog:
            raise ValueError("prepared benchmark task catalog must not be empty")
        if len(self.task_catalog) != len(self.task_ids):
            raise ValueError("task_catalog and task_ids must have the same length")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("prepared benchmark contains duplicate task IDs")
        if not all(isinstance(task, self.task_type) for task in self.task_catalog):
            raise TypeError("prepared benchmark contains an unexpected task type")

    def first_tasks(self, count: int) -> "PreparedBenchmark":
        """Return a stable task prefix while retaining the official split identity."""
        if count < 1:
            raise ValueError("prepared benchmark task count must be positive")
        if count >= len(self.task_ids):
            return self
        return replace(
            self,
            task_catalog=self.task_catalog[:count],
            task_ids=self.task_ids[:count],
        )
