from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from tau3_evolver.benchmarks.executor import BenchmarkExecutor
from tau3_evolver.config import ProjectConfig
from tau3_evolver.execution_mode import ExecutionMode


class BenchmarkDefinition(Protocol):
    """Immutable registration data that can prepare one benchmark runtime."""

    name: str
    default_memory_namespace: str

    def credential_requirements(
        self,
        config: ProjectConfig,
    ) -> tuple[tuple[str, str], ...]: ...

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
    """Public metadata plus one opaque benchmark-specific executor."""

    name: str
    task_ids: tuple[str, ...]
    split_name: str
    split_hash: str
    runtime_origin: RuntimeOrigin
    default_memory_namespace: str
    task_group: str
    executor: BenchmarkExecutor

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("prepared benchmark task IDs must not be empty")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("prepared benchmark contains duplicate task IDs")

    def first_tasks(self, count: int) -> "PreparedBenchmark":
        """Return a stable task prefix while retaining the official split identity."""
        if count < 1:
            raise ValueError("prepared benchmark task count must be positive")
        if count >= len(self.task_ids):
            return self
        return replace(
            self,
            task_ids=self.task_ids[:count],
        )
