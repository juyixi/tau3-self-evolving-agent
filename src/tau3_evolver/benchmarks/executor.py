from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from tau3_evolver.config import ProjectConfig


@dataclass(frozen=True, slots=True)
class BenchmarkAgentSpec:
    """Benchmark-independent dependencies used to host the Fast Loop Agent."""

    policy: Any
    repository: Any
    retriever: Any
    config: Any
    memory_source_namespace: str | None
    cross_domain_memory: bool
    propose_experience: bool


BenchmarkContextFactory = Callable[[int, Any], Any]


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionRequest:
    """One prepared task batch passed to a benchmark-specific executor."""

    task_ids: tuple[str, ...]
    project_config: ProjectConfig
    agent: BenchmarkAgentSpec
    context_factory: BenchmarkContextFactory

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("benchmark execution requires at least one task")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("benchmark execution contains duplicate task IDs")


@dataclass(frozen=True, slots=True)
class BenchmarkEpisode:
    """A normalized successful task plus its internal lifecycle trace."""

    episode: Any
    events: tuple[dict[str, Any], ...]
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkTaskFailure:
    task_id: str
    stage: str
    error_type: str
    seed: int


@dataclass(frozen=True, slots=True)
class BenchmarkExecutionResult:
    episodes: tuple[BenchmarkEpisode, ...]
    failures: tuple[BenchmarkTaskFailure, ...]


class BenchmarkExecutor(Protocol):
    """Opaque execution boundary owned by one prepared benchmark runtime."""

    def execute(
        self,
        request: BenchmarkExecutionRequest,
    ) -> BenchmarkExecutionResult: ...


__all__ = [
    "BenchmarkAgentSpec",
    "BenchmarkContextFactory",
    "BenchmarkEpisode",
    "BenchmarkExecutionRequest",
    "BenchmarkExecutionResult",
    "BenchmarkExecutor",
    "BenchmarkTaskFailure",
]
