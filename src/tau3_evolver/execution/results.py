from __future__ import annotations

from dataclasses import dataclass

from tau3_evolver.agent.policy import EpisodeResult


@dataclass(frozen=True, slots=True)
class BatchFailure:
    task_id: str
    stage: str
    error_type: str


@dataclass(frozen=True, slots=True)
class BatchResult:
    episodes: tuple[EpisodeResult, ...]
    failures: tuple[BatchFailure, ...]
    input_memory_snapshot_id: str | None
    output_memory_snapshot_id: str | None

    @property
    def successful(self) -> bool:
        return not self.failures
