from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tau3_evolver.agent.policy import EpisodeResult


@dataclass(frozen=True, slots=True)
class BatchFailure:
    task_id: str
    stage: str
    error_type: str


@dataclass(frozen=True, slots=True)
class MaintenanceFailure:
    maintenance_round: int
    trigger_task_index: int
    error_type: str


@dataclass(frozen=True, slots=True)
class MaintenanceBatchResult:
    period: int
    completed_train_tasks_before: int
    completed_train_tasks_after: int
    records: tuple[Mapping[str, Any], ...] = ()
    failures: tuple[MaintenanceFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchResult:
    episodes: tuple[EpisodeResult, ...]
    failures: tuple[BatchFailure, ...]
    input_memory_snapshot_id: str | None
    output_memory_snapshot_id: str | None
    maintenance: MaintenanceBatchResult | None = None

    @property
    def successful(self) -> bool:
        return not self.failures and not (
            self.maintenance is not None and self.maintenance.failures
        )
