from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.memory.paths import (
    evaluation_quarantine_root,
    training_memory_root,
)
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository


class EvaluationProtocol(StrEnum):
    NO_MEMORY = "no_memory"
    TEST_STATIC = "test_static"
    TEST_STREAMING = "test_streaming"


@dataclass(frozen=True, slots=True)
class EvaluationCapabilities:
    optimizer_create: bool = False
    attribution: bool = False
    dataset_write: bool = False
    checkpoint_write: bool = False
    train_memory_write: bool = False
    memory_write: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "optimizer_create": self.optimizer_create,
            "attribution": self.attribution,
            "dataset_write": self.dataset_write,
            "checkpoint_write": self.checkpoint_write,
            "train_memory_write": self.train_memory_write,
            "memory_write": self.memory_write,
        }


MemoryRepositoryView = MemoryRepository | ReadOnlyMemoryRepository


@dataclass(frozen=True, slots=True)
class EvaluationMemory:
    repository: MemoryRepositoryView | None
    root: Path | None
    memory_snapshot_id: str | None


@dataclass(frozen=True, slots=True)
class EvaluationGuard:
    protocol: EvaluationProtocol
    run_id: str
    agent_id: str
    project_root: Path
    split: str = "test"
    official_base_reproduction: bool = False
    memory_snapshot_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol")
        resolved_root = Path(self.project_root).resolve()
        object.__setattr__(self, "project_root", resolved_root)

        # These helpers also validate run_id and agent_id before any directory is made.
        evaluation_quarantine_root(
            self.run_id,
            self.agent_id,
            root=resolved_root,
        )
        if self.split == "test":
            if self.official_base_reproduction:
                raise ValueError(
                    "official-base-reproduction requires the base split"
                )
        elif self.split == "base":
            if not self.official_base_reproduction:
                raise ValueError(
                    "base split requires --official-base-reproduction"
                )
            if self.protocol is not EvaluationProtocol.NO_MEMORY:
                raise ValueError(
                    "official base reproduction supports only no_memory"
                )
        else:
            raise ValueError("evaluation split must be test or explicit base")

        if self.protocol is EvaluationProtocol.TEST_STATIC:
            if self.memory_snapshot_path is None:
                raise ValueError("test_static requires a training Memory snapshot")
            snapshot = Path(self.memory_snapshot_path).resolve()
            expected_parent = (
                training_memory_root(self.agent_id, root=resolved_root)
                / "snapshots"
            ).resolve()
            if snapshot.parent != expected_parent:
                raise ValueError(
                    "test_static requires a training Memory snapshot"
                )
            object.__setattr__(self, "memory_snapshot_path", snapshot)
        elif self.memory_snapshot_path is not None:
            raise ValueError(
                f"{self.protocol.value} does not accept a Memory snapshot"
            )

    @property
    def capabilities(self) -> EvaluationCapabilities:
        return EvaluationCapabilities(
            memory_write=self.protocol is EvaluationProtocol.TEST_STREAMING
        )

    @property
    def quarantine_root(self) -> Path:
        return evaluation_quarantine_root(
            self.run_id,
            self.agent_id,
            root=self.project_root,
        )

    def source_memory_snapshot_id(self) -> str | None:
        if self.protocol is not EvaluationProtocol.TEST_STATIC:
            return None
        assert self.memory_snapshot_path is not None
        return ReadOnlyMemoryRepository(
            self.memory_snapshot_path
        ).memory_snapshot_id

    def open_memory(self, *, trial_index: int) -> EvaluationMemory:
        _require_trial_index(trial_index)
        if self.protocol is EvaluationProtocol.NO_MEMORY:
            return EvaluationMemory(
                repository=None,
                root=None,
                memory_snapshot_id=None,
            )
        if self.protocol is EvaluationProtocol.TEST_STATIC:
            assert self.memory_snapshot_path is not None
            repository = ReadOnlyMemoryRepository(self.memory_snapshot_path)
            return EvaluationMemory(
                repository=repository,
                root=repository.root,
                memory_snapshot_id=repository.memory_snapshot_id,
            )

        trial_root = self.quarantine_root / f"trial-{trial_index:03d}"
        if trial_root.exists():
            raise FileExistsError(
                f"refusing to reuse evaluation trial directory: {trial_root}"
            )
        repository = MemoryRepository(trial_root)
        snapshot = repository.snapshot()
        return EvaluationMemory(
            repository=repository,
            root=trial_root,
            memory_snapshot_id=snapshot.memory_snapshot_id,
        )

    def validate_episode(
        self,
        context: RunContext,
        memory: EvaluationMemory,
        *,
        trial_index: int,
    ) -> None:
        _require_trial_index(trial_index)
        if context.mode is not RunMode.EVALUATE:
            raise ValueError("evaluation episode requires EVALUATE mode")
        if context.split != self.split:
            raise ValueError("evaluation context split does not match guard")
        if context.run_id != self.run_id:
            raise ValueError("evaluation context run ID does not match guard")
        if context.trial_index != trial_index:
            raise ValueError("evaluation context trial index does not match")

        if self.protocol is EvaluationProtocol.NO_MEMORY:
            if memory.repository is not None or memory.root is not None:
                raise ValueError("no_memory evaluation requires no repository")
            if context.memory_snapshot_id is not None:
                raise ValueError("no_memory evaluation has no Memory snapshot")
            return

        if self.protocol is EvaluationProtocol.TEST_STATIC:
            if not isinstance(memory.repository, ReadOnlyMemoryRepository):
                raise ValueError("test_static requires a read-only repository")
            expected_snapshot = memory.repository.memory_snapshot_id
            if context.memory_snapshot_id != expected_snapshot:
                raise ValueError("test_static Memory snapshot provenance mismatch")
            if memory.root != self.memory_snapshot_path:
                raise ValueError("test_static repository path is not the requested snapshot")
            return

        if not isinstance(memory.repository, MemoryRepository):
            raise ValueError("test_streaming requires a mutable repository")
        expected_root = self.quarantine_root / f"trial-{trial_index:03d}"
        if memory.root != expected_root or memory.repository.root != expected_root:
            raise ValueError("test_streaming repository is outside its quarantine trial")
        current_snapshot = memory.repository.snapshot().memory_snapshot_id
        if context.memory_snapshot_id != current_snapshot:
            raise ValueError("test_streaming Memory snapshot provenance mismatch")


def reject_evaluation_artifact_for_training(path: Path) -> Path:
    resolved = Path(path).resolve()
    lowered = tuple(part.casefold() for part in resolved.parts)
    if any(
        lowered[index : index + 2] == ("history", "evaluations")
        for index in range(len(lowered) - 1)
    ):
        raise ValueError(
            f"training input is inside the evaluation quarantine: {resolved}"
        )
    return resolved


def _require_trial_index(trial_index: int) -> None:
    if type(trial_index) is not int or trial_index < 0:
        raise ValueError("trial index must be a non-negative integer")
