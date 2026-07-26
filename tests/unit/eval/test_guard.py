from __future__ import annotations

from pathlib import Path

import pytest

from tau3_retail_evolver.eval.guard import (
    EvaluationGuard,
    EvaluationProtocol,
    reject_evaluation_artifact_for_training,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.memory.paths import (
    evaluation_quarantine_root,
    training_memory_root,
)
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository


class EventCollector:
    def append(self, _event: dict[str, object]) -> None:
        pass


def _training_snapshot(project_root: Path):
    repository = MemoryRepository(
        training_memory_root("retail", root=project_root)
    )
    repository.add(
        tier="tip",
        content="Verify the order before changing it.",
        source_task_ids=("0",),
        created_round=0,
    )
    return repository.snapshot()


def _context(**overrides: object) -> RunContext:
    values: dict[str, object] = {
        "run_id": "eval-001",
        "iteration": 2,
        "split": "test",
        "model_revision": "qwen-revision",
        "adapter_revision": "adapter-2",
        "memory_snapshot_id": None,
        "seed": 42,
        "event_writer": EventCollector(),
        "mode": RunMode.EVALUATE,
        "trial_index": 0,
    }
    values.update(overrides)
    return RunContext(**values)


def test_static_evaluation_opens_only_a_read_only_training_snapshot(
    tmp_path: Path,
) -> None:
    snapshot = _training_snapshot(tmp_path)
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STATIC,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
        memory_snapshot_path=snapshot.path,
    )

    memory = guard.open_memory(trial_index=0)

    assert isinstance(memory.repository, ReadOnlyMemoryRepository)
    assert memory.memory_snapshot_id == snapshot.memory_snapshot_id
    assert memory.root == snapshot.path
    assert guard.capabilities.memory_write is False
    assert guard.capabilities.train_memory_write is False
    with pytest.raises(PermissionError, match="read-only"):
        memory.repository.add(tier="tip", content="test-only")


def test_static_evaluation_rejects_a_snapshot_from_quarantine(
    tmp_path: Path,
) -> None:
    quarantine = evaluation_quarantine_root(
        "old-eval",
        "retail",
        root=tmp_path,
    )
    snapshot = MemoryRepository(quarantine).snapshot()

    with pytest.raises(ValueError, match="training Memory snapshot"):
        EvaluationGuard(
            protocol=EvaluationProtocol.TEST_STATIC,
            run_id="eval-001",
            agent_id="retail",
            project_root=tmp_path,
            split="test",
            memory_snapshot_path=snapshot.path,
        )


def test_streaming_trials_are_empty_and_isolated_under_quarantine(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )

    first = guard.open_memory(trial_index=0)
    assert isinstance(first.repository, MemoryRepository)
    assert first.repository.list() == []
    first.repository.add(
        tier="tip",
        content="Test-stream experience",
        source_task_ids=("75",),
        created_round=0,
    )

    second = guard.open_memory(trial_index=1)

    expected = evaluation_quarantine_root(
        "eval-001",
        "retail",
        root=tmp_path,
    )
    assert first.root == expected / "trial-000"
    assert second.root == expected / "trial-001"
    assert second.repository is not None
    assert second.repository.list() == []
    assert guard.capabilities.memory_write is True
    assert guard.capabilities.train_memory_write is False


def test_streaming_refuses_to_reuse_a_trial_directory(tmp_path: Path) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    guard.open_memory(trial_index=0)

    with pytest.raises(FileExistsError, match="trial"):
        guard.open_memory(trial_index=0)


def test_no_memory_protocol_has_no_memory_side_effects(tmp_path: Path) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.NO_MEMORY,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )

    memory = guard.open_memory(trial_index=0)

    assert memory.repository is None
    assert memory.root is None
    assert memory.memory_snapshot_id is None
    assert guard.capabilities.memory_write is False
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize(
    "field",
    (
        "optimizer_create",
        "attribution",
        "dataset_write",
        "checkpoint_write",
        "train_memory_write",
    ),
)
def test_all_evaluation_protocols_disable_learning_side_effects(
    tmp_path: Path,
    field: str,
) -> None:
    for protocol in EvaluationProtocol:
        snapshot = (
            _training_snapshot(tmp_path)
            if protocol is EvaluationProtocol.TEST_STATIC
            else None
        )
        guard = EvaluationGuard(
            protocol=protocol,
            run_id=f"eval-{protocol.value.replace('_', '-')}",
            agent_id="retail",
            project_root=tmp_path,
            split="test",
            memory_snapshot_path=snapshot.path if snapshot else None,
        )
        assert getattr(guard.capabilities, field) is False


def test_guard_requires_test_evaluate_context_before_environment_reset(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.NO_MEMORY,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    memory = guard.open_memory(trial_index=0)

    with pytest.raises(ValueError, match="EVALUATE"):
        guard.validate_episode(
            _context(mode=RunMode.LEARN),
            memory,
            trial_index=0,
        )
    with pytest.raises(ValueError, match="split"):
        guard.validate_episode(
            _context(split="train"),
            memory,
            trial_index=0,
        )


def test_official_base_reproduction_is_explicit_and_no_memory_only(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="official-base-reproduction"):
        EvaluationGuard(
            protocol=EvaluationProtocol.NO_MEMORY,
            run_id="eval-base",
            agent_id="retail",
            project_root=tmp_path,
            split="base",
        )
    with pytest.raises(ValueError, match="no_memory"):
        EvaluationGuard(
            protocol=EvaluationProtocol.TEST_STREAMING,
            run_id="eval-base",
            agent_id="retail",
            project_root=tmp_path,
            split="base",
            official_base_reproduction=True,
        )

    guard = EvaluationGuard(
        protocol=EvaluationProtocol.NO_MEMORY,
        run_id="eval-base",
        agent_id="retail",
        project_root=tmp_path,
        split="base",
        official_base_reproduction=True,
    )
    assert guard.split == "base"


def test_training_artifact_guard_rejects_evaluation_quarantine(
    tmp_path: Path,
) -> None:
    quarantined = (
        tmp_path
        / "history"
        / "evaluations"
        / "eval-001"
        / "retail"
        / "quarantine"
        / "trial-000"
    )

    with pytest.raises(ValueError, match="evaluation quarantine"):
        reject_evaluation_artifact_for_training(quarantined)

    accepted = tmp_path / "runs" / "train-001"
    assert reject_evaluation_artifact_for_training(accepted) == accepted.resolve()
