from pathlib import Path

import pytest

from tau3_evolver.memory.factory import open_training_memory


def test_same_namespace_accumulates_across_reopens(tmp_path: Path) -> None:
    first = open_training_memory("retail", root=tmp_path)
    created = first.add(
        tier="tip",
        content="Confirm identity before issuing a refund.",
        source_task_ids=("retail-task-1",),
        created_round=1,
    )

    reopened = open_training_memory("retail", root=tmp_path)

    assert reopened.get(created.id) == created


def test_namespaces_are_isolated(tmp_path: Path) -> None:
    retail = open_training_memory("retail", root=tmp_path)
    created = retail.add(
        tier="skill",
        content="Inspect the order before modification.",
        source_task_ids=("retail-task-1",),
        created_round=1,
    )

    assert open_training_memory("airline", root=tmp_path).get(created.id) is None


def test_rejects_unsafe_namespace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="namespace"):
        open_training_memory("../escape", root=tmp_path)
