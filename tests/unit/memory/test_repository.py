from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.types import MEMORY_TIERS, MemoryStatus
import tau3_retail_evolver.memory.json_store as json_store


def test_persists_all_tiers_with_stable_ids_and_provenance(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")

    items = [
        repository.add(
            tier=tier,
            content=f"{tier} retail guidance",
            source_task_ids=("task-2", "task-1", "task-2"),
            created_round=3,
            metadata={"reward": 1.0},
        )
        for tier in MEMORY_TIERS
    ]

    reopened = MemoryRepository(tmp_path / "memory")
    assert [item.id for item in reopened.list()] == sorted(item.id for item in items)
    assert {item.tier for item in reopened.list()} == set(MEMORY_TIERS)
    assert all(item.source_task_ids == ("task-1", "task-2") for item in items)
    assert all(item.version == 1 and item.status == MemoryStatus.ACTIVE for item in items)

    for tier in MEMORY_TIERS:
        payload = json.loads(
            (tmp_path / "memory" / f"{tier}_memory.json").read_text(encoding="utf-8")
        )
        assert payload["schema_version"] == 1
        assert payload["tier"] == tier
        assert payload["count"] == 1
        assert payload["items"][0]["id"].startswith(f"mem_{tier}_")


def test_rejects_duplicate_content_without_changing_file(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    repository.add(
        tier="tip",
        content="Confirm identity before a refund.",
        source_task_ids=("task-1",),
        created_round=0,
    )
    path = tmp_path / "memory" / "tip_memory.json"
    original = path.read_bytes()

    with pytest.raises(ValueError, match="duplicate memory"):
        repository.add(
            tier="tip",
            content="  Confirm identity before a refund.  ",
            source_task_ids=("task-2",),
            created_round=1,
        )

    assert path.read_bytes() == original


def test_status_change_increments_version_and_filters_retired_items(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="skill",
        content="Authenticate, inspect the order, then request confirmation.",
        source_task_ids=("task-3",),
        created_round=2,
    )

    retired = repository.update_status(
        item.id,
        MemoryStatus.RETIRED,
        updated_round=7,
    )

    assert retired.version == 2
    assert retired.status == MemoryStatus.RETIRED
    assert retired.created_round == 2
    assert retired.updated_round == 7
    assert repository.get(item.id) == retired
    assert repository.list() == []
    assert repository.list(status=None) == [retired]


def test_atomic_replace_failure_preserves_previous_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    repository.add(
        tier="tool",
        content="Use get_order before modifying an order.",
        source_task_ids=("task-1",),
        created_round=0,
    )
    path = tmp_path / "memory" / "tool_memory.json"
    original = path.read_bytes()

    def fail_replace(_source: str | Path, _target: str | Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(json_store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        repository.add(
            tier="tool",
            content="Use cancel_order only after confirmation.",
            source_task_ids=("task-2",),
            created_round=1,
        )

    assert path.read_bytes() == original
    assert len(MemoryRepository(tmp_path / "memory").list(tier="tool")) == 1


def test_snapshot_is_deterministic_and_readable_after_repository_reopens(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    repository.add(
        tier="tip",
        content="Ask for confirmation before changing an address.",
        source_task_ids=("task-8",),
        created_round=1,
    )
    repository.add(
        tier="trajectory",
        content="Task: exchange an item. Outcome: success.",
        source_task_ids=("task-4",),
        created_round=1,
    )

    first = repository.snapshot()
    second = MemoryRepository(tmp_path / "memory").snapshot()

    assert first.memory_snapshot_id == second.memory_snapshot_id
    assert first.path == second.path
    assert first.path.name == first.memory_snapshot_id
    manifest = json.loads((first.path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["memory_snapshot_id"] == first.memory_snapshot_id
    assert manifest["counts"] == {"skill": 0, "tip": 1, "tool": 0, "trajectory": 1}
    assert sorted(path.name for path in first.path.glob("*.json")) == [
        "manifest.json",
        "skill_memory.json",
        "tip_memory.json",
        "tool_memory.json",
        "trajectory_memory.json",
    ]


def test_rejects_corrupt_or_mismatched_tier_store(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "tip_memory.json").write_text(
        '{"schema_version":1,"tier":"skill","count":0,"items":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tier mismatch"):
        MemoryRepository(root)
