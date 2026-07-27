from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tau3_retail_evolver.memory.operations import (
    DeleteCommand,
    LookupCommand,
    MemoryOperations,
    MergeCommand,
)
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.tier_contracts import TipPayload, render_tier_payload
from tau3_retail_evolver.memory.types import MemoryStatus, MemoryTier, stable_memory_id


def _seed(repository: MemoryRepository):
    first = repository.add(
        tier="tip",
        content="Verify identity before refund.",
        source_task_ids=("task-1",),
        created_round=0,
        metadata={"kind": "identity"},
    )
    second = repository.add(
        tier="tip",
        content="Ask for confirmation before refund.",
        source_task_ids=("task-2",),
        created_round=0,
        metadata={"kind": "confirmation"},
    )
    tool = repository.add(
        tier="tool",
        content="Refund tool helper.",
        source_task_ids=("task-3",),
        created_round=0,
    )
    return first, second, tool


def test_lookup_returns_requested_active_memories_without_mutation(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)

    result = MemoryOperations(repository).apply_batch(
        [LookupCommand(memory_ids=(second.id, first.id))]
    )

    assert [item.id for item in result.looked_up] == [second.id, first.id]
    assert result.created_ids == ()
    assert result.updated_ids == ()
    assert len(repository.list()) == 3


def test_rejects_mixed_lookup_and_write_batch_without_mutation(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, _, _ = _seed(repository)

    with pytest.raises(ValueError, match="lookup commands cannot be mixed"):
        MemoryOperations(repository).apply_batch(
            [
                LookupCommand(memory_ids=(first.id,)),
                DeleteCommand(
                    memory_ids=(first.id,),
                    updated_round=1,
                    reason="maintenance",
                ),
            ]
        )

    assert repository.get(first.id).status == MemoryStatus.ACTIVE


def test_merge_creates_same_tier_memory_and_retires_sources(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)

    result = MemoryOperations(repository).apply_batch(
        [
            MergeCommand(
                source_ids=(first.id, second.id),
                content="Verify identity and obtain confirmation before refund.",
                updated_round=5,
                metadata={"reason": "combine refund safeguards"},
            )
        ]
    )

    assert len(result.created_ids) == 1
    assert set(result.updated_ids) == {first.id, second.id}
    merged = repository.get(result.created_ids[0])
    assert merged.tier == "tip"
    assert merged.source_task_ids == ("task-1", "task-2")
    assert merged.created_round == 5
    assert merged.version == 1
    assert merged.metadata["merged_from"] == sorted([first.id, second.id])
    assert repository.get(first.id).status == MemoryStatus.RETIRED
    assert repository.get(first.id).version == 2
    assert len(MemoryRepository(tmp_path / "memory").list(tier="tip")) == 1


def test_rejects_cross_tier_merge_without_changing_any_file(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, _, tool = _seed(repository)
    before = {
        path.name: path.read_bytes() for path in (tmp_path / "memory").glob("*_memory.json")
    }

    with pytest.raises(ValueError, match="same tier"):
        MemoryOperations(repository).apply_batch(
            [
                MergeCommand(
                    source_ids=(first.id, tool.id),
                    content="Invalid cross-tier merge.",
                    updated_round=2,
                )
            ]
        )

    assert {
        path.name: path.read_bytes() for path in (tmp_path / "memory").glob("*_memory.json")
    } == before


def test_rejects_free_text_merge_for_v2_memories(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    memories = []
    for task_id, guidance in (
        ("task-1", "Verify the order ID."),
        ("task-2", "Confirm the requested operation."),
    ):
        payload = TipPayload(guidance=guidance)
        memories.append(
            repository.add(
                tier=MemoryTier.TIP,
                tier_schema_version=2,
                payload=payload.model_dump(mode="json"),
                content=render_tier_payload(MemoryTier.TIP, payload),
                source_task_ids=(task_id,),
                created_round=0,
            )
        )

    with pytest.raises(ValueError, match="typed tier payload"):
        MemoryOperations(repository).apply_batch(
            [
                MergeCommand(
                    source_ids=tuple(memory.id for memory in memories),
                    content="Verify the order and confirm the operation.",
                    updated_round=1,
                )
            ]
        )

    assert all(repository.get(memory.id).status == MemoryStatus.ACTIVE for memory in memories)


def test_v2_merge_creates_typed_memory_and_retires_sources(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    memories = []
    for task_id, guidance in (
        ("task-1", "Verify the order ID."),
        ("task-2", "Confirm the requested operation."),
    ):
        payload = TipPayload(guidance=guidance)
        memories.append(
            repository.add(
                tier=MemoryTier.TIP,
                tier_schema_version=2,
                payload=payload.model_dump(mode="json"),
                content=render_tier_payload(MemoryTier.TIP, payload),
                source_task_ids=(task_id,),
                created_round=0,
            )
        )
    merged_payload = TipPayload(
        guidance="Verify the order and confirm the requested operation."
    )

    result = MemoryOperations(repository).apply_batch(
        [
            MergeCommand(
                source_ids=tuple(memory.id for memory in memories),
                content=render_tier_payload(MemoryTier.TIP, merged_payload),
                payload=merged_payload.model_dump(mode="json"),
                updated_round=1,
            )
        ]
    )

    assert len(result.created_ids) == 1
    merged = repository.get(result.created_ids[0])
    assert merged is not None
    assert merged.tier_schema_version == 2
    assert merged.payload == merged_payload.model_dump(mode="json")
    assert merged.source_task_ids == ("task-1", "task-2")
    assert all(
        repository.get(memory.id).status == MemoryStatus.RETIRED
        for memory in memories
    )


def test_delete_is_a_versioned_soft_delete_with_reason(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, _, _ = _seed(repository)

    result = MemoryOperations(repository).apply_batch(
        [DeleteCommand(memory_ids=(first.id,), updated_round=4, reason="superseded")]
    )

    retired = repository.get(first.id)
    assert result.updated_ids == (first.id,)
    assert retired.status == MemoryStatus.RETIRED
    assert retired.version == 2
    assert retired.metadata["retired_reason"] == "superseded"
    assert repository.list(tier="tip") == [
        item for item in repository.list(tier="tip", status=None) if item.id != first.id
    ]


def test_invalid_late_command_prevents_earlier_command_from_being_written(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, tool = _seed(repository)
    before = {
        path.name: path.read_bytes() for path in (tmp_path / "memory").glob("*_memory.json")
    }

    with pytest.raises(ValueError, match="same tier"):
        MemoryOperations(repository).apply_batch(
            [
                DeleteCommand(memory_ids=(first.id,), updated_round=3, reason="candidate"),
                MergeCommand(
                    source_ids=(second.id, tool.id),
                    content="Invalid cross-tier merge.",
                    updated_round=3,
                ),
            ]
        )

    assert repository.get(first.id).status == MemoryStatus.ACTIVE
    assert {
        path.name: path.read_bytes() for path in (tmp_path / "memory").glob("*_memory.json")
    } == before


def test_replays_cross_tier_batch_after_later_tier_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "memory"
    repository = MemoryRepository(root)
    first, _, tool = _seed(repository)
    commands = [
        DeleteCommand(memory_ids=(first.id,), updated_round=4, reason="maintenance"),
        DeleteCommand(memory_ids=(tool.id,), updated_round=4, reason="maintenance"),
    ]
    tool_store = repository._stores[MemoryTier.TOOL]
    original_write = tool_store.write
    failed = False

    def fail_once(items: object) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated second-tier failure")
        original_write(items)  # type: ignore[arg-type]

    monkeypatch.setattr(tool_store, "write", fail_once)

    with pytest.raises(OSError, match="second-tier failure"):
        MemoryOperations(repository).apply_batch(commands)

    reopened = MemoryRepository(root)
    assert reopened.get(first.id).status == MemoryStatus.RETIRED
    assert reopened.get(tool.id).status == MemoryStatus.ACTIVE

    MemoryOperations(reopened).apply_batch(commands)

    assert reopened.get(first.id).status == MemoryStatus.RETIRED
    assert reopened.get(first.id).version == 2
    assert reopened.get(tool.id).status == MemoryStatus.RETIRED


def test_replaying_merge_preserves_existing_versions(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)
    command = MergeCommand(
        source_ids=(first.id, second.id),
        content="Verify identity and obtain confirmation before refund.",
        updated_round=5,
    )

    initial = MemoryOperations(repository).apply_batch([command])
    replayed = MemoryOperations(repository).apply_batch([command])

    assert replayed.created_ids == initial.created_ids
    assert replayed.updated_ids == initial.updated_ids
    assert repository.get(first.id).version == 2
    assert repository.get(second.id).version == 2


def test_replays_merge_followed_by_delete_of_merged_target(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, second, _ = _seed(repository)
    merged_content = "Verify identity and obtain confirmation before refund."
    merge = MergeCommand(
        source_ids=(first.id, second.id),
        content=merged_content,
        updated_round=5,
    )
    target_id = stable_memory_id(MemoryTier.TIP, merged_content)
    commands = [
        merge,
        DeleteCommand(memory_ids=(target_id,), updated_round=6, reason="superseded"),
    ]

    initial = MemoryOperations(repository).apply_batch(commands)
    replayed = MemoryOperations(repository).apply_batch(commands)

    assert replayed == initial
    assert repository.get(target_id).status == MemoryStatus.RETIRED
    assert repository.get(target_id).version == 2


def test_commands_reject_duplicate_or_incomplete_ids() -> None:
    with pytest.raises(ValidationError):
        MergeCommand(source_ids=("same", "same"), content="merge", updated_round=1)
    with pytest.raises(ValidationError):
        DeleteCommand(memory_ids=(), updated_round=1, reason="empty")


def test_operations_reject_round_regression(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first = repository.add(
        tier="tip",
        content="Verify identity before refund.",
        source_task_ids=("task-1",),
        created_round=4,
    )
    second = repository.add(
        tier="tip",
        content="Ask for confirmation before refund.",
        source_task_ids=("task-2",),
        created_round=5,
    )

    with pytest.raises(ValueError, match="must not move backwards"):
        MemoryOperations(repository).apply_batch(
            [DeleteCommand(memory_ids=(first.id,), updated_round=3, reason="stale")]
        )
    with pytest.raises(ValueError, match="must not move backwards"):
        MemoryOperations(repository).apply_batch(
            [
                MergeCommand(
                    source_ids=(first.id, second.id),
                    content="Combined safeguard.",
                    updated_round=4,
                )
            ]
        )


def test_read_only_snapshot_validates_hashes_and_rejects_writes(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first, _, _ = _seed(repository)
    snapshot = repository.snapshot()
    read_only = ReadOnlyMemoryRepository(snapshot.path)

    assert read_only.get(first.id) == repository.get(first.id)
    assert [item.id for item in read_only.list()] == [item.id for item in repository.list()]
    with pytest.raises(PermissionError, match="read-only"):
        read_only.add(
            tier="tip",
            content="forbidden",
            source_task_ids=("test-task",),
            created_round=0,
        )
    with pytest.raises(PermissionError, match="read-only"):
        read_only.update_embeddings({first.id: ((1.0, 0.0), "new-revision")})

    tip_path = snapshot.path / "tip_memory.json"
    payload = json.loads(tip_path.read_text(encoding="utf-8"))
    payload["items"][0]["content"] = "tampered"
    tip_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ReadOnlyMemoryRepository(snapshot.path)


def test_read_only_snapshot_rejects_unhashed_json_file(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    _seed(repository)
    snapshot = repository.snapshot()
    (snapshot.path / "unexpected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="file set mismatch"):
        ReadOnlyMemoryRepository(snapshot.path)


def test_read_only_snapshot_recomputes_active_counts(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    _seed(repository)
    snapshot = repository.snapshot()
    source_manifest = json.loads(
        (snapshot.path / "manifest.json").read_text(encoding="utf-8")
    )
    counts = {**source_manifest["counts"], "tip": 99}
    hash_payload = json.dumps(
        {
            "schema_version": 1,
            "counts": counts,
            "files": source_manifest["files"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    snapshot_id = hashlib.sha256(hash_payload).hexdigest()
    forged = tmp_path / snapshot_id
    forged.mkdir()
    for name in source_manifest["files"]:
        (forged / name).write_bytes((snapshot.path / name).read_bytes())
    manifest = {
        **source_manifest,
        "memory_snapshot_id": snapshot_id,
        "counts": counts,
    }
    (forged / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="count mismatch"):
        ReadOnlyMemoryRepository(forged)
