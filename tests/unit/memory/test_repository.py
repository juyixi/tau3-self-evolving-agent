from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from pathlib import Path
from threading import Event, Lock
from typing import Any

import pytest

from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.types import MEMORY_TIERS, MemoryStatus, MemoryTier
import tau3_evolver.memory.json_store as json_store
import tau3_evolver.persistence.atomic as atomic


def _add_memory_in_process(
    root: str,
    content: str,
    started: Any,
    finished: Any,
    results: Any,
    entered_write: Any | None = None,
    release_write: Any | None = None,
) -> None:
    started.set()
    try:
        repository = MemoryRepository(Path(root))
        if entered_write is not None and release_write is not None:
            store = repository._stores[MemoryTier.TIP]
            original_write = store.write

            def blocking_write(items: object) -> None:
                entered_write.set()
                if not release_write.wait(timeout=10):
                    raise TimeoutError("timed out waiting to release first process write")
                original_write(items)  # type: ignore[arg-type]

            store.write = blocking_write  # type: ignore[method-assign]
        repository.add(
            tier="tip",
            content=content,
            source_task_ids=(content,),
            created_round=0,
        )
        results.put(None)
    except BaseException as error:
        results.put(f"{type(error).__name__}: {error}")
    finally:
        finished.set()


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


def test_fresh_repository_initializes_all_canonical_tier_files(tmp_path: Path) -> None:
    root = tmp_path / "memory"

    MemoryRepository(root)

    assert sorted(path.name for path in root.glob("*_memory.json")) == [
        "skill_memory.json",
        "tip_memory.json",
        "tool_memory.json",
        "trajectory_memory.json",
    ]
    for tier in MemoryTier:
        payload = json.loads((root / f"{tier.value}_memory.json").read_text(encoding="utf-8"))
        assert payload == {
            "schema_version": 1,
            "tier": tier.value,
            "count": 0,
            "items": [],
        }


def test_repository_rejects_missing_canonical_tier_files(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    root.mkdir()
    (root / "tip_memory.json").write_bytes(json_store.serialize_tier(MemoryTier.TIP, []))

    with pytest.raises(ValueError, match="missing memory tier files"):
        MemoryRepository(root)


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


def test_status_change_validates_round_without_changing_file(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="skill",
        content="Authenticate before accessing an order.",
        source_task_ids=("task-3",),
        created_round=2,
    )
    path = tmp_path / "memory" / "skill_memory.json"
    original = path.read_bytes()

    with pytest.raises(ValueError):
        repository.update_status(
            item.id,
            MemoryStatus.RETIRED,
            updated_round=-1,
        )

    assert path.read_bytes() == original
    assert repository.get(item.id) == item


def test_status_change_rejects_round_regression(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="skill",
        content="Authenticate before accessing an order.",
        source_task_ids=("task-3",),
        created_round=3,
    )

    with pytest.raises(ValueError, match="must not move backwards"):
        repository.update_status(item.id, MemoryStatus.RETIRED, updated_round=2)

    assert repository.get(item.id) == item


def test_serializes_concurrent_writes_to_the_same_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    store = repository._stores[next(tier for tier in MEMORY_TIERS if tier == "tip")]
    original_write = store.write
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    call_lock = Lock()
    call_count = 0

    def blocking_write(items: object) -> None:
        nonlocal call_count
        with call_lock:
            call_count += 1
            current_call = call_count
        if current_call == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        original_write(items)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "write", blocking_write)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            repository.add,
            tier="tip",
            content="Confirm the order number before a return.",
            source_task_ids=("task-1",),
            created_round=0,
        )
        assert first_entered.wait(timeout=1)
        second = pool.submit(
            repository.add,
            tier="tip",
            content="Confirm the item before an exchange.",
            source_task_ids=("task-2",),
            created_round=0,
        )
        try:
            assert not second_entered.wait(timeout=0.1)
        finally:
            release_first.set()
        first.result()
        second.result()

    assert second_entered.is_set()
    reopened = MemoryRepository(tmp_path / "memory")
    assert len(reopened.list(tier="tip")) == 2


def test_repository_instances_do_not_overwrite_each_others_updates(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    first = MemoryRepository(root)
    second = MemoryRepository(root)

    first.add(
        tier="tip",
        content="Confirm the order number before a return.",
        source_task_ids=("task-1",),
        created_round=0,
    )
    second.add(
        tier="tip",
        content="Confirm the item before an exchange.",
        source_task_ids=("task-2",),
        created_round=0,
    )

    assert len(first.list(tier="tip")) == 2
    assert len(second.list(tier="tip")) == 2
    assert len(MemoryRepository(root).list(tier="tip")) == 2


def test_processes_do_not_overwrite_each_others_updates(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "memory"
    first_started = context.Event()
    first_finished = context.Event()
    first_entered_write = context.Event()
    release_first_write = context.Event()
    second_started = context.Event()
    second_finished = context.Event()
    results = context.Queue()
    first = context.Process(
        target=_add_memory_in_process,
        args=(
            str(root),
            "Confirm the order number before a return.",
            first_started,
            first_finished,
            results,
            first_entered_write,
            release_first_write,
        ),
    )
    second = context.Process(
        target=_add_memory_in_process,
        args=(
            str(root),
            "Confirm the item before an exchange.",
            second_started,
            second_finished,
            results,
        ),
    )

    processes = [first]
    timed_out: list[int | None] = []
    first.start()
    try:
        assert first_started.wait(timeout=5)
        assert first_entered_write.wait(timeout=5)
        second.start()
        processes.append(second)
        assert second_started.wait(timeout=5)
        second_finished.wait(timeout=2)
    finally:
        release_first_write.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                timed_out.append(process.pid)
                process.terminate()
                process.join(timeout=5)

    assert not timed_out, f"child processes did not exit: {timed_out}"
    for process in processes:
        assert process.exitcode == 0

    assert [results.get(timeout=2) for _ in range(2)] == [None, None]
    assert len(MemoryRepository(root).list(tier="tip")) == 2


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

    monkeypatch.setattr(atomic.os, "replace", fail_replace)

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


def test_rejects_memory_id_that_does_not_match_content(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    repository = MemoryRepository(root)
    repository.add(
        tier="tip",
        content="Confirm identity before a refund.",
        source_task_ids=("task-1",),
        created_round=0,
    )
    path = root / "tip_memory.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["id"] = "mem_tip_invalid"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="stable memory id mismatch"):
        MemoryRepository(root)
