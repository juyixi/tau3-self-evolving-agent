from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from tau3_evolver.persistence.atomic import write_bytes_atomic
from tau3_evolver.persistence.locking import reentrant_process_lock


_STATE_FILE = "batch_state.json"


@dataclass(frozen=True, slots=True)
class MemoryExecutionState:
    """Execution progress associated with a writable Memory namespace."""

    committed_batches: int = 0
    completed_tasks: int = 0
    last_snapshot_id: str | None = None

    @property
    def next_generation(self) -> int:
        return self.committed_batches + 1


def load_memory_state(memory_root: Path) -> MemoryExecutionState:
    path = memory_root / _STATE_FILE
    if not path.exists():
        return MemoryExecutionState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported Memory execution state schema")
        state = MemoryExecutionState(
            committed_batches=payload["committed_batches"],
            completed_tasks=payload["completed_tasks"],
            last_snapshot_id=payload.get("last_snapshot_id"),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Memory execution state: {path}") from error
    if state.committed_batches < 0 or state.completed_tasks < 0:
        raise ValueError(f"invalid Memory execution counters: {path}")
    return state


def commit_memory_state(
    memory_root: Path,
    *,
    expected_generation: int,
    completed_tasks: int,
    snapshot_id: str,
) -> MemoryExecutionState:
    if completed_tasks < 0:
        raise ValueError("completed_tasks must be non-negative")
    with reentrant_process_lock(memory_root, namespace="memory-execution-state"):
        current = load_memory_state(memory_root)
        if current.next_generation != expected_generation:
            raise RuntimeError(
                "Memory generation changed while the batch was executing"
            )
        updated = MemoryExecutionState(
            committed_batches=expected_generation,
            completed_tasks=current.completed_tasks + completed_tasks,
            last_snapshot_id=snapshot_id,
        )
        payload = {
            "schema_version": 1,
            "committed_batches": updated.committed_batches,
            "completed_tasks": updated.completed_tasks,
            "last_snapshot_id": updated.last_snapshot_id,
        }
        write_bytes_atomic(
            memory_root / _STATE_FILE,
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        return updated


__all__ = ["MemoryExecutionState", "commit_memory_state", "load_memory_state"]
