from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tau3_evolver.benchmarks.types import PreparedBenchmark
from tau3_evolver.execution.memory_state import load_memory_state
from tau3_evolver.execution.request import ExecutionMode, ExecutionRequest
from tau3_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.persistence.layout import project_root, training_memory_root


@dataclass(frozen=True, slots=True)
class ResolvedMemory:
    source: ReadOnlyMemoryRepository | None
    destination: MemoryRepository | None
    source_namespace: str | None
    destination_namespace: str | None
    input_snapshot_id: str | None
    generation: int


def resolve_memory(
    request: ExecutionRequest,
    prepared: PreparedBenchmark,
    *,
    root: Path | None = None,
) -> ResolvedMemory:
    """Resolve run-specific Memory sources without leaking execution into Memory."""
    if not request.memory_enabled:
        return ResolvedMemory(None, None, None, None, None, 0)

    workspace = (root or project_root()).resolve()
    source_namespace = request.resolved_memory_source(
        prepared.default_memory_namespace
    )
    assert source_namespace is not None
    destination_namespace = request.destination_memory_namespace(
        prepared.default_memory_namespace
    )
    destination: MemoryRepository | None = None
    generation = 0
    if request.mode is ExecutionMode.TRAIN:
        destination = MemoryRepository(
            training_memory_root(destination_namespace, root=workspace)
        )
        generation = load_memory_state(destination.root).next_generation

    if request.memory_snapshot is not None:
        snapshot_path = resolve_snapshot_path(
            request.memory_snapshot,
            namespace=source_namespace,
            root=workspace,
        )
    elif destination is not None and source_namespace == destination_namespace:
        snapshot_path = destination.snapshot().path
    else:
        raise ValueError("the selected Memory source requires a frozen snapshot")

    source = ReadOnlyMemoryRepository(snapshot_path)
    return ResolvedMemory(
        source=source,
        destination=destination,
        source_namespace=source_namespace,
        destination_namespace=(
            destination_namespace if destination is not None else None
        ),
        input_snapshot_id=source.memory_snapshot_id,
        generation=generation,
    )


def resolve_snapshot_path(value: Path, *, namespace: str, root: Path) -> Path:
    supplied = value.expanduser()
    candidates = [supplied]
    if not supplied.is_absolute():
        candidates.append(
            training_memory_root(namespace, root=root) / "snapshots" / supplied
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    rendered = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise ValueError(f"Memory snapshot does not exist; checked: {rendered}")


__all__ = ["ResolvedMemory", "resolve_memory", "resolve_snapshot_path"]
