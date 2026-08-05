from __future__ import annotations

from pathlib import Path

from tau3_evolver.memory.paths import training_memory_root
from tau3_evolver.memory.repository import MemoryRepository


def open_training_memory(
    namespace: str,
    *,
    root: Path | None = None,
) -> MemoryRepository:
    return MemoryRepository(training_memory_root(namespace, root=root))
