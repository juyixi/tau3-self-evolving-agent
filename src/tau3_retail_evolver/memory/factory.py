from __future__ import annotations

from pathlib import Path

from tau3_retail_evolver.config import MemoryConfig
from tau3_retail_evolver.memory.paths import training_memory_root
from tau3_retail_evolver.memory.repository import MemoryRepository


def open_training_memory(
    config: MemoryConfig,
    *,
    root: Path | None = None,
) -> MemoryRepository:
    return MemoryRepository(training_memory_root(config.agent_id, root=root))
