from __future__ import annotations

from pathlib import Path

from tau3_retail_evolver.config import MemoryConfig
from tau3_retail_evolver.memory.factory import open_training_memory


def test_same_agent_accumulates_memory_across_repository_reopens(tmp_path: Path) -> None:
    config = MemoryConfig(agent_id="retail")
    first_round = open_training_memory(config, root=tmp_path)
    created = first_round.add(
        tier="tip",
        content="Confirm identity before issuing a refund.",
        source_task_ids=("retail-task-1",),
        created_round=1,
    )

    next_round = open_training_memory(config, root=tmp_path)

    assert next_round.root == tmp_path.resolve() / "history" / "agents" / "retail" / "memory"
    assert next_round.get(created.id) == created


def test_different_agents_do_not_share_memory(tmp_path: Path) -> None:
    retail = open_training_memory(MemoryConfig(agent_id="retail"), root=tmp_path)
    created = retail.add(
        tier="skill",
        content="Inspect the retail order before modification.",
        source_task_ids=("retail-task-1",),
        created_round=1,
    )

    airline = open_training_memory(MemoryConfig(agent_id="airline"), root=tmp_path)

    assert airline.get(created.id) is None
    assert airline.list() == []
