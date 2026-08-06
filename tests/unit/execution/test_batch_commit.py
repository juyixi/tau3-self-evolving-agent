from pathlib import Path

import pytest

from tau3_evolver.fast_loop.contracts import PendingEpisode
from tau3_evolver.execution.batch import commit_pending_experience
from tau3_evolver.fast_loop.contracts import EpisodeResult
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.types import MemoryTier, stable_memory_id


def _episode(task_id: str, content: str) -> PendingEpisode:
    memory_id = stable_memory_id(MemoryTier.TIP, content)
    proposal = {
        "memory_id": memory_id,
        "add_kwargs": {
            "tier": MemoryTier.TIP,
            "content": content,
            "source_task_ids": (task_id,),
            "created_round": 1,
        },
        "evidence": {"memory_id": memory_id},
    }
    return PendingEpisode(
        result=EpisodeResult(
            task_id=task_id,
            final_reward=1.0,
            steps=1,
            terminal_evaluation={},
            selected_memory_ids=(),
            written_memory_ids=(),
            truncated=False,
        ),
        proposals=(proposal,),
    )


def test_pending_experience_is_not_visible_until_batch_commit(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    episodes = [_episode("task-1", "Check the order first."), _episode("task-2", "Confirm identity.")]

    assert repository.list() == []
    commits = commit_pending_experience(repository, episodes)

    assert tuple(len(commit.written_ids) for commit in commits) == (1, 1)
    assert {item.id for item in repository.list()} == {
        memory_id for commit in commits for memory_id in commit.written_ids
    }


def test_identical_batch_proposals_are_coalesced_with_replay_evidence(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first = _episode("task-1", "Shared guidance.")
    second = _episode("task-2", "Shared guidance.")

    commits = commit_pending_experience(repository, (first, second))

    assert len(repository.list()) == 1
    assert commits[0].replayed_ids == ()
    assert commits[1].replayed_ids == commits[1].written_ids


def test_conflicting_batch_is_rejected_before_any_new_write(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first = _episode("task-1", "Shared guidance.")
    conflict = _episode("task-2", "Shared guidance.")
    conflict.proposals[0]["add_kwargs"]["content"] = "Different content."

    with pytest.raises(ValueError, match="conflicting"):
        commit_pending_experience(repository, (first, conflict))

    assert repository.list() == []
