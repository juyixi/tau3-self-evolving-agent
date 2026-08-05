import pytest
from pydantic import ValidationError

from tau3_evolver.slow_loop.evidence import EpisodeEvidence, EvidenceLedger


def _episode(**overrides: object) -> EpisodeEvidence:
    values = {
        "episode_id": "run-1:task-1",
        "run_id": "run-1",
        "source_episode_row": 1,
        "source_episode_sha256": "a" * 64,
        "memory_generation": 1,
        "task_id": "task-1",
        "task_group": "airline",
        "model_revision": "Qwen/Qwen3.5-9B",
        "adapter_revision": None,
        "runtime_revision": "tau2-runtime",
        "split_hash": "b" * 64,
        "memory_namespace": "airline",
        "memory_snapshot_id": "snapshot-1",
        "seed": 7,
        "policy": {},
        "tools": (),
        "initial_observation": "start",
        "query_hash": "c" * 64,
        "retriever_revision": "embedding-v1",
        "candidates": (),
        "selected_memory_ids": (),
        "trajectory": (),
        "terminal_evaluation": {},
        "final_reward": 1.0,
        "terminated": True,
        "truncated": False,
        "write_proposals": (),
        "proposed_memory_ids": (),
        "committed_new_memory_ids": (),
        "replayed_memory_ids": (),
    }
    values.update(overrides)
    return EpisodeEvidence.model_validate(values)


def test_evidence_lineage_uses_memory_generation_not_iteration() -> None:
    episode = _episode()
    ledger = EvidenceLedger(
        memory_generation=1,
        model_revision=episode.model_revision,
        adapter_revision=None,
        runtime_revision=episode.runtime_revision,
        split_hash=episode.split_hash,
        memory_namespace="airline",
        source_run_ids=("run-1",),
        episodes=(episode,),
        maintenance=(),
    )

    assert ledger.memory_generation == 1
    assert not hasattr(ledger, "iteration")


def test_evidence_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _episode(iteration=1)
