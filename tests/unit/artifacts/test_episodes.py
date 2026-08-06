from tau3_evolver.artifacts.contracts import (
    CompletedEpisodeProjection,
    FailedEpisodeProjection,
)
from tau3_evolver.artifacts.episodes import build_completed_episode, build_failed_episode


def test_collapses_internal_events_without_repeating_selected_candidate() -> None:
    common = {"task_id": "task-1", "task_group": "retail", "seed": 7}
    candidate = {
        "memory_id": "tip-1",
        "memory_version": 1,
        "tier": "tip",
        "rank": 1,
        "similarity": 0.9,
    }
    proposal = {
        "memory_id": "tip-2",
        "tier": "tip",
        "content": "Check the order.",
    }
    events = [
        {
            **common,
            "event_type": "EpisodeStarted",
            "observation": "start",
            "policy": {},
            "tools": [],
        },
        {
            **common,
            "event_type": "MemoryCandidatesRetrieved",
            "query_hash": "a" * 64,
            "retriever_revision": "embed-v1",
            "candidates": [candidate],
        },
        {
            **common,
            "event_type": "MemorySelected",
            "selected_memory_ids": ["tip-1"],
            "selected": [candidate],
            "latency_s": 0.1,
        },
        {
            **common,
            "event_type": "DecisionMade",
            "turn": 0,
            "observation": "start",
            "parsed_action": "done",
        },
        {
            **common,
            "event_type": "EnvironmentStepped",
            "turn": 0,
            "action": "done",
            "observation": "end",
            "reward": 1.0,
            "done": True,
            "terminated": True,
            "truncated": False,
            "public_info": {},
        },
        {
            **common,
            "event_type": "EpisodeFinished",
            "final_reward": 1.0,
        },
        {
            **common,
            "event_type": "MemoryWriteProposed",
            "proposals": [proposal],
        },
        {
            **common,
            "event_type": "MemoryWriteCommitted",
            "written_memory_ids": ["tip-2"],
            "replayed_memory_ids": [],
        },
    ]
    result = CompletedEpisodeProjection(
        task_id="task-1",
        final_reward=1.0,
        steps=1,
        terminal_evaluation={"reward": 1.0},
        truncated=False,
    )

    row = build_completed_episode(result, events)

    assert row["memory"]["retrieval"]["candidates"] == [candidate]
    assert row["memory"]["selected_memory_ids"] == ["tip-1"]
    assert "selected" not in row["memory"]["selection"]
    assert row["memory"]["writes"][0]["disposition"] == "created"
    assert "simulation_result" not in row["outcome"]


def test_failure_is_one_bounded_episode_row() -> None:
    row = build_failed_episode(
        FailedEpisodeProjection("task-1", "run_domain", "Timeout"),
        task_group="retail",
        seed=3,
    )

    assert row == {
        "schema_version": 1,
        "task_id": "task-1",
        "task_group": "retail",
        "seed": 3,
        "status": "failed",
        "failure": {"stage": "run_domain", "error_type": "Timeout"},
    }
