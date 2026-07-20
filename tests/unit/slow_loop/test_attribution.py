from __future__ import annotations

import hashlib
import math

import pytest

from tau3_retail_evolver.memory.types import MemoryTier
from tau3_retail_evolver.slow_loop.attribution import compute_memory_scores
from tau3_retail_evolver.slow_loop.evidence import (
    EpisodeEvidence,
    EvidenceLedger,
    MemoryCandidateEvidence,
    WriteProposalEvidence,
)


PRIORS = {"trajectory": 0.9, "tip": 0.8, "skill": 1.0, "tool": 1.2}


def _candidate(memory_id: str, *, tier: str = "tip", version: int = 1):
    content = f"Public content for {memory_id}"
    return MemoryCandidateEvidence(
        memory_id=memory_id,
        memory_version=version,
        tier=tier,
        rank=1,
        similarity=0.8,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _episode(
    episode_id: str,
    *,
    group: str,
    reward: float,
    memory_id: str | None,
    selected: bool = False,
    tier: str = "tip",
    committed_new: bool = False,
) -> EpisodeEvidence:
    candidates = (_candidate(memory_id, tier=tier),) if memory_id is not None else ()
    selected_ids = (memory_id,) if memory_id is not None and selected else ()
    proposals = ()
    committed_ids = ()
    proposed_ids = ()
    if committed_new:
        assert memory_id is not None
        proposal = WriteProposalEvidence(
            memory_id=memory_id,
            tier=tier,
            content=f"Public content for {memory_id}",
            retrieval_text=f"Public content for {memory_id}",
            metadata={},
            source_task_ids=(episode_id,),
            created_round=3,
        )
        proposals = (proposal,)
        proposed_ids = (memory_id,)
        committed_ids = (memory_id,)
    return EpisodeEvidence(
        episode_id=episode_id,
        run_id="run-a",
        source_event_start=1,
        source_event_end=8,
        source_event_sha256="e" * 64,
        iteration=3,
        task_id=episode_id,
        task_group=group,
        model_revision="model-a",
        adapter_revision="adapter-a",
        tau2_commit="c" * 40,
        split_hash="d" * 64,
        memory_agent_id="retail",
        memory_snapshot_id="snapshot-a",
        seed=17,
        policy="public policy",
        tools=(),
        initial_observation="hello",
        query_hash="a" * 64,
        retriever_revision="embedding-a",
        candidates=candidates,
        selected_memory_ids=selected_ids,
        trajectory=(),
        terminal_evaluation={},
        simulation_result={},
        final_reward=reward,
        terminated=True,
        truncated=False,
        write_proposals=proposals,
        proposed_memory_ids=proposed_ids,
        committed_new_memory_ids=committed_ids,
        replayed_memory_ids=(),
    )


def _ledger(episodes: list[EpisodeEvidence]) -> EvidenceLedger:
    return EvidenceLedger(
        iteration=3,
        model_revision="model-a",
        adapter_revision="adapter-a",
        tau2_commit="c" * 40,
        split_hash="d" * 64,
        memory_agent_id="retail",
        source_run_ids=("run-a",),
        episodes=tuple(episodes),
        maintenance=(),
    )


def test_compute_memory_scores_matches_paper_equations() -> None:
    memory_id = "mem-tip-a"
    observations = [
        ("returns", True, 1.0),
        ("returns", True, 0.5),
        ("returns", False, 0.0),
        ("returns", False, 0.5),
        ("exchange", True, 1.0),
        ("exchange", False, 0.0),
    ]
    ledger = _ledger(
        [
            _episode(
                f"episode-{index}",
                group=group,
                reward=reward,
                memory_id=memory_id,
                selected=selected,
            )
            for index, (group, selected, reward) in enumerate(observations)
        ]
    )

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.01
    )[0]

    expected_a_hat = (2 / 4) * (0.75 - 0.25) + (1 / 2) * (1.0 - 0.0)
    expected_gamma = 1.0 - 1.0 / math.sqrt(1.0 + 3)
    assert score.attribution == pytest.approx(expected_a_hat)
    assert score.confidence == pytest.approx(expected_gamma)
    assert score.value == pytest.approx(0.8 * expected_gamma * expected_a_hat)
    assert score.retrieved_count == 6
    assert score.selected_count == 3
    assert [group.group for group in score.groups] == ["exchange", "returns"]
    assert score.qualified_for_supervision is True


def test_unretrieved_tasks_never_enter_candidate_control() -> None:
    memory_id = "mem-tip-a"
    ledger = _ledger(
        [
            _episode("selected", group="returns", reward=1.0, memory_id=memory_id, selected=True),
            _episode("control", group="returns", reward=0.0, memory_id=memory_id),
            _episode("unretrieved", group="returns", reward=1000.0, memory_id=None),
        ]
    )

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.01
    )[0]

    assert score.retrieved_count == 2
    assert score.groups[0].not_selected_reward_mean == 0.0
    assert score.source_episode_ids == ("selected", "control")


def test_one_sided_group_is_omitted_and_no_groups_is_null() -> None:
    ledger = _ledger(
        [
            _episode(
                "selected-only",
                group="returns",
                reward=1.0,
                memory_id="mem-tip-a",
                selected=True,
            )
        ]
    )

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.01
    )[0]

    assert score.groups == ()
    assert score.status == "insufficient_evidence"
    assert score.attribution is None
    assert score.value is None
    assert score.selected_count == 1
    assert score.confidence == pytest.approx(1 - 1 / math.sqrt(2))
    assert score.qualified_for_supervision is False


def test_negative_value_is_retained_but_not_qualified() -> None:
    memory_id = "mem-tool-a"
    ledger = _ledger(
        [
            _episode(
                "selected",
                group="returns",
                reward=0.0,
                memory_id=memory_id,
                selected=True,
                tier="tool",
            ),
            _episode(
                "control",
                group="returns",
                reward=1.0,
                memory_id=memory_id,
                tier="tool",
            ),
        ]
    )

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.01
    )[0]

    assert score.status == "scored"
    assert score.value is not None and score.value < 0
    assert score.qualified_for_supervision is False


def test_value_equal_to_threshold_is_qualified() -> None:
    memory_id = "mem-tip-threshold"
    ledger = _ledger(
        [
            _episode(
                "selected",
                group="returns",
                reward=1.0,
                memory_id=memory_id,
                selected=True,
            ),
            _episode(
                "control",
                group="returns",
                reward=0.0,
                memory_id=memory_id,
            ),
        ]
    )
    value = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.0
    )[0].value
    assert value is not None

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=value
    )[0]

    assert score.value == value
    assert score.qualified_for_supervision is True


def test_creator_episode_cannot_value_its_own_write() -> None:
    memory_id = "mem-skill-new"
    ledger = _ledger(
        [
            _episode(
                "creator",
                group="returns",
                reward=100.0,
                memory_id=memory_id,
                selected=True,
                tier="skill",
                committed_new=True,
            ),
            _episode(
                "future-selected",
                group="returns",
                reward=1.0,
                memory_id=memory_id,
                selected=True,
                tier="skill",
            ),
            _episode(
                "future-not-selected",
                group="returns",
                reward=0.0,
                memory_id=memory_id,
                tier="skill",
            ),
        ]
    )

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.01
    )[0]

    assert score.creator_episode_id == "creator"
    assert score.source_episode_ids == ("future-selected", "future-not-selected")
    assert score.retrieved_count == 2
    assert score.attribution == pytest.approx(1.0 / 2.0)


def test_committed_memory_without_future_retrieval_is_preserved_as_insufficient() -> None:
    memory_id = "mem-tip-new"
    ledger = _ledger(
        [
            _episode(
                "creator",
                group="returns",
                reward=1.0,
                memory_id=memory_id,
                tier="tip",
                committed_new=True,
            )
        ]
    )

    score = compute_memory_scores(
        ledger, tier_priors=PRIORS, score_threshold=0.01
    )[0]

    assert score.memory_id == memory_id
    assert score.retrieved_count == 0
    assert score.value is None
