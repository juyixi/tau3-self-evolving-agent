from __future__ import annotations

from tau3_retail_evolver.memory.outcomes import (
    MemoryOutcomeClass,
    MemoryPolarity,
    classify_episode_memory,
    memory_outcome_labels,
)
from tau3_retail_evolver.memory.types import MemoryItem, MemoryTier


def test_success_allows_positive_memory_tiers() -> None:
    policy = classify_episode_memory(
        final_reward=1.0,
        terminal_evaluation={"reward": 1.0},
        truncated=False,
    )

    assert policy.should_generate is True
    assert policy.outcome_class is MemoryOutcomeClass.SUCCESS
    assert policy.polarity is MemoryPolarity.POSITIVE
    assert set(policy.allowed_tiers) == set(MemoryTier)


def test_task_failure_allows_only_caution_tip_and_trajectory() -> None:
    policy = classify_episode_memory(
        final_reward=0.0,
        terminal_evaluation={"reward": 0.0},
        truncated=False,
    )

    assert policy.should_generate is True
    assert policy.outcome_class is MemoryOutcomeClass.TASK_FAILURE
    assert policy.polarity is MemoryPolarity.CAUTION
    assert policy.allowed_tiers == (MemoryTier.TIP, MemoryTier.TRAJECTORY)


def test_incomplete_or_unevaluated_episode_skips_memory() -> None:
    truncated = classify_episode_memory(
        final_reward=0.0,
        terminal_evaluation={"reward": 0.0},
        truncated=True,
    )
    missing_evaluation = classify_episode_memory(
        final_reward=0.0,
        terminal_evaluation={},
        truncated=False,
    )

    assert truncated.should_generate is False
    assert truncated.outcome_class is MemoryOutcomeClass.INCOMPLETE
    assert truncated.skip_reason == "episode_truncated"
    assert missing_evaluation.should_generate is False
    assert missing_evaluation.outcome_class is MemoryOutcomeClass.INFRA_FAILURE
    assert missing_evaluation.skip_reason == "missing_terminal_evaluation"


def test_legacy_failure_metadata_derives_caution_labels() -> None:
    item = MemoryItem(
        id="legacy-failure",
        tier=MemoryTier.TIP,
        content="Avoid the failed action.",
        retrieval_text="failed action",
        metadata={"source_final_reward": 0.0},
        source_task_ids=("task-1",),
        created_round=0,
        updated_round=0,
    )

    assert memory_outcome_labels(item) == (
        MemoryPolarity.CAUTION,
        MemoryOutcomeClass.TASK_FAILURE,
    )
