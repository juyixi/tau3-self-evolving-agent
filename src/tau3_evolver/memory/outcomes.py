from __future__ import annotations

from enum import StrEnum
import math

from tau3_evolver.memory.types import MemoryItem, MemoryTier


class MemoryPolarity(StrEnum):
    POSITIVE = "positive"
    CAUTION = "caution"


class MemoryOutcomeClass(StrEnum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    INFRA_FAILURE = "infra_failure"
    INCOMPLETE = "incomplete"


def memory_outcome_labels(item: MemoryItem) -> tuple[MemoryPolarity, MemoryOutcomeClass]:
    explicit_polarity = item.metadata.get("polarity")
    explicit_outcome = item.metadata.get("outcome_class")
    try:
        if explicit_polarity is not None and explicit_outcome is not None:
            return (
                MemoryPolarity(str(explicit_polarity)),
                MemoryOutcomeClass(str(explicit_outcome)),
            )
    except ValueError:
        pass

    source_reward = item.metadata.get("source_final_reward")
    if (
        isinstance(source_reward, (int, float))
        and not isinstance(source_reward, bool)
        and math.isfinite(float(source_reward))
        and not math.isclose(float(source_reward), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        return MemoryPolarity.CAUTION, MemoryOutcomeClass.TASK_FAILURE
    return MemoryPolarity.POSITIVE, MemoryOutcomeClass.SUCCESS


def is_retrieval_eligible(item: MemoryItem) -> bool:
    polarity, _ = memory_outcome_labels(item)
    return not (
        polarity is MemoryPolarity.CAUTION
        and item.tier in {MemoryTier.SKILL, MemoryTier.TOOL}
    )


__all__ = [
    "MemoryOutcomeClass",
    "MemoryPolarity",
    "is_retrieval_eligible",
    "memory_outcome_labels",
]
