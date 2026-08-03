from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Any

from tau3_retail_evolver.memory.types import MemoryItem, MemoryTier


class MemoryPolarity(StrEnum):
    POSITIVE = "positive"
    CAUTION = "caution"


class MemoryOutcomeClass(StrEnum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    INFRA_FAILURE = "infra_failure"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class EpisodeMemoryPolicy:
    outcome_class: MemoryOutcomeClass
    polarity: MemoryPolarity | None
    allowed_tiers: tuple[MemoryTier, ...]
    final_reward: float
    skip_reason: str | None = None

    @property
    def should_generate(self) -> bool:
        return self.skip_reason is None

    def prompt_payload(self) -> dict[str, Any]:
        if not self.should_generate or self.polarity is None:
            raise ValueError("skipped Memory outcomes do not have a write prompt")
        guidance = (
            "Extract reusable successful behavior."
            if self.polarity is MemoryPolarity.POSITIVE
            else (
                "Extract only failure reflections: describe what to avoid and the "
                "corrective behavior. Do not present failed behavior as a success strategy."
            )
        )
        return {
            "final_reward": self.final_reward,
            "outcome_class": self.outcome_class.value,
            "polarity": self.polarity.value,
            "allowed_tiers": [tier.value for tier in self.allowed_tiers],
            "guidance": guidance,
        }


# Trajectories are runtime-owned records of every rollout.  They are intentionally
# absent from the LLM writer contract; the writer extracts only reusable knowledge.
_SUCCESS_TIERS = (MemoryTier.TIP, MemoryTier.SKILL, MemoryTier.TOOL)
_FAILURE_TIERS = (MemoryTier.TIP,)


def classify_episode_memory(
    *,
    final_reward: float,
    terminal_evaluation: Mapping[str, Any],
    truncated: bool,
) -> EpisodeMemoryPolicy:
    reward = _finite_reward(final_reward)
    if truncated:
        return EpisodeMemoryPolicy(
            outcome_class=MemoryOutcomeClass.INCOMPLETE,
            polarity=None,
            allowed_tiers=(),
            final_reward=reward,
            skip_reason="episode_truncated",
        )
    if not terminal_evaluation:
        return EpisodeMemoryPolicy(
            outcome_class=MemoryOutcomeClass.INFRA_FAILURE,
            polarity=None,
            allowed_tiers=(),
            final_reward=reward,
            skip_reason="missing_terminal_evaluation",
        )
    if math.isclose(reward, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return EpisodeMemoryPolicy(
            outcome_class=MemoryOutcomeClass.SUCCESS,
            polarity=MemoryPolarity.POSITIVE,
            allowed_tiers=_SUCCESS_TIERS,
            final_reward=reward,
        )
    return EpisodeMemoryPolicy(
        outcome_class=MemoryOutcomeClass.TASK_FAILURE,
        polarity=MemoryPolarity.CAUTION,
        allowed_tiers=_FAILURE_TIERS,
        final_reward=reward,
    )


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


def _finite_reward(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("final reward must be finite")
    return float(value)
