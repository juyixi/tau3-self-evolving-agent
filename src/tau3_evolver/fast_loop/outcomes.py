from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from tau3_evolver.memory.outcomes import MemoryOutcomeClass, MemoryPolarity
from tau3_evolver.memory.types import MemoryTier


@dataclass(frozen=True, slots=True)
class EpisodeMemoryPolicy:
    """Fast Loop decision describing whether an episode may produce Memory."""

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


def _finite_reward(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("final reward must be finite")
    return float(value)


__all__ = ["EpisodeMemoryPolicy", "classify_episode_memory"]
