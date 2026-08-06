from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
import unicodedata

from tau3_evolver.credential_policy import is_credential_key
from tau3_evolver.fast_loop.context import LifecycleContext
from tau3_evolver.fast_loop.contracts import WriteDecision
from tau3_evolver.fast_loop.settings import FastLoopConfig
from tau3_evolver.fast_loop.outcomes import EpisodeMemoryPolicy
from tau3_evolver.memory.tier_contracts import materialize_tier_memory


def validate_write_decision(
    decision: WriteDecision,
    *,
    tools: Sequence[Mapping[str, Any]],
    task_id: str,
    context: LifecycleContext,
    final_reward: float,
    trajectory: Sequence[Mapping[str, Any]],
    memory_policy: EpisodeMemoryPolicy,
) -> WriteDecision:
    if not memory_policy.should_generate:
        raise ValueError("skipped outcomes cannot validate Memory writes")
    disallowed = sorted(
        {
            memory.tier.value
            for memory in decision.memories
            if memory.tier not in memory_policy.allowed_tiers
        }
    )
    if disallowed:
        allowed = ", ".join(tier.value for tier in memory_policy.allowed_tiers)
        raise ValueError(
            f"{memory_policy.outcome_class.value} may write only {allowed}; "
            f"disallowed tiers: {', '.join(disallowed)}"
        )
    for memory in decision.memories:
        _validate_write_metadata(memory.metadata)
        materialize_tier_memory(
            tier=memory.tier,
            payload=memory.payload,
            retrieval_text=memory.retrieval_text,
            tools=tools,
            run_id=context.run_id,
            task_id=task_id,
            task_group=context.task_group_for(task_id),
            final_reward=final_reward,
            trajectory=trajectory,
        )
    return decision


def apply_write_quotas(
    decision: WriteDecision,
    config: FastLoopConfig,
) -> tuple[tuple[Any, ...], dict[str, int]]:
    """Keep the model's stable proposal order while bounding memory growth."""
    accepted: list[Any] = []
    accepted_by_tier: dict[str, int] = {}
    dropped_by_tier: dict[str, int] = {}
    for memory in decision.memories:
        tier = memory.tier.value
        if accepted_by_tier.get(tier, 0) >= config.write_quota_for(tier):
            dropped_by_tier[tier] = dropped_by_tier.get(tier, 0) + 1
            continue
        accepted_by_tier[tier] = accepted_by_tier.get(tier, 0) + 1
        accepted.append(memory)
    return tuple(accepted), dropped_by_tier


def empty_write_decision(error: str) -> WriteDecision:
    normalized = unicodedata.normalize("NFKC", error).casefold()
    if "credential" in normalized or "attribution score" in normalized:
        raise ValueError(f"invalid write decision after repair: {error}")
    return WriteDecision(memories=())


def _validate_write_metadata(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = unicodedata.normalize("NFKC", str(key))
            compact_key = "".join(
                character
                for character in normalized_key.casefold()
                if character.isalnum()
            )
            if "attributionscore" in compact_key:
                raise ValueError("write metadata must not contain attribution score")
            if is_credential_key(normalized_key):
                raise ValueError("write metadata must not contain credential fields")
            _validate_write_metadata(nested)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            _validate_write_metadata(nested)


__all__ = [
    "apply_write_quotas",
    "empty_write_decision",
    "validate_write_decision",
]
