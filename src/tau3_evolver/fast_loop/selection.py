from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from tau3_evolver.fast_loop.contracts import SelectionDecision
from tau3_evolver.fast_loop.settings import FastLoopConfig
from tau3_evolver.memory.retrieval import MemoryCandidate


def build_retrieval_query(
    task_instruction: str,
    policy: Any,
    tools: Sequence[Any],
    observation: str,
) -> str:
    tool_names = sorted(set(_find_named_values(tools)))
    return json.dumps(
        {
            "task_instruction": task_instruction,
            "policy": policy,
            "tool_names": tool_names,
            "observation": observation,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def candidate_evidence(candidate: MemoryCandidate) -> dict[str, Any]:
    return {
        "memory_id": candidate.memory_id,
        "memory_version": candidate.memory_version,
        "tier": candidate.tier.value,
        "rank": candidate.rank,
        "similarity": candidate.similarity,
    }


def validate_selection_limits(
    decision: SelectionDecision,
    candidates: Sequence[MemoryCandidate],
    config: FastLoopConfig,
) -> SelectionDecision:
    if len(decision.memory_ids) > config.selection_max_total:
        raise ValueError("selected memory count exceeds the configured total limit")
    tier_by_id = {
        candidate.memory_id: candidate.tier.value for candidate in candidates
    }
    selected_by_tier: dict[str, int] = {}
    for memory_id in decision.memory_ids:
        tier = tier_by_id[memory_id]
        selected_by_tier[tier] = selected_by_tier.get(tier, 0) + 1
    limits = config.selection_tier_limits()
    for tier, count in selected_by_tier.items():
        if count > limits[tier]:
            raise ValueError(
                f"selected {tier} memory count exceeds the configured limit"
            )
    return decision


def _find_named_values(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "name" and isinstance(nested, str):
                names.append(nested)
            else:
                names.extend(_find_named_values(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            names.extend(_find_named_values(nested))
    return names


__all__ = [
    "build_retrieval_query",
    "candidate_evidence",
    "query_hash",
    "validate_selection_limits",
]
