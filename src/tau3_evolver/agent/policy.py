from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol, TypeVar
import unicodedata

from tau3_evolver.credential_policy import is_credential_key
from tau3_evolver.agent.decisions import Decision, SelectionDecision, WriteDecision, parse_decision
from tau3_evolver.agent.prompts import LifecyclePrompt
from tau3_evolver.execution.events import ExecutionContext
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.outcomes import EpisodeMemoryPolicy
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.retrieval import MemoryCandidate
from tau3_evolver.memory.tier_contracts import materialize_tier_memory
from tau3_evolver.memory.types import MemoryItem, canonical_content


@dataclass(frozen=True, slots=True)
class FastLoopConfig:
    retrieve_top_k: int = 50
    max_episode_steps: int = 40
    memory_enabled: bool = True
    max_new_tips_per_episode: int = 2
    max_new_skills_per_episode: int = 1
    max_new_tools_per_episode: int = 1
    max_new_trajectories_per_episode: int = 1
    maintenance_tip_capacity: int = 240
    maintenance_similarity_threshold: float = 0.92
    maintenance_priority_pair_limit: int = 24
    retrieval_mmr_lambda_tip: float = 0.65
    retrieval_mmr_lambda_skill: float = 0.80
    retrieval_mmr_lambda_tool: float = 0.85
    retrieval_mmr_lambda_trajectory: float = 0.75
    retrieval_global_mmr_lambda: float = 0.75
    retrieval_quota_tip: int = 18
    retrieval_quota_skill: int = 18
    retrieval_quota_tool: int = 6
    retrieval_quota_trajectory: int = 4
    selection_max_total: int = 20
    selection_max_tip: int = 7
    selection_max_skill: int = 8
    selection_max_tool: int = 3
    selection_max_trajectory: int = 2

    def __post_init__(self) -> None:
        if type(self.memory_enabled) is not bool:
            raise ValueError("memory_enabled must be a bool")
        if self.retrieve_top_k < 1 or self.max_episode_steps < 1:
            raise ValueError("fast-loop limits must be positive")
        if any(
            type(limit) is not int or limit < 0
            for limit in (
                self.max_new_tips_per_episode,
                self.max_new_skills_per_episode,
                self.max_new_tools_per_episode,
                self.max_new_trajectories_per_episode,
            )
        ):
            raise ValueError("memory write quotas must be non-negative integers")
        if self.maintenance_tip_capacity < 1:
            raise ValueError("maintenance tip capacity must be positive")
        if not -1.0 <= self.maintenance_similarity_threshold <= 1.0:
            raise ValueError("maintenance similarity threshold must be between -1 and 1")
        if self.maintenance_priority_pair_limit < 0:
            raise ValueError("maintenance priority pair limit must be non-negative")
        if any(
            not 0.0 <= value <= 1.0
            for value in (*self.retrieval_mmr_lambdas().values(), self.retrieval_global_mmr_lambda)
        ):
            raise ValueError("retrieval MMR lambdas must be between 0 and 1")
        if any(value < 0 for value in self.retrieval_tier_quotas().values()):
            raise ValueError("retrieval tier quotas must be non-negative")
        if self.selection_max_total < 1 or any(
            value < 0 for value in self.selection_tier_limits().values()
        ):
            raise ValueError("selection limits must be non-negative with a positive total")

    def write_quota_for(self, tier: str) -> int:
        return {
            "tip": self.max_new_tips_per_episode,
            "skill": self.max_new_skills_per_episode,
            "tool": self.max_new_tools_per_episode,
            "trajectory": self.max_new_trajectories_per_episode,
        }[tier]

    def retrieval_tier_quotas(self) -> dict[str, int]:
        return {
            "tip": self.retrieval_quota_tip,
            "skill": self.retrieval_quota_skill,
            "tool": self.retrieval_quota_tool,
            "trajectory": self.retrieval_quota_trajectory,
        }

    def retrieval_mmr_lambdas(self) -> dict[str, float]:
        return {
            "tip": self.retrieval_mmr_lambda_tip,
            "skill": self.retrieval_mmr_lambda_skill,
            "tool": self.retrieval_mmr_lambda_tool,
            "trajectory": self.retrieval_mmr_lambda_trajectory,
        }

    def selection_tier_limits(self) -> dict[str, int]:
        return {
            "tip": self.selection_max_tip,
            "skill": self.selection_max_skill,
            "tool": self.selection_max_tool,
            "trajectory": self.selection_max_trajectory,
        }


@dataclass(frozen=True, slots=True)
class LifecycleResponse:
    raw_output: str
    sampling_params: Mapping[str, float]
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")


class FastLoopPolicy(Protocol):
    def generate(self, prompt: LifecyclePrompt) -> LifecycleResponse: ...

    def repair(
        self,
        prompt: LifecyclePrompt,
        raw_output: str,
        error: str,
    ) -> LifecycleResponse: ...


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    task_id: str
    final_reward: float
    steps: int
    terminal_evaluation: Mapping[str, Any]
    selected_memory_ids: tuple[str, ...]
    written_memory_ids: tuple[str, ...]
    truncated: bool
    parse_error_count: int = 0
    response_parse_error_count: int = 0
    response_count: int = 0
    completed: bool = True
    project_truncated: bool = False
    agent_prompt_tokens: int | None = None
    agent_completion_tokens: int | None = None


DecisionT = TypeVar("DecisionT", bound=Decision)
def _retrieval_query(
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


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _candidate_evidence(candidate: MemoryCandidate) -> dict[str, Any]:
    return {
        "memory_id": candidate.memory_id,
        "memory_version": candidate.memory_version,
        "tier": candidate.tier.value,
        "rank": candidate.rank,
        "similarity": candidate.similarity,
    }


def _validate_selection_limits(
    decision: SelectionDecision,
    candidates: Sequence[MemoryCandidate],
    config: FastLoopConfig,
) -> SelectionDecision:
    if len(decision.memory_ids) > config.selection_max_total:
        raise ValueError("selected memory count exceeds the configured total limit")
    tier_by_id = {candidate.memory_id: candidate.tier.value for candidate in candidates}
    selected_by_tier: dict[str, int] = {}
    for memory_id in decision.memory_ids:
        tier = tier_by_id[memory_id]
        selected_by_tier[tier] = selected_by_tier.get(tier, 0) + 1
    limits = config.selection_tier_limits()
    for tier, count in selected_by_tier.items():
        if count > limits[tier]:
            raise ValueError(f"selected {tier} memory count exceeds the configured limit")
    return decision


def _generate_decision(
    policy: FastLoopPolicy,
    prompt: LifecyclePrompt,
    decision_type: type[DecisionT],
    *,
    candidate_ids: Sequence[str] | None = None,
    validator: Callable[[DecisionT], Any] | None = None,
    invalid_fallback: Callable[[str], DecisionT] | None = None,
    label: str,
) -> tuple[DecisionT, dict[str, Any]]:
    response = policy.generate(prompt)
    responses = [response]
    result = parse_decision(
        response.raw_output,
        decision_type,
        validator=validator,
        candidate_ids=candidate_ids,
    )
    repaired_output: str | None = None
    initial_error = result.error
    if result.decision is None:
        repair = policy.repair(prompt, response.raw_output, result.error or "invalid output")
        responses.append(repair)
        repaired_output = repair.raw_output
        result = parse_decision(
            repair.raw_output,
            decision_type,
            validator=validator,
            candidate_ids=candidate_ids,
        )
    fallback_used = False
    decision = result.decision
    if decision is None:
        terminal_error = result.error or "invalid output"
        if invalid_fallback is None:
            raise ValueError(f"invalid {label} decision after repair: {terminal_error}")
        decision = invalid_fallback(terminal_error)
        fallback_used = True
    prompt_tokens, completion_tokens = _combined_token_usage(responses)
    return decision, {
        "raw_output": response.raw_output,
        "repaired_output": repaired_output,
        "error": initial_error,
        "fallback_used": fallback_used,
        "sampling_params": dict(response.sampling_params),
        "latency_s": sum(item.latency_s for item in responses),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _combined_token_usage(
    responses: Sequence[LifecycleResponse],
) -> tuple[int | None, int | None]:
    if any(
        response.prompt_tokens is None or response.completion_tokens is None
        for response in responses
    ):
        return None, None
    return (
        sum(response.prompt_tokens or 0 for response in responses),
        sum(response.completion_tokens or 0 for response in responses),
    )


def _accumulate_token_usage(
    prompt_total: int,
    completion_total: int,
    complete: bool,
    audit: Mapping[str, Any],
) -> tuple[int, int, bool]:
    prompt_tokens = audit["prompt_tokens"]
    completion_tokens = audit["completion_tokens"]
    if prompt_tokens is None or completion_tokens is None:
        return prompt_total, completion_total, False
    return (
        prompt_total + prompt_tokens,
        completion_total + completion_tokens,
        complete,
    )


def _validate_write_decision(
    decision: WriteDecision,
    *,
    tools: Sequence[Mapping[str, Any]],
    task_id: str,
    context: ExecutionContext,
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


def _apply_write_quotas(
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


def _empty_write_decision(error: str) -> WriteDecision:
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
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _validate_write_metadata(nested)


def _persist_proposals(
    repository: MemoryRepository,
    proposals: Sequence[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    written: list[str] = []
    committed: list[str] = []
    replayed: list[str] = []
    for proposal in proposals:
        try:
            item = repository.add(**proposal["add_kwargs"])
        except ValueError as error:
            try:
                existing = repository.get(proposal["memory_id"])
            except BaseException as lookup_error:
                _attach_write_progress(lookup_error, committed, replayed)
                lookup_error.add_note(
                    f"Replay lookup followed rejected add ({type(error).__name__})"
                )
                raise
            if not _is_safe_replay(existing, proposal):
                _attach_write_progress(error, committed, replayed)
                raise
            item = existing
            replayed.append(item.id)
        except BaseException as error:
            _attach_write_progress(error, committed, replayed)
            raise
        else:
            committed.append(item.id)
        written.append(item.id)
    return tuple(written), tuple(replayed)


def _is_safe_replay(existing: MemoryItem | None, proposal: Mapping[str, Any]) -> bool:
    if existing is None:
        return False
    kwargs = proposal["add_kwargs"]
    return (
        existing.id == proposal["memory_id"]
        and existing.tier == kwargs["tier"]
        and existing.tier_schema_version == kwargs["tier_schema_version"]
        and existing.payload == kwargs["payload"]
        and canonical_content(existing.content) == canonical_content(kwargs["content"])
        and existing.source_task_ids == tuple(kwargs["source_task_ids"])
    )


def _attach_write_progress(
    error: BaseException,
    committed: Sequence[str],
    replayed: Sequence[str],
) -> None:
    try:
        setattr(error, "_fast_loop_committed_ids", tuple(committed))
        setattr(error, "_fast_loop_replayed_ids", tuple(replayed))
    except Exception:
        error.add_note("Fast-loop write progress was recorded before failure")
