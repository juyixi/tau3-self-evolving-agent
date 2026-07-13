from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from tau3_retail_evolver.memory.operations import DeleteCommand, LookupCommand, MergeCommand
from tau3_retail_evolver.memory.retrieval import MemoryCandidate
from tau3_retail_evolver.memory.types import MemoryItem


PromptKind = Literal["selection", "action", "write", "maintenance"]
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "attribution_score",
        "embeddings",
        "embedding",
        "evaluation_criteria",
        "evaluator_metadata",
        "metadata",
        "privileged_hindsight",
        "source_task_ids",
        "task_id",
        "test_task_id",
    }
)


class LifecyclePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: PromptKind
    payload: dict[str, Any]
    command_schemas: tuple[dict[str, Any], ...]

    @field_validator("payload", "command_schemas")
    @classmethod
    def value_must_be_json_safe(cls, value: Any) -> Any:
        _require_json_safe(value, "lifecycle prompt")
        return value


def build_selection_prompt(
    *,
    task_instruction: str,
    policy: Any,
    tools: Sequence[Any],
    observation: str,
    history: Sequence[Mapping[str, Any]] = (),
    candidates: Sequence[MemoryCandidate | MemoryItem | Mapping[str, Any]] = (),
) -> LifecyclePrompt:
    return LifecyclePrompt(
        kind="selection",
        payload={
            **_public_context(
                task_instruction=task_instruction,
                policy=policy,
                tools=tools,
                observation=observation,
                history=history,
            ),
            "memories": [_public_memory(candidate) for candidate in candidates],
        },
        command_schemas=(),
    )


def build_action_prompt(
    *,
    task_instruction: str,
    policy: Any,
    tools: Sequence[Any],
    observation: str,
    history: Sequence[Mapping[str, Any]] = (),
    memories: Sequence[MemoryCandidate | MemoryItem | Mapping[str, Any]] = (),
) -> LifecyclePrompt:
    return LifecyclePrompt(
        kind="action",
        payload={
            **_public_context(
                task_instruction=task_instruction,
                policy=policy,
                tools=tools,
                observation=observation,
                history=history,
            ),
            "memories": [_public_memory(memory) for memory in memories],
        },
        command_schemas=(),
    )


def build_write_prompt(
    *,
    task_instruction: str,
    policy: Any,
    tools: Sequence[Any],
    observation: str,
    history: Sequence[Mapping[str, Any]] = (),
    trajectory: Sequence[Mapping[str, Any]],
    terminal_evaluation: Mapping[str, Any],
) -> LifecyclePrompt:
    context = _public_context(
        task_instruction=task_instruction,
        policy=policy,
        tools=tools,
        observation=observation,
        history=history,
    )
    normalized_trajectory = _json_copy(trajectory, "trajectory")
    normalized_evaluation = _json_copy(terminal_evaluation, "terminal evaluation")
    _reject_forbidden_fields(normalized_trajectory, "trajectory")
    _reject_forbidden_fields(normalized_evaluation, "terminal evaluation")
    return LifecyclePrompt(
        kind="write",
        payload={
            **context,
            "trajectory": normalized_trajectory,
            "terminal_evaluation": normalized_evaluation,
        },
        command_schemas=(),
    )


def build_maintenance_prompt(*, diagnostics: Mapping[str, Any]) -> LifecyclePrompt:
    normalized_diagnostics = _json_copy(diagnostics, "diagnostics")
    _reject_forbidden_fields(normalized_diagnostics, "diagnostics")
    return LifecyclePrompt(
        kind="maintenance",
        payload={"diagnostics": normalized_diagnostics},
        command_schemas=(
            {"operation": "lookup", "schema": LookupCommand.model_json_schema()},
            {"operation": "merge", "schema": MergeCommand.model_json_schema()},
            {"operation": "delete", "schema": DeleteCommand.model_json_schema()},
        ),
    )


def _public_context(
    *,
    task_instruction: str,
    policy: Any,
    tools: Sequence[Any],
    observation: str,
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise ValueError("task instruction must be a nonblank string")
    if not isinstance(observation, str) or not observation.strip():
        raise ValueError("observation must be a nonblank string")
    if isinstance(tools, (str, bytes)) or isinstance(history, (str, bytes)):
        raise ValueError("tools and history must be sequences")
    context = {
        "task_instruction": task_instruction.strip(),
        "policy": _json_copy(policy, "policy"),
        "tools": _json_copy(list(tools), "tools"),
        "observation": observation.strip(),
        "history": _json_copy(list(history), "history"),
    }
    _reject_forbidden_fields(context, "public prompt")
    return context


def _public_memory(candidate: MemoryCandidate | MemoryItem | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, MemoryCandidate):
        item = candidate.item
        return {
            "id": item.id,
            "tier": item.tier.value,
            "content": item.content,
            "version": item.version,
            "rank": candidate.rank,
            "similarity": candidate.similarity,
        }
    if isinstance(candidate, MemoryItem):
        return {
            "id": candidate.id,
            "tier": candidate.tier.value,
            "content": candidate.content,
            "version": candidate.version,
        }
    if not isinstance(candidate, Mapping):
        raise ValueError("memory candidates must be MemoryCandidate, MemoryItem, or mappings")
    required = ("id", "tier", "content", "version")
    if any(key not in candidate for key in required):
        raise ValueError("memory mappings require id, tier, content, and version")
    public = {key: candidate[key] for key in required}
    for key in ("rank", "similarity"):
        if key in candidate:
            public[key] = candidate[key]
    _require_json_safe(public, "public memory")
    return _json_copy(public, "public memory")


def _json_copy(value: Any, label: str) -> Any:
    _require_json_safe(value, label)
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _require_json_safe(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON serializable") from error


def _reject_forbidden_fields(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden hidden field in {label}: {key}")
            _reject_forbidden_fields(nested, label)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_fields(nested, label)
