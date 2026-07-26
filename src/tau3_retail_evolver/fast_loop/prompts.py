from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Literal
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from tau3_retail_evolver.fast_loop.baseline_prompt import normalize_tool_schema
from tau3_retail_evolver.fast_loop.decisions import maintenance_command_schemas
from tau3_retail_evolver.memory.retrieval import MemoryCandidate
from tau3_retail_evolver.memory.types import MemoryItem
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


PromptKind = Literal["selection", "action", "write", "maintenance"]
MAX_DIAGNOSTIC_ITEMS_PER_TIER = 100
MAX_DIAGNOSTIC_CONTENT_CHARS = 8_000
CUMULATIVE_TRAJECTORY_FORMAT = "final_observation_plus_actions_v1"
_FORBIDDEN_PUBLIC_KEY_NAMES = frozenset(
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


def _normalize_key(key: str) -> str:
    normalized = unicodedata.normalize("NFKC", key).casefold()
    return "".join(character for character in normalized if character.isalnum())


_FORBIDDEN_PUBLIC_KEYS = frozenset(
    _normalize_key(key) for key in _FORBIDDEN_PUBLIC_KEY_NAMES
)


class _PromptModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MaintenanceDiagnosticItem(_PromptModel):
    id: str = Field(min_length=1, max_length=256)
    tier: Literal["trajectory", "tip", "skill", "tool"]
    content: str = Field(min_length=1, max_length=MAX_DIAGNOSTIC_CONTENT_CHARS)
    version: int = Field(ge=1)
    status: Literal["active", "retired"]

    @field_validator("id", "content")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("diagnostic text must not be blank")
        return value


class MaintenanceTierDiagnostics(_PromptModel):
    items: tuple[MaintenanceDiagnosticItem, ...] = Field(
        max_length=MAX_DIAGNOSTIC_ITEMS_PER_TIER
    )


class MaintenanceDiagnostics(_PromptModel):
    trajectory: MaintenanceTierDiagnostics
    tip: MaintenanceTierDiagnostics
    skill: MaintenanceTierDiagnostics
    tool: MaintenanceTierDiagnostics


class LifecyclePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: PromptKind
    payload: dict[str, Any]
    command_schemas: tuple[dict[str, Any], ...]

    @field_validator("payload")
    @classmethod
    def payload_must_be_public_and_json_safe(cls, value: Any) -> Any:
        _require_json_safe(value, "lifecycle prompt")
        _reject_forbidden_fields(value, "lifecycle prompt")
        return value

    @field_validator("command_schemas")
    @classmethod
    def command_schemas_must_be_json_safe(cls, value: Any) -> Any:
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
            **project_public_context(
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
    include_memory_context: bool = True,
) -> LifecyclePrompt:
    payload = project_public_context(
        task_instruction=task_instruction,
        policy=policy,
        tools=tools,
        observation=observation,
        history=history,
    )
    if include_memory_context:
        payload["memories"] = [_public_memory(memory) for memory in memories]
    return LifecyclePrompt(
        kind="action",
        payload=payload,
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
    context = project_public_context(
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
    prompt_trajectory = _compact_cumulative_trajectory(
        normalized_trajectory,
        final_observation=context["observation"],
    )
    trajectory_payload: dict[str, Any] = {"trajectory": prompt_trajectory}
    if prompt_trajectory is not normalized_trajectory:
        trajectory_payload["trajectory_format"] = CUMULATIVE_TRAJECTORY_FORMAT
    return LifecyclePrompt(
        kind="write",
        payload={
            **context,
            **trajectory_payload,
            "terminal_evaluation": normalized_evaluation,
        },
        command_schemas=(),
    )


def build_maintenance_prompt(*, diagnostics: Mapping[str, Any]) -> LifecyclePrompt:
    normalized_diagnostics = _json_copy(diagnostics, "diagnostics")
    _reject_forbidden_fields(normalized_diagnostics, "diagnostics")
    try:
        validated_diagnostics = MaintenanceDiagnostics.model_validate_json(
            json.dumps(normalized_diagnostics, allow_nan=False)
        )
    except ValidationError as error:
        raise ValueError(f"invalid maintenance diagnostics: {error}") from error
    return LifecyclePrompt(
        kind="maintenance",
        payload={"diagnostics": validated_diagnostics.model_dump(mode="json")},
        command_schemas=maintenance_command_schemas(),
    )


def project_public_context(
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
    normalized_policy = _json_copy(sanitize_artifact_data(policy), "policy")
    normalized_tools = _json_copy(
        sanitize_artifact_data([normalize_tool_schema(tool) for tool in tools]),
        "tools",
    )
    _reject_forbidden_fields(normalized_policy, "policy")
    _reject_forbidden_fields(normalized_tools, "tools")
    context = {
        "task_instruction": sanitize_artifact_data(task_instruction).strip(),
        "policy": normalized_policy,
        "tools": normalized_tools,
        "observation": sanitize_artifact_data(observation).strip(),
        "history": _public_history(history),
    }
    _reject_forbidden_fields(context, "public prompt")
    return context


def _public_memory(candidate: MemoryCandidate | MemoryItem | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, MemoryCandidate):
        item = candidate.item
        public = {
            "id": item.id,
            "tier": item.tier.value,
            "content": item.content,
            "version": item.version,
            "rank": candidate.rank,
            "similarity": candidate.similarity,
        }
        if item.tier_schema_version == 2:
            public.update(
                tier_schema_version=2,
                payload=_json_copy(item.payload, "memory tier payload"),
            )
        return public
    if isinstance(candidate, MemoryItem):
        public = {
            "id": candidate.id,
            "tier": candidate.tier.value,
            "content": candidate.content,
            "version": candidate.version,
        }
        if candidate.tier_schema_version == 2:
            public.update(
                tier_schema_version=2,
                payload=_json_copy(candidate.payload, "memory tier payload"),
            )
        return public
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


def _compact_cumulative_trajectory(
    trajectory: list[Any],
    *,
    final_observation: str,
) -> list[Any]:
    if not trajectory:
        return trajectory

    compact: list[dict[str, Any]] = []
    previous_observation: str | None = None
    for turn, step in enumerate(trajectory):
        if not isinstance(step, Mapping):
            return trajectory
        observation = step.get("observation")
        next_observation = step.get("next_observation")
        if not isinstance(observation, str) or not isinstance(next_observation, str):
            return trajectory
        if previous_observation is not None and observation != previous_observation:
            return trajectory
        terminal_without_observation = not next_observation.strip()
        if terminal_without_observation:
            if (
                turn != len(trajectory) - 1
                or step.get("done") is not True
                or observation.strip() != final_observation
            ):
                return trajectory
        elif not next_observation.startswith(observation):
            return trajectory

        compact.append(
            {
                "turn": turn,
                **{
                    key: value
                    for key, value in step.items()
                    if key not in {"turn", "observation", "next_observation"}
                },
            }
        )
        previous_observation = (
            observation if terminal_without_observation else next_observation
        )

    if previous_observation is None or previous_observation.strip() != final_observation:
        return trajectory
    return compact


def _require_json_safe(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON serializable") from error


def _public_history(history: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in history:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError("history messages may contain only role and content")
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("history role and content must be strings")
        messages.append({"role": role, "content": content})
    return messages


def _reject_forbidden_fields(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and _normalize_key(key) in _FORBIDDEN_PUBLIC_KEYS:
                raise ValueError(f"forbidden hidden field in {label}: {key}")
            _reject_forbidden_fields(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_forbidden_fields(nested, label)
