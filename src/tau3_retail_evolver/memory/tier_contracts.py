from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from tau3_retail_evolver.memory.types import MemoryTier


TIER_SCHEMA_VERSION = 2
_ORDERED_LIST = re.compile(r"(?:^|\n)\s*\d+[.)]\s+\S")


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TipPayload(_ContractModel):
    condition: str | None = None
    guidance: str
    rationale: str | None = None
    scope: tuple[str, ...] = ()

    @field_validator("condition", "guidance", "rationale")
    @classmethod
    def text_must_be_nonblank(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "tip text")

    @field_validator("scope")
    @classmethod
    def scope_must_be_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_nonblank_values(value, "tip scope")

    @model_validator(mode="after")
    def guidance_must_be_atomic(self) -> TipPayload:
        if _ORDERED_LIST.search(self.guidance):
            raise ValueError("tip guidance must be one atomic rule, not an ordered workflow")
        return self


class SkillStep(_ContractModel):
    order: int = Field(ge=1)
    instruction: str
    success_signal: str | None = None

    @field_validator("instruction", "success_signal")
    @classmethod
    def text_must_be_nonblank(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "skill step text")


class SkillPayload(_ContractModel):
    goal: str
    preconditions: tuple[str, ...] = ()
    steps: tuple[SkillStep, ...] = Field(min_length=2)
    success_condition: str
    recovery: tuple[str, ...] = ()

    @field_validator("goal", "success_condition")
    @classmethod
    def text_must_be_nonblank(cls, value: str) -> str:
        return _nonblank(value, "skill text")

    @field_validator("preconditions", "recovery")
    @classmethod
    def sequences_must_be_nonblank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_nonblank_values(value, "skill sequence")

    @model_validator(mode="after")
    def step_order_must_be_contiguous(self) -> SkillPayload:
        expected = tuple(range(1, len(self.steps) + 1))
        actual = tuple(step.order for step in self.steps)
        if actual != expected:
            raise ValueError("skill step order must start at 1 and be contiguous")
        return self


class ToolCallExample(_ContractModel):
    name: str
    arguments: dict[str, Any]

    @field_validator("name")
    @classmethod
    def name_must_be_nonblank(cls, value: str) -> str:
        return _nonblank(value, "tool example name")


class ToolPayload(_ContractModel):
    tool_name: str
    purpose: str
    method: str | None = None
    preconditions: tuple[str, ...] = Field(min_length=1)
    argument_rules: dict[str, str] = Field(default_factory=dict)
    expected_effect: str
    example: ToolCallExample | None = None

    @field_validator("tool_name", "purpose", "method", "expected_effect")
    @classmethod
    def text_must_be_nonblank(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "tool text")

    @field_validator("preconditions")
    @classmethod
    def preconditions_must_be_nonblank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        return _canonical_nonblank_values(value, "tool preconditions")

    @field_validator("argument_rules")
    @classmethod
    def argument_rules_must_be_nonblank(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_name, raw_rule in value.items():
            name = _nonblank(raw_name, "tool argument name")
            rule = _nonblank(raw_rule, "tool argument rule")
            if name in normalized:
                raise ValueError(f"duplicate tool argument rule: {name}")
            normalized[name] = rule
        return normalized


class TrajectoryDraftPayload(_ContractModel):
    initial_state: str
    lesson: str

    @field_validator("initial_state", "lesson")
    @classmethod
    def text_must_be_nonblank(cls, value: str) -> str:
        return _nonblank(value, "trajectory text")


class TrajectoryStepPayload(_ContractModel):
    order: int = Field(ge=1)
    observation: str | None = None
    action: str
    action_name: str | None = None
    action_arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    reward: float
    done: bool
    terminated: bool | None = None
    truncated: bool | None = None

    @field_validator("observation", "action", "action_name", "result")
    @classmethod
    def text_must_be_nonblank(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "trajectory step text")

    @field_validator("reward")
    @classmethod
    def reward_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("trajectory reward must be finite")
        return value


class TrajectoryPayload(_ContractModel):
    source_episode_id: str
    task_group: str
    task_instruction: str | None = None
    initial_state: str
    steps: tuple[TrajectoryStepPayload, ...] = Field(min_length=1)
    final_reward: float
    result: Literal["success", "partial", "failure"]
    outcome_class: Literal["success", "task_failure", "infra_failure", "incomplete"] | None = None
    lesson: str | None = None

    @field_validator(
        "source_episode_id", "task_group", "task_instruction", "initial_state", "lesson"
    )
    @classmethod
    def text_must_be_nonblank(cls, value: str | None) -> str | None:
        return _optional_nonblank(value, "trajectory text")

    @field_validator("final_reward")
    @classmethod
    def final_reward_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("trajectory final reward must be finite")
        return value

    @model_validator(mode="after")
    def trajectory_must_be_consistent(self) -> TrajectoryPayload:
        expected = tuple(range(1, len(self.steps) + 1))
        if tuple(step.order for step in self.steps) != expected:
            raise ValueError("trajectory step order must start at 1 and be contiguous")
        if not math.isclose(
            self.steps[-1].reward,
            self.final_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("trajectory final reward must match the final step")
        if self.result != _reward_result(self.final_reward):
            raise ValueError("trajectory result does not match final reward")
        return self


DraftTierPayload = TipPayload | SkillPayload | ToolPayload | TrajectoryDraftPayload
StoredTierPayload = TipPayload | SkillPayload | ToolPayload | TrajectoryPayload
_STORED_ADAPTER = TypeAdapter(
    Annotated[StoredTierPayload, Field(union_mode="left_to_right")]
)


@dataclass(frozen=True, slots=True)
class MaterializedTierMemory:
    tier: MemoryTier
    payload: dict[str, Any]
    content: str
    retrieval_text: str
    classification_rule: str


def validate_draft_tier_payload(
    tier: MemoryTier | str,
    payload: DraftTierPayload,
) -> None:
    resolved = MemoryTier(tier)
    expected = {
        MemoryTier.TIP: TipPayload,
        MemoryTier.SKILL: SkillPayload,
        MemoryTier.TOOL: ToolPayload,
        MemoryTier.TRAJECTORY: TrajectoryDraftPayload,
    }[resolved]
    if not isinstance(payload, expected):
        raise ValueError(
            f"{resolved.value} memory requires {expected.__name__}, "
            f"not {type(payload).__name__}"
        )


def validate_stored_tier_payload(
    tier: MemoryTier | str,
    payload: Mapping[str, Any],
) -> StoredTierPayload:
    resolved = MemoryTier(tier)
    expected = {
        MemoryTier.TIP: TipPayload,
        MemoryTier.SKILL: SkillPayload,
        MemoryTier.TOOL: ToolPayload,
        MemoryTier.TRAJECTORY: TrajectoryPayload,
    }[resolved]
    validated = expected.model_validate_json(
        json.dumps(dict(payload), allow_nan=False, separators=(",", ":"))
    )
    return _STORED_ADAPTER.validate_python(validated)


def validate_tool_payload_against_tools(
    payload: ToolPayload | Mapping[str, Any],
    tools: Sequence[Mapping[str, Any]],
) -> ToolPayload:
    validated = (
        ToolPayload.model_validate_json(
            json.dumps(dict(payload), allow_nan=False, separators=(",", ":"))
        )
        if isinstance(payload, Mapping)
        else payload
    )
    _validate_tool_payload(validated, tools)
    return validated


def materialize_tier_memory(
    *,
    tier: MemoryTier | str,
    payload: DraftTierPayload,
    retrieval_text: str | None,
    tools: Sequence[Mapping[str, Any]],
    run_id: str,
    task_id: str,
    task_group: str,
    final_reward: float,
    trajectory: Sequence[Mapping[str, Any]],
) -> MaterializedTierMemory:
    resolved = MemoryTier(tier)
    validate_draft_tier_payload(resolved, payload)
    if resolved is MemoryTier.TOOL:
        assert isinstance(payload, ToolPayload)
        if payload.method is None:
            raise ValueError("new tool memory requires a reusable executable method")
        if payload.example is None:
            raise ValueError("new tool memory requires a valid example call")
        _validate_tool_payload(payload, tools)
        stored: StoredTierPayload = payload
    elif resolved is MemoryTier.TRAJECTORY:
        assert isinstance(payload, TrajectoryDraftPayload)
        stored = _materialize_trajectory(
            payload,
            run_id=run_id,
            task_id=task_id,
            task_group=task_group,
            final_reward=final_reward,
            trajectory=trajectory,
        )
    else:
        stored = payload
    content = render_tier_payload(resolved, stored)
    resolved_retrieval = (
        _nonblank(retrieval_text, "memory retrieval text")
        if retrieval_text is not None
        else default_retrieval_text(resolved, stored)
    )
    return MaterializedTierMemory(
        tier=resolved,
        payload=stored.model_dump(mode="json"),
        content=content,
        retrieval_text=resolved_retrieval,
        classification_rule=f"{resolved.value}-contract-v2",
    )


def materialize_rule_trajectory_memory(
    *,
    task_instruction: str,
    run_id: str,
    task_id: str,
    task_group: str,
    final_reward: float,
    outcome_class: Literal["success", "task_failure", "infra_failure", "incomplete"],
    trajectory: Sequence[Mapping[str, Any]],
) -> MaterializedTierMemory:
    """Build an immutable trajectory memory solely from observed runtime evidence."""
    if not trajectory:
        raise ValueError("trajectory memory requires at least one observed step")
    steps = tuple(
        _rule_trajectory_step(index, step)
        for index, step in enumerate(trajectory, start=1)
    )
    resolved_reward = _finite_number(final_reward, "final reward")
    stored = TrajectoryPayload(
        source_episode_id=f"{_nonblank(run_id, 'run ID')}:{_nonblank(task_id, 'task ID')}",
        task_group=_nonblank(task_group, "task group"),
        task_instruction=_bounded_text(task_instruction, "task instruction"),
        initial_state=_bounded_text(trajectory[0].get("observation"), "initial observation"),
        steps=steps,
        final_reward=resolved_reward,
        result=_reward_result(resolved_reward),
        outcome_class=outcome_class,
        lesson=None,
    )
    content = render_tier_payload(MemoryTier.TRAJECTORY, stored)
    return MaterializedTierMemory(
        tier=MemoryTier.TRAJECTORY,
        payload=stored.model_dump(mode="json"),
        content=content,
        retrieval_text=default_retrieval_text(MemoryTier.TRAJECTORY, stored),
        classification_rule="trajectory-runtime-record-v2",
    )


def render_tier_payload(
    tier: MemoryTier | str,
    payload: StoredTierPayload | Mapping[str, Any],
) -> str:
    resolved = MemoryTier(tier)
    stored = (
        validate_stored_tier_payload(resolved, payload)
        if isinstance(payload, Mapping)
        else payload
    )
    if resolved is MemoryTier.TIP:
        assert isinstance(stored, TipPayload)
        parts = []
        if stored.condition is not None:
            parts.append(f"Condition: {stored.condition}")
        parts.append(f"Guidance: {stored.guidance}")
        if stored.rationale is not None:
            parts.append(f"Rationale: {stored.rationale}")
        return "\n".join(parts)
    if resolved is MemoryTier.SKILL:
        assert isinstance(stored, SkillPayload)
        lines = [f"Goal: {stored.goal}"]
        if stored.preconditions:
            lines.append("Preconditions: " + "; ".join(stored.preconditions))
        lines.append("Steps:")
        lines.extend(
            f"{step.order}. {step.instruction}"
            + (
                f" Success signal: {step.success_signal}"
                if step.success_signal is not None
                else ""
            )
            for step in stored.steps
        )
        lines.append(f"Success condition: {stored.success_condition}")
        if stored.recovery:
            lines.append("Recovery: " + "; ".join(stored.recovery))
        return "\n".join(lines)
    if resolved is MemoryTier.TOOL:
        assert isinstance(stored, ToolPayload)
        lines = [
            f"Tool: {stored.tool_name}",
            f"Purpose: {stored.purpose}",
        ]
        if stored.method is not None:
            lines.append(f"Executable method: {stored.method}")
        lines.extend((
            "Preconditions: " + "; ".join(stored.preconditions),
            "Argument rules:",
        ))
        lines.extend(
            f"- {name}: {rule}" for name, rule in sorted(stored.argument_rules.items())
        )
        lines.append(f"Expected effect: {stored.expected_effect}")
        if stored.example is not None:
            lines.append(
                "Example call: "
                f"{stored.example.name}("
                f"{json.dumps(stored.example.arguments, ensure_ascii=False, sort_keys=True)}"
                ")"
            )
        return "\n".join(lines)
    assert isinstance(stored, TrajectoryPayload)
    lines = []
    if stored.task_instruction is not None:
        lines.append(f"Task: {stored.task_instruction}")
    lines.extend([
        f"Case: {stored.initial_state}",
        f"Episode: {stored.source_episode_id}",
        f"Task group: {stored.task_group}",
        "Observed steps:",
    ])
    for step in stored.steps:
        if step.observation is not None:
            lines.append(f"{step.order}. Observation: {step.observation}")
            lines.append(f"   Action: {step.action}")
            if step.result is not None:
                lines.append(f"   Result: {step.result}")
            lines.append(
                "   Outcome: "
                f"reward={step.reward}, done={str(step.done).lower()}, "
                f"terminated={str(bool(step.terminated)).lower()}, "
                f"truncated={str(bool(step.truncated)).lower()}"
            )
        else:
            lines.append(
                f"{step.order}. {step.action} "
                f"[reward={step.reward}, done={str(step.done).lower()}]"
            )
    outcome_suffix = (
        f", outcome_class={stored.outcome_class}"
        if stored.outcome_class is not None
        else ""
    )
    lines.append(
        f"Result: {stored.result} (final_reward={stored.final_reward}{outcome_suffix})"
    )
    if stored.lesson is not None:
        lines.append(f"Lesson: {stored.lesson}")
    return "\n".join(lines)


def default_retrieval_text(
    tier: MemoryTier | str,
    payload: StoredTierPayload | Mapping[str, Any],
) -> str:
    resolved = MemoryTier(tier)
    stored = (
        validate_stored_tier_payload(resolved, payload)
        if isinstance(payload, Mapping)
        else payload
    )
    if isinstance(stored, TipPayload):
        return " ".join(
            part for part in (stored.condition, stored.guidance) if part is not None
        )
    if isinstance(stored, SkillPayload):
        return f"{stored.goal} {' '.join(step.instruction for step in stored.steps)}"
    if isinstance(stored, ToolPayload):
        return " ".join(
            part for part in (stored.tool_name, stored.purpose, stored.method) if part
        )
    assert isinstance(stored, TrajectoryPayload)
    return " ".join(
        part
        for part in (
            stored.task_instruction,
            stored.initial_state,
            stored.outcome_class,
            stored.lesson,
        )
        if part
    )


def _materialize_trajectory(
    draft: TrajectoryDraftPayload,
    *,
    run_id: str,
    task_id: str,
    task_group: str,
    final_reward: float,
    trajectory: Sequence[Mapping[str, Any]],
) -> TrajectoryPayload:
    if not trajectory:
        raise ValueError("trajectory memory requires at least one observed step")
    steps = tuple(
        TrajectoryStepPayload(
            order=index,
            action=_nonblank(step.get("action"), "trajectory action"),
            reward=_finite_number(step.get("reward"), "trajectory reward"),
            done=_strict_bool(step.get("done"), "trajectory done"),
        )
        for index, step in enumerate(trajectory, start=1)
    )
    resolved_reward = _finite_number(final_reward, "final reward")
    return TrajectoryPayload(
        source_episode_id=f"{_nonblank(run_id, 'run ID')}:{_nonblank(task_id, 'task ID')}",
        task_group=_nonblank(task_group, "task group"),
        initial_state=draft.initial_state,
        steps=steps,
        final_reward=resolved_reward,
        result=_reward_result(resolved_reward),
        lesson=draft.lesson,
    )


def _rule_trajectory_step(
    order: int,
    step: Mapping[str, Any],
) -> TrajectoryStepPayload:
    action = _nonblank(step.get("action"), "trajectory action")
    action_name, action_arguments = _structured_action(action)
    return TrajectoryStepPayload(
        order=order,
        observation=_bounded_text(step.get("observation"), "trajectory observation"),
        action=action,
        action_name=action_name,
        action_arguments=action_arguments,
        result=_optional_bounded_text(
            step.get("next_observation"), "trajectory result"
        ),
        reward=_finite_number(step.get("reward"), "trajectory reward"),
        done=_strict_bool(step.get("done"), "trajectory done"),
        terminated=_strict_bool(step.get("terminated"), "trajectory terminated"),
        truncated=_strict_bool(step.get("truncated"), "trajectory truncated"),
    )


def _structured_action(action: str) -> tuple[str | None, dict[str, Any]]:
    try:
        parsed = json.loads(action)
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(parsed, Mapping):
        return None, {}
    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, Mapping):
        return None, {}
    return name.strip(), dict(arguments)


def _bounded_text(value: Any, label: str, *, limit: int = 500) -> str:
    return _nonblank(value, label)[:limit]


def _optional_bounded_text(
    value: Any,
    label: str,
    *,
    limit: int = 500,
) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _bounded_text(value, label, limit=limit)


def _validate_tool_payload(
    payload: ToolPayload,
    tools: Sequence[Mapping[str, Any]],
) -> None:
    registry = _tool_registry(tools)
    schema = registry.get(payload.tool_name)
    if schema is None:
        raise ValueError(
            f"tool memory references an unavailable environment tool: {payload.tool_name}"
        )
    parameters = schema.get("parameters")
    parameters = parameters if isinstance(parameters, Mapping) else {}
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    declared = set(properties)
    referenced = set(payload.argument_rules)
    if declared:
        unknown = sorted(referenced - declared)
        if unknown:
            raise ValueError(
                "tool memory references undeclared arguments: " + ", ".join(unknown)
            )
        required = parameters.get("required")
        required_names = (
            {item for item in required if isinstance(item, str)}
            if isinstance(required, Sequence) and not isinstance(required, (str, bytes))
            else set()
        )
        missing = sorted(required_names - referenced)
        if missing:
            raise ValueError(
                "tool memory omits required argument rules: " + ", ".join(missing)
            )
    elif referenced:
        raise ValueError(
            f"tool {payload.tool_name} declares no arguments but argument_rules is not empty"
        )
    if payload.example is None:
        return
    if payload.example.name != payload.tool_name:
        raise ValueError("tool example name must match tool_name")
    arguments = payload.example.arguments
    unknown = sorted(set(arguments) - declared)
    if unknown:
        raise ValueError("tool example has undeclared arguments: " + ", ".join(unknown))
    required = parameters.get("required")
    required_names = (
        {item for item in required if isinstance(item, str)}
        if isinstance(required, Sequence) and not isinstance(required, (str, bytes))
        else set()
    )
    missing = sorted(required_names - set(arguments))
    if missing:
        raise ValueError("tool example omits required arguments: " + ", ".join(missing))
    for name, value in arguments.items():
        expected = properties.get(name)
        if isinstance(expected, Mapping) and not _matches_json_type(value, expected.get("type")):
            raise ValueError(f"tool example argument has invalid type: {name}")


def _tool_registry(
    tools: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    registry: dict[str, Mapping[str, Any]] = {}
    for raw in tools:
        if not isinstance(raw, Mapping):
            raise ValueError("environment tool schemas must be objects")
        function = raw.get("function")
        schema = function if isinstance(function, Mapping) else raw
        name = schema.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("environment tool schema is missing a name")
        normalized = name.strip()
        if normalized in registry:
            raise ValueError(f"duplicate environment tool schema: {normalized}")
        registry[normalized] = schema
    return registry


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(value, str) and value.startswith("<") and value.endswith(">"):
        return True
    if expected is None:
        return True
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    return {
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": lambda: type(value) is bool,
        "array": lambda: isinstance(value, list),
        "object": lambda: isinstance(value, dict),
        "null": lambda: value is None,
    }.get(expected, lambda: True)()


def _reward_result(value: float) -> Literal["success", "partial", "failure"]:
    if math.isclose(value, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return "success"
    if value <= 0.0:
        return "failure"
    return "partial"


def _canonical_nonblank_values(
    values: tuple[str, ...],
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(_nonblank(value, label) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} values must be unique")
    return normalized


def _optional_nonblank(value: str | None, label: str) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _nonblank(value, label)


def _nonblank(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value.strip()


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value
