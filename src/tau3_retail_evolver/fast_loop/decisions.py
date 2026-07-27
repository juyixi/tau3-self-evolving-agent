from __future__ import annotations

from collections.abc import Callable, Collection
import json
from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from tau3_retail_evolver.memory.operations import (
    DeleteCommand,
    LookupCommand,
    MemoryCommand,
    MergeCommand,
)
from tau3_retail_evolver.memory.tier_contracts import (
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryDraftPayload,
)
from tau3_retail_evolver.memory.types import MemoryTier


class _DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SelectionDecision(_DecisionModel):
    memory_ids: tuple[str, ...]

    @field_validator("memory_ids")
    @classmethod
    def ids_must_be_unique_and_nonblank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(memory_id.strip() for memory_id in value)
        if any(not memory_id for memory_id in canonical):
            raise ValueError("memory IDs must not be blank")
        if len(set(canonical)) != len(canonical):
            raise ValueError("memory IDs must be unique")
        return canonical

    def validate_candidates(self, candidate_ids: Collection[str]) -> SelectionDecision:
        allowed = {memory_id.strip() for memory_id in candidate_ids}
        unknown = sorted(set(self.memory_ids) - allowed)
        if unknown:
            raise ValueError(f"selected memory IDs are not a candidate: {', '.join(unknown)}")
        return self


class ActionDecision(_DecisionModel):
    action: str

    @field_validator("action")
    @classmethod
    def action_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("action must not be blank")
        return value


class _MemoryWriteBase(_DecisionModel):
    retrieval_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("retrieval_text")
    @classmethod
    def text_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        _require_json_safe(value, "memory metadata")
        return value


class TipMemoryWrite(_MemoryWriteBase):
    tier: Literal[MemoryTier.TIP]
    payload: TipPayload


class SkillMemoryWrite(_MemoryWriteBase):
    tier: Literal[MemoryTier.SKILL]
    payload: SkillPayload


class ToolMemoryWrite(_MemoryWriteBase):
    tier: Literal[MemoryTier.TOOL]
    payload: ToolPayload


class TrajectoryMemoryWrite(_MemoryWriteBase):
    tier: Literal[MemoryTier.TRAJECTORY]
    payload: TrajectoryDraftPayload


MemoryWrite = Annotated[
    TipMemoryWrite | SkillMemoryWrite | ToolMemoryWrite | TrajectoryMemoryWrite,
    Field(discriminator="tier"),
]


class WriteDecision(_DecisionModel):
    memories: tuple[MemoryWrite, ...]


class _StrictLookupCommand(_DecisionModel):
    operation: Literal["lookup"] = "lookup"
    memory_ids: tuple[str, ...] = Field(min_length=1)


class _StrictMergeCommand(_DecisionModel):
    operation: Literal["merge"] = "merge"
    source_ids: tuple[str, ...] = Field(min_length=2)
    content: str
    updated_round: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _StrictDeleteCommand(_DecisionModel):
    operation: Literal["delete"] = "delete"
    memory_ids: tuple[str, ...] = Field(min_length=1)
    updated_round: int = Field(ge=0)
    reason: str


_StrictMemoryCommand = Annotated[
    _StrictLookupCommand | _StrictMergeCommand | _StrictDeleteCommand,
    Field(discriminator="operation"),
]
_STRICT_COMMAND_ADAPTER = TypeAdapter(_StrictMemoryCommand)


class MaintenanceDecision(_DecisionModel):
    commands: tuple[Annotated[MemoryCommand, Field(discriminator="operation")], ...]
    reviews: tuple["MaintenanceReview", ...] = ()

    @model_validator(mode="before")
    @classmethod
    def commands_must_use_strict_input_types(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if isinstance(value.get("commands"), (list, tuple)):
            commands = []
            for command in value["commands"]:
                payload = (
                    command.model_dump(mode="python")
                    if isinstance(command, BaseModel)
                    else command
                )
                encoded = json.dumps(payload, allow_nan=False)
                strict_command = _STRICT_COMMAND_ADAPTER.validate_json(encoded)
                commands.append(strict_command.model_dump(mode="python"))
            normalized["commands"] = tuple(commands)
        if isinstance(value.get("reviews"), (list, tuple)):
            normalized["reviews"] = tuple(
                {
                    **review,
                    "memory_ids": tuple(review["memory_ids"]),
                }
                if isinstance(review, dict)
                and isinstance(review.get("memory_ids"), (list, tuple))
                else review
                for review in value["reviews"]
            )
        return normalized


class MaintenanceReview(_DecisionModel):
    """An auditable disposition for a memory inspected during maintenance."""

    memory_ids: tuple[str, ...] = Field(min_length=1)
    disposition: Literal["keep", "merge", "retire"]
    reason: str = Field(min_length=1, max_length=1_000)

    @field_validator("memory_ids")
    @classmethod
    def review_ids_must_be_unique_and_nonblank(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        canonical = tuple(memory_id.strip() for memory_id in value)
        if any(not memory_id for memory_id in canonical):
            raise ValueError("review memory IDs must not be blank")
        if len(set(canonical)) != len(canonical):
            raise ValueError("review memory IDs must be unique")
        return canonical


Decision = SelectionDecision | ActionDecision | WriteDecision | MaintenanceDecision
DecisionT = TypeVar("DecisionT", bound=Decision)
Validator = Callable[[DecisionT], Any]
Repair = Callable[[str, str], str]


class DecisionParseResult(_DecisionModel, Generic[DecisionT]):
    decision: DecisionT | None = None
    raw_output: str
    repaired_output: str | None = None
    error: str | None = None


def parse_decision(
    raw_output: str,
    decision_type: type[DecisionT],
    validator: Validator[DecisionT] | None = None,
    repair: Repair | None = None,
    *,
    candidate_ids: Collection[str] | None = None,
) -> DecisionParseResult[DecisionT]:
    """Parse exactly one JSON object, with at most one caller-provided repair."""
    if issubclass(decision_type, SelectionDecision) and candidate_ids is None:
        return DecisionParseResult(
            raw_output=raw_output,
            error="candidate_ids is required when parsing SelectionDecision",
        )
    parsed, error = _parse_once(raw_output, decision_type, validator, candidate_ids)
    if error is None:
        return DecisionParseResult(decision=parsed, raw_output=raw_output)
    if repair is None:
        return DecisionParseResult(raw_output=raw_output, error=error)

    try:
        repaired_output = repair(raw_output, error)
    except Exception as repair_error:
        return DecisionParseResult(
            raw_output=raw_output,
            error=f"repair failed: {repair_error}",
        )
    if not isinstance(repaired_output, str):
        return DecisionParseResult(
            raw_output=raw_output,
            error="repair did not return a string",
        )

    repaired, repaired_error = _parse_once(
        repaired_output,
        decision_type,
        validator,
        candidate_ids,
    )
    if repaired_error is None:
        return DecisionParseResult(
            decision=repaired,
            raw_output=raw_output,
            repaired_output=repaired_output,
        )
    return DecisionParseResult(
        raw_output=raw_output,
        repaired_output=repaired_output,
        error=repaired_error,
    )


def _parse_once(
    raw_output: str,
    decision_type: type[DecisionT],
    validator: Validator[DecisionT] | None,
    candidate_ids: Collection[str] | None,
) -> tuple[DecisionT | None, str | None]:
    if not isinstance(raw_output, str):
        return None, "model output must be a string"
    try:
        payload = json.loads(raw_output)
    except (TypeError, ValueError) as error:
        return None, f"invalid JSON: {error}"
    if not isinstance(payload, dict):
        return None, "decision output must be one JSON object"
    try:
        decision = decision_type.model_validate_json(json.dumps(payload, allow_nan=False))
        if isinstance(decision, SelectionDecision):
            if candidate_ids is None:
                raise ValueError("candidate_ids is required when parsing SelectionDecision")
            decision.validate_candidates(candidate_ids)
        if validator is not None:
            validator(decision)
    except (TypeError, ValueError, ValidationError) as error:
        return None, str(error)
    return decision, None


def _require_json_safe(value: Any, label: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON serializable") from error


def maintenance_command_schemas() -> tuple[dict[str, Any], ...]:
    return (
        {"operation": "lookup", "schema": _StrictLookupCommand.model_json_schema()},
        {"operation": "merge", "schema": _StrictMergeCommand.model_json_schema()},
        {"operation": "delete", "schema": _StrictDeleteCommand.model_json_schema()},
    )
