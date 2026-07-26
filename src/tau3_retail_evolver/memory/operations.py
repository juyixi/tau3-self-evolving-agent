from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.types import (
    MemoryItem,
    MemoryStatus,
    MemoryTier,
    stable_memory_id,
)


class _Command(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LookupCommand(_Command):
    operation: Literal["lookup"] = "lookup"
    memory_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("memory_ids")
    @classmethod
    def ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank_ids(value)


class MergeCommand(_Command):
    operation: Literal["merge"] = "merge"
    source_ids: tuple[str, ...] = Field(min_length=2)
    content: str
    updated_round: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_ids")
    @classmethod
    def source_ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank_ids(value)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("merge content must not be blank")
        return value

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("merge metadata must be JSON serializable") from error
        return value


class DeleteCommand(_Command):
    operation: Literal["delete"] = "delete"
    memory_ids: tuple[str, ...] = Field(min_length=1)
    updated_round: int = Field(ge=0)
    reason: str

    @field_validator("memory_ids")
    @classmethod
    def ids_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank_ids(value)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("delete reason must not be blank")
        return value


MemoryCommand = Annotated[
    LookupCommand | MergeCommand | DeleteCommand,
    Field(discriminator="operation"),
]


class OperationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    looked_up: tuple[MemoryItem, ...] = ()
    created_ids: tuple[str, ...] = ()
    updated_ids: tuple[str, ...] = ()


class MemoryOperations:
    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def apply_batch(self, commands: list[MemoryCommand]) -> OperationResult:
        with self.repository._lock:
            return self._apply_batch(commands)

    def _apply_batch(self, commands: list[MemoryCommand]) -> OperationResult:
        if any(isinstance(command, LookupCommand) for command in commands) and any(
            not isinstance(command, LookupCommand) for command in commands
        ):
            raise ValueError("lookup commands cannot be mixed with write commands")
        state = {item.id: item for item in self.repository.list(status=None)}
        looked_up: list[MemoryItem] = []
        created_ids: list[str] = []
        updated_ids: list[str] = []
        modified_tiers: set[MemoryTier] = set()

        for command in commands:
            if isinstance(command, LookupCommand):
                looked_up.extend(_require_active(state, memory_id) for memory_id in command.memory_ids)
            elif isinstance(command, DeleteCommand):
                for memory_id in command.memory_ids:
                    current = _require(state, memory_id)
                    if current.status == MemoryStatus.RETIRED:
                        if (
                            current.updated_round == command.updated_round
                            and current.metadata.get("retired_reason") == command.reason
                        ):
                            _append_once(updated_ids, memory_id)
                            continue
                        raise ValueError(f"memory is not active: {memory_id}")
                    _validate_updated_round(current, command.updated_round)
                    state[memory_id] = _copy_item(
                        current,
                        status=MemoryStatus.RETIRED,
                        version=current.version + 1,
                        updated_round=command.updated_round,
                        metadata={**current.metadata, "retired_reason": command.reason},
                    )
                    modified_tiers.add(current.tier)
                    _append_once(updated_ids, memory_id)
            elif isinstance(command, MergeCommand):
                sources = [_require(state, memory_id) for memory_id in command.source_ids]
                tiers = {source.tier for source in sources}
                if len(tiers) != 1:
                    raise ValueError("merge sources must belong to the same tier")
                if any(source.tier_schema_version == 2 for source in sources):
                    raise ValueError(
                        "V2 memory merge requires a typed tier payload"
                    )
                tier = sources[0].tier
                target_id = stable_memory_id(tier, command.content)
                if target_id in state:
                    if _is_replayed_merge(state, sources, target_id, command):
                        _append_once(created_ids, target_id)
                        for source in sources:
                            _append_once(updated_ids, source.id)
                        continue
                    raise ValueError(f"duplicate memory: {target_id}")
                if any(source.status != MemoryStatus.ACTIVE for source in sources):
                    retired = next(
                        source.id for source in sources if source.status != MemoryStatus.ACTIVE
                    )
                    raise ValueError(f"memory is not active: {retired}")
                for source in sources:
                    _validate_updated_round(source, command.updated_round)
                source_ids = tuple(sorted(source.id for source in sources))
                merged = MemoryItem(
                    id=target_id,
                    tier=tier,
                    content=command.content,
                    retrieval_text=command.content,
                    metadata={**command.metadata, "merged_from": list(source_ids)},
                    source_task_ids=tuple(
                        sorted({task_id for source in sources for task_id in source.source_task_ids})
                    ),
                    created_round=command.updated_round,
                    updated_round=command.updated_round,
                )
                state[target_id] = merged
                created_ids.append(target_id)
                for source in sources:
                    state[source.id] = _copy_item(
                        source,
                        status=MemoryStatus.RETIRED,
                        version=source.version + 1,
                        updated_round=command.updated_round,
                        metadata={
                            **source.metadata,
                            "retired_reason": f"merged into {target_id}",
                        },
                    )
                    _append_once(updated_ids, source.id)
                modified_tiers.add(tier)
            else:
                raise TypeError(f"unsupported memory command: {type(command).__name__}")

        if modified_tiers:
            self.repository._commit_tier_states(
                {
                    tier: [item for item in state.values() if item.tier == tier]
                    for tier in modified_tiers
                }
            )
        return OperationResult(
            looked_up=tuple(looked_up),
            created_ids=tuple(created_ids),
            updated_ids=tuple(updated_ids),
        )


def _require_active(state: dict[str, MemoryItem], memory_id: str) -> MemoryItem:
    item = _require(state, memory_id)
    if item.status != MemoryStatus.ACTIVE:
        raise ValueError(f"memory is not active: {memory_id}")
    return item


def _require(state: dict[str, MemoryItem], memory_id: str) -> MemoryItem:
    item = state.get(memory_id)
    if item is None:
        raise KeyError(f"unknown memory: {memory_id}")
    return item


def _is_replayed_merge(
    state: dict[str, MemoryItem],
    sources: list[MemoryItem],
    target_id: str,
    command: MergeCommand,
) -> bool:
    target = state[target_id]
    source_ids = tuple(sorted(source.id for source in sources))
    expected_metadata = {**command.metadata, "merged_from": list(source_ids)}
    expected_tasks = tuple(
        sorted({task_id for source in sources for task_id in source.source_task_ids})
    )
    expected_reason = f"merged into {target_id}"
    return (
        target.created_round == command.updated_round
        and target.updated_round >= command.updated_round
        and all(target.metadata.get(key) == value for key, value in expected_metadata.items())
        and target.source_task_ids == expected_tasks
        and all(
            source.status == MemoryStatus.RETIRED
            and source.updated_round == command.updated_round
            and source.metadata.get("retired_reason") == expected_reason
            for source in sources
        )
    )


def _copy_item(item: MemoryItem, **updates: Any) -> MemoryItem:
    payload = item.model_dump(mode="python")
    payload.update(updates)
    return MemoryItem.model_validate(payload)


def _unique_nonblank_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    canonical = tuple(value.strip() for value in values)
    if any(not value for value in canonical):
        raise ValueError("memory IDs must not be blank")
    if len(set(canonical)) != len(canonical):
        raise ValueError("memory IDs must be unique")
    return canonical


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _validate_updated_round(item: MemoryItem, updated_round: int) -> None:
    if updated_round < item.updated_round:
        raise ValueError("updated_round must not move backwards")
