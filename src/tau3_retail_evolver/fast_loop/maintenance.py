from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
import unicodedata

from tau3_retail_evolver.credential_policy import is_credential_key
from tau3_retail_evolver.fast_loop.decisions import (
    MaintenanceDecision,
    parse_decision,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.prompts import (
    MAX_DIAGNOSTIC_CONTENT_CHARS,
    MAX_DIAGNOSTIC_ITEMS_PER_TIER,
    build_maintenance_prompt,
)
from tau3_retail_evolver.fast_loop.runner import FastLoopPolicy
from tau3_retail_evolver.memory.json_store import write_bytes_atomic
from tau3_retail_evolver.memory.locking import reentrant_process_lock
from tau3_retail_evolver.memory.operations import (
    DeleteCommand,
    LookupCommand,
    MemoryCommand,
    MemoryOperations,
    MergeCommand,
)
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.tier_contracts import (
    SkillPayload,
    TipPayload,
    ToolPayload,
    TrajectoryPayload,
    render_tier_payload,
    validate_stored_tier_payload,
)
from tau3_retail_evolver.memory.types import MemoryTier
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


_STATE_FILENAME = "maintenance_state.json"
_STATE_SCHEMA_VERSION = 2
_LOCK_NAMESPACE = "fast-loop-maintenance"


@dataclass(frozen=True, slots=True)
class MaintenanceState:
    schema_version: int = _STATE_SCHEMA_VERSION
    completed_rounds: tuple[int, ...] = ()
    review_cursor_by_tier: tuple[tuple[str, str | None], ...] = ()

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported maintenance state schema")
        if not isinstance(self.completed_rounds, tuple) or any(
            type(round_number) is not int or round_number <= 0
            for round_number in self.completed_rounds
        ):
            raise ValueError("maintenance state rounds must be positive integers")
        if self.completed_rounds != tuple(sorted(set(self.completed_rounds))):
            raise ValueError("maintenance state rounds must be sorted and unique")
        expected = {tier.value for tier in MemoryTier}
        cursor_keys = {tier for tier, _ in self.review_cursor_by_tier}
        if cursor_keys and cursor_keys != expected:
            raise ValueError("maintenance cursors must cover every tier")
        if any(
            tier not in expected or (cursor is not None and not cursor.strip())
            for tier, cursor in self.review_cursor_by_tier
        ):
            raise ValueError("maintenance cursors must be valid tier/id pairs")

    def cursor_for(self, tier: MemoryTier) -> str | None:
        return dict(self.review_cursor_by_tier).get(tier.value)


@dataclass(frozen=True, slots=True)
class MaintenanceResult:
    due: bool
    executed: bool
    maintenance_round: int
    commands: tuple[MemoryCommand, ...] = ()
    looked_up_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    updated_ids: tuple[str, ...] = ()


def bounded_diagnostics(
    repository: MemoryRepository,
    *,
    per_tier_limit: int = 100,
) -> dict[str, Any]:
    if (
        type(per_tier_limit) is not int
        or not 1 <= per_tier_limit <= MAX_DIAGNOSTIC_ITEMS_PER_TIER
    ):
        raise ValueError(
            "per_tier_limit must be between 1 and "
            f"{MAX_DIAGNOSTIC_ITEMS_PER_TIER}"
        )

    diagnostics, _ = _paged_diagnostics(
        repository,
        per_tier_limit=per_tier_limit,
        cursors={},
        similarity_threshold=1.0,
        priority_pair_limit=0,
    )
    return diagnostics


def _paged_diagnostics(
    repository: MemoryRepository,
    *,
    per_tier_limit: int,
    cursors: Mapping[str, str | None],
    similarity_threshold: float,
    priority_pair_limit: int,
) -> tuple[dict[str, Any], dict[str, str | None]]:
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if type(priority_pair_limit) is not int or priority_pair_limit < 0:
        raise ValueError("priority_pair_limit must be non-negative")
    diagnostics: dict[str, Any] = {}
    next_cursors: dict[str, str | None] = {}
    with repository.read_transaction():
        for tier in MemoryTier:
            items = repository.list(tier=tier)
            page, next_cursor = _rotated_page(
                items,
                cursor=cursors.get(tier.value),
                limit=per_tier_limit,
            )
            priority_ids = _priority_ids(
                items,
                threshold=similarity_threshold,
                pair_limit=priority_pair_limit,
            )
            selected_by_id = {item.id: item for item in page}
            for item in items:
                if item.id in priority_ids:
                    selected_by_id.setdefault(item.id, item)
            ordered_ids = [memory_id for memory_id in priority_ids if memory_id in selected_by_id]
            ordered_ids.extend(item.id for item in page if item.id not in priority_ids)
            selected = [selected_by_id[memory_id] for memory_id in ordered_ids[:per_tier_limit]]
            diagnostics[tier.value] = {
                "items": [
                    {
                        "id": item.id,
                        "tier": item.tier.value,
                        "content": item.content[:MAX_DIAGNOSTIC_CONTENT_CHARS],
                        "version": item.version,
                        "status": item.status.value,
                    }
                    for item in selected
                ]
            }
            next_cursors[tier.value] = next_cursor
    return diagnostics, next_cursors


def _rotated_page(
    items: list[Any], *, cursor: str | None, limit: int
) -> tuple[list[Any], str | None]:
    if not items:
        return [], None
    start = 0
    if cursor is not None:
        start = next((index + 1 for index, item in enumerate(items) if item.id == cursor), 0)
        if start >= len(items):
            start = 0
    page = (items[start:] + items[:start])[:limit]
    return page, page[-1].id


def _priority_ids(
    items: list[Any], *, threshold: float, pair_limit: int
) -> list[str]:
    pairs: list[tuple[float, str, str]] = []
    if pair_limit == 0:
        return []
    for left_index, left in enumerate(items):
        if left.embedding is None:
            continue
        for right in items[left_index + 1 :]:
            if right.embedding is None:
                continue
            similarity = _cosine(left.embedding, right.embedding)
            if similarity >= threshold:
                pairs.append((similarity, left.id, right.id))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    return list(dict.fromkeys(memory_id for _, left, right in pairs[:pair_limit] for memory_id in (left, right)))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def run_due_maintenance(
    *,
    completed_train_tasks: int,
    period: int,
    repository: MemoryRepository,
    policy: FastLoopPolicy,
    context: RunContext,
    per_tier_limit: int = 100,
    tip_capacity: int = 240,
    similarity_threshold: float = 0.92,
    priority_pair_limit: int = 24,
) -> MaintenanceResult:
    if type(completed_train_tasks) is not int or completed_train_tasks < 0:
        raise ValueError("completed_train_tasks must be a non-negative integer")
    if type(period) is not int or period <= 0:
        raise ValueError("period must be a positive integer")
    if context.split != "train" or context.mode != RunMode.LEARN:
        raise ValueError("maintenance requires train split and learn mode")
    return _run_due_maintenance(
        completed_tasks=completed_train_tasks,
        period=period,
        repository=repository,
        policy=policy,
        context=context,
        per_tier_limit=per_tier_limit,
        tip_capacity=tip_capacity,
        similarity_threshold=similarity_threshold,
        priority_pair_limit=priority_pair_limit,
    )


def run_evaluation_maintenance(
    *,
    completed_tasks: int,
    period: int,
    repository: MemoryRepository,
    policy: FastLoopPolicy,
    context: RunContext,
    per_tier_limit: int = 100,
) -> MaintenanceResult:
    if type(completed_tasks) is not int or completed_tasks < 0:
        raise ValueError("completed tasks must be a non-negative integer")
    if type(period) is not int or period <= 0:
        raise ValueError("period must be a positive integer")
    if context.split not in {"test", "base"} or context.mode != RunMode.EVALUATE:
        raise ValueError("evaluation maintenance requires test/base and EVALUATE mode")
    return _run_due_maintenance(
        completed_tasks=completed_tasks,
        period=period,
        repository=repository,
        policy=policy,
        context=context,
        per_tier_limit=per_tier_limit,
    )


def _run_due_maintenance(
    *,
    completed_tasks: int,
    period: int,
    repository: MemoryRepository,
    policy: FastLoopPolicy,
    context: RunContext,
    per_tier_limit: int,
    tip_capacity: int = 240,
    similarity_threshold: float = 0.92,
    priority_pair_limit: int = 24,
) -> MaintenanceResult:
    if not isinstance(repository, MemoryRepository) or repository.is_read_only:
        raise TypeError("maintenance requires a mutable MemoryRepository")

    maintenance_round = completed_tasks // period
    state_path = repository.root / _STATE_FILENAME
    scheduler_lock = reentrant_process_lock(
        repository.root,
        namespace=_LOCK_NAMESPACE,
    )
    with scheduler_lock:
        state = _load_state(state_path, maintenance_round)
        if maintenance_round == 0:
            return MaintenanceResult(
                due=False,
                executed=False,
                maintenance_round=maintenance_round,
            )
        if maintenance_round in state.completed_rounds:
            return MaintenanceResult(
                due=True,
                executed=False,
                maintenance_round=maintenance_round,
            )
        return _execute_round(
            completed_train_tasks=completed_tasks,
            period=period,
            maintenance_round=maintenance_round,
            repository=repository,
            policy=policy,
            context=context,
            per_tier_limit=per_tier_limit,
            tip_capacity=tip_capacity,
            similarity_threshold=similarity_threshold,
            priority_pair_limit=priority_pair_limit,
            state=state,
            state_path=state_path,
        )


def _execute_round(
    *,
    completed_train_tasks: int,
    period: int,
    maintenance_round: int,
    repository: MemoryRepository,
    policy: FastLoopPolicy,
    context: RunContext,
    per_tier_limit: int,
    tip_capacity: int,
    similarity_threshold: float,
    priority_pair_limit: int,
    state: MaintenanceState,
    state_path: Path,
) -> MaintenanceResult:
    task_key = f"maintenance-round-{maintenance_round}"
    try:
        diagnostics, next_cursors = _paged_diagnostics(
            repository,
            per_tier_limit=per_tier_limit,
            cursors={tier.value: state.cursor_for(tier) for tier in MemoryTier},
            similarity_threshold=similarity_threshold,
            priority_pair_limit=priority_pair_limit,
        )
        diagnostics = sanitize_artifact_data(diagnostics)
        active_tip_count = len(repository.list(tier=MemoryTier.TIP))
        requires_tip_reduction = active_tip_count > tip_capacity
        required_tip_reduction_count = max(0, active_tip_count - tip_capacity)
        prompt = build_maintenance_prompt(
            diagnostics=diagnostics,
            maintenance_context={
                "active_counts": {tier.value: len(repository.list(tier=tier)) for tier in MemoryTier},
                "tip_capacity": tip_capacity,
                "requires_tip_reduction": requires_tip_reduction,
                "required_tip_reduction_count": required_tip_reduction_count,
                "review_instruction": (
                    "Priority items appear first in each tier. Review at least one item "
                    "unless no safe disposition can be made."
                ),
            },
        )
        public_diagnostics = prompt.payload["diagnostics"]
        _emit(
            context,
            task_key,
            "MaintenanceStarted",
            maintenance_round=maintenance_round,
            completed_train_tasks=completed_train_tasks,
            period=period,
            per_tier_counts={
                tier: len(tier_diagnostics["items"])
                for tier, tier_diagnostics in public_diagnostics.items()
            },
            diagnostics=public_diagnostics,
            review_cursor_by_tier={tier.value: state.cursor_for(tier) for tier in MemoryTier},
        )
        decision, repair_used = _generate_decision(
            policy,
            prompt,
            maintenance_round,
            reviewed_ids={
                item["id"]
                for tier in public_diagnostics.values()
                for item in tier["items"]
            },
            reviewed_tip_ids={item["id"] for item in public_diagnostics["tip"]["items"]},
            requires_tip_reduction=requires_tip_reduction,
        )
        decision = _bind_command_round(decision, maintenance_round)
        decision = _prepare_merge_commands(decision, repository)
        canonical_commands = [
            command.model_dump(mode="json") for command in decision.commands
        ]
        _emit(
            context,
            task_key,
            "MaintenanceProposed",
            maintenance_round=maintenance_round,
            commands=canonical_commands,
            reviews=[review.model_dump(mode="json") for review in decision.reviews],
            repair_used=repair_used,
        )
        operation_result = MemoryOperations(repository).apply_batch(
            list(decision.commands)
        )
        completed_state = MaintenanceState(
            completed_rounds=tuple(
                sorted((*state.completed_rounds, maintenance_round))
            ),
            review_cursor_by_tier=tuple(sorted(next_cursors.items())),
        )
        write_bytes_atomic(state_path, _encode_state(completed_state))
        looked_up_ids = tuple(item.id for item in operation_result.looked_up)
        _emit(
            context,
            task_key,
            "MaintenanceCommitted",
            maintenance_round=maintenance_round,
            looked_up_ids=list(looked_up_ids),
            created_ids=list(operation_result.created_ids),
            updated_ids=list(operation_result.updated_ids),
            completed_rounds=list(completed_state.completed_rounds),
            review_cursor_by_tier=dict(completed_state.review_cursor_by_tier),
        )
        return MaintenanceResult(
            due=True,
            executed=True,
            maintenance_round=maintenance_round,
            commands=decision.commands,
            looked_up_ids=looked_up_ids,
            created_ids=operation_result.created_ids,
            updated_ids=operation_result.updated_ids,
        )
    except BaseException as error:
        _emit_failure(context, task_key, maintenance_round, error)
        raise


def _generate_decision(
    policy: FastLoopPolicy,
    prompt: Any,
    maintenance_round: int,
    reviewed_ids: set[str],
    reviewed_tip_ids: set[str],
    requires_tip_reduction: bool,
) -> tuple[MaintenanceDecision, bool]:
    response = policy.generate(prompt)
    result = parse_decision(
        response.raw_output,
        MaintenanceDecision,
        validator=lambda decision: _validate_commands(
            decision, reviewed_ids, reviewed_tip_ids, requires_tip_reduction
        ),
    )
    if result.decision is not None:
        return result.decision, False

    repair = policy.repair(
        prompt,
        response.raw_output,
        result.error or "invalid output",
    )
    repaired_result = parse_decision(
        repair.raw_output,
        MaintenanceDecision,
        validator=lambda decision: _validate_commands(
            decision, reviewed_ids, reviewed_tip_ids, requires_tip_reduction
        ),
    )
    if repaired_result.decision is None:
        raise ValueError(
            "invalid maintenance decision after repair: "
            f"{repaired_result.error}"
        )
    return repaired_result.decision, True


def _bind_command_round(
    decision: MaintenanceDecision, maintenance_round: int
) -> MaintenanceDecision:
    """Keep the runtime-owned maintenance round out of the model's control."""
    commands = tuple(
        command.model_copy(update={"updated_round": maintenance_round})
        if isinstance(command, (MergeCommand, DeleteCommand))
        else command
        for command in decision.commands
    )
    return decision.model_copy(update={"commands": commands})


def _prepare_merge_commands(
    decision: MaintenanceDecision,
    repository: MemoryRepository,
) -> MaintenanceDecision:
    dispositions = {
        memory_id: review.disposition
        for review in decision.reviews
        for memory_id in review.memory_ids
    }
    filtered_merges: dict[int, MergeCommand] = {}
    for index, command in enumerate(decision.commands):
        if not isinstance(command, MergeCommand):
            continue
        source_ids = tuple(
            memory_id
            for memory_id in command.source_ids
            if dispositions.get(memory_id) in {None, "merge"}
        )
        if len(source_ids) < 2:
            continue
        filtered_merges[index] = command.model_copy(update={"source_ids": source_ids})
    merge_source_ids = {
        memory_id
        for command in filtered_merges.values()
        for memory_id in command.source_ids
    }
    seen_delete_ids: set[str] = set()
    normalized: list[MemoryCommand] = []
    for index, command in enumerate(decision.commands):
        if isinstance(command, DeleteCommand):
            remaining = tuple(
                memory_id
                for memory_id in command.memory_ids
                if memory_id not in merge_source_ids
                and memory_id not in seen_delete_ids
                and dispositions.get(memory_id) in {None, "retire"}
            )
            if not remaining:
                continue
            command = command.model_copy(update={"memory_ids": remaining})
            seen_delete_ids.update(remaining)
        if isinstance(command, MergeCommand):
            command = filtered_merges.get(index)
            if command is None:
                continue
            command = _materialize_v2_merge(command, repository)
        normalized.append(command)
    return decision.model_copy(update={"commands": tuple(normalized)})


def _materialize_v2_merge(
    command: MergeCommand,
    repository: MemoryRepository,
) -> MergeCommand:
    sources = []
    for memory_id in command.source_ids:
        item = repository.get(memory_id)
        if item is None:
            raise KeyError(f"unknown memory: {memory_id}")
        sources.append(item)
    tiers = {source.tier for source in sources}
    if len(tiers) != 1:
        raise ValueError("merge sources must belong to the same tier")
    schema_versions = {source.tier_schema_version for source in sources}
    if schema_versions == {1}:
        return command.model_copy(update={"payload": None})
    if schema_versions != {2}:
        raise ValueError("merge sources must use the same tier schema version")

    representative = max(
        sources,
        key=lambda item: (
            item.success_count,
            item.usage_count,
            item.updated_round,
            item.version,
            item.id,
        ),
    )
    assert representative.payload is not None
    payload = validate_stored_tier_payload(
        representative.tier,
        representative.payload,
    )
    if isinstance(payload, TipPayload):
        payload = payload.model_copy(update={"guidance": command.content})
    elif isinstance(payload, SkillPayload):
        payload = payload.model_copy(update={"goal": command.content})
    elif isinstance(payload, ToolPayload):
        tool_names = {
            validate_stored_tier_payload(source.tier, source.payload).tool_name
            for source in sources
            if source.payload is not None
        }
        if len(tool_names) != 1:
            raise ValueError("tool merge sources must describe the same tool")
        payload = payload.model_copy(update={"purpose": command.content})
    elif isinstance(payload, TrajectoryPayload):
        raise ValueError(
            "trajectory memories are immutable runtime records and cannot be merged"
        )
    else:
        raise TypeError(f"unsupported V2 merge payload: {type(payload).__name__}")

    rendered = render_tier_payload(representative.tier, payload)
    return command.model_copy(
        update={
            "content": rendered,
            "payload": payload.model_dump(mode="json"),
            "metadata": {
                **command.metadata,
                "representative_memory_id": representative.id,
            },
        }
    )


def _validate_commands(
    decision: MaintenanceDecision,
    reviewed_ids: set[str],
    reviewed_tip_ids: set[str],
    requires_tip_reduction: bool,
) -> None:
    has_lookup = any(isinstance(command, LookupCommand) for command in decision.commands)
    has_write = any(not isinstance(command, LookupCommand) for command in decision.commands)
    if has_lookup and has_write:
        raise ValueError("lookup commands cannot be mixed with write commands")
    for command in decision.commands:
        if isinstance(command, MergeCommand):
            _reject_forbidden_metadata(command.metadata)
    reviewed = [memory_id for review in decision.reviews for memory_id in review.memory_ids]
    if any(memory_id not in reviewed_ids for memory_id in reviewed):
        raise ValueError("maintenance reviews may only reference provided diagnostics")
    if len(set(reviewed)) != len(reviewed):
        raise ValueError("maintenance reviews must not repeat memory IDs")
    reducing_tip_command = any(
        isinstance(command, MergeCommand) and set(command.source_ids).issubset(reviewed_tip_ids)
        or isinstance(command, DeleteCommand) and set(command.memory_ids).issubset(reviewed_tip_ids)
        for command in decision.commands
    )
    if requires_tip_reduction and not reducing_tip_command:
        raise ValueError("tip capacity requires at least one merge or delete command")
    if requires_tip_reduction and not decision.reviews:
        raise ValueError("tip capacity requires explicit keep, merge, or retire reviews")


def _reject_forbidden_metadata(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized_key = unicodedata.normalize("NFKC", str(key))
            compact_key = "".join(
                character
                for character in normalized_key.casefold()
                if character.isalnum()
            )
            if "attributionscore" in compact_key:
                raise ValueError(
                    "maintenance merge metadata must not contain attribution score"
                )
            if is_credential_key(normalized_key):
                raise ValueError(
                    "maintenance merge metadata must not contain credential fields"
                )
            _reject_forbidden_metadata(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_forbidden_metadata(nested)


def _load_state(path: Path, current_round: int) -> MaintenanceState:
    if not path.exists():
        return MaintenanceState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state must be an object")
        if type(payload["schema_version"]) is not int:
            raise ValueError("schema_version must be an integer")
        rounds = payload["completed_rounds"]
        if not isinstance(rounds, list):
            raise ValueError("completed_rounds must be a list")
        if payload["schema_version"] == 1 and set(payload) == {
            "schema_version",
            "completed_rounds",
        }:
            cursors: tuple[tuple[str, str | None], ...] = ()
        elif payload["schema_version"] == _STATE_SCHEMA_VERSION and set(payload) == {
            "schema_version",
            "completed_rounds",
            "review_cursor_by_tier",
        } and isinstance(payload["review_cursor_by_tier"], dict):
            cursors = tuple(sorted(payload["review_cursor_by_tier"].items()))
        else:
            raise ValueError("state fields do not match a supported schema")
        state = MaintenanceState(
            schema_version=_STATE_SCHEMA_VERSION,
            completed_rounds=tuple(rounds),
            review_cursor_by_tier=cursors,
        )
        if any(round_number > current_round for round_number in state.completed_rounds):
            raise ValueError("maintenance state contains future completed rounds")
        return state
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid maintenance state: {path}") from error


def _encode_state(state: MaintenanceState) -> bytes:
    payload = {
        "schema_version": state.schema_version,
        "completed_rounds": list(state.completed_rounds),
        "review_cursor_by_tier": dict(state.review_cursor_by_tier),
    }
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _emit(
    context: RunContext,
    task_key: str,
    event_type: str,
    **payload: Any,
) -> None:
    context.event_writer.append(
        sanitize_artifact_data(context.event(event_type, task_key, **payload))
    )


def _emit_failure(
    context: RunContext,
    task_key: str,
    maintenance_round: int,
    error: BaseException,
) -> None:
    try:
        _emit(
            context,
            task_key,
            "MaintenanceFailed",
            maintenance_round=maintenance_round,
            error={"type": type(error).__name__, "message": "operation failed"},
        )
    except BaseException as evidence_error:
        error.add_note(f"Maintenance failure evidence also failed: {evidence_error}")
