from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
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
from tau3_retail_evolver.memory.types import MemoryTier
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


_STATE_FILENAME = "maintenance_state.json"
_STATE_SCHEMA_VERSION = 1
_LOCK_NAMESPACE = "fast-loop-maintenance"


@dataclass(frozen=True, slots=True)
class MaintenanceState:
    schema_version: int = _STATE_SCHEMA_VERSION
    completed_rounds: tuple[int, ...] = ()

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

    diagnostics: dict[str, Any] = {}
    with repository.read_transaction():
        for tier in MemoryTier:
            items = repository.list(tier=tier)[:per_tier_limit]
            diagnostics[tier.value] = {
                "items": [
                    {
                        "id": item.id,
                        "content": item.content[:MAX_DIAGNOSTIC_CONTENT_CHARS],
                        "version": item.version,
                        "usage_count": item.usage_count,
                        "success_count": item.success_count,
                        "last_used": item.last_used,
                    }
                    for item in items
                ]
            }
    return diagnostics


def run_due_maintenance(
    *,
    completed_train_tasks: int,
    period: int,
    repository: MemoryRepository,
    policy: FastLoopPolicy,
    context: RunContext,
    per_tier_limit: int = 100,
) -> MaintenanceResult:
    if type(completed_train_tasks) is not int or completed_train_tasks < 0:
        raise ValueError("completed_train_tasks must be a non-negative integer")
    if type(period) is not int or period <= 0:
        raise ValueError("period must be a positive integer")
    if context.split != "train" or context.mode != RunMode.LEARN:
        raise ValueError("maintenance requires train split and learn mode")
    if not isinstance(repository, MemoryRepository) or repository.is_read_only:
        raise TypeError("maintenance requires a mutable MemoryRepository")

    maintenance_round = completed_train_tasks // period
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
            completed_train_tasks=completed_train_tasks,
            period=period,
            maintenance_round=maintenance_round,
            repository=repository,
            policy=policy,
            context=context,
            per_tier_limit=per_tier_limit,
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
    state: MaintenanceState,
    state_path: Path,
) -> MaintenanceResult:
    task_key = f"maintenance-round-{maintenance_round}"
    try:
        diagnostics = sanitize_artifact_data(
            bounded_diagnostics(
                repository,
                per_tier_limit=per_tier_limit,
            )
        )
        prompt = build_maintenance_prompt(diagnostics=diagnostics)
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
        )
        decision, repair_used = _generate_decision(
            policy,
            prompt,
            maintenance_round,
        )
        canonical_commands = [
            command.model_dump(mode="json") for command in decision.commands
        ]
        _emit(
            context,
            task_key,
            "MaintenanceProposed",
            maintenance_round=maintenance_round,
            commands=canonical_commands,
            repair_used=repair_used,
        )
        operation_result = MemoryOperations(repository).apply_batch(
            list(decision.commands)
        )
        completed_state = MaintenanceState(
            completed_rounds=tuple(
                sorted((*state.completed_rounds, maintenance_round))
            )
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
) -> tuple[MaintenanceDecision, bool]:
    response = policy.generate(prompt)
    result = parse_decision(
        response.raw_output,
        MaintenanceDecision,
        validator=lambda decision: _validate_commands(decision, maintenance_round),
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
        validator=lambda decision: _validate_commands(decision, maintenance_round),
    )
    if repaired_result.decision is None:
        raise ValueError(
            "invalid maintenance decision after repair: "
            f"{repaired_result.error}"
        )
    return repaired_result.decision, True


def _validate_commands(
    decision: MaintenanceDecision,
    maintenance_round: int,
) -> None:
    has_lookup = any(isinstance(command, LookupCommand) for command in decision.commands)
    has_write = any(not isinstance(command, LookupCommand) for command in decision.commands)
    if has_lookup and has_write:
        raise ValueError("lookup commands cannot be mixed with write commands")
    for command in decision.commands:
        if isinstance(command, (MergeCommand, DeleteCommand)) and (
            command.updated_round != maintenance_round
        ):
            raise ValueError(
                "maintenance command updated_round must equal maintenance_round"
            )
        if isinstance(command, MergeCommand):
            _reject_forbidden_metadata(command.metadata)


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
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "completed_rounds",
        }:
            raise ValueError("state must contain exactly the canonical fields")
        if type(payload["schema_version"]) is not int:
            raise ValueError("schema_version must be an integer")
        rounds = payload["completed_rounds"]
        if not isinstance(rounds, list):
            raise ValueError("completed_rounds must be a list")
        state = MaintenanceState(
            schema_version=payload["schema_version"],
            completed_rounds=tuple(rounds),
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
