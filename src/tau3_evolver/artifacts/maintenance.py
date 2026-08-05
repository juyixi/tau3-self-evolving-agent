from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from tau3_evolver.artifacts.sanitize import sanitize_artifact_data


MAINTENANCE_RECORD_SCHEMA_VERSION = 1


def build_completed_maintenance(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collapse one internal maintenance lifecycle into one run-level record."""
    if tuple(event.get("event_type") for event in events) != (
        "MaintenanceStarted",
        "MaintenanceProposed",
        "MaintenanceCommitted",
    ):
        raise ValueError("incomplete maintenance lifecycle")
    task_ids = {event.get("task_id") for event in events}
    snapshots = {event.get("memory_snapshot_id") for event in events}
    rounds = {event.get("maintenance_round") for event in events}
    if len(task_ids) != 1 or len(snapshots) != 1 or len(rounds) != 1:
        raise ValueError("maintenance lifecycle crosses provenance")

    started, proposed, committed = events
    maintenance_round = started.get("maintenance_round")
    trigger_task_index = started.get("completed_train_tasks")
    period = started.get("period")
    snapshot_id = started.get("memory_snapshot_id")
    if type(maintenance_round) is not int or maintenance_round <= 0:
        raise ValueError("maintenance round must be a positive integer")
    if type(trigger_task_index) is not int or trigger_task_index <= 0:
        raise ValueError("maintenance trigger task index must be positive")
    if type(period) is not int or period <= 0:
        raise ValueError("maintenance period must be positive")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("maintenance snapshot ID must be non-blank")

    record = sanitize_artifact_data(
        {
            "schema_version": MAINTENANCE_RECORD_SCHEMA_VERSION,
            "maintenance_round": maintenance_round,
            "trigger_task_index": trigger_task_index,
            "period": period,
            "memory_snapshot_id": snapshot_id,
            "diagnostics": started.get("diagnostics"),
            "commands": proposed.get("commands", []),
            "looked_up_ids": committed.get("looked_up_ids", []),
            "created_ids": committed.get("created_ids", []),
            "updated_ids": committed.get("updated_ids", []),
        }
    )
    return {**record, "record_sha256": maintenance_record_sha256(record)}


def maintenance_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


__all__ = [
    "MAINTENANCE_RECORD_SCHEMA_VERSION",
    "build_completed_maintenance",
    "maintenance_record_sha256",
]
