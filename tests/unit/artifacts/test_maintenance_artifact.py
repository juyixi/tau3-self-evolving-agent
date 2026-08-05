from tau3_evolver.artifacts.maintenance import (
    build_completed_maintenance,
    maintenance_record_sha256,
)


def test_collapses_maintenance_lifecycle_without_run_context_duplication() -> None:
    common = {
        "task_id": "maintenance-round-1",
        "memory_snapshot_id": "snapshot-1",
        "maintenance_round": 1,
        "run_id": "run-1",
    }
    record = build_completed_maintenance(
        (
            {
                **common,
                "event_type": "MaintenanceStarted",
                "completed_train_tasks": 30,
                "period": 30,
                "diagnostics": {
                    tier: {"items": []}
                    for tier in ("trajectory", "tip", "skill", "tool")
                },
            },
            {
                **common,
                "event_type": "MaintenanceProposed",
                "commands": [],
            },
            {
                **common,
                "event_type": "MaintenanceCommitted",
                "looked_up_ids": [],
                "created_ids": [],
                "updated_ids": [],
            },
        )
    )

    assert record["maintenance_round"] == 1
    assert record["trigger_task_index"] == 30
    assert "run_id" not in record
    assert record["record_sha256"] == maintenance_record_sha256(record)
