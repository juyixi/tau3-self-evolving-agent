from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_retail_evolver.slow_loop.task_grouping import (
    RETAIL_TASK_GROUP,
    RetailTaskGroups,
    canonicalize_retail_task_group,
)


def _write_tasks(tmp_path: Path, tasks: list[dict[str, object]]) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def _task(task_id: str, actions: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": task_id,
        "evaluation_criteria": {
            "actions": actions,
            "nl_assertions": [{"contains": "private rubric"}],
        },
    }


def test_group_signature_is_shared_by_all_retail_tasks(
    tmp_path: Path,
) -> None:
    path = _write_tasks(
        tmp_path,
        [
            _task(
                "7",
                [
                    {"name": "get_user_details", "arguments": {"user_id": "secret-a"}},
                    {
                        "name": "return_delivered_order_items",
                        "arguments": {"item_ids": ["private-a"]},
                    },
                    {
                        "name": "return_delivered_order_items",
                        "arguments": {"item_ids": ["private-b"]},
                    },
                ],
            ),
            _task(
                "8",
                [
                    {
                        "name": "return_delivered_order_items",
                        "arguments": {"item_ids": ["different"]},
                    },
                    {"name": "get_order_details", "arguments": {"order_id": "other"}},
                ],
            ),
        ],
    )

    groups = RetailTaskGroups.from_file(path, task_ids=("7", "8"))

    assert groups.signature_for("7") == RETAIL_TASK_GROUP
    assert groups.signature_for("8") == RETAIL_TASK_GROUP
    assert groups.task_ids == ("7", "8")
    assert "private" not in repr(groups)


def test_legacy_action_signature_is_normalized_to_domain_group() -> None:
    assert canonicalize_retail_task_group(
        "retail-actions-v1:" + "a" * 64
    ) == RETAIL_TASK_GROUP
    with pytest.raises(ValueError, match="unsupported Retail task group"):
        canonicalize_retail_task_group("airline-v2")


def test_requested_task_must_exist(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_task("7", [])])

    with pytest.raises(ValueError, match="missing requested task ID"):
        RetailTaskGroups.from_file(path, task_ids=("8",))


def test_duplicate_catalog_task_id_fails_closed(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_task("7", []), _task("7", [])])

    with pytest.raises(ValueError, match="duplicate task ID"):
        RetailTaskGroups.from_file(path, task_ids=("7",))
