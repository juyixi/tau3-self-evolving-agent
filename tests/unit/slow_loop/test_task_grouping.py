from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_retail_evolver.slow_loop.task_grouping import RetailTaskGroups


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


def test_group_signature_uses_only_unique_sorted_mutating_action_names(
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

    expected = (
        "retail-actions-v1:"
        "98d1978041aa4796c5464cfbc0b07ace74a2967b88a0934b6a039fcf1a688395"
    )
    assert groups.signature_for("7") == expected
    assert groups.signature_for("8") == expected
    assert groups.task_ids == ("7", "8")
    assert "private" not in repr(groups)


@pytest.mark.parametrize(
    "actions",
    [
        [{"name": "new_unclassified_action"}],
        [{"arguments": {}}],
        "not-a-list",
    ],
)
def test_malformed_or_unknown_actions_fail_closed(tmp_path: Path, actions: object) -> None:
    path = _write_tasks(tmp_path, [_task("7", actions)])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="action|actions"):
        RetailTaskGroups.from_file(path, task_ids=("7",))


def test_requested_task_must_exist(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_task("7", [])])

    with pytest.raises(ValueError, match="missing requested task ID"):
        RetailTaskGroups.from_file(path, task_ids=("8",))


def test_duplicate_catalog_task_id_fails_closed(tmp_path: Path) -> None:
    path = _write_tasks(tmp_path, [_task("7", []), _task("7", [])])

    with pytest.raises(ValueError, match="duplicate task ID"):
        RetailTaskGroups.from_file(path, task_ids=("7",))
