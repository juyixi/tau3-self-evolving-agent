from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any


GROUPING_REVISION = "retail-actions-v1"

READ_ONLY_ACTIONS = frozenset(
    {
        "calculate",
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_item_details",
        "get_order_details",
        "get_product_details",
        "get_user_details",
    }
)

MUTATING_ACTIONS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "transfer_to_human_agents",
    }
)


@dataclass(frozen=True, slots=True)
class RetailTaskGroups:
    """Anonymous task-group signatures derived from selected golden action names."""

    task_ids: tuple[str, ...]
    _signatures: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_signatures", MappingProxyType(dict(self._signatures)))

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        task_ids: Sequence[str],
    ) -> "RetailTaskGroups":
        requested = tuple(str(task_id) for task_id in task_ids)
        if not requested:
            raise ValueError("task grouping requires at least one requested task ID")
        if any(not task_id.strip() for task_id in requested):
            raise ValueError("task grouping task IDs must not be blank")
        if len(requested) != len(set(requested)):
            raise ValueError("task grouping task IDs must be unique")

        tasks = _load_tasks(path)
        requested_set = set(requested)
        selected: dict[str, Mapping[str, Any]] = {}
        seen: set[str] = set()
        for raw_task in tasks:
            if not isinstance(raw_task, Mapping):
                raise ValueError(f"tasks JSON file {path} contains a non-object task")
            raw_id = raw_task.get("id", raw_task.get("task_id"))
            if not isinstance(raw_id, (str, int)) or isinstance(raw_id, bool):
                raise ValueError(f"tasks JSON file {path} contains a task without an ID")
            task_id = str(raw_id)
            if task_id in seen:
                raise ValueError(f"tasks JSON file {path} contains duplicate task ID {task_id}")
            seen.add(task_id)
            if task_id in requested_set:
                selected[task_id] = raw_task

        missing = requested_set - selected.keys()
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ValueError(f"tasks JSON file {path} is missing requested task ID(s): {rendered}")

        signatures = {
            task_id: _task_signature(selected[task_id], path, task_id)
            for task_id in requested
        }
        return cls(task_ids=requested, _signatures=signatures)

    def signature_for(self, task_id: str) -> str:
        try:
            return self._signatures[task_id]
        except KeyError as error:
            raise ValueError(f"task group is unavailable for task ID {task_id!r}") from error


def _load_tasks(path: Path) -> list[Any]:
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"tasks JSON file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"tasks JSON file is invalid: {path}") from error

    tasks = data.get("tasks") if isinstance(data, Mapping) else data
    if not isinstance(tasks, list):
        raise ValueError(f"tasks JSON file {path} must contain a task list")
    return tasks


def _task_signature(task: Mapping[str, Any], path: Path, task_id: str) -> str:
    criteria = task.get("evaluation_criteria")
    if not isinstance(criteria, Mapping):
        raise ValueError(
            f"task {task_id} in {path} must contain object evaluation_criteria"
        )
    actions = criteria.get("actions")
    if not isinstance(actions, list):
        raise ValueError(f"task {task_id} in {path} must contain an actions list")

    mutating_names: set[str] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise ValueError(f"task {task_id} action {index} in {path} must be an object")
        name = action.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"task {task_id} action {index} in {path} has no action name")
        if name in READ_ONLY_ACTIONS:
            continue
        if name not in MUTATING_ACTIONS:
            raise ValueError(
                f"task {task_id} in {path} contains unclassified retail action {name!r}"
            )
        mutating_names.add(name)

    canonical = json.dumps(
        {
            "domain": "retail",
            "grouping_revision": GROUPING_REVISION,
            "action_names": sorted(mutating_names),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return f"{GROUPING_REVISION}:{digest}"
