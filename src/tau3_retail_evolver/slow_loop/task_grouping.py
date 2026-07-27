from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


GROUPING_REVISION = "retail-v2"
RETAIL_TASK_GROUP = GROUPING_REVISION
MAINTENANCE_TASK_GROUP = f"{GROUPING_REVISION}:maintenance"
_LEGACY_ACTION_SIGNATURE = re.compile(r"^retail-actions-v1:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RetailTaskGroups:
    """One attribution group for all Retail tasks in the same domain."""

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

        available = _load_task_ids(path)
        missing = set(requested) - available
        if missing:
            rendered = ", ".join(sorted(missing))
            raise ValueError(f"tasks JSON file {path} is missing requested task ID(s): {rendered}")
        return cls(
            task_ids=requested,
            _signatures={task_id: RETAIL_TASK_GROUP for task_id in requested},
        )

    def signature_for(self, task_id: str) -> str:
        try:
            return self._signatures[task_id]
        except KeyError as error:
            raise ValueError(f"task group is unavailable for task ID {task_id!r}") from error


def canonicalize_retail_task_group(value: object) -> str:
    """Map legacy action signatures into the current domain-level group."""

    if value == RETAIL_TASK_GROUP or (
        isinstance(value, str) and _LEGACY_ACTION_SIGNATURE.fullmatch(value)
    ):
        return RETAIL_TASK_GROUP
    raise ValueError(f"unsupported Retail task group: {value!r}")


def is_supported_retail_task_group(value: object) -> bool:
    try:
        canonicalize_retail_task_group(value)
    except ValueError:
        return False
    return True


def _load_task_ids(path: Path) -> set[str]:
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
    return seen
