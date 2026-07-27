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
_DOMAIN_NAME = re.compile(r"^[a-z0-9_-]+$")


def domain_task_group(domain: str) -> str:
    if not isinstance(domain, str) or not _DOMAIN_NAME.fullmatch(domain):
        raise ValueError(f"invalid Tau2 domain for task grouping: {domain!r}")
    return f"{domain}-v2"


def maintenance_task_group(domain: str) -> str:
    return f"{domain_task_group(domain)}:maintenance"


@dataclass(frozen=True, slots=True)
class RetailTaskGroups:
    """One attribution group for every task in a Tau2 domain.

    The historical class name remains compatible with Retail callers.
    """

    task_ids: tuple[str, ...]
    _signatures: Mapping[str, str] = field(repr=False)
    domain: str = "retail"

    def __post_init__(self) -> None:
        object.__setattr__(self, "_signatures", MappingProxyType(dict(self._signatures)))

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        task_ids: Sequence[str],
        domain: str = "retail",
    ) -> "RetailTaskGroups":
        group = domain_task_group(domain)
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
            _signatures={task_id: group for task_id in requested},
            domain=domain,
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


def canonicalize_domain_task_group(value: object, *, domain: str) -> str:
    expected = domain_task_group(domain)
    if value == expected:
        return expected
    if domain == "retail" and isinstance(value, str) and _LEGACY_ACTION_SIGNATURE.fullmatch(
        value
    ):
        return expected
    raise ValueError(f"unsupported {domain} task group: {value!r}")


def is_supported_domain_task_group(value: object, *, domain: str) -> bool:
    try:
        canonicalize_domain_task_group(value, domain=domain)
    except ValueError:
        return False
    return True


def canonicalize_tau2_task_group(value: object) -> str:
    for domain in ("retail", "airline"):
        try:
            return canonicalize_domain_task_group(value, domain=domain)
        except ValueError:
            continue
    raise ValueError(f"unsupported Tau2 task group: {value!r}")


Tau2TaskGroups = RetailTaskGroups


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
