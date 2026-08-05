from __future__ import annotations

import re


GROUPING_REVISION = "benchmark-domain-v1"
MAINTENANCE_TASK_GROUP = "maintenance"
_SAFE_GROUP = re.compile(r"^[a-z0-9][a-z0-9_.:-]*$")


def canonicalize_task_group(value: object) -> str:
    """Validate a benchmark-provided semantic grouping label."""
    if not isinstance(value, str):
        raise ValueError(f"task group must be a string: {value!r}")
    normalized = value.strip().casefold()
    if not _SAFE_GROUP.fullmatch(normalized):
        raise ValueError(f"unsupported task group: {value!r}")
    return normalized


def is_supported_task_group(value: object) -> bool:
    try:
        canonicalize_task_group(value)
    except ValueError:
        return False
    return True
