from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


def normalize_tool_schema(tool: Any) -> dict[str, Any]:
    schema = tool if isinstance(tool, Mapping) else getattr(tool, "openai_schema", None)
    if not isinstance(schema, Mapping):
        raise ValueError("tools must be mappings or expose an openai_schema mapping")
    return deepcopy(dict(schema))


__all__ = ["normalize_tool_schema"]
