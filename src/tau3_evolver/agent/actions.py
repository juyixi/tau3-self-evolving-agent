from __future__ import annotations

import ast
from collections.abc import Mapping
import json
from typing import Any

from tau3_evolver.artifacts.sanitize import sanitize_artifact_data


def assistant_for_action(
    *,
    assistant_type: type,
    tool_call_type: type,
    action: str,
    turn: int,
    audit: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> Any:
    usage = None
    if audit["prompt_tokens"] is not None and audit["completion_tokens"] is not None:
        usage = {
            "prompt_tokens": audit["prompt_tokens"],
            "completion_tokens": audit["completion_tokens"],
        }
    tool = parse_tool_action(action)
    common = {
        "role": "assistant",
        "usage": usage,
        "generation_time_seconds": audit["latency_s"],
        "raw_data": {"tau3_agent": sanitize_artifact_data(dict(marker))},
    }
    if tool is None:
        return assistant_type(**common, content=action)
    name, arguments = tool
    return assistant_type(
        **common,
        tool_calls=[
            tool_call_type(
                id=f"tau3-tool-{turn}",
                name=name,
                arguments=arguments,
                requestor="assistant",
            )
        ],
    )


def parse_tool_action(action: str) -> tuple[str, dict[str, Any]] | None:
    try:
        value = json.loads(action)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, Mapping):
        name = value.get("name")
        arguments = value.get("arguments")
        if isinstance(name, str) and isinstance(arguments, Mapping):
            return name, dict(arguments)

    try:
        expression = ast.parse(action, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        return None
    if expression.args or any(keyword.arg is None for keyword in expression.keywords):
        return None
    try:
        arguments = {
            str(keyword.arg): ast.literal_eval(keyword.value)
            for keyword in expression.keywords
        }
    except (ValueError, TypeError):
        return None
    return expression.func.id, arguments
