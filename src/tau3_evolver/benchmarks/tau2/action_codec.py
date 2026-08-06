from __future__ import annotations

import ast
from collections.abc import Mapping
import json
import re
from typing import Any


_THINKING_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_FUNCTION_PREFIX = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")
TAU2_STOP_ACTION = "###STOP###"


class Tau2ActionCodec:
    """Validate model output while preserving Tau2's accepted action formats."""

    @staticmethod
    def decode(model_output: str, tool_names: set[str]) -> str:
        if not isinstance(model_output, str):
            raise ValueError("model output must be text")

        action = _THINKING_BLOCK.sub("", model_output).strip()
        if not action:
            raise ValueError("model output has no final action")
        if "<think>" in action.casefold():
            raise ValueError("unterminated thinking block")
        if action == "stop":
            return TAU2_STOP_ACTION
        if action == TAU2_STOP_ACTION:
            return action

        if action.startswith("{"):
            _validate_json_tool_call(action, tool_names)
            return action

        function_match = _FUNCTION_PREFIX.match(action)
        if function_match is not None:
            _validate_function_tool_call(action, function_match.group(1), tool_names)
            return action

        return action


def _validate_json_tool_call(action: str, tool_names: set[str]) -> None:
    try:
        tool_call = json.loads(action)
    except json.JSONDecodeError as error:
        raise ValueError("malformed JSON tool call") from error
    if not isinstance(tool_call, Mapping):
        raise ValueError("JSON tool call must be an object")

    tool_name = tool_call.get("name")
    if not isinstance(tool_name, str):
        raise ValueError("JSON tool call must contain a string name")
    _validate_tool_name(tool_name, tool_names)

    arguments = tool_call.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")


def _validate_function_tool_call(action: str, tool_name: str, tool_names: set[str]) -> None:
    _validate_tool_name(tool_name, tool_names)
    try:
        expression = ast.parse(action, mode="eval").body
    except SyntaxError as error:
        raise ValueError("tool arguments are malformed") from error
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name):
        raise ValueError("tool call must use a simple function name")
    if expression.args or any(keyword.arg is None for keyword in expression.keywords):
        raise ValueError("tool arguments must be named literals")
    try:
        for keyword in expression.keywords:
            ast.literal_eval(keyword.value)
    except ValueError as error:
        raise ValueError("tool arguments must be literals") from error


def _validate_tool_name(tool_name: str, tool_names: set[str]) -> None:
    if tool_name not in tool_names:
        raise ValueError(f"tool {tool_name!r} is not available")
