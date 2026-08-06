from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any

from tau3_evolver.fast_loop.tools import normalize_tool_schema


@dataclass(frozen=True, slots=True)
class BaselinePrompt:
    """OpenAI-compatible messages and tools derived from a Tau2 reset."""

    messages: tuple[dict[str, str], ...]
    tools: tuple[dict[str, Any], ...]


def build_baseline_prompt(
    observation: str,
    reset_info: Mapping[str, Any],
    history: Sequence[Mapping[str, str]] = (),
) -> BaselinePrompt:
    """Build a public decision prompt from Tau2's official reset payload."""
    policy = reset_info.get("policy")
    if policy is None:
        raise ValueError("Tau2 reset info is missing policy")

    tools = reset_info.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise ValueError("Tau2 reset info must contain a sequence of tools")
    normalized_tools = tuple(normalize_tool_schema(tool) for tool in tools)

    messages = [{"role": "system", "content": _policy_content(policy)}]
    messages.extend(_public_history(history))
    messages.append({"role": "user", "content": observation})
    return BaselinePrompt(
        messages=tuple(messages),
        tools=normalized_tools,
    )


def _policy_content(policy: Any) -> str:
    if isinstance(policy, str):
        return policy
    try:
        return json.dumps(policy, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Tau2 reset policy must be JSON serializable") from error


def _public_history(history: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if isinstance(history, (str, bytes)):
        raise ValueError("prior message history must be a sequence of messages")

    messages: list[dict[str, str]] = []
    for message in history:
        if not isinstance(message, Mapping):
            raise ValueError("prior message history must contain mappings")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("prior messages must contain string role and content")
        messages.append({"role": role, "content": content})
    return messages
