from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class BaselinePrompt:
    """OpenAI-compatible messages and tools derived from a Tau2 reset."""

    messages: tuple[dict[str, str], ...]
    tools: tuple[dict[str, Any], ...]


def build_baseline_prompt(observation: str, reset_info: Mapping[str, Any]) -> BaselinePrompt:
    """Build a public decision prompt from Tau2's official reset payload."""
    policy = reset_info.get("policy")
    if policy is None:
        raise ValueError("Tau2 reset info is missing policy")

    tools = reset_info.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        raise ValueError("Tau2 reset info must contain a sequence of tools")
    if not all(isinstance(tool, Mapping) for tool in tools):
        raise ValueError("Tau2 tools must be mappings")

    return BaselinePrompt(
        messages=(
            {"role": "system", "content": _policy_content(policy)},
            {"role": "user", "content": observation},
        ),
        tools=tuple(deepcopy(dict(tool)) for tool in tools),
    )


def _policy_content(policy: Any) -> str:
    if isinstance(policy, str):
        return policy
    try:
        return json.dumps(policy, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Tau2 reset policy must be JSON serializable") from error
