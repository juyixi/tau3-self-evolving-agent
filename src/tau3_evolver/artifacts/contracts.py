from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompletedEpisodeProjection:
    """Stable task outcome fields required by the episode artifact projector."""

    task_id: str
    final_reward: float
    steps: int
    terminal_evaluation: Mapping[str, Any]
    truncated: bool
    project_truncated: bool = False
    parse_error_count: int = 0
    response_parse_error_count: int = 0
    response_count: int = 0
    agent_prompt_tokens: int | None = None
    agent_completion_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class FailedEpisodeProjection:
    """Bounded failure fields that are safe to publish as an artifact."""

    task_id: str
    stage: str
    error_type: str


__all__ = ["CompletedEpisodeProjection", "FailedEpisodeProjection"]
