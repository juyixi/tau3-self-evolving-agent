from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """Everything a rollout policy needs to make one Tau2 decision."""

    observation: str
    reset_info: Mapping[str, Any]
    temperature: float
    top_p: float
    history: Sequence[Mapping[str, str]] = ()


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    """Auditable output from one policy decision."""

    raw_output: str
    parsed_action: str
    sampling_params: Mapping[str, float]
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Policy(ABC):
    """Generate one action from a Tau2 observation and reset contract."""

    @abstractmethod
    def generate(self, request: DecisionRequest) -> DecisionResponse:
        """Return an auditable decision for a single environment step."""
