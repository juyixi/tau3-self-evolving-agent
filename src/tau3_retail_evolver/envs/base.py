from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResetResult:
    observation: str
    info: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: str
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
