from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tau3_evolver.memory.retrieval import MemoryCandidate


class Tau3AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[Any] = Field(default_factory=list)
    selected: tuple[MemoryCandidate, ...] = ()
    started: bool = False
    turn: int = 0
