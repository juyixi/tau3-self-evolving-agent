from __future__ import annotations

from enum import StrEnum
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryTier(StrEnum):
    TRAJECTORY = "trajectory"
    TIP = "tip"
    SKILL = "skill"
    TOOL = "tool"


MEMORY_TIERS = tuple(tier.value for tier in MemoryTier)


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    tier: MemoryTier
    content: str
    retrieval_text: str
    embedding: tuple[float, ...] | None = None
    embedding_model_revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_task_ids: tuple[str, ...]
    created_round: int = Field(ge=0)
    updated_round: int = Field(ge=0)
    version: int = Field(default=1, ge=1)
    status: MemoryStatus = MemoryStatus.ACTIVE
    usage_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    last_used: str | None = None

    @field_validator("content", "retrieval_text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("memory text must not be blank")
        return value

    @field_validator("source_task_ids")
    @classmethod
    def sources_must_be_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(sorted({source.strip() for source in value if source.strip()}))
        if not canonical:
            raise ValueError("memory requires at least one source task")
        return canonical

    @field_validator("embedding")
    @classmethod
    def embedding_must_be_finite(
        cls, value: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        if value is not None and (not value or not all(math.isfinite(component) for component in value)):
            raise ValueError("embedding must contain finite values")
        return value

    @field_validator("metadata")
    @classmethod
    def metadata_must_be_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("memory metadata must be JSON serializable") from error
        return value


class MemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_snapshot_id: str
    path: Path
    counts: dict[str, int]


def canonical_content(content: str) -> str:
    return " ".join(content.split())


def stable_memory_id(tier: MemoryTier, content: str) -> str:
    digest = hashlib.sha256(f"{tier.value}\0{canonical_content(content)}".encode("utf-8")).hexdigest()
    return f"mem_{tier.value}_{digest[:20]}"
