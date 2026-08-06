from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from tau3_evolver.memory.types import MemoryItem, MemoryTier, stable_memory_id
from tau3_evolver.persistence.atomic import write_bytes_atomic


MEMORY_SCHEMA_VERSION = 1


class JsonTierStore:
    def __init__(self, path: Path, tier: MemoryTier) -> None:
        self.path = path
        self.tier = tier

    def load(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid memory store: {self.path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"memory store must be an object: {self.path}")
        if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema version: {self.path}")
        if payload.get("tier") != self.tier.value:
            raise ValueError(f"memory tier mismatch: {self.path}")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list) or payload.get("count") != len(raw_items):
            raise ValueError(f"memory count mismatch: {self.path}")
        items = [MemoryItem.model_validate(item) for item in raw_items]
        self._validate_items(items)
        return sorted(items, key=lambda item: item.id)

    def write(self, items: Iterable[MemoryItem]) -> None:
        serialized = serialize_tier(self.tier, items)
        write_bytes_atomic(self.path, serialized)

    def _validate_items(self, items: Iterable[MemoryItem]) -> None:
        seen: set[str] = set()
        for item in items:
            if item.tier != self.tier:
                raise ValueError(f"memory tier mismatch for {item.id}")
            if item.id != stable_memory_id(item.tier, item.content):
                raise ValueError(f"stable memory id mismatch for {item.id}")
            if item.id in seen:
                raise ValueError(f"duplicate memory id: {item.id}")
            seen.add(item.id)


def serialize_tier(tier: MemoryTier, items: Iterable[MemoryItem]) -> bytes:
    ordered = sorted(items, key=lambda item: item.id)
    JsonTierStore(Path("unused"), tier)._validate_items(ordered)
    payload = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "tier": tier.value,
        "count": len(ordered),
        "items": [item.model_dump(mode="json") for item in ordered],
    }
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError("memory store must be JSON serializable") from error
    return f"{text}\n".encode("utf-8")
