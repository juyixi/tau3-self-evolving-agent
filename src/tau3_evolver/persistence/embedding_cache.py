from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path

from tau3_evolver.persistence.atomic import write_bytes_atomic
from tau3_evolver.persistence.locking import reentrant_process_lock


class JsonEmbeddingCache:
    """Durable cache for derived embedding vectors."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = reentrant_process_lock(path, namespace="embedding-cache")
        with self._lock:
            self._entries = self._load()

    def get(self, model_revision: str, text: str) -> tuple[float, ...] | None:
        value = self._entries.get(_cache_key(model_revision, text))
        return tuple(value) if value is not None else None

    def put_many(
        self,
        model_revision: str,
        values: Sequence[tuple[str, Sequence[float]]],
    ) -> None:
        with self._lock:
            replacement = self._load()
            for text, embedding in values:
                vector = _validate_vector(embedding)
                replacement[_cache_key(model_revision, text)] = list(vector)
            payload = {
                "schema_version": 1,
                "count": len(replacement),
                "entries": dict(sorted(replacement.items())),
            }
            serialized = (
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            write_bytes_atomic(self.path, serialized)
            self._entries = replacement

    def _load(self) -> dict[str, list[float]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid embedding cache: {self.path}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError(f"unsupported embedding cache schema: {self.path}")
        entries = payload.get("entries")
        if not isinstance(entries, dict) or payload.get("count") != len(entries):
            raise ValueError(f"embedding cache count mismatch: {self.path}")
        return {
            str(key): list(_validate_vector(value)) for key, value in entries.items()
        }


def _cache_key(model_revision: str, text: str) -> str:
    return hashlib.sha256(f"{model_revision}\0{text}".encode("utf-8")).hexdigest()


def _validate_vector(value: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(component) for component in value)
    if not vector or not all(math.isfinite(component) for component in vector):
        raise ValueError("embedding must contain finite values")
    if math.sqrt(sum(component * component for component in vector)) == 0.0:
        raise ValueError("embedding must not be zero-norm")
    return vector


__all__ = ["JsonEmbeddingCache"]
