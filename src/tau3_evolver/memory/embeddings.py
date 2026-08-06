from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Retrieval-facing contract implemented by embedding infrastructure."""

    model_revision: str
    dimension: int

    def embed(self, text: str) -> tuple[float, ...]: ...

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]: ...


def validate_embedding(
    value: Sequence[float], *, dimension: int | None = None
) -> tuple[float, ...]:
    vector = tuple(float(component) for component in value)
    if not vector or not all(math.isfinite(component) for component in vector):
        raise ValueError("embedding must contain finite values")
    if dimension is not None and len(vector) != dimension:
        raise ValueError(
            f"embedding dimension mismatch: expected {dimension}, got {len(vector)}"
        )
    if math.sqrt(sum(component * component for component in vector)) == 0.0:
        raise ValueError("embedding must not be zero-norm")
    return vector


__all__ = ["EmbeddingProvider", "validate_embedding"]
