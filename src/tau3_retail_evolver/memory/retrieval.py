from __future__ import annotations

import hashlib
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import EmbeddingProvider
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.types import MEMORY_TIERS, MemoryItem, MemoryTier


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    memory_id: str
    memory_version: int
    tier: MemoryTier
    rank: int = Field(ge=1)
    similarity: float
    retriever_revision: str
    query_hash: str
    item: MemoryItem


class Retriever:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        candidate_writer: JsonlWriter | None = None,
    ) -> None:
        self.provider = provider
        self.candidate_writer = candidate_writer

    def retrieve(
        self,
        query: str,
        repository: MemoryRepository,
        *,
        top_k: int = 50,
        event_context: dict[str, Any] | None = None,
    ) -> list[MemoryCandidate]:
        query = query.strip()
        if not query:
            raise ValueError("retrieval query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        tier_order = {tier: index for index, tier in enumerate(MEMORY_TIERS)}
        items = sorted(
            repository.list(),
            key=lambda item: (tier_order[item.tier.value], item.id),
        )
        stale = [
            item
            for item in items
            if item.embedding is None
            or item.embedding_model_revision != self.provider.model_revision
            or len(item.embedding) != self.provider.dimension
        ]
        if stale:
            generated = self.provider.embed_batch([item.retrieval_text for item in stale])
            if len(generated) != len(stale):
                raise ValueError("embedding provider returned the wrong batch size")
            repository.update_embeddings(
                {
                    item.id: (embedding, self.provider.model_revision)
                    for item, embedding in zip(stale, generated, strict=True)
                }
            )
            items = [repository.get(item.id) for item in items]
        query_embedding = self.provider.embed(query) if items else ()
        ranked = sorted(
            (
                (_cosine(query_embedding, item.embedding or ()), item)
                for item in items
            ),
            key=lambda pair: (-pair[0], pair[1].id),
        )[:top_k]
        candidates = [
            MemoryCandidate(
                memory_id=item.id,
                memory_version=item.version,
                tier=item.tier,
                rank=rank,
                similarity=similarity,
                retriever_revision=self.provider.model_revision,
                query_hash=query_hash,
                item=item,
            )
            for rank, (similarity, item) in enumerate(ranked, start=1)
        ]
        if self.candidate_writer is not None:
            self.candidate_writer.append(
                {
                    **(event_context or {}),
                    "schema_version": 1,
                    "event_type": "MemoryCandidatesRetrieved",
                    "query_hash": query_hash,
                    "retriever_revision": self.provider.model_revision,
                    "candidates": [
                        candidate.model_dump(
                            mode="json",
                            exclude={"item", "query_hash", "retriever_revision"},
                        )
                        for candidate in candidates
                    ],
                }
            )
        return candidates


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    left_norm = math.sqrt(sum(component * component for component in left))
    right_norm = math.sqrt(sum(component * component for component in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("zero-norm embedding")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
