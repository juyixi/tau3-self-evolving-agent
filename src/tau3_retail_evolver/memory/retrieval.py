from __future__ import annotations

import hashlib
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import EmbeddingProvider, validate_embedding
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
        with repository.read_transaction():
            return self._retrieve(
                query,
                repository,
                top_k=top_k,
                event_context=event_context,
            )

    def _retrieve(
        self,
        query: str,
        repository: MemoryRepository,
        *,
        top_k: int,
        event_context: dict[str, Any] | None,
    ) -> list[MemoryCandidate]:
        provider = self.provider
        if repository.is_read_only:
            read_only_view = getattr(provider, "read_only_view", None)
            if callable(read_only_view):
                provider = read_only_view()
        query = query.strip()
        if not query:
            raise ValueError("retrieval query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        prepare = getattr(provider, "prepare", None)
        if callable(prepare):
            prepare()
        model_revision = provider.model_revision
        dimension = provider.dimension
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
            or item.embedding_model_revision != model_revision
            or not _is_valid_embedding(item.embedding, dimension)
        ]
        if stale:
            generated = provider.embed_batch([item.retrieval_text for item in stale])
            _assert_provider_identity(provider, model_revision, dimension)
            if len(generated) != len(stale):
                raise ValueError("embedding provider returned the wrong batch size")
            updates = {
                item.id: (
                    validate_embedding(embedding, dimension=dimension),
                    model_revision,
                )
                for item, embedding in zip(stale, generated, strict=True)
            }
            if repository.is_read_only:
                replacements = {}
                for item in stale:
                    payload = item.model_dump(mode="python")
                    payload.update(
                        embedding=updates[item.id][0],
                        embedding_model_revision=updates[item.id][1],
                    )
                    replacements[item.id] = MemoryItem.model_validate(payload)
                items = [replacements.get(item.id, item) for item in items]
            else:
                repository.update_embeddings(updates)
                items = [repository.get(item.id) for item in items]
        query_embedding = (
            validate_embedding(provider.embed(query), dimension=dimension)
            if items
            else ()
        )
        _assert_provider_identity(provider, model_revision, dimension)
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
                retriever_revision=model_revision,
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
                    "retriever_revision": model_revision,
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


def _is_valid_embedding(embedding: tuple[float, ...], dimension: int) -> bool:
    try:
        validate_embedding(embedding, dimension=dimension)
    except ValueError:
        return False
    return True


def _assert_provider_identity(
    provider: EmbeddingProvider,
    model_revision: str,
    dimension: int,
) -> None:
    if provider.model_revision != model_revision or provider.dimension != dimension:
        raise RuntimeError("embedding provider identity changed during retrieval")
