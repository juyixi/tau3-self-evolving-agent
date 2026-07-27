from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import EmbeddingProvider, validate_embedding
from tau3_retail_evolver.memory.outcomes import is_retrieval_eligible
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
        tier_quotas: Mapping[str, int] | None = None,
        mmr_lambdas: Mapping[str, float] | None = None,
        global_mmr_lambda: float = 0.75,
        event_context: dict[str, Any] | None = None,
    ) -> list[MemoryCandidate]:
        with repository.read_transaction():
            return self._retrieve(
                query,
                repository,
                top_k=top_k,
                tier_quotas=tier_quotas,
                mmr_lambdas=mmr_lambdas,
                global_mmr_lambda=global_mmr_lambda,
                event_context=event_context,
            )

    def _retrieve(
        self,
        query: str,
        repository: MemoryRepository,
        *,
        top_k: int,
        tier_quotas: Mapping[str, int] | None,
        mmr_lambdas: Mapping[str, float] | None,
        global_mmr_lambda: float,
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
            (item for item in repository.list() if is_retrieval_eligible(item)),
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
        )
        if tier_quotas is not None:
            ranked = _tiered_mmr_rank(
                ranked,
                top_k=top_k,
                tier_quotas=tier_quotas,
                mmr_lambdas=mmr_lambdas or {},
                global_mmr_lambda=global_mmr_lambda,
            )
        else:
            ranked = ranked[:top_k]
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


_TIER_SELECTION_ORDER = (
    MemoryTier.SKILL,
    MemoryTier.TIP,
    MemoryTier.TOOL,
    MemoryTier.TRAJECTORY,
)


def _tiered_mmr_rank(
    ranked: list[tuple[float, MemoryItem]],
    *,
    top_k: int,
    tier_quotas: Mapping[str, int],
    mmr_lambdas: Mapping[str, float],
    global_mmr_lambda: float,
) -> list[tuple[float, MemoryItem]]:
    if not 0.0 <= global_mmr_lambda <= 1.0:
        raise ValueError("global_mmr_lambda must be between 0 and 1")
    quotas = _scaled_tier_quotas(tier_quotas, top_k)
    for tier in MemoryTier:
        if tier.value in mmr_lambdas and not 0.0 <= mmr_lambdas[tier.value] <= 1.0:
            raise ValueError(f"MMR lambda for {tier.value} must be between 0 and 1")

    selected: list[tuple[float, MemoryItem]] = []
    selected_ids: set[str] = set()
    for tier in _TIER_SELECTION_ORDER:
        tier_ranked = [pair for pair in ranked if pair[1].tier == tier]
        picks = _mmr_select(
            tier_ranked,
            count=quotas[tier.value],
            selected_items=(),
            relevance_weight=mmr_lambdas.get(tier.value, 0.75),
        )
        selected.extend(picks)
        selected_ids.update(item.id for _, item in picks)

    remaining = [pair for pair in ranked if pair[1].id not in selected_ids]
    selected.extend(
        _mmr_select(
            remaining,
            count=top_k - len(selected),
            selected_items=tuple(item for _, item in selected),
            relevance_weight=global_mmr_lambda,
        )
    )
    return selected


def _scaled_tier_quotas(tier_quotas: Mapping[str, int], top_k: int) -> dict[str, int]:
    values: dict[str, int] = {}
    for tier in MemoryTier:
        value = tier_quotas.get(tier.value, 0)
        if type(value) is not int or value < 0:
            raise ValueError(f"tier quota for {tier.value} must be a non-negative integer")
        values[tier.value] = value
    total = sum(values.values())
    if total <= top_k:
        return values
    scaled = {tier: values[tier] * top_k / total for tier in values}
    floors = {tier: int(scaled[tier]) for tier in values}
    remaining = top_k - sum(floors.values())
    order = sorted(
        values,
        key=lambda tier: (-(scaled[tier] - floors[tier]), tier),
    )
    for tier in order[:remaining]:
        floors[tier] += 1
    return floors


def _mmr_select(
    ranked: list[tuple[float, MemoryItem]],
    *,
    count: int,
    selected_items: tuple[MemoryItem, ...],
    relevance_weight: float,
) -> list[tuple[float, MemoryItem]]:
    if count <= 0 or not ranked:
        return []
    available = list(ranked)
    chosen: list[tuple[float, MemoryItem]] = []
    comparison = list(selected_items)
    while available and len(chosen) < count:
        def score(pair: tuple[float, MemoryItem]) -> tuple[float, float, str]:
            relevance, item = pair
            redundancy = max(
                (_cosine(item.embedding or (), other.embedding or ()) for other in comparison),
                default=0.0,
            )
            return (relevance_weight * relevance - (1.0 - relevance_weight) * redundancy, relevance, item.id)

        best = sorted(
            available,
            key=lambda pair: (-score(pair)[0], -score(pair)[1], score(pair)[2]),
        )[0]
        chosen.append(best)
        comparison.append(best[1])
        available.remove(best)
    return chosen


def _assert_provider_identity(
    provider: EmbeddingProvider,
    model_revision: str,
    dimension: int,
) -> None:
    if provider.model_revision != model_revision or provider.dimension != dimension:
        raise RuntimeError("embedding provider identity changed during retrieval")
