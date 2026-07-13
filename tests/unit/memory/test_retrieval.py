from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Event

import pytest

from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import CachedEmbeddingProvider, JsonEmbeddingCache
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever
from tau3_retail_evolver.memory.types import MemoryStatus


class FakeEmbeddingProvider:
    model_revision = "fake-embedding@revision-1"
    dimension = 2

    def __init__(self, vectors: dict[str, tuple[float, float]]) -> None:
        self.vectors = vectors
        self.batches: list[list[str]] = []

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.batches.append(list(texts))
        return [self.vectors[text] for text in texts]


class ResolvingEmbeddingProvider(FakeEmbeddingProvider):
    model_revision = "fake-embedding@alias"

    def prepare(self) -> None:
        self.model_revision = "fake-embedding@resolved-commit"


def test_retrieval_backfills_embeddings_and_returns_auditable_candidates(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    tip = repository.add(
        tier="tip",
        content="Confirm identity before refund",
        source_task_ids=("task-1",),
        created_round=0,
    )
    skill = repository.add(
        tier="skill",
        content="Inspect order and request confirmation",
        source_task_ids=("task-2",),
        created_round=0,
    )
    tool = repository.add(
        tier="tool",
        content="Cancel order helper",
        source_task_ids=("task-3",),
        created_round=0,
    )
    repository.update_status(tool.id, MemoryStatus.RETIRED, updated_round=1)
    provider = FakeEmbeddingProvider(
        {
            "refund request": (1.0, 0.0),
            tip.retrieval_text: (1.0, 0.0),
            skill.retrieval_text: (0.6, 0.8),
        }
    )

    candidates = Retriever(provider).retrieve("refund request", repository, top_k=50)

    assert [candidate.memory_id for candidate in candidates] == [tip.id, skill.id]
    assert [candidate.rank for candidate in candidates] == [1, 2]
    assert candidates[0].similarity == pytest.approx(1.0)
    assert candidates[1].similarity == pytest.approx(0.6)
    assert candidates[0].tier == "tip"
    assert candidates[0].memory_version == 1
    assert candidates[0].retriever_revision == provider.model_revision
    assert candidates[0].query_hash == hashlib.sha256(b"refund request").hexdigest()
    assert provider.batches == [[tip.retrieval_text, skill.retrieval_text]]

    reopened = MemoryRepository(tmp_path / "memory")
    assert reopened.get(tip.id).embedding == (1.0, 0.0)
    assert reopened.get(tip.id).embedding_model_revision == provider.model_revision
    assert reopened.get(tool.id).embedding is None


def test_retrieval_recomputes_stale_embeddings_and_breaks_ties_deterministically(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first = repository.add(
        tier="trajectory",
        content="A trajectory",
        source_task_ids=("task-1",),
        created_round=0,
        embedding=(0.0, 1.0),
        embedding_model_revision="old-revision",
    )
    second = repository.add(
        tier="tip",
        content="A tip",
        source_task_ids=("task-2",),
        created_round=0,
    )
    provider = FakeEmbeddingProvider(
        {"query": (1.0, 0.0), first.retrieval_text: (1.0, 0.0), second.retrieval_text: (1.0, 0.0)}
    )

    candidates = Retriever(provider).retrieve("query", repository, top_k=1)

    assert len(candidates) == 1
    assert candidates[0].memory_id == min(first.id, second.id)
    assert provider.batches == [[first.retrieval_text, second.retrieval_text]]


def test_retrieval_appends_complete_candidate_evidence(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="tip",
        content="Verify the order status",
        source_task_ids=("task-1",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@revision-1",
    )
    provider = FakeEmbeddingProvider({"query": (1.0, 0.0)})
    path = tmp_path / "rollouts" / "events.jsonl"

    Retriever(provider, candidate_writer=JsonlWriter(path)).retrieve(
        "query",
        repository,
        top_k=50,
        event_context={"run_id": "run-1", "task_id": "task-9", "split": "train"},
    )

    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["event_type"] == "MemoryCandidatesRetrieved"
    assert event["run_id"] == "run-1"
    assert event["task_id"] == "task-9"
    assert event["split"] == "train"
    assert event["query_hash"] == hashlib.sha256(b"query").hexdigest()
    assert event["retriever_revision"] == provider.model_revision
    assert event["candidates"] == [
        {
            "memory_id": item.id,
            "memory_version": 1,
            "rank": 1,
            "similarity": 1.0,
            "tier": "tip",
        }
    ]


def test_read_only_snapshot_retrieves_without_persisting_missing_embeddings(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="tip",
        content="Verify identity before refund",
        source_task_ids=("task-1",),
        created_round=0,
    )
    snapshot = repository.snapshot()
    tip_path = snapshot.path / "tip_memory.json"
    original = tip_path.read_bytes()
    provider = FakeEmbeddingProvider(
        {"refund request": (1.0, 0.0), item.retrieval_text: (1.0, 0.0)}
    )

    candidates = Retriever(provider).retrieve(
        "refund request",
        ReadOnlyMemoryRepository(snapshot.path),
    )

    assert [candidate.memory_id for candidate in candidates] == [item.id]
    assert candidates[0].item.embedding == (1.0, 0.0)
    assert tip_path.read_bytes() == original


def test_read_only_retrieval_uses_cache_hits_without_persisting_misses(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="tip",
        content="Verify identity before refund",
        source_task_ids=("task-1",),
        created_round=0,
    )
    snapshot = repository.snapshot()
    cache = JsonEmbeddingCache(tmp_path / "embedding_cache.json")
    cache.put_many(
        FakeEmbeddingProvider.model_revision,
        [(item.retrieval_text, (1.0, 0.0))],
    )
    original_cache = cache.path.read_bytes()
    provider = FakeEmbeddingProvider({"refund request": (1.0, 0.0)})

    candidates = Retriever(CachedEmbeddingProvider(provider, cache)).retrieve(
        "refund request",
        ReadOnlyMemoryRepository(snapshot.path),
    )

    assert [candidate.memory_id for candidate in candidates] == [item.id]
    assert provider.batches == [["refund request"]]
    assert cache.path.read_bytes() == original_cache


def test_retrieval_holds_a_consistent_view_during_embedding_backfill(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="tip",
        content="Verify identity before refund",
        source_task_ids=("task-1",),
        created_round=0,
    )
    provider = FakeEmbeddingProvider(
        {"refund request": (1.0, 0.0), item.retrieval_text: (1.0, 0.0)}
    )
    entered = Event()
    release = Event()
    retired = Event()
    original_embed_batch = provider.embed_batch

    def blocking_embed_batch(texts: list[str]) -> list[tuple[float, ...]]:
        entered.set()
        assert release.wait(timeout=2)
        return original_embed_batch(texts)

    provider.embed_batch = blocking_embed_batch  # type: ignore[method-assign]

    def retire() -> None:
        repository.update_status(item.id, MemoryStatus.RETIRED, updated_round=1)
        retired.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        retrieval = pool.submit(Retriever(provider).retrieve, "refund request", repository)
        assert entered.wait(timeout=1)
        maintenance = pool.submit(retire)
        try:
            assert not retired.wait(timeout=0.1)
        finally:
            release.set()
        candidates = retrieval.result()
        maintenance.result()

    assert [candidate.memory_id for candidate in candidates] == [item.id]
    assert repository.get(item.id).status == MemoryStatus.RETIRED


def test_retrieval_resolves_revision_before_classifying_stale_embeddings(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="tip",
        content="Verify identity before refund",
        source_task_ids=("task-1",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@alias",
    )
    provider = ResolvingEmbeddingProvider(
        {"refund request": (1.0, 0.0), item.retrieval_text: (0.6, 0.8)}
    )

    candidates = Retriever(provider).retrieve("refund request", repository)

    assert provider.batches == [[item.retrieval_text]]
    assert candidates[0].retriever_revision == "fake-embedding@resolved-commit"
    assert repository.get(item.id).embedding == (0.6, 0.8)
    assert repository.get(item.id).embedding_model_revision == provider.model_revision


def test_retrieval_recomputes_invalid_stored_vector_and_rejects_invalid_query(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    item = repository.add(
        tier="tip",
        content="Verify identity before refund",
        source_task_ids=("task-1",),
        created_round=0,
        embedding=(0.0, 0.0),
        embedding_model_revision="fake-embedding@revision-1",
    )
    provider = FakeEmbeddingProvider(
        {"bad query": (float("nan"), 0.0), item.retrieval_text: (1.0, 0.0)}
    )

    with pytest.raises(ValueError, match="finite"):
        Retriever(provider).retrieve("bad query", repository)

    assert provider.batches == [[item.retrieval_text]]
    assert repository.get(item.id).embedding == (1.0, 0.0)
