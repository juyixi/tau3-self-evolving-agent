from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tau3_retail_evolver.io.jsonl import JsonlWriter
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
