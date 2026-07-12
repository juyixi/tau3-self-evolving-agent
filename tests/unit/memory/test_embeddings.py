from __future__ import annotations

import json
from pathlib import Path

from tau3_retail_evolver.memory.embeddings import CachedEmbeddingProvider, JsonEmbeddingCache


class CountingProvider:
    dimension = 2

    def __init__(self, revision: str) -> None:
        self.model_revision = revision
        self.calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.calls.append(list(texts))
        return [(float(len(text)), 1.0) for text in texts]

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_batch([text])[0]


def test_cache_avoids_duplicate_embedding_calls_and_persists(tmp_path: Path) -> None:
    cache_path = tmp_path / "embedding_cache.json"
    provider = CountingProvider("embedding@rev-a")
    cached = CachedEmbeddingProvider(provider, JsonEmbeddingCache(cache_path))

    first = cached.embed_batch(["refund", "exchange", "refund"])
    second = CachedEmbeddingProvider(
        CountingProvider("embedding@rev-a"), JsonEmbeddingCache(cache_path)
    ).embed("refund")

    assert first[0] == first[2] == second
    assert provider.calls == [["refund", "exchange"]]
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["count"] == 2
    assert len(payload["entries"]) == 2


def test_cache_key_isolated_by_model_revision(tmp_path: Path) -> None:
    cache = JsonEmbeddingCache(tmp_path / "embedding_cache.json")
    first_provider = CountingProvider("embedding@rev-a")
    second_provider = CountingProvider("embedding@rev-b")

    CachedEmbeddingProvider(first_provider, cache).embed("same text")
    CachedEmbeddingProvider(second_provider, cache).embed("same text")

    assert first_provider.calls == [["same text"]]
    assert second_provider.calls == [["same text"]]
    assert json.loads(cache.path.read_text(encoding="utf-8"))["count"] == 2
