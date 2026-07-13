from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_retail_evolver.config import MemoryConfig
from tau3_retail_evolver.memory.embeddings import (
    CachedEmbeddingProvider,
    JsonEmbeddingCache,
    LocalQwenEmbeddingProvider,
    build_embedding_provider,
)


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


class ResolvingProvider(CountingProvider):
    def __init__(self) -> None:
        super().__init__("embedding@alias")
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True
        self.model_revision = "embedding@resolved-commit"

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        assert self.prepared
        return super().embed_batch(texts)


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


def test_cache_instances_merge_updates_against_authoritative_file(tmp_path: Path) -> None:
    cache_path = tmp_path / "embedding_cache.json"
    first = JsonEmbeddingCache(cache_path)
    second = JsonEmbeddingCache(cache_path)

    first.put_many("embedding@rev-a", [("refund", (1.0, 0.0))])
    second.put_many("embedding@rev-a", [("exchange", (0.0, 1.0))])

    reopened = JsonEmbeddingCache(cache_path)
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert reopened.get("embedding@rev-a", "refund") == (1.0, 0.0)
    assert reopened.get("embedding@rev-a", "exchange") == (0.0, 1.0)
    assert payload["count"] == len(payload["entries"]) == 2


def test_cache_hit_must_match_provider_dimension(tmp_path: Path) -> None:
    cache = JsonEmbeddingCache(tmp_path / "embedding_cache.json")
    cache.put_many("embedding@rev-a", [("refund", (1.0, 0.0, 0.0))])

    with pytest.raises(ValueError, match="dimension mismatch"):
        CachedEmbeddingProvider(CountingProvider("embedding@rev-a"), cache).embed("refund")


def test_rejects_zero_norm_embedding_before_caching(tmp_path: Path) -> None:
    provider = CountingProvider("embedding@rev-a")
    provider.embed_batch = lambda _texts: [(0.0, 0.0)]  # type: ignore[method-assign]
    cache = JsonEmbeddingCache(tmp_path / "embedding_cache.json")

    with pytest.raises(ValueError, match="zero-norm"):
        CachedEmbeddingProvider(provider, cache).embed("refund")

    assert not cache.path.exists()


def test_resolves_model_revision_before_cache_lookup(tmp_path: Path) -> None:
    cache_path = tmp_path / "embedding_cache.json"
    first = ResolvingProvider()
    second = ResolvingProvider()

    generated = CachedEmbeddingProvider(first, JsonEmbeddingCache(cache_path)).embed("refund")
    cached = CachedEmbeddingProvider(second, JsonEmbeddingCache(cache_path)).embed("refund")

    assert generated == cached
    assert first.calls == [["refund"]]
    assert second.calls == []
    assert first.model_revision == second.model_revision == "embedding@resolved-commit"


def test_builds_local_qwen_provider_and_json_cache_from_config(tmp_path: Path) -> None:
    config = MemoryConfig(
        embedding_model="Qwen/test-embedding",
        embedding_device="cpu",
        embedding_dtype="float32",
        embedding_max_length=512,
        embedding_batch_size=4,
        embedding_cache=True,
    )

    provider = build_embedding_provider(config, tmp_path / "memory")

    assert isinstance(provider, CachedEmbeddingProvider)
    assert isinstance(provider.provider, LocalQwenEmbeddingProvider)
    assert provider.provider.model_id == "Qwen/test-embedding"
    assert provider.provider.device == "cpu"
    assert provider.provider.dtype == "float32"
    assert provider.provider.max_length == 512
    assert provider.provider.batch_size == 4
    assert provider.cache.path == tmp_path / "memory" / "embedding_cache.json"


def test_can_disable_embedding_cache(tmp_path: Path) -> None:
    provider = build_embedding_provider(
        MemoryConfig(embedding_cache=False),
        tmp_path / "memory",
    )

    assert isinstance(provider, LocalQwenEmbeddingProvider)
