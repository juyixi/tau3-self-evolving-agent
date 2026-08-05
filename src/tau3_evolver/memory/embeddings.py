from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Protocol

from tau3_evolver.config import MemoryConfig
from tau3_evolver.memory.json_store import write_bytes_atomic
from tau3_evolver.memory.locking import reentrant_process_lock


class EmbeddingProvider(Protocol):
    model_revision: str
    dimension: int

    def embed(self, text: str) -> tuple[float, ...]: ...

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]: ...


class JsonEmbeddingCache:
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
                vector = validate_embedding(embedding)
                replacement[_cache_key(model_revision, text)] = list(vector)
            payload = {
                "schema_version": 1,
                "count": len(replacement),
                "entries": dict(sorted(replacement.items())),
            }
            serialized = (
                json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
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
        return {str(key): list(validate_embedding(value)) for key, value in entries.items()}


class CachedEmbeddingProvider:
    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: JsonEmbeddingCache,
        *,
        write_cache: bool = True,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self._write_cache = write_cache

    @property
    def model_revision(self) -> str:
        return self.provider.model_revision

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_batch([text])[0]

    def prepare(self) -> None:
        prepare = getattr(self.provider, "prepare", None)
        if callable(prepare):
            prepare()

    def read_only_view(self) -> CachedEmbeddingProvider:
        return CachedEmbeddingProvider(self.provider, self.cache, write_cache=False)

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        self.prepare()
        model_revision = self.model_revision
        resolved: dict[str, tuple[float, ...]] = {}
        missing: list[str] = []
        for text in texts:
            cached = self.cache.get(model_revision, text)
            if cached is not None:
                resolved[text] = validate_embedding(cached, dimension=self.dimension)
            elif text not in resolved and text not in missing:
                missing.append(text)
        if missing:
            generated = self.provider.embed_batch(missing)
            if self.model_revision != model_revision:
                raise RuntimeError("embedding model revision changed during generation")
            if len(generated) != len(missing):
                raise ValueError("embedding provider returned the wrong batch size")
            additions: list[tuple[str, tuple[float, ...]]] = []
            for text, embedding in zip(missing, generated, strict=True):
                vector = validate_embedding(embedding, dimension=self.dimension)
                resolved[text] = vector
                additions.append((text, vector))
            if self._write_cache:
                self.cache.put_many(model_revision, additions)
        return [resolved[text] for text in texts]


class LocalQwenEmbeddingProvider:
    """Lazy local Qwen3 embedding provider matching the official OPD-Evolver pooling path."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        *,
        revision: str | None = None,
        device: str = "cuda",
        dtype: str = "float16",
        max_length: int = 2048,
        batch_size: int = 16,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_revision = f"{model_id}@{revision}" if revision else model_id
        self.dimension = 1024
        self._model = None
        self._tokenizer = None

    def embed(self, text: str) -> tuple[float, ...]:
        return self.embed_batch([text])[0]

    def prepare(self) -> None:
        self._load()

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        model, tokenizer, torch = self._load()
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            pooled = self._last_token_pool(
                outputs.last_hidden_state,
                batch["attention_mask"],
                torch,
            )
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(
                tuple(float(component) for component in row)
                for row in normalized.detach().cpu().to(torch.float32).tolist()
            )
        return vectors

    def _load(self):
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as error:
                raise ImportError(
                    "local embeddings require torch and transformers"
                ) from error
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(self.dtype)
            if dtype is None:
                raise ValueError(f"unsupported embedding dtype: {self.dtype}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                padding_side="left",
            )
            self._model = AutoModel.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=dtype,
            ).to(self.device)
            self._model.eval()
            self.dimension = int(getattr(self._model.config, "hidden_size", 1024))
            resolved = getattr(self._model.config, "_commit_hash", None)
            if resolved:
                self.model_revision = f"{self.model_id}@{resolved}"
        else:
            import torch
        return self._model, self._tokenizer, torch

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask, torch):
        if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def build_embedding_provider(
    config: MemoryConfig,
    cache_root: Path | None = None,
) -> EmbeddingProvider:
    if config.embedding_provider != "local":
        raise ValueError(f"unsupported embedding provider: {config.embedding_provider}")
    provider = LocalQwenEmbeddingProvider(
        model_id=config.embedding_model,
        device=config.embedding_device,
        dtype=config.embedding_dtype,
        max_length=config.embedding_max_length,
        batch_size=config.embedding_batch_size,
    )
    if not config.embedding_cache or cache_root is None:
        return provider
    return CachedEmbeddingProvider(
        provider,
        JsonEmbeddingCache(cache_root / "embedding_cache.json"),
    )


def _cache_key(model_revision: str, text: str) -> str:
    return hashlib.sha256(f"{model_revision}\0{text}".encode("utf-8")).hexdigest()


def validate_embedding(
    value: Sequence[float], *, dimension: int | None = None
) -> tuple[float, ...]:
    vector = tuple(float(component) for component in value)
    if not vector or not all(math.isfinite(component) for component in vector):
        raise ValueError("embedding must contain finite values")
    if dimension is not None and len(vector) != dimension:
        raise ValueError(f"embedding dimension mismatch: expected {dimension}, got {len(vector)}")
    if math.sqrt(sum(component * component for component in vector)) == 0.0:
        raise ValueError("embedding must not be zero-norm")
    return vector
