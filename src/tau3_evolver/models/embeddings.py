from __future__ import annotations

from pathlib import Path

from tau3_evolver.config import MemoryConfig
from tau3_evolver.memory.embeddings import EmbeddingProvider, validate_embedding
from tau3_evolver.persistence.embedding_cache import JsonEmbeddingCache


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
    """Lazy local Qwen3 embedding provider matching the OPD-Evolver pooling path."""

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


__all__ = [
    "CachedEmbeddingProvider",
    "LocalQwenEmbeddingProvider",
    "build_embedding_provider",
]
