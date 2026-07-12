from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from tau3_retail_evolver.io.jsonl import _fsync_directory
from tau3_retail_evolver.memory.json_store import JsonTierStore, serialize_tier
from tau3_retail_evolver.memory.types import (
    MEMORY_TIERS,
    MemoryItem,
    MemorySnapshot,
    MemoryStatus,
    MemoryTier,
    stable_memory_id,
)


class MemoryRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._stores = {
            tier: JsonTierStore(self.root / f"{tier.value}_memory.json", tier)
            for tier in MemoryTier
        }
        self._items: dict[str, MemoryItem] = {}
        for store in self._stores.values():
            for item in store.load():
                if item.id in self._items:
                    raise ValueError(f"duplicate memory id across tiers: {item.id}")
                self._items[item.id] = item

    def add(
        self,
        *,
        tier: MemoryTier | str,
        content: str,
        source_task_ids: Iterable[str],
        created_round: int,
        metadata: dict[str, Any] | None = None,
        retrieval_text: str | None = None,
        embedding: Iterable[float] | None = None,
        embedding_model_revision: str | None = None,
    ) -> MemoryItem:
        memory_tier = MemoryTier(tier)
        memory_id = stable_memory_id(memory_tier, content)
        if memory_id in self._items:
            raise ValueError(f"duplicate memory: {memory_id}")
        item = MemoryItem(
            id=memory_id,
            tier=memory_tier,
            content=content,
            retrieval_text=retrieval_text or content,
            embedding=tuple(embedding) if embedding is not None else None,
            embedding_model_revision=embedding_model_revision,
            metadata=metadata or {},
            source_task_ids=tuple(source_task_ids),
            created_round=created_round,
            updated_round=created_round,
        )
        replacement = self._tier_items(memory_tier) + [item]
        self._stores[memory_tier].write(replacement)
        self._items[item.id] = item
        return item

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items.get(memory_id)

    def list(
        self,
        *,
        tier: MemoryTier | str | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
    ) -> list[MemoryItem]:
        memory_tier = MemoryTier(tier) if tier is not None else None
        return sorted(
            (
                item
                for item in self._items.values()
                if (memory_tier is None or item.tier == memory_tier)
                and (status is None or item.status == status)
            ),
            key=lambda item: item.id,
        )

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        updated_round: int,
    ) -> MemoryItem:
        current = self._require(memory_id)
        replacement = current.model_copy(
            update={
                "status": MemoryStatus(status),
                "version": current.version + 1,
                "updated_round": updated_round,
            }
        )
        tier_items = [replacement if item.id == memory_id else item for item in self._tier_items(current.tier)]
        self._stores[current.tier].write(tier_items)
        self._items[memory_id] = replacement
        return replacement

    def update_embeddings(
        self,
        updates: Mapping[str, tuple[Sequence[float], str]],
    ) -> list[MemoryItem]:
        replacements: dict[str, MemoryItem] = {}
        for memory_id, (embedding, model_revision) in updates.items():
            current = self._require(memory_id)
            payload = current.model_dump(mode="python")
            payload.update(
                embedding=tuple(embedding),
                embedding_model_revision=model_revision,
            )
            replacements[memory_id] = MemoryItem.model_validate(payload)
        for tier in MemoryTier:
            tier_replacements = {
                memory_id: item
                for memory_id, item in replacements.items()
                if item.tier == tier
            }
            if not tier_replacements:
                continue
            tier_items = [
                tier_replacements.get(item.id, item) for item in self._tier_items(tier)
            ]
            self._stores[tier].write(tier_items)
            self._items.update(tier_replacements)
        return [replacements[memory_id] for memory_id in sorted(replacements)]

    def snapshot(self) -> MemorySnapshot:
        files = {
            f"{tier.value}_memory.json": serialize_tier(tier, self._tier_items(tier))
            for tier in MemoryTier
        }
        counts = {
            tier: len(self.list(tier=tier, status=MemoryStatus.ACTIVE)) for tier in MEMORY_TIERS
        }
        file_hashes = {
            name: hashlib.sha256(content).hexdigest() for name, content in sorted(files.items())
        }
        hash_payload = json.dumps(
            {"schema_version": 1, "counts": counts, "files": file_hashes},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        snapshot_id = hashlib.sha256(hash_payload).hexdigest()
        snapshot_path = self.root / "snapshots" / snapshot_id
        manifest = {
            "schema_version": 1,
            "memory_snapshot_id": snapshot_id,
            "counts": counts,
            "files": file_hashes,
        }
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        expected = {**files, "manifest.json": manifest_bytes}
        if snapshot_path.exists():
            self._verify_snapshot(snapshot_path, expected)
        else:
            self._publish_snapshot(snapshot_path, expected)
        return MemorySnapshot(
            memory_snapshot_id=snapshot_id,
            path=snapshot_path,
            counts=counts,
        )

    def _require(self, memory_id: str) -> MemoryItem:
        item = self.get(memory_id)
        if item is None:
            raise KeyError(f"unknown memory: {memory_id}")
        return item

    def _tier_items(self, tier: MemoryTier) -> list[MemoryItem]:
        return sorted(
            (item for item in self._items.values() if item.tier == tier),
            key=lambda item: item.id,
        )

    @staticmethod
    def _verify_snapshot(path: Path, expected: dict[str, bytes]) -> None:
        if {file.name for file in path.glob("*.json")} != set(expected):
            raise ValueError(f"snapshot contents mismatch: {path}")
        for name, content in expected.items():
            if (path / name).read_bytes() != content:
                raise ValueError(f"snapshot contents mismatch: {path / name}")

    @staticmethod
    def _publish_snapshot(path: Path, files: dict[str, bytes]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=path.parent))
        try:
            for name, content in files.items():
                with (temporary / name).open("wb") as destination:
                    destination.write(content)
                    destination.flush()
                    os.fsync(destination.fileno())
            _fsync_directory(temporary)
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
