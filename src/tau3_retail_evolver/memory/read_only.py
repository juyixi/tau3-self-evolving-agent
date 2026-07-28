from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from typing import Any

from tau3_retail_evolver.memory.json_store import JsonTierStore
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.types import (
    MEMORY_TIERS,
    MemoryItem,
    MemoryStatus,
    MemoryTier,
)


class ReadOnlyMemoryRepository:
    is_read_only = True

    def __init__(self, snapshot_path: Path) -> None:
        self.root = snapshot_path
        self.memory_snapshot_id = self._verify_snapshot(snapshot_path)
        self._repository = MemoryRepository(snapshot_path)
        self._items = tuple(self._repository.list(status=None))
        self._items_by_id = {item.id: item for item in self._items}

    def get(self, memory_id: str) -> MemoryItem | None:
        return self._items_by_id.get(memory_id)

    def list(
        self,
        *,
        tier: MemoryTier | str | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
    ) -> list[MemoryItem]:
        memory_tier = MemoryTier(tier) if tier is not None else None
        return [
            item
            for item in self._items
            if (memory_tier is None or item.tier == memory_tier)
            and (status is None or item.status == status)
        ]

    def read_transaction(self):
        return nullcontext()

    def add(self, **_kwargs: Any) -> None:
        self._deny_write()

    def update_status(self, *_args: Any, **_kwargs: Any) -> None:
        self._deny_write()

    def update_embeddings(self, *_args: Any, **_kwargs: Any) -> None:
        self._deny_write()

    def snapshot(self) -> None:
        self._deny_write()

    @staticmethod
    def _deny_write() -> None:
        raise PermissionError("memory snapshot is read-only")

    @staticmethod
    def _verify_snapshot(path: Path) -> str:
        manifest_path = path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid memory snapshot manifest: {manifest_path}") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported memory snapshot schema: {manifest_path}")
        snapshot_id = manifest.get("memory_snapshot_id")
        files = manifest.get("files")
        counts = manifest.get("counts")
        if not isinstance(snapshot_id, str) or not isinstance(files, dict) or not isinstance(counts, dict):
            raise ValueError(f"invalid memory snapshot manifest: {manifest_path}")
        if path.name != snapshot_id:
            raise ValueError(f"memory snapshot ID mismatch: {path}")
        expected_files = {f"{tier.value}_memory.json" for tier in MemoryTier}
        if set(files) != expected_files:
            raise ValueError(f"memory snapshot file set mismatch: {path}")
        actual_json_files = {file.name for file in path.glob("*.json")}
        if actual_json_files != expected_files | {"manifest.json"}:
            raise ValueError(f"memory snapshot file set mismatch: {path}")
        for name, expected_hash in files.items():
            if not isinstance(expected_hash, str):
                raise ValueError(f"invalid memory snapshot manifest: {manifest_path}")
            file_path = path / name
            if not file_path.is_file():
                raise ValueError(f"memory snapshot file missing: {file_path}")
            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"memory snapshot hash mismatch: {file_path}")
        hash_payload = json.dumps(
            {"schema_version": 1, "counts": counts, "files": files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(hash_payload).hexdigest() != snapshot_id:
            raise ValueError(f"memory snapshot ID hash mismatch: {path}")
        if set(counts) != set(MEMORY_TIERS) or any(
            not isinstance(count, int) or count < 0 for count in counts.values()
        ):
            raise ValueError(f"invalid memory snapshot counts: {manifest_path}")
        actual_counts = {
            tier.value: sum(
                item.status == MemoryStatus.ACTIVE
                for item in JsonTierStore(
                    path / f"{tier.value}_memory.json",
                    tier,
                ).load()
            )
            for tier in MemoryTier
        }
        if counts != actual_counts:
            raise ValueError(f"memory snapshot count mismatch: {path}")
        return snapshot_id
