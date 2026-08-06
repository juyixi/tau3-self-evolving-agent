from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from tau3_evolver.persistence.atomic import write_bytes_atomic
from tau3_evolver.persistence.jsonl import iter_jsonl_objects
from tau3_evolver.security.redaction import redact_public_data


RUN_SCHEMA_VERSION = 1


def episode_artifact_metadata(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    rows = tuple(iter_jsonl_objects(path))
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(rows),
    }


def write_run_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite run record: {path}")
    normalized = redact_public_data(
        {"schema_version": RUN_SCHEMA_VERSION, **dict(record)}
    )
    serialized = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    write_bytes_atomic(path, f"{serialized}\n".encode("utf-8"))
    return normalized


__all__ = ["RUN_SCHEMA_VERSION", "episode_artifact_metadata", "write_run_record"]
