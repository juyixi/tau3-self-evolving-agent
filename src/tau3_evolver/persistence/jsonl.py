from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from tau3_evolver.persistence.atomic import fsync_directory


class JsonlWriter:
    """Append canonical JSON objects without truncating existing records."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise TypeError("record must be JSON serializable") from error

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(f"{line}\n")
            destination.flush()
            os.fsync(destination.fileno())
        fsync_directory(self.path.parent)


def iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects while preserving path and line diagnostics."""
    try:
        source = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"unable to read JSONL file: {path}") from error
    with source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row must be a JSON object at {path}:{line_number}"
                )
            yield value


__all__ = ["JsonlWriter", "iter_jsonl_objects"]
