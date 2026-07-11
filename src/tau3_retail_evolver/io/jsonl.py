from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonlWriter:
    """Append canonical JSON objects one at a time without truncating existing events."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: dict[str, Any]) -> None:
        try:
            line = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("event must be JSON serializable") from error

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as destination:
            destination.write(f"{line}\n")
            destination.flush()
            os.fsync(destination.fileno())
