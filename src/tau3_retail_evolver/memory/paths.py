from __future__ import annotations

from pathlib import Path
import re


_SAFE_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def training_memory_root(agent_id: str, *, root: Path | None = None) -> Path:
    namespace = _validated_slug(agent_id, field="agent_id")
    base = (root or project_root()).resolve()
    return base / "history" / "agents" / namespace / "memory"


def evaluation_quarantine_root(
    run_id: str,
    agent_id: str,
    *,
    root: Path | None = None,
) -> Path:
    evaluation = _validated_slug(run_id, field="run_id")
    namespace = _validated_slug(agent_id, field="agent_id")
    base = (root or project_root()).resolve()
    return base / "history" / "evaluations" / evaluation / namespace / "quarantine"


def _validated_slug(value: str, *, field: str) -> str:
    if not _SAFE_SLUG.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{field} must contain only ASCII letters, digits, '-' or '_'")
    return value
