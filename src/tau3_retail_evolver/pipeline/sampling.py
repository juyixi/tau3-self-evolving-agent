from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any


OPD_KINDS = ("sel", "act", "write", "maint")


@dataclass(frozen=True, slots=True)
class KindSample:
    epoch: int
    round_index: int
    kind: str
    index: int


def select_train_tasks(
    train_task_ids: Sequence[str],
    *,
    task_count: int,
    seed: int,
    iteration: int,
    shuffle: bool,
    explicit_task_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    official = tuple(train_task_ids)
    if not official or len(official) != len(set(official)):
        raise ValueError("official train task IDs must be nonempty and unique")
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if not 1 <= task_count <= len(official):
        raise ValueError("task_count must fit within the official train split")

    if explicit_task_ids:
        selected = tuple(explicit_task_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("explicit task IDs must be unique")
        unknown = [task_id for task_id in selected if task_id not in set(official)]
        if unknown:
            raise ValueError(
                "requested task IDs are not in the official train split: "
                + ", ".join(unknown)
            )
        return selected

    selected = list(official)
    if shuffle:
        random.Random(f"{seed}:{iteration}:tau3-retail-stage7").shuffle(selected)
    return tuple(selected[:task_count])


def balanced_kind_schedule(
    kind_counts: Mapping[str, int],
    *,
    num_epochs: int,
    kind_order: Sequence[str] = OPD_KINDS,
) -> tuple[KindSample, ...]:
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    unknown = set(kind_counts) - set(kind_order)
    if unknown:
        raise ValueError(f"unknown OPD kinds: {', '.join(sorted(unknown))}")
    if any(type(count) is not int or count < 0 for count in kind_counts.values()):
        raise ValueError("kind counts must be non-negative integers")

    active = tuple(kind for kind in kind_order if kind_counts.get(kind, 0) > 0)
    if not active:
        return ()
    rounds = max(kind_counts[kind] for kind in active)
    return tuple(
        KindSample(
            epoch=epoch,
            round_index=round_index,
            kind=kind,
            index=round_index % kind_counts[kind],
        )
        for epoch in range(num_epochs)
        for round_index in range(rounds)
        for kind in active
    )


def assert_train_only_artifacts(
    root: Path,
    *,
    train_task_ids: Sequence[str],
) -> None:
    allowed = set(train_task_ids)
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if path.suffix == ".json":
            values = (_read_json(path),)
        else:
            values = tuple(_read_jsonl(path))
        for value in values:
            _assert_value_task_ids(value, allowed=allowed, source=path)


def _assert_value_task_ids(value: Any, *, allowed: set[str], source: Path) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if normalized == "task_id" or normalized.endswith("_task_id"):
                _require_task_id(nested, allowed=allowed, source=source)
            elif normalized == "task_ids" or normalized.endswith("_task_ids"):
                if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
                    raise ValueError(f"task ID collection is invalid: {source}")
                for task_id in nested:
                    _require_task_id(task_id, allowed=allowed, source=source)
            else:
                _assert_value_task_ids(nested, allowed=allowed, source=source)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _assert_value_task_ids(nested, allowed=allowed, source=source)


def _require_task_id(value: Any, *, allowed: set[str], source: Path) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"learning artifact contains non-train task ID {value!r}: {source}")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to inspect JSON artifact: {path}") from error


def _read_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"unable to inspect JSONL artifact: {path}:{line_number}"
                    ) from error
    except OSError as error:
        raise ValueError(f"unable to inspect JSONL artifact: {path}") from error
    return values
