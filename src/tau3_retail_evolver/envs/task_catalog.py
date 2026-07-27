from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal


SplitName = Literal["train", "test", "base"]
_REQUIRED_SPLITS = ("train", "test", "base")
OFFICIAL_SPLIT_COUNTS = {"train": 74, "test": 40, "base": 114}
OFFICIAL_SPLIT_SHA256 = "235237983dd826c6c16989e90797e9d58f8ed52059020c9079e60069288147eb"
OFFICIAL_TRAIN_TASK_IDS = (
    "0", "1", "2", "3", "4", "6", "7", "8", "10", "11", "13", "14",
    "15", "16", "19", "20", "21", "22", "23", "24", "25", "28", "29",
    "30", "31", "34", "35", "37", "41", "43", "44", "46", "47", "48",
    "50", "52", "54", "57", "58", "59", "63", "66", "67", "69", "72",
    "73", "75", "76", "78", "80", "81", "82", "83", "84", "85", "87",
    "88", "89", "91", "92", "93", "95", "96", "98", "99", "103", "104",
    "105", "106", "107", "109", "110", "112", "113",
)
OFFICIAL_AIRLINE_SPLIT_COUNTS = {"train": 30, "test": 20, "base": 50}
OFFICIAL_AIRLINE_SPLIT_SHA256 = (
    "46e2ced1b82b193a5c0057a471c4884cece06105ea0a94a726f9b24acb090051"
)
OFFICIAL_AIRLINE_TRAIN_TASK_IDS = (
    "0", "1", "3", "4", "5", "7", "9", "10", "11", "12", "14", "15",
    "17", "20", "21", "23", "27", "28", "33", "34", "36", "38", "39",
    "40", "41", "42", "43", "46", "47", "49",
)
_OFFICIAL_DOMAIN_SPLITS = {
    "retail": (
        OFFICIAL_SPLIT_COUNTS,
        OFFICIAL_SPLIT_SHA256,
        OFFICIAL_TRAIN_TASK_IDS,
    ),
    "airline": (
        OFFICIAL_AIRLINE_SPLIT_COUNTS,
        OFFICIAL_AIRLINE_SPLIT_SHA256,
        OFFICIAL_AIRLINE_TRAIN_TASK_IDS,
    ),
}


@dataclass(frozen=True)
class RetailTaskCatalog:
    """Validated Tau2 domain task IDs sourced from a pinned checkout.

    The historical name is retained for callers that only use Retail.
    """

    _task_ids_by_split: dict[SplitName, tuple[str, ...]]
    split_sha256: str
    domain: str = "retail"

    @classmethod
    def from_files(
        cls,
        tasks_path: Path,
        split_path: Path,
        *,
        domain: str = "retail",
    ) -> "RetailTaskCatalog":
        if domain not in _OFFICIAL_DOMAIN_SPLITS:
            raise ValueError(f"unsupported Tau2 domain: {domain!r}")
        tasks_data = _load_json(tasks_path, "tasks")
        split_data = _load_json(split_path, "split")
        task_ids = _task_ids(tasks_data, tasks_path)
        split_ids = _split_task_ids(split_data, split_path)

        missing_task_ids = set().union(*split_ids.values()) - task_ids
        if missing_task_ids:
            missing = ", ".join(sorted(missing_task_ids, key=_task_sort_key))
            raise ValueError(
                f"split file {split_path} references task IDs absent from "
                f"catalog {tasks_path}: {missing}"
            )

        train_ids = set(split_ids["train"])
        test_ids = set(split_ids["test"])
        base_ids = set(split_ids["base"])
        if not train_ids.isdisjoint(test_ids):
            overlap = ", ".join(sorted(train_ids & test_ids, key=_task_sort_key))
            raise ValueError(f"split file {split_path} overlaps train and test: {overlap}")
        if base_ids != train_ids | test_ids:
            raise ValueError(f"split file {split_path} must define base as train union test")

        fingerprint_data = json.dumps(
            split_data, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return cls(
            _task_ids_by_split=split_ids,
            split_sha256=hashlib.sha256(fingerprint_data).hexdigest(),
            domain=domain,
        )

    def task_ids(self, split: SplitName) -> tuple[str, ...]:
        try:
            return self._task_ids_by_split[split]
        except KeyError as error:
            raise ValueError(f"unknown {self.domain} split: {split!r}") from error

    def require_official_compatibility(self) -> None:
        expected_counts, expected_hash, expected_train_ids = _OFFICIAL_DOMAIN_SPLITS[
            self.domain
        ]
        actual_counts = {
            split: len(self._task_ids_by_split[split]) for split in _REQUIRED_SPLITS
        }
        if actual_counts != expected_counts:
            expected = ", ".join(
                f"{split}={expected_counts[split]}" for split in _REQUIRED_SPLITS
            )
            actual = ", ".join(
                f"{split}={actual_counts[split]}" for split in _REQUIRED_SPLITS
            )
            raise RuntimeError(
                f"Tau2 {self.domain} split count mismatch: expected {expected}; "
                f"resolved {actual}. "
                "Restore the official split_tasks.json."
            )
        if self.split_sha256 != expected_hash:
            raise RuntimeError(
                f"Tau2 {self.domain} split hash mismatch: expected "
                f"{expected_hash}, resolved {self.split_sha256}. "
                "Restore the official split_tasks.json."
            )
        if self._task_ids_by_split["train"] != expected_train_ids:
            raise RuntimeError(
                f"Tau2 {self.domain} train task IDs do not match the pinned official order. "
                "Restore the official split_tasks.json."
            )


Tau2TaskCatalog = RetailTaskCatalog


def _load_json(path: Path, label: str) -> Any:
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError as error:
        raise ValueError(f"{label} JSON file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} JSON file is invalid: {path}") from error


def _task_ids(tasks_data: Any, tasks_path: Path) -> set[str]:
    tasks = tasks_data.get("tasks") if isinstance(tasks_data, dict) else tasks_data
    if not isinstance(tasks, list):
        raise ValueError(f"tasks JSON file {tasks_path} must contain a task list")

    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError(f"tasks JSON file {tasks_path} contains a non-object task")
        task_id = task.get("id", task.get("task_id"))
        if not isinstance(task_id, (str, int)) or isinstance(task_id, bool):
            raise ValueError(f"tasks JSON file {tasks_path} contains a task without an ID")
        normalized_id = str(task_id)
        if normalized_id in task_ids:
            raise ValueError(f"tasks JSON file {tasks_path} contains duplicate task ID {normalized_id}")
        task_ids.add(normalized_id)
    return task_ids


def _split_task_ids(split_data: Any, split_path: Path) -> dict[SplitName, tuple[str, ...]]:
    if not isinstance(split_data, dict):
        raise ValueError(f"split JSON file {split_path} must contain an object")

    split_ids: dict[SplitName, tuple[str, ...]] = {}
    for split in _REQUIRED_SPLITS:
        raw_ids = split_data.get(split)
        if not isinstance(raw_ids, list):
            raise ValueError(f"split JSON file {split_path} must contain a {split!r} list")
        if not all(isinstance(task_id, str) for task_id in raw_ids):
            raise ValueError(f"split JSON file {split_path} has non-string {split!r} task IDs")
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"split JSON file {split_path} has duplicate {split!r} task IDs")
        split_ids[split] = tuple(raw_ids)
    return split_ids


def _task_sort_key(task_id: str) -> tuple[int, str]:
    return (int(task_id), task_id) if task_id.isdigit() else (0, task_id)
