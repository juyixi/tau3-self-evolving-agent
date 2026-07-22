from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_retail_evolver.pipeline.sampling import (
    assert_train_only_artifacts,
    balanced_kind_schedule,
    select_train_tasks,
)


TRAIN_IDS = tuple(str(index) for index in range(10))


def test_train_task_selection_is_deterministic_per_iteration() -> None:
    first = select_train_tasks(
        TRAIN_IDS,
        task_count=5,
        seed=42,
        iteration=3,
        shuffle=True,
    )
    repeated = select_train_tasks(
        TRAIN_IDS,
        task_count=5,
        seed=42,
        iteration=3,
        shuffle=True,
    )
    next_iteration = select_train_tasks(
        TRAIN_IDS,
        task_count=5,
        seed=42,
        iteration=4,
        shuffle=True,
    )

    assert first == repeated
    assert len(first) == 5
    assert len(set(first)) == 5
    assert next_iteration != first
    assert set(first) <= set(TRAIN_IDS)


def test_explicit_task_selection_preserves_order_and_rejects_non_train_ids() -> None:
    assert select_train_tasks(
        TRAIN_IDS,
        task_count=10,
        seed=42,
        iteration=0,
        shuffle=True,
        explicit_task_ids=("3", "1"),
    ) == ("3", "1")

    with pytest.raises(ValueError, match="official train split"):
        select_train_tasks(
            TRAIN_IDS,
            task_count=10,
            seed=42,
            iteration=0,
            shuffle=True,
            explicit_task_ids=("3", "test-1"),
        )


@pytest.mark.parametrize("task_count", (0, 11))
def test_task_count_must_fit_the_train_split(task_count: int) -> None:
    with pytest.raises(ValueError, match="task_count"):
        select_train_tasks(
            TRAIN_IDS,
            task_count=task_count,
            seed=42,
            iteration=0,
            shuffle=False,
        )


def test_balanced_kind_schedule_cycles_only_existing_examples() -> None:
    schedule = balanced_kind_schedule(
        {"sel": 2, "act": 1, "write": 0, "maint": 3},
        num_epochs=2,
    )

    assert [sample.kind for sample in schedule[:3]] == ["sel", "act", "maint"]
    assert len(schedule) == 18
    assert all(sample.kind != "write" for sample in schedule)
    assert {sample.index for sample in schedule if sample.kind == "act"} == {0}
    assert {sample.index for sample in schedule if sample.kind == "maint"} == {0, 1, 2}


def test_artifact_scan_rejects_test_task_ids(tmp_path: Path) -> None:
    artifact = tmp_path / "manifest.json"
    artifact.write_text(
        json.dumps({"task_ids": ["0", "test-1"]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-train task ID"):
        assert_train_only_artifacts(tmp_path, train_task_ids=TRAIN_IDS)


def test_artifact_scan_accepts_train_ids_in_json_and_jsonl(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"task_ids": ["0", "1"]}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"task_id": "2", "source_task_ids": ["3"]}) + "\n",
        encoding="utf-8",
    )

    assert_train_only_artifacts(tmp_path, train_task_ids=TRAIN_IDS)
