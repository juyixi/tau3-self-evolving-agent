from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tau3_retail_evolver.slow_loop.source_runs import load_source_runs


GROUP = f"retail-actions-v1:{'a' * 64}"


def _write_source_run(
    root: Path,
    run_id: str,
    *,
    task_id: str,
    before: int,
    input_snapshot: str,
    output_snapshot: str,
    adapter_revision: str | None = "adapter-a",
) -> Path:
    run_path = root / run_id
    events_path = run_path / "rollouts" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 3,
        "model_revision": "model-a",
        "adapter_revision": adapter_revision,
        "memory_snapshot_id": input_snapshot,
        "tau2_commit": "c" * 40,
        "split": "train",
        "split_hash": "d" * 64,
        "task_ids": [task_id],
        "seed": 17,
        "environment_options": {"domain": "retail"},
        "rollout_options": {
            "memory_enabled": True,
            "memory_agent_id": "retail",
        },
    }
    summary = {
        "run_id": run_id,
        "episode_count": 1,
        "completed_train_tasks_before": before,
        "completed_train_tasks_after": before + 1,
        "input_memory_snapshot_id": input_snapshot,
        "output_memory_snapshot_id": output_snapshot,
        "memory_enabled": True,
        "successful_task_ids": [task_id],
        "maintenance_rounds_executed": [],
        "total_terminal_reward": 1.0,
    }
    common = {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 3,
        "split": "train",
        "mode": "learn",
        "task_id": task_id,
        "task_group": GROUP,
        "model_revision": "model-a",
        "adapter_revision": adapter_revision,
        "memory_snapshot_id": input_snapshot,
        "seed": 17,
    }
    events = [
        {**common, "event_type": "EpisodeStarted"},
        {**common, "event_type": "MemoryCandidatesRetrieved", "candidates": []},
        {**common, "event_type": "MemorySelected", "selected_memory_ids": []},
        {**common, "event_type": "DecisionMade", "turn": 0},
        {**common, "event_type": "EnvironmentStepped", "turn": 0},
        {
            **common,
            "event_type": "EpisodeFinished",
            "steps": 1,
            "final_reward": 1.0,
        },
        {**common, "event_type": "MemoryWriteProposed", "proposals": []},
        {
            **common,
            "event_type": "MemoryWriteCommitted",
            "written_memory_ids": [],
            "replayed_memory_ids": [],
        },
    ]
    (run_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_path / "fast_loop_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    events_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8"
    )
    return run_path


def _memory_root(tmp_path: Path, *snapshot_ids: str) -> Path:
    root = tmp_path / "history" / "agents" / "retail" / "memory"
    for snapshot_id in snapshot_ids:
        (root / "snapshots" / snapshot_id).mkdir(parents=True)
    return root


def _catalog(*task_ids: str) -> SimpleNamespace:
    return SimpleNamespace(task_ids=lambda split: task_ids, split_sha256="d" * 64)


def _rewrite(path: Path, mutator: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutator(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_runs_require_same_policy_and_continuous_snapshots(
    tmp_path: Path,
) -> None:
    first = _write_source_run(
        tmp_path, "run-a", task_id="1", before=0, input_snapshot="s0", output_snapshot="s1"
    )
    second = _write_source_run(
        tmp_path, "run-b", task_id="2", before=1, input_snapshot="s1", output_snapshot="s2"
    )

    loaded = load_source_runs(
        [second, first],
        catalog=_catalog("1", "2"),
        memory_root=_memory_root(tmp_path, "s0", "s1", "s2"),
    )

    assert [run.run_id for run in loaded.runs] == ["run-a", "run-b"]
    assert loaded.iteration == 3
    assert loaded.model_revision == "model-a"
    assert loaded.adapter_revision == "adapter-a"
    assert loaded.memory_agent_id == "retail"
    assert loaded.runs[0].manifest_sha256 != loaded.runs[0].events_sha256
    with pytest.raises(TypeError):
        loaded.runs[0].manifest["iteration"] = 4  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("test_split", "train split"),
        ("memory_disabled", "memory-enabled"),
        ("schema_1_event", "event schema"),
        ("failure_event", "failure event"),
        ("incomplete_lifecycle", "lifecycle"),
        ("provenance_mismatch", "event provenance"),
        ("snapshot_mismatch", "episode snapshot"),
        ("invalid_event_type", "event type"),
    ],
)
def test_source_run_fails_closed_on_invalid_artifacts(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    run_path = _write_source_run(
        tmp_path, "run-a", task_id="1", before=0, input_snapshot="s0", output_snapshot="s1"
    )
    manifest_path = run_path / "manifest.json"
    events_path = run_path / "rollouts" / "events.jsonl"
    if mutation == "test_split":
        _rewrite(manifest_path, lambda payload: payload.update(split="test"))
    elif mutation == "memory_disabled":
        _rewrite(
            manifest_path,
            lambda payload: payload["rollout_options"].update(memory_enabled=False),
        )
    else:
        events = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]
        if mutation == "schema_1_event":
            events[0]["schema_version"] = 1
        elif mutation == "failure_event":
            events.append({**events[0], "event_type": "EpisodeFailed"})
        elif mutation == "incomplete_lifecycle":
            events = [event for event in events if event["event_type"] != "MemorySelected"]
        elif mutation == "provenance_mismatch":
            events[0]["model_revision"] = "other-model"
        elif mutation == "snapshot_mismatch":
            events[1]["memory_snapshot_id"] = "s1"
        elif mutation == "invalid_event_type":
            events[0]["event_type"] = ["EpisodeStarted"]
        events_path.write_text(
            "".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8"
        )

    with pytest.raises(ValueError, match=message):
        load_source_runs(
            [run_path],
            catalog=_catalog("1"),
            memory_root=_memory_root(tmp_path, "s0", "s1"),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("adapter_mismatch", "policy lineage"),
        ("snapshot_gap", "snapshot continuity"),
        ("task_range_gap", "task range continuity"),
    ],
)
def test_source_run_set_fails_closed_on_invalid_lineage(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    first = _write_source_run(
        tmp_path, "run-a", task_id="1", before=0, input_snapshot="s0", output_snapshot="s1"
    )
    second_before = 2 if mutation == "task_range_gap" else 1
    second_input = "sx" if mutation == "snapshot_gap" else "s1"
    second_adapter = "adapter-b" if mutation == "adapter_mismatch" else "adapter-a"
    second = _write_source_run(
        tmp_path,
        "run-b",
        task_id="2",
        before=second_before,
        input_snapshot=second_input,
        output_snapshot="s2",
        adapter_revision=second_adapter,
    )

    with pytest.raises(ValueError, match=message):
        load_source_runs(
            [first, second],
            catalog=_catalog("1", "2"),
            memory_root=_memory_root(tmp_path, "s0", "s1", "s2", "sx"),
        )


def test_source_runs_allow_the_same_task_in_distinct_on_policy_passes(
    tmp_path: Path,
) -> None:
    first = _write_source_run(
        tmp_path,
        "run-pass-1",
        task_id="1",
        before=0,
        input_snapshot="s0",
        output_snapshot="s1",
    )
    second = _write_source_run(
        tmp_path,
        "run-pass-2",
        task_id="1",
        before=1,
        input_snapshot="s1",
        output_snapshot="s2",
    )

    loaded = load_source_runs(
        [first, second],
        catalog=_catalog("1"),
        memory_root=_memory_root(tmp_path, "s0", "s1", "s2"),
    )

    assert [run.run_id for run in loaded.runs] == ["run-pass-1", "run-pass-2"]
    assert [run.manifest["task_ids"] for run in loaded.runs] == [("1",), ("1",)]


def test_source_runs_reject_evaluation_quarantine_paths(tmp_path: Path) -> None:
    run_path = _write_source_run(
        tmp_path / "history" / "evaluations",
        "run-a",
        task_id="1",
        before=0,
        input_snapshot="s0",
        output_snapshot="s1",
    )

    with pytest.raises(ValueError, match="evaluation quarantine"):
        load_source_runs(
            [run_path],
            catalog=_catalog("1"),
            memory_root=_memory_root(tmp_path, "s0", "s1"),
        )
