from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tau3_retail_evolver.slow_loop.canonicalize import (
    CanonicalizeRequest,
    canonicalize_opd_sources,
)
from tau3_retail_evolver.slow_loop.source_runs import load_source_runs


def test_canonicalizer_deduplicates_fragments_and_preserves_failed_task_markers(
    tmp_path: Path,
) -> None:
    run_a = _write_run(
        tmp_path,
        "run-a",
        seed=42,
        task_ids=["1", "2"],
        events=[
            *_episode("run-a", seed=42, task_id="1", snapshot="s0", reward=1.0),
            *_failed_episode("run-a", seed=42, task_id="2", snapshot="s0"),
        ],
        manifest_snapshot="s0",
    )
    run_a_retry = _write_run(
        tmp_path,
        "run-a-retry",
        seed=42,
        task_ids=["2"],
        events=_episode(
            "run-a-retry",
            seed=42,
            task_id="2",
            snapshot="s1",
            reward=0.0,
        ),
        manifest_snapshot="s1",
    )
    run_b = _write_run(
        tmp_path,
        "run-b",
        seed=43,
        task_ids=["3", "4"],
        events=[
            *_episode("run-b", seed=43, task_id="3", snapshot="s2", reward=1.0),
            _task_failed("run-b", seed=43, task_id="4", snapshot="s2"),
        ],
        manifest_snapshot="s2",
    )
    maintenance_failed = _write_event_file(
        tmp_path / "maintenance-failed" / "rollouts" / "events.jsonl",
        [
            _maintenance_common(
                "maintenance-failed",
                event_type="MaintenanceTaskFailed",
                round_number=1,
                snapshot="sm1",
                seed=42,
            )
        ],
    )
    maintenance_round_1 = _write_event_file(
        tmp_path / "maintenance-1" / "rollouts" / "events.jsonl",
        _maintenance("maintenance-1", round_number=1, snapshot="sm1", seed=42),
    )
    maintenance_round_2 = _write_event_file(
        tmp_path / "maintenance-2" / "rollouts" / "events.jsonl",
        _maintenance("maintenance-2", round_number=2, snapshot="sm2", seed=43),
    )
    memory_root = _memory_root(tmp_path, "s0", "s1", "s2", "sf", "sm1", "sm2")
    catalog = _catalog("1", "2", "3", "4")

    result = canonicalize_opd_sources(
        CanonicalizeRequest(
            source_run_paths=(run_a, run_a_retry, run_b),
            maintenance_event_paths=(
                maintenance_failed,
                maintenance_round_1,
                maintenance_round_2,
            ),
            output_root=tmp_path / "canonical",
            build_id="opd-source",
            final_memory_snapshot_id="sf",
            maintenance_period=2,
            expected_seeds=(42, 43),
            catalog=catalog,
            memory_root=memory_root,
            deep_validate=False,
        )
    )

    assert result.index["coverage"] == {
        "logical_task_count": 4,
        "included_episode_count": 3,
        "excluded_failure_count": 1,
        "duplicate_logical_key_count": 1,
        "selected_maintenance_count": 2,
        "failed_maintenance_candidate_count": 1,
    }
    assert result.index["validation"]["logical_keys_unique"] is True
    assert result.index["validation"]["test_leakage_detected"] is False
    assert [path.name for path in result.source_run_paths] == [
        "run-a",
        "run-a-retry",
        "run-b",
    ]

    loaded = load_source_runs(
        result.source_run_paths,
        catalog=catalog,
        memory_root=memory_root,
    )
    assert [run.summary["completed_train_tasks_before"] for run in loaded.runs] == [
        0,
        1,
        2,
    ]
    assert [run.summary["completed_train_tasks_after"] for run in loaded.runs] == [
        1,
        2,
        4,
    ]
    assert loaded.runs[-1].summary["failed_task_ids"] == ("4",)
    assert loaded.runs[-1].summary["episode_count"] == 1

    retry_events = _read_jsonl(
        result.root / "run-a-retry" / "rollouts" / "events.jsonl"
    )
    maintenance_started = next(
        event
        for event in retry_events
        if event["event_type"] == "MaintenanceStarted"
    )
    assert maintenance_started["completed_train_tasks"] == 2
    assert maintenance_started["period"] == 2
    assert maintenance_started["run_id"] == "run-a-retry"

    failed_event = _read_jsonl(
        result.root / "run-b" / "rollouts" / "events.jsonl"
    )[-4]
    assert failed_event["event_type"] == "TaskFailed"
    assert failed_event["canonical_exclusion_reason"] == "task_failed"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        canonicalize_opd_sources(
            CanonicalizeRequest(
                source_run_paths=(run_a, run_a_retry, run_b),
                maintenance_event_paths=(maintenance_round_1, maintenance_round_2),
                output_root=tmp_path / "canonical",
                build_id="opd-source",
                final_memory_snapshot_id="sf",
                maintenance_period=2,
                expected_seeds=(42, 43),
                catalog=catalog,
                memory_root=memory_root,
                deep_validate=False,
            )
        )


def _write_run(
    root: Path,
    run_id: str,
    *,
    seed: int,
    task_ids: list[str],
    events: list[dict[str, Any]],
    manifest_snapshot: str,
) -> Path:
    run = root / run_id
    _write_event_file(run / "rollouts" / "events.jsonl", events)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 0,
        "model_revision": "model-a",
        "adapter_revision": "adapter-a",
        "memory_snapshot_id": manifest_snapshot,
        "tau2_commit": "c" * 40,
        "split": "train",
        "split_hash": "d" * 64,
        "task_ids": task_ids,
        "seed": seed,
        "environment_options": {"domain": "retail"},
        "rollout_options": {
            "memory_enabled": True,
            "memory_agent_id": "retail",
            "task_order_seed": seed,
        },
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def _episode(
    run_id: str,
    *,
    seed: int,
    task_id: str,
    snapshot: str,
    reward: float,
) -> list[dict[str, Any]]:
    common = _event_common(
        run_id,
        seed=seed,
        task_id=task_id,
        snapshot=snapshot,
    )
    return [
        {**common, "event_type": "EpisodeStarted"},
        {**common, "event_type": "MemoryCandidatesRetrieved", "candidates": []},
        {
            **common,
            "event_type": "MemorySelected",
            "selected": [],
            "selected_memory_ids": [],
        },
        {**common, "event_type": "DecisionMade", "turn": 0},
        {**common, "event_type": "EnvironmentStepped", "turn": 0},
        {
            **common,
            "event_type": "EpisodeFinished",
            "steps": 1,
            "final_reward": reward,
        },
        {**common, "event_type": "MemoryWriteProposed", "proposals": []},
        {
            **common,
            "event_type": "MemoryWriteCommitted",
            "written_memory_ids": [],
            "replayed_memory_ids": [],
        },
    ]


def _failed_episode(
    run_id: str,
    *,
    seed: int,
    task_id: str,
    snapshot: str,
) -> list[dict[str, Any]]:
    common = _event_common(
        run_id,
        seed=seed,
        task_id=task_id,
        snapshot=snapshot,
    )
    return [
        {**common, "event_type": "EpisodeStarted"},
        {**common, "event_type": "MemoryCandidatesRetrieved", "candidates": []},
        {
            **common,
            "event_type": "EpisodeFailed",
            "error": "tokenizer unavailable",
        },
    ]


def _task_failed(
    run_id: str,
    *,
    seed: int,
    task_id: str,
    snapshot: str,
) -> dict[str, Any]:
    return {
        **_event_common(
            run_id,
            seed=seed,
            task_id=task_id,
            snapshot=snapshot,
        ),
        "event_type": "TaskFailed",
        "error": "task failed",
    }


def _event_common(
    run_id: str,
    *,
    seed: int,
    task_id: str,
    snapshot: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 0,
        "split": "train",
        "mode": "learn",
        "task_id": task_id,
        "task_group": "retail-v2",
        "model_revision": "model-a",
        "adapter_revision": "adapter-a",
        "memory_snapshot_id": snapshot,
        "seed": seed,
    }


def _maintenance(
    run_id: str,
    *,
    round_number: int,
    snapshot: str,
    seed: int,
) -> list[dict[str, Any]]:
    return [
        _maintenance_common(
            run_id,
            event_type="MaintenanceStarted",
            round_number=round_number,
            snapshot=snapshot,
            seed=seed,
        ),
        _maintenance_common(
            run_id,
            event_type="MaintenanceProposed",
            round_number=round_number,
            snapshot=snapshot,
            seed=seed,
        ),
        _maintenance_common(
            run_id,
            event_type="MaintenanceCommitted",
            round_number=round_number,
            snapshot=snapshot,
            seed=seed,
        ),
    ]


def _maintenance_common(
    run_id: str,
    *,
    event_type: str,
    round_number: int,
    snapshot: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 0,
        "split": "train",
        "mode": "learn",
        "task_id": f"maintenance-round-{round_number}",
        "task_group": "retail-v2:maintenance",
        "model_revision": "model-a",
        "adapter_revision": "adapter-a",
        "memory_snapshot_id": snapshot,
        "seed": seed,
        "event_type": event_type,
        "maintenance_round": round_number,
    }


def _write_event_file(path: Path, events: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _memory_root(tmp_path: Path, *snapshot_ids: str) -> Path:
    root = tmp_path / "history" / "agents" / "retail" / "memory"
    for snapshot_id in snapshot_ids:
        (root / "snapshots" / snapshot_id).mkdir(parents=True)
    return root


def _catalog(*task_ids: str) -> SimpleNamespace:
    return SimpleNamespace(task_ids=lambda split: task_ids, split_sha256="d" * 64)
