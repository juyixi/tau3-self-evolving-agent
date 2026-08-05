from __future__ import annotations

from pathlib import Path

import pytest

from tau3_evolver.artifacts.jsonl import JsonlWriter
from tau3_evolver.artifacts.maintenance import build_completed_maintenance
from tau3_evolver.artifacts.run import episode_artifact_metadata, write_run_record
from tau3_evolver.memory.paths import training_memory_root
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.slow_loop.evidence import build_evidence
from tau3_evolver.slow_loop.source_runs import load_source_runs


def _source_run(
    root: Path,
    *,
    benchmark: str = "retail",
    run_id: str = "run-1",
    task_scope: str = "full",
    with_maintenance: bool = False,
) -> Path:
    namespace = f"{benchmark}-debug" if task_scope == "debug" else benchmark
    memory = MemoryRepository(training_memory_root(namespace, root=root))
    input_snapshot = memory.snapshot().memory_snapshot_id
    output_snapshot = memory.snapshot().memory_snapshot_id
    maintenance = None
    if with_maintenance:
        common = {
            "task_id": "maintenance-round-1",
            "memory_snapshot_id": output_snapshot,
            "maintenance_round": 1,
        }
        record = build_completed_maintenance(
            (
                {
                    **common,
                    "event_type": "MaintenanceStarted",
                    "completed_train_tasks": 1,
                    "period": 1,
                    "diagnostics": {
                        tier: {"items": []}
                        for tier in ("trajectory", "tip", "skill", "tool")
                    },
                },
                {
                    **common,
                    "event_type": "MaintenanceProposed",
                    "commands": [],
                },
                {
                    **common,
                    "event_type": "MaintenanceCommitted",
                    "looked_up_ids": [],
                    "created_ids": [],
                    "updated_ids": [],
                },
            )
        )
        maintenance = {
            "period": 1,
            "completed_train_tasks_before": 0,
            "completed_train_tasks_after": 1,
            "records": [record],
            "failures": [],
        }
    run = root / "runs" / run_id
    run.mkdir(parents=True)
    episodes = run / "episodes.jsonl"
    JsonlWriter(episodes).append(
        {
        "schema_version": 1,
            "task_id": "1",
            "task_group": benchmark,
            "seed": 7,
            "status": "completed",
            "task": {
                "initial_observation": "start",
                "policy": {},
                "tools": [],
            },
            "trajectory": [
                {
                    "turn": 0,
                    "observation": "start",
                    "action": "done",
                    "next_observation": "done",
                    "reward": 1.0,
                    "done": True,
                    "terminated": True,
                    "truncated": False,
                    "public_info": {},
                    "decision": {},
                }
            ],
            "outcome": {
                "final_reward": 1.0,
                "steps": 1,
                "terminal_evaluation": {},
                "truncated": False,
                "project_truncated": False,
                "parse_error_count": 0,
                "response_parse_error_count": 0,
                "response_count": 1,
                "agent_prompt_tokens": None,
                "agent_completion_tokens": None,
            },
            "memory": {
                "enabled": True,
                "retrieval": {
                    "query_hash": "a" * 64,
                    "retriever_revision": "embed",
                    "candidates": [],
                },
                "selected_memory_ids": [],
                "selection": {},
                "writes": [],
                "write_audit": {},
            },
        }
    )
    write_run_record(
        run / "run.json",
        {
            "run_id": run_id,
            "status": "completed",
            "execution": {
                "benchmark": benchmark,
                "mode": "train",
                "split": "train",
                "split_hash": "split-hash",
                "task_scope": task_scope,
                "planned_task_count": 1,
            },
            "runtime": {
                "source_root": "C:/tau2",
                "package_version": "0.1",
                "git_commit": None,
            },
            "policy": {
                "model_revision": "Qwen/Qwen3.5-9B",
                "checkpoint": None,
            },
            "memory": {
                "enabled": True,
                "generation": 1,
                "source_namespace": namespace,
                "destination_namespace": namespace,
                "input_snapshot_id": input_snapshot,
                "output_snapshot_id": output_snapshot,
                "cross_domain": False,
                "maintenance": maintenance,
            },
            "config": {},
            "summary": {
                "metrics": {
                    "task_count": 1,
                    "completed_count": 1,
                    "failure_count": 0,
                    "mean_reward": 1.0,
                    "pass_rate": 1.0,
                }
            },
            "artifacts": {"episodes": episode_artifact_metadata(episodes)},
        },
    )
    return run


@pytest.mark.parametrize("benchmark", ("retail", "airline"))
def test_loads_generic_two_file_source_run(tmp_path: Path, benchmark: str) -> None:
    run = _source_run(tmp_path, benchmark=benchmark)

    loaded = load_source_runs(
        (run,),
        benchmark=benchmark,
        official_train_task_ids=("1", "2"),
        split_hash="split-hash",
        project_root=tmp_path,
    )

    assert loaded.benchmark == benchmark
    assert loaded.memory_generation == 1
    assert loaded.memory_namespace == benchmark
    assert loaded.adapter_revision == "zero-impact-init-v1"
    assert not hasattr(loaded, "iteration")
    assert loaded.runs[0].summary["episode_count"] == 1
    evidence = build_evidence(
        loaded,
        memory_root=training_memory_root(benchmark, root=tmp_path),
    )
    assert evidence.episodes[0].source_episode_row == 1
    assert len(evidence.episodes[0].source_episode_sha256) == 64


def test_builds_maintainer_evidence_from_compressed_run_record(
    tmp_path: Path,
) -> None:
    run = _source_run(tmp_path, with_maintenance=True)
    loaded = load_source_runs(
        (run,),
        benchmark="retail",
        official_train_task_ids=("1",),
        split_hash="split-hash",
        project_root=tmp_path,
    )

    evidence = build_evidence(
        loaded,
        memory_root=training_memory_root("retail", root=tmp_path),
    )

    assert loaded.runs[0].summary["maintenance_rounds_executed"] == (1,)
    assert len(evidence.maintenance) == 1
    assert evidence.maintenance[0].source_record_index == 1
    assert evidence.maintenance[0].trigger_task_index == 1


def test_rejects_source_for_a_different_benchmark(tmp_path: Path) -> None:
    run = _source_run(tmp_path, benchmark="retail")

    with pytest.raises(ValueError, match="benchmark"):
        load_source_runs(
            (run,),
            benchmark="airline",
            official_train_task_ids=("1",),
            split_hash="split-hash",
            project_root=tmp_path,
        )


def test_rejects_legacy_or_extra_run_artifacts(tmp_path: Path) -> None:
    run = _source_run(tmp_path)
    (run / "results.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly run.json and episodes.jsonl"):
        load_source_runs(
            (run,),
            benchmark="retail",
            official_train_task_ids=("1",),
            split_hash="split-hash",
            project_root=tmp_path,
        )


def test_rejects_debug_run_as_slow_loop_source(tmp_path: Path) -> None:
    run = _source_run(tmp_path, task_scope="debug")

    with pytest.raises(ValueError, match="debug runs cannot be used"):
        load_source_runs(
            (run,),
            benchmark="retail",
            official_train_task_ids=("1",),
            split_hash="split-hash",
            project_root=tmp_path,
        )


def test_explicit_debug_scope_loads_only_debug_memory(tmp_path: Path) -> None:
    run = _source_run(tmp_path, task_scope="debug")

    loaded = load_source_runs(
        (run,),
        benchmark="retail",
        official_train_task_ids=("1",),
        split_hash="split-hash",
        project_root=tmp_path,
        task_scope="debug",
    )

    assert loaded.task_scope == "debug"
    assert loaded.memory_namespace == "retail-debug"
