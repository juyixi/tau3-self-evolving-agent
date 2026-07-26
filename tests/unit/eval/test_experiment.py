from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from tau3_retail_evolver.eval.experiment import (
    BASE_NO_MEMORY,
    BASE_WITH_MEMORY,
    OPD_NO_MEMORY,
    OPD_WITH_MEMORY,
    build_stage8_experiment_report,
)
from tau3_retail_evolver.eval.guard import EvaluationProtocol
from tau3_retail_evolver.eval.metrics import (
    EvaluationProvenance,
    build_evaluation_report,
)
from tau3_retail_evolver.eval.runner import EvaluationRunResult, TrialEpisode
from tau3_retail_evolver.eval.visualization import render_stage8_dashboard
from tau3_retail_evolver.fast_loop.runner import EpisodeResult
from tau3_retail_evolver.memory.repository import MemoryRepository


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _memory_chain(tmp_path: Path) -> tuple[list[Any], list[str]]:
    repository = MemoryRepository(
        tmp_path / "history" / "agents" / "retail" / "memory"
    )
    snapshots = [repository.snapshot()]
    memory_ids = []
    for index in range(3):
        item = repository.add(
            tier="tip",
            content=f"Reusable retail lesson {index}",
            source_task_ids=(str(index),),
            created_round=index,
        )
        memory_ids.append(item.id)
        snapshots.append(repository.snapshot())
    return snapshots, memory_ids


def _train_runs(
    tmp_path: Path,
    snapshots: list[Any],
    memory_ids: list[str],
) -> list[Path]:
    task_ids = [str(index) for index in range(74)]
    paths = []
    for index in range(3):
        run_id = f"stage8-train-pass-{index + 1}"
        root = tmp_path / "runs" / run_id
        ordered_task_ids = task_ids[index:] + task_ids[:index]
        paths.append(root)
        _write_json(
            root / "manifest.json",
            {
                "run_id": run_id,
                "iteration": 0,
                "split": "train",
                "task_ids": ordered_task_ids,
                "seed": 42 + index,
                "model_revision": "qwen-sha",
                "adapter_revision": None,
                "tau2_commit": "tau2-sha",
                "split_hash": "split-sha",
                "memory_snapshot_id": snapshots[index].memory_snapshot_id,
            },
        )
        _write_json(
            root / "fast_loop_summary.json",
            {
                "run_id": run_id,
                "episode_count": 74,
                "memory_enabled": True,
                "completed_train_tasks_before": index * 74,
                "completed_train_tasks_after": (index + 1) * 74,
                "input_memory_snapshot_id": snapshots[
                    index
                ].memory_snapshot_id,
                "output_memory_snapshot_id": snapshots[
                    index + 1
                ].memory_snapshot_id,
                "input_memory_counts": snapshots[index].counts,
                "output_memory_counts": snapshots[index + 1].counts,
                "maintenance_rounds_executed": [index * 2 + 1, index * 2 + 2],
                "token_usage_episode_count": 74,
                "mean_agent_tokens": 100 + index * 10,
            },
        )
        events = []
        for task_index, task_id in enumerate(ordered_task_ids):
            events.append(
                {
                    "event_type": "EpisodeStarted",
                    "task_id": task_id,
                    "memory_snapshot_id": snapshots[index].memory_snapshot_id,
                }
            )
            if index and task_index == 0:
                events.append(
                    {
                        "event_type": "MemorySelected",
                        "selected_memory_ids": memory_ids[:index],
                    }
                )
            events.append(
                {
                    "event_type": "EpisodeFinished",
                    "task_id": task_id,
                    "final_reward": (
                        1.0 if int(task_id) % 2 == 0 else 0.0
                    ),
                }
            )
        events.append(
            {
                "event_type": "MemoryWriteCommitted",
                "written_memory_ids": [memory_ids[index]],
            }
        )
        _write_jsonl(root / "rollouts" / "events.jsonl", events)
    return paths


def _dataset(
    tmp_path: Path,
    run_dirs: list[Path],
    snapshots: list[Any],
) -> Path:
    root = tmp_path / "datasets" / "stage8"
    _write_json(
        root / "dataset_manifest.json",
        {
            "dataset_build_id": "stage8-dataset",
            "counts": {
                "evidence_episodes": 222,
                "evidence_maintenance": 6,
                "memory_scores": 3,
                "sel": 120,
                "act": 240,
                "write": 60,
                "maint": 6,
            },
            "skip_reasons": {"insufficient_selected_control": 1},
            "source_runs": [
                {"run_id": path.name} for path in run_dirs
            ],
            "policy_lineage": {
                "iteration": 0,
                "model_revision": "qwen-sha",
                "adapter_revision": None,
                "tau2_commit": "tau2-sha",
            },
            "official_split": {
                "name": "train",
                "sha256": "split-sha",
            },
            "memory": {
                "snapshot_chain": [
                    snapshot.memory_snapshot_id for snapshot in snapshots
                ]
            },
        },
    )
    _write_json(
        root / "audit_report.json",
        {
            "audit_schema_version": 1,
            "dataset_build_id": "stage8-dataset",
            "passed": True,
            "checked_artifacts": [],
            "errors": [],
        },
    )
    return root


def _training(tmp_path: Path) -> Path:
    root = tmp_path / "training"
    checkpoint = root / "checkpoints" / "step-00000010"
    metric_rows = []
    for kind, count, value in (
        ("sel", 120, 0.4),
        ("act", 240, 0.3),
        ("write", 60, 0.2),
        ("maint", 6, 0.1),
    ):
        for _ in range(count):
            metric_rows.append(
                {
                    "sequence_index": len(metric_rows),
                    "epoch": 0,
                    "kind": kind,
                    "metrics": {"forward_kl": value},
                }
            )
    source_lineage = {
        "model_revision": "qwen-sha",
        "adapter_revision": None,
    }
    _write_json(
        root / "training_manifest.json",
        {
            "status": "complete",
            "latest_checkpoint": "checkpoints/step-00000010",
            "adapter_revision": "adapter-opd",
            "dataset_build_id": "stage8-dataset",
            "completed_examples": 426,
            "total_examples": 426,
            "optimizer_steps": 10,
            "training_config": {
                "loss_type": "forward_kl",
                "num_train_epochs": 1,
            },
            "source_lineage": source_lineage,
        },
    )
    _write_json(
        checkpoint / "checkpoint_manifest.json",
        {
            "status": "checkpoint",
            "adapter_revision": "adapter-opd",
            "dataset_build_id": "stage8-dataset",
            "source_lineage": source_lineage,
        },
    )
    _write_jsonl(
        root / "training_metrics.jsonl",
        metric_rows,
    )
    _write_jsonl(
        root / "training_generations.jsonl",
        [{"response_ids": [1, 2]} for _ in metric_rows],
    )
    return root


def _evaluation_report(
    *,
    label: str,
    rewards: tuple[float, ...],
    tokens: int,
    snapshot: Any,
    memory_ids: list[str],
    checkpoint_path: Path,
) -> dict[str, Any]:
    trained = label in {OPD_WITH_MEMORY, OPD_NO_MEMORY}
    with_memory = label in {BASE_WITH_MEMORY, OPD_WITH_MEMORY}
    protocol = (
        EvaluationProtocol.TEST_STATIC
        if with_memory
        else EvaluationProtocol.NO_MEMORY
    )
    provenance = EvaluationProvenance(
        run_id=f"eval-{label}",
        protocol=protocol,
        official_base_reproduction=False,
        split="test",
        checkpoint=(
            str(checkpoint_path)
            if trained
            else None
        ),
        base_model="Qwen/Qwen3.5-9B",
        model_revision="qwen-sha",
        adapter_revision="adapter-opd" if trained else None,
        tau2_commit="tau2-sha",
        split_hash="test-split-sha",
        task_ids=("75", "76"),
        seeds=(42, 43),
        user_simulator_config={"model": "deepseek/deepseek-v4-pro"},
        nl_evaluator={"model": "openrouter/openai/gpt-4.1"},
        memory_snapshot_id=(
            snapshot.memory_snapshot_id if with_memory else None
        ),
        max_episode_steps=40,
        model_serving_contract={"max_tokens": 8192},
        capabilities={"memory_write": False},
        memory_counts=(
            snapshot.counts
            if with_memory
            else {
                "trajectory": 0,
                "tip": 0,
                "skill": 0,
                "tool": 0,
            }
        ),
    )
    task_ids = ("75", "76", "75", "76")
    episodes = tuple(
        TrialEpisode(
            trial_index=index // 2,
            seed=(42, 43)[index // 2],
            result=EpisodeResult(
                task_id=task_id,
                final_reward=reward,
                steps=2,
                terminal_evaluation={
                    "reward": reward,
                    "reward_breakdown": {"DB": reward},
                },
                simulation_result={"termination_reason": "agent_stop"},
                selected_memory_ids=(
                    tuple(memory_ids[:2]) if with_memory else ()
                ),
                written_memory_ids=(),
                truncated=False,
                agent_prompt_tokens=tokens - 20,
                agent_completion_tokens=20,
            ),
        )
        for index, (task_id, reward) in enumerate(
            zip(task_ids, rewards, strict=True)
        )
    )
    return build_evaluation_report(
        provenance,
        EvaluationRunResult(
            episodes=episodes,
            maintenance_rounds_by_trial=((), ()),
            output_memory_snapshot_ids=(
                (
                    snapshot.memory_snapshot_id,
                    snapshot.memory_snapshot_id,
                )
                if with_memory
                else (None, None)
            ),
        ),
    )


def _artifacts(tmp_path: Path) -> dict[str, Any]:
    snapshots, memory_ids = _memory_chain(tmp_path)
    runs = _train_runs(tmp_path, snapshots, memory_ids)
    dataset = _dataset(tmp_path, runs, snapshots)
    training = _training(tmp_path)
    checkpoint = training / "checkpoints" / "step-00000010"
    reports = {
        BASE_NO_MEMORY: _evaluation_report(
            label=BASE_NO_MEMORY,
            rewards=(0.0, 0.0, 1.0, 0.0),
            tokens=100,
            snapshot=snapshots[-1],
            memory_ids=memory_ids,
            checkpoint_path=checkpoint,
        ),
        BASE_WITH_MEMORY: _evaluation_report(
            label=BASE_WITH_MEMORY,
            rewards=(1.0, 0.0, 1.0, 0.0),
            tokens=120,
            snapshot=snapshots[-1],
            memory_ids=memory_ids,
            checkpoint_path=checkpoint,
        ),
        OPD_WITH_MEMORY: _evaluation_report(
            label=OPD_WITH_MEMORY,
            rewards=(1.0, 1.0, 1.0, 0.0),
            tokens=110,
            snapshot=snapshots[-1],
            memory_ids=memory_ids,
            checkpoint_path=checkpoint,
        ),
        OPD_NO_MEMORY: _evaluation_report(
            label=OPD_NO_MEMORY,
            rewards=(1.0, 0.0, 1.0, 0.0),
            tokens=90,
            snapshot=snapshots[-1],
            memory_ids=memory_ids,
            checkpoint_path=checkpoint,
        ),
    }
    return {
        "snapshot": snapshots[-1],
        "runs": runs,
        "dataset": dataset,
        "training": training,
        "reports": reports,
    }


def test_builds_controlled_two_by_two_experiment_and_dashboard(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)

    report = build_stage8_experiment_report(
        experiment_id="stage8-test",
        evaluation_reports=artifacts["reports"],
        train_run_dirs=artifacts["runs"],
        dataset_dir=artifacts["dataset"],
        training_dir=artifacts["training"],
        memory_snapshot_path=artifacts["snapshot"].path,
        expected_test_tasks=2,
        expected_test_trials=2,
        bootstrap_samples=100,
    )
    dashboard = render_stage8_dashboard(report)

    assert report["design"]["train_passes"] == 3
    assert report["fast_loop"]["episode_count"] == 222
    assert report["fast_loop"]["final_memory_item_count"] == 3
    assert report["fast_loop"]["memory_reuse_coverage"] == 1.0
    assert report["evaluation"]["cells"][BASE_NO_MEMORY]["pass_at_1"] == 0.25
    assert report["evaluation"]["cells"][OPD_WITH_MEMORY]["pass_at_1"] == 0.75
    assert report["evaluation"]["contrasts"]["full_system_gain"][
        "pass_at_1_delta"
    ] == pytest.approx(0.5)
    assert report["opd_dataset"]["example_count"] == 426
    assert report["opd_training"]["forward_kl_mean"] == pytest.approx(
        (120 * 0.4 + 240 * 0.3 + 60 * 0.2 + 6 * 0.1) / 426
    )
    assert report["opd_training"]["generated_response_token_count"] == 852
    assert "<svg" in dashboard
    assert "Test pass@1" in dashboard
    assert "Forward KL during OPD training" in dashboard
    assert dashboard.index(">sel<") < dashboard.index(">act<")
    assert dashboard.index(">act<") < dashboard.index(">write<")
    assert dashboard.index(">write<") < dashboard.index(">maint<")


def test_rejects_trained_cells_with_different_checkpoints(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    reports = dict(artifacts["reports"])
    changed = json.loads(json.dumps(reports[OPD_NO_MEMORY]))
    changed["provenance"]["checkpoint"] = "other/step-00000011"
    reports[OPD_NO_MEMORY] = changed

    with pytest.raises(ValueError, match="checkpoint"):
        build_stage8_experiment_report(
            experiment_id="stage8-test",
            evaluation_reports=reports,
            train_run_dirs=artifacts["runs"],
            dataset_dir=artifacts["dataset"],
            training_dir=artifacts["training"],
            memory_snapshot_path=artifacts["snapshot"].path,
            expected_test_tasks=2,
            expected_test_trials=2,
            bootstrap_samples=100,
        )


def test_rejects_less_than_three_train_passes(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="exactly 3"):
        build_stage8_experiment_report(
            experiment_id="stage8-test",
            evaluation_reports=artifacts["reports"],
            train_run_dirs=artifacts["runs"][:2],
            dataset_dir=artifacts["dataset"],
            training_dir=artifacts["training"],
            memory_snapshot_path=artifacts["snapshot"].path,
            expected_test_tasks=2,
            expected_test_trials=2,
            bootstrap_samples=100,
        )


def test_rejects_incomplete_test_token_telemetry(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    reports = json.loads(json.dumps(artifacts["reports"]))
    reports[BASE_NO_MEMORY]["summary"]["token_usage_episode_count"] = 3

    with pytest.raises(ValueError, match="token_usage_episode_count"):
        build_stage8_experiment_report(
            experiment_id="stage8-test",
            evaluation_reports=reports,
            train_run_dirs=artifacts["runs"],
            dataset_dir=artifacts["dataset"],
            training_dir=artifacts["training"],
            memory_snapshot_path=artifacts["snapshot"].path,
            expected_test_tasks=2,
            expected_test_trials=2,
            bootstrap_samples=100,
        )


def test_rejects_incomplete_test_snapshot_telemetry(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    reports = json.loads(json.dumps(artifacts["reports"]))
    reports[BASE_NO_MEMORY]["provenance"]["output_memory_snapshot_ids"] = [None]

    with pytest.raises(ValueError, match="output Memory snapshot count"):
        build_stage8_experiment_report(
            experiment_id="stage8-test",
            evaluation_reports=reports,
            train_run_dirs=artifacts["runs"],
            dataset_dir=artifacts["dataset"],
            training_dir=artifacts["training"],
            memory_snapshot_path=artifacts["snapshot"].path,
            expected_test_tasks=2,
            expected_test_trials=2,
            bootstrap_samples=100,
        )


def test_rejects_incomplete_train_episode_stream(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    events_path = artifacts["runs"][0] / "rollouts" / "events.jsonl"
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    removed = False
    filtered = []
    for row in reversed(rows):
        if not removed and row.get("event_type") == "EpisodeFinished":
            removed = True
            continue
        filtered.append(row)
    _write_jsonl(events_path, list(reversed(filtered)))

    with pytest.raises(ValueError, match="finished episode"):
        build_stage8_experiment_report(
            experiment_id="stage8-test",
            evaluation_reports=artifacts["reports"],
            train_run_dirs=artifacts["runs"],
            dataset_dir=artifacts["dataset"],
            training_dir=artifacts["training"],
            memory_snapshot_path=artifacts["snapshot"].path,
            expected_test_tasks=2,
            expected_test_trials=2,
            bootstrap_samples=100,
        )


def test_rejects_same_named_checkpoint_from_another_directory(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    reports = json.loads(json.dumps(artifacts["reports"]))
    wrong_checkpoint = str(tmp_path / "other" / "step-00000010")
    reports[OPD_WITH_MEMORY]["provenance"]["checkpoint"] = wrong_checkpoint
    reports[OPD_NO_MEMORY]["provenance"]["checkpoint"] = wrong_checkpoint

    with pytest.raises(ValueError, match="completed OPD checkpoint"):
        build_stage8_experiment_report(
            experiment_id="stage8-test",
            evaluation_reports=reports,
            train_run_dirs=artifacts["runs"],
            dataset_dir=artifacts["dataset"],
            training_dir=artifacts["training"],
            memory_snapshot_path=artifacts["snapshot"].path,
            expected_test_tasks=2,
            expected_test_trials=2,
            bootstrap_samples=100,
        )
