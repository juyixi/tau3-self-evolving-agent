from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.run_iteration as cli
from tau3_retail_evolver.pipeline.iteration import IterationRequest, IterationResult
from tau3_retail_evolver.pipeline.state import IterationState


def test_parse_args_supports_parent_resume_and_explicit_smoke_tasks() -> None:
    args = cli.parse_args(
        [
            "--iteration-id",
            "iteration-0002",
            "--iteration",
            "2",
            "--model-revision",
            "model-a",
            "--parent-iteration-dir",
            "runs/iterations/iteration-0001",
            "--task-id",
            "3",
            "--task-id",
            "1",
            "--stop-after",
            "dataset_complete",
        ]
    )

    assert args.iteration_id == "iteration-0002"
    assert args.iteration == 2
    assert args.parent_iteration_dir == Path("runs/iterations/iteration-0001")
    assert args.task_ids == ["3", "1"]
    assert args.stop_after == "dataset_complete"
    assert args.output_root == Path("runs/iterations")


def test_main_builds_request_runs_iteration_and_prints_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    request = IterationRequest(
        iteration_id="iteration-0000",
        iteration=0,
        iteration_dir=tmp_path / "iteration-0000",
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        model_revision="model-a",
        adapter_revision="adapter-a",
        parent_checkpoint=None,
        parent_iteration_dir=None,
        input_memory_snapshot_id="memory-0",
        memory_snapshots_dir=tmp_path / "history" / "snapshots",
        task_ids=("0",),
        official_train_task_ids=("0",),
        completed_train_tasks_before=0,
    )
    monkeypatch.setattr(cli, "_build_request", lambda args: request)
    executor = object()
    monkeypatch.setattr(cli, "CommandIterationExecutor", lambda: executor)
    monkeypatch.setattr(
        cli,
        "run_iteration",
        lambda actual_request, actual_executor, stop_after=None: IterationResult(
            iteration_dir=request.iteration_dir,
            state=IterationState.PROMOTED,
            promotion_manifest={
                "child": {"adapter_revision": "adapter-b"},
                "memory": {"output_snapshot_id": "memory-1"},
            },
        ),
    )

    exit_code = cli.main(
        [
            "--iteration-id",
            "iteration-0000",
            "--iteration",
            "0",
            "--model-revision",
            "model-a",
            "--adapter-revision",
            "adapter-a",
            "--completed-train-tasks-before",
            "0",
        ]
    )

    assert exit_code == 0
    assert '"state":"promoted"' in capsys.readouterr().out


def test_child_request_inherits_promoted_parent_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parent = tmp_path / "iterations" / "iteration-0000"
    parent.mkdir(parents=True)
    (parent / "promotion_manifest.json").write_text(
        """{
          "status": "promoted",
          "iteration_id": "iteration-0000",
          "iteration": 0,
          "completed_train_tasks_after": 2,
          "child": {
            "model_revision": "model-a",
            "adapter_revision": "adapter-b",
            "checkpoint": "C:/checkpoints/adapter-b"
          },
          "memory": {"output_snapshot_id": "memory-1"}
        }""",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        pipeline=SimpleNamespace(iteration_task_count=2, shuffle_train_tasks=False),
        training=SimpleNamespace(seed=42),
        memory=SimpleNamespace(),
        tau2=SimpleNamespace(repo_path=Path("external/tau2-bench")),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "_load_official_train_tasks",
        lambda config, project_root: ("0", "1", "2"),
    )
    monkeypatch.setattr(
        cli,
        "open_training_memory",
        lambda config, root: SimpleNamespace(
            snapshot=lambda: SimpleNamespace(
                memory_snapshot_id="memory-1",
                path=tmp_path / "history" / "snapshots" / "memory-1",
            )
        ),
    )
    args = cli.parse_args(
        [
            "--iteration-id",
            "iteration-0001",
            "--iteration",
            "1",
            "--model-revision",
            "model-a",
            "--parent-iteration-dir",
            str(parent),
            "--project-root",
            str(tmp_path),
        ]
    )

    request = cli._build_request(args)

    assert request.adapter_revision == "adapter-b"
    assert request.parent_checkpoint == Path("C:/checkpoints/adapter-b")
    assert request.input_memory_snapshot_id == "memory-1"
    assert request.completed_train_tasks_before == 2
    assert request.task_ids == ("0", "1")


def test_initial_iteration_resume_reuses_persisted_input_memory_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    iteration_dir = tmp_path / "iterations" / "iteration-0000"
    iteration_dir.mkdir(parents=True)
    (iteration_dir / "iteration_state.json").write_text(
        json.dumps(
            {
                "identity": {
                    "input_memory_snapshot_id": "memory-before-failed-rollout"
                }
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        pipeline=SimpleNamespace(iteration_task_count=1, shuffle_train_tasks=False),
        training=SimpleNamespace(seed=42),
        memory=SimpleNamespace(),
        tau2=SimpleNamespace(repo_path=Path("external/tau2-bench")),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "_load_official_train_tasks",
        lambda config, project_root: ("0",),
    )
    monkeypatch.setattr(
        cli,
        "open_training_memory",
        lambda config, root: SimpleNamespace(
            snapshot=lambda: SimpleNamespace(
                memory_snapshot_id="memory-mutated-by-failed-rollout",
                path=tmp_path
                / "history"
                / "snapshots"
                / "memory-mutated-by-failed-rollout",
            )
        ),
    )
    args = cli.parse_args(
        [
            "--iteration-id",
            "iteration-0000",
            "--iteration",
            "0",
            "--model-revision",
            "model-a",
            "--adapter-revision",
            "adapter-a",
            "--completed-train-tasks-before",
            "0",
            "--output-root",
            "iterations",
            "--project-root",
            str(tmp_path),
        ]
    )

    request = cli._build_request(args)

    assert (
        request.input_memory_snapshot_id
        == "memory-before-failed-rollout"
    )
