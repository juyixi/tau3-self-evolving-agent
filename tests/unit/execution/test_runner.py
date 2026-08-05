from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tau3_evolver.execution.runner as runner
from tau3_evolver.agent.policy import EpisodeResult
from tau3_evolver.benchmarks.types import PreparedBenchmark, RuntimeOrigin
from tau3_evolver.config import load_config
from tau3_evolver.execution.request import ExecutionRequest
from tau3_evolver.execution.results import BatchResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_execute_publishes_only_run_and_episode_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret")
    prepared = SimpleNamespace(
        name="retail",
        split_name="test",
        split_hash="s" * 64,
        task_ids=("task-1",),
        runtime_origin=RuntimeOrigin(Path("tau2"), "0.1", "c" * 40),
        default_memory_namespace="retail",
    )
    definition = SimpleNamespace(prepare=lambda _config, _mode: prepared)
    monkeypatch.setattr(
        runner.benchmark_registry,
        "resolve",
        lambda _benchmark: definition,
    )
    monkeypatch.setattr(
        runner,
        "resolve_memory",
        lambda _request, _prepared: SimpleNamespace(
            source=None,
            destination=None,
            source_namespace=None,
            destination_namespace=None,
            input_snapshot_id=None,
            generation=0,
        ),
    )
    monkeypatch.setattr(runner, "_policy", lambda _config: SimpleNamespace())

    result = BatchResult(
        episodes=(
            EpisodeResult(
                task_id="task-1",
                final_reward=1.0,
                steps=1,
                terminal_evaluation={"reward": 1.0},
                selected_memory_ids=(),
                written_memory_ids=(),
                truncated=False,
            ),
        ),
        failures=(),
        input_memory_snapshot_id=None,
        output_memory_snapshot_id=None,
    )

    def run_batch(**values):
        values["episode_writer"].append(
            {
                "schema_version": 1,
                "task_id": "task-1",
                "task_group": "retail",
                "seed": 0,
                "status": "completed",
                "task": {},
                "trajectory": [],
                "outcome": {"final_reward": 1.0},
                "memory": {"enabled": False, "writes": []},
            }
        )
        return result

    monkeypatch.setattr(runner, "run_batch", run_batch)
    request = ExecutionRequest(
        benchmark="retail",
        mode="test",
        memory_enabled=False,
        run_id="run-1",
        output_root=tmp_path,
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
    )

    runner.execute(request)

    run_dir = tmp_path / "run-1"
    assert {path.name for path in run_dir.iterdir()} == {
        "episodes.jsonl",
        "run.json",
    }
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert run_record["artifacts"]["episodes"]["rows"] == 1
    assert run_record["summary"]["metrics"]["pass_rate"] == 1.0
    assert run_record["execution"]["task_scope"] == "full"
    assert run_record["memory"]["destination_namespace"] is None
    assert "failures" not in run_record


def test_execute_rejects_missing_online_credentials_before_benchmark_prepare(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prepared = False

    def resolve_benchmark(_benchmark: str):
        nonlocal prepared
        prepared = True
        raise AssertionError("benchmark preparation must not start")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        runner,
        "load_project_environment",
        lambda: SimpleNamespace(path=tmp_path / ".env", loaded_names=()),
    )
    monkeypatch.setattr(runner.benchmark_registry, "resolve", resolve_benchmark)
    request = ExecutionRequest(
        benchmark="retail",
        mode="train",
        memory_enabled=False,
        run_id="missing-credentials",
        output_root=tmp_path,
        config_path=PROJECT_ROOT / "configs" / "default.yaml",
    )

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        runner.execute(request)

    assert not prepared
    assert not (tmp_path / request.run_id).exists()


def test_debug_selects_one_concurrency_sized_stable_batch() -> None:
    tasks = tuple(SimpleNamespace(id=f"task-{index}") for index in range(5))
    prepared = PreparedBenchmark(
        name="retail",
        task_type=SimpleNamespace,
        task_catalog=tasks,
        task_ids=tuple(task.id for task in tasks),
        split_name="train",
        split_hash="s" * 64,
        environment_factory=lambda: None,
        runtime=None,
        run_domain=lambda config: config,
        text_run_config_type=dict,
        registry=object(),
        runtime_origin=RuntimeOrigin(Path("tau2"), None, None),
        default_memory_namespace="retail",
        task_group="retail",
    )
    request = ExecutionRequest(
        benchmark="retail",
        mode="train",
        debug=True,
        memory_enabled=False,
        run_id="debug-1",
    )
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    selected = runner._select_execution_tasks(
        prepared,
        request=request,
        config=config,
    )

    assert selected.task_ids == ("task-0", "task-1", "task-2")
    assert selected.task_catalog == tasks[:3]
    assert selected.split_hash == prepared.split_hash

    single_task_config = config.model_copy(
        update={
            "execution": config.execution.model_copy(
                update={"max_concurrency": 1}
            )
        }
    )
    single = runner._select_execution_tasks(
        prepared,
        request=request,
        config=single_task_config,
    )

    assert single.task_ids == ("task-0",)
    assert single.task_catalog == tasks[:1]
