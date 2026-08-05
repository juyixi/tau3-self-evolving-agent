from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import tau3_evolver.execution.runner as runner
from tau3_evolver.agent.policy import EpisodeResult
from tau3_evolver.benchmarks.types import RuntimeOrigin
from tau3_evolver.execution.request import ExecutionRequest
from tau3_evolver.execution.results import BatchResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_execute_publishes_only_run_and_episode_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
    assert "failures" not in run_record
