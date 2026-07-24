from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.pipeline.executor import CommandIterationExecutor
from tau3_retail_evolver.pipeline.iteration import IterationRequest


def _value(command: Sequence[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class FakeCommandRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: Sequence[str], *, cwd: Path) -> dict[str, Any]:
        command = tuple(command)
        self.commands.append(command)
        module = _value(command, "-m")
        if module == "scripts.run_fast_loop":
            root = Path(_value(command, "--output-root")) / _value(command, "--run-id")
            summary = {
                "completed_train_tasks_after": 2,
                "input_memory_snapshot_id": "memory-0",
                "output_memory_snapshot_id": "memory-1",
            }
            _write_json(root / "manifest.json", {"task_ids": ["0", "1"]})
            _write_json(root / "fast_loop_summary.json", summary)
            return summary
        if module == "scripts.build_opd_dataset":
            root = Path(_value(command, "--output-root")) / "dataset" / "slow_loop"
            _write_json(
                root / "dataset_manifest.json",
                {
                    "dataset_build_id": "dataset",
                    "policy_lineage": {
                        "iteration": 0,
                        "model_revision": "model-a",
                        "adapter_revision": "adapter-a",
                    },
                },
            )
            return {"dataset_build_id": "dataset", "dataset_dir": str(root)}
        if module == "scripts.audit_opd_dataset":
            return {"passed": True, "errors": []}
        if module == "scripts.train_opd_lora":
            root = Path(_value(command, "--output-dir"))
            checkpoint = root / "checkpoints" / "step-00000001"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            _write_json(adapter / "adapter_config.json", {"r": 32})
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            _write_json(
                checkpoint / "checkpoint_manifest.json",
                {
                    "adapter_path": "adapter",
                    "adapter_revision": "adapter-b",
                    "dataset_build_id": "dataset",
                    "source_lineage": {
                        "model_revision": "model-a",
                        "adapter_revision": "adapter-a",
                    },
                    "status": "checkpoint",
                },
            )
            _write_json(
                root / "training_manifest.json",
                {
                    "adapter_revision": "adapter-b",
                    "dataset_build_id": "dataset",
                    "latest_checkpoint": "checkpoints/step-00000001",
                    "source_lineage": {
                        "model_revision": "model-a",
                        "adapter_revision": "adapter-a",
                    },
                    "status": "complete",
                },
            )
            return {"latest_checkpoint": str(checkpoint), "status": "complete"}
        raise AssertionError(f"unexpected module: {module}")


def _request(tmp_path: Path) -> IterationRequest:
    snapshot = MemoryRepository(tmp_path / "history").snapshot()
    return IterationRequest(
        iteration_id="iteration-0000",
        iteration=0,
        iteration_dir=tmp_path / "iterations" / "iteration-0000",
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        model_revision="model-a",
        adapter_revision="adapter-a",
        parent_checkpoint=None,
        parent_iteration_dir=None,
        input_memory_snapshot_id=snapshot.memory_snapshot_id,
        memory_snapshots_dir=snapshot.path.parent,
        task_ids=("0", "1"),
        official_train_task_ids=("0", "1", "2"),
        completed_train_tasks_before=0,
        qwen_base_url="http://127.0.0.1:8000/v1",
    )


def test_command_executor_maps_each_stage_to_existing_cli(tmp_path: Path) -> None:
    runner = FakeCommandRunner()
    executor = CommandIterationExecutor(command_runner=runner)
    request = _request(tmp_path)
    request.iteration_dir.mkdir(parents=True)

    rollout = executor.rollout(request)
    rollout_again = executor.rollout(request)
    record: dict[str, Any] = {
        "stages": {
            "rollout": {"artifacts": {}, "metadata": dict(rollout.metadata)},
        }
    }
    dataset = executor.build_dataset(request, record)
    record["stages"]["attribution"] = {
        "artifacts": {},
        "metadata": dict(dataset.metadata),
    }
    audit = executor.audit_dataset(request, record)
    record["stages"]["dataset"] = {
        "artifacts": {},
        "metadata": dict(audit.metadata),
    }
    training = executor.train(request, record)

    modules = [_value(command, "-m") for command in runner.commands]
    assert modules == [
        "scripts.run_fast_loop",
        "scripts.build_opd_dataset",
        "scripts.audit_opd_dataset",
        "scripts.train_opd_lora",
    ]
    rollout_command = runner.commands[0]
    assert [
        rollout_command[index + 1]
        for index, value in enumerate(rollout_command)
        if value == "--task-id"
    ] == ["0", "1"]
    assert rollout.metadata["output_memory_snapshot_id"] == "memory-1"
    assert rollout_again.metadata == rollout.metadata
    assert dataset.metadata["dataset_build_id"] == "dataset"
    assert audit.metadata["passed"] is True
    assert training.metadata["child_adapter_revision"] == "adapter-b"


def test_rollout_failure_and_retry_restore_input_memory_snapshot(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "history"
    repository = MemoryRepository(memory_root)
    initial = repository.snapshot()
    request = replace(
        _request(tmp_path),
        input_memory_snapshot_id=initial.memory_snapshot_id,
        memory_snapshots_dir=initial.path.parent,
    )
    request.iteration_dir.mkdir(parents=True)

    def failing_runner(
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> dict[str, Any]:
        repository.add(
            tier="tip",
            content="Partial failed rollout write",
            source_task_ids=("0",),
            created_round=0,
        )
        raise RuntimeError("simulated rollout failure")

    with pytest.raises(RuntimeError, match="simulated rollout failure"):
        CommandIterationExecutor(command_runner=failing_runner).rollout(request)

    assert MemoryRepository(memory_root).snapshot().memory_snapshot_id == (
        initial.memory_snapshot_id
    )

    repository.add(
        tier="tip",
        content="Interrupted process write",
        source_task_ids=("0",),
        created_round=0,
    )
    (request.iteration_dir / ".work" / "rollout").mkdir(parents=True)

    def retry_runner(
        command: Sequence[str],
        *,
        cwd: Path,
    ) -> dict[str, Any]:
        assert MemoryRepository(memory_root).snapshot().memory_snapshot_id == (
            initial.memory_snapshot_id
        )
        root = Path(_value(command, "--output-root")) / _value(command, "--run-id")
        summary = {
            "completed_train_tasks_after": 2,
            "input_memory_snapshot_id": initial.memory_snapshot_id,
            "output_memory_snapshot_id": initial.memory_snapshot_id,
        }
        _write_json(root / "fast_loop_summary.json", summary)
        return summary

    result = CommandIterationExecutor(command_runner=retry_runner).rollout(request)

    assert result.metadata["input_memory_snapshot_id"] == initial.memory_snapshot_id
