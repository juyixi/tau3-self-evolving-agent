from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from tau3_retail_evolver.memory.json_store import write_bytes_atomic
from tau3_retail_evolver.memory.locking import reentrant_process_lock
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.types import MemoryTier
from tau3_retail_evolver.pipeline.iteration import IterationRequest, StageResult


CommandRunner = Callable[..., Mapping[str, Any]]


class CommandIterationExecutor:
    """Execute an iteration by composing the existing Stage 4-6 CLIs."""

    def __init__(self, *, command_runner: CommandRunner | None = None) -> None:
        self.command_runner = command_runner or _run_json_command

    def rollout(self, request: IterationRequest) -> StageResult:
        target = request.iteration_dir / "artifacts" / "rollout"
        if target.is_dir():
            return _rollout_stage_result(target)
        work_root = request.iteration_dir / ".work" / "rollout"
        _restore_input_memory_snapshot(request)
        if work_root.exists():
            shutil.rmtree(work_root)
        command = [
            sys.executable,
            "-m",
            "scripts.run_fast_loop",
            "--config",
            str(request.config_path),
            "--split",
            "train",
            "--run-id",
            "rollout",
            "--output-root",
            str(work_root),
            "--iteration",
            str(request.iteration),
            "--model-revision",
            request.model_revision,
            "--adapter-revision",
            request.adapter_revision,
            "--project-root",
            str(request.project_root),
            "--completed-train-tasks-before",
            str(request.completed_train_tasks_before),
        ]
        for task_id in request.task_ids:
            command.extend(("--task-id", task_id))
        if request.qwen_base_url is not None:
            command.extend(("--qwen-base-url", request.qwen_base_url))
        try:
            self.command_runner(command, cwd=request.project_root)
            source = work_root / "rollout"
            work_result = _rollout_stage_result(source)
            _publish_directory(source, target)
        except BaseException:
            try:
                _restore_input_memory_snapshot(request)
            except Exception as restore_error:
                raise RuntimeError(
                    "rollout failed and input Memory snapshot could not be restored"
                ) from restore_error
            raise
        return StageResult(
            artifacts={"rollout": target},
            metadata=work_result.metadata,
        )

    def build_dataset(
        self,
        request: IterationRequest,
        record: dict[str, Any],
    ) -> StageResult:
        output_root = request.iteration_dir / "artifacts"
        dataset_root = output_root / "dataset"
        dataset_dir = dataset_root / "slow_loop"
        if not dataset_dir.exists():
            command = [
                sys.executable,
                "-m",
                "scripts.build_opd_dataset",
                "--config",
                str(request.config_path),
                "--source-run",
                str(request.iteration_dir / "artifacts" / "rollout"),
                "--dataset-build-id",
                "dataset",
                "--output-root",
                str(output_root),
                "--project-root",
                str(request.project_root),
            ]
            self.command_runner(command, cwd=request.project_root)
        manifest = _read_json(dataset_dir / "dataset_manifest.json")
        return StageResult(
            artifacts={"dataset": dataset_dir},
            metadata={"dataset_build_id": manifest.get("dataset_build_id")},
        )

    def audit_dataset(
        self,
        request: IterationRequest,
        record: dict[str, Any],
    ) -> StageResult:
        dataset_dir = request.iteration_dir / "artifacts" / "dataset" / "slow_loop"
        command = [
            sys.executable,
            "-m",
            "scripts.audit_opd_dataset",
            "--dataset-dir",
            str(dataset_dir),
            "--project-root",
            str(request.project_root),
        ]
        report = dict(self.command_runner(command, cwd=request.project_root))
        report_path = request.iteration_dir / "artifacts" / "dataset_audit.json"
        _write_json(report_path, report)
        return StageResult(
            artifacts={"dataset_audit": report_path},
            metadata={"passed": report.get("passed") is True},
        )

    def train(
        self,
        request: IterationRequest,
        record: dict[str, Any],
    ) -> StageResult:
        dataset_dir = request.iteration_dir / "artifacts" / "dataset" / "slow_loop"
        output_dir = request.iteration_dir / "artifacts" / "training"
        command = [
            sys.executable,
            "-m",
            "scripts.train_opd_lora_suite",
            "--config",
            str(request.config_path),
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(output_dir),
            "--model-revision",
            request.model_revision,
            "--adapter-revision",
            request.adapter_revision,
        ]
        self.command_runner(command, cwd=request.project_root)
        manifest = _read_json(output_dir / "training_manifest.json")
        bundle_revision = manifest.get("adapter_bundle_revision")
        if not isinstance(bundle_revision, str) or not bundle_revision:
            raise ValueError("training manifest adapter bundle revision is missing")
        return StageResult(
            artifacts={"training": output_dir},
            metadata={
                "adapter_checkpoints": manifest.get("adapter_checkpoints"),
                "child_adapter_revision": bundle_revision,
                "latest_checkpoint": str(output_dir.resolve()),
            },
        )


def _run_json_command(command: Sequence[str], *, cwd: Path) -> Mapping[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            cwd=Path(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(f"unable to execute iteration stage: {command[0]}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no command output"
        raise RuntimeError(
            f"iteration stage command failed with exit code {completed.returncode}: {detail}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("iteration stage command did not return JSON")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError("iteration stage command returned invalid JSON") from error
    if not isinstance(value, Mapping):
        raise RuntimeError("iteration stage command JSON must be an object")
    return value


def _publish_directory(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"iteration stage output is missing: {source}")
    if target.exists():
        raise FileExistsError(f"iteration stage output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)


def _restore_input_memory_snapshot(request: IterationRequest) -> None:
    memory_root = Path(request.memory_snapshots_dir).resolve().parent
    snapshot_path = (
        Path(request.memory_snapshots_dir).resolve()
        / request.input_memory_snapshot_id
    )
    snapshot = ReadOnlyMemoryRepository(snapshot_path)
    if snapshot.memory_snapshot_id != request.input_memory_snapshot_id:
        raise ValueError("input Memory snapshot ID does not match iteration")
    files = {
        f"{tier.value}_memory.json": (
            snapshot_path / f"{tier.value}_memory.json"
        ).read_bytes()
        for tier in MemoryTier
    }
    with reentrant_process_lock(memory_root, namespace="memory-repository"):
        for name, content in files.items():
            write_bytes_atomic(memory_root / name, content)
        restored = MemoryRepository(memory_root).snapshot()
    if restored.memory_snapshot_id != request.input_memory_snapshot_id:
        raise ValueError("restored Memory snapshot does not match iteration")


def _rollout_stage_result(root: Path) -> StageResult:
    summary = _read_json(root / "fast_loop_summary.json")
    return StageResult(
        artifacts={"rollout": root},
        metadata={
            "completed_train_tasks_after": summary.get("completed_train_tasks_after"),
            "input_memory_snapshot_id": summary.get("input_memory_snapshot_id"),
            "output_memory_snapshot_id": summary.get("output_memory_snapshot_id"),
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read iteration stage JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"iteration stage JSON must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, payload)
