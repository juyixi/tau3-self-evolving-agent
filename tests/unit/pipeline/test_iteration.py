from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tau3_retail_evolver.pipeline.iteration import (
    IterationRequest,
    StageResult,
    run_iteration,
)
from tau3_retail_evolver.pipeline.state import IterationState, load_iteration_record


TRAIN_IDS = ("0", "1", "2", "3")


class FakeExecutor:
    def __init__(
        self,
        *,
        fail_on: str | None = None,
        incomplete_adapter: bool = False,
        incomplete_memory: bool = False,
    ) -> None:
        self.fail_on = fail_on
        self.incomplete_adapter = incomplete_adapter
        self.incomplete_memory = incomplete_memory
        self.calls: list[str] = []

    def _enter(self, stage: str) -> None:
        self.calls.append(stage)
        if self.fail_on == stage:
            raise RuntimeError(f"forced {stage} failure")

    def rollout(self, request: IterationRequest) -> StageResult:
        self._enter("rollout")
        root = request.iteration_dir / "artifacts" / "rollout"
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "manifest.json", {"task_ids": list(request.task_ids)})
        _write_json(
            root / "fast_loop_summary.json",
            {
                "completed_train_tasks_before": request.completed_train_tasks_before,
                "completed_train_tasks_after": (
                    request.completed_train_tasks_before + len(request.task_ids)
                ),
                "input_memory_snapshot_id": request.input_memory_snapshot_id,
                "output_memory_snapshot_id": f"memory-{request.iteration + 1}",
                "successful_task_ids": list(request.task_ids),
            },
        )
        if not self.incomplete_memory:
            _write_snapshot(request.memory_snapshots_dir, f"memory-{request.iteration + 1}")
        return StageResult(
            artifacts={"rollout": root},
            metadata={
                "completed_train_tasks_after": (
                    request.completed_train_tasks_before + len(request.task_ids)
                ),
                "input_memory_snapshot_id": request.input_memory_snapshot_id,
                "output_memory_snapshot_id": f"memory-{request.iteration + 1}",
            },
        )

    def build_dataset(self, request: IterationRequest, record: dict[str, Any]) -> StageResult:
        self._enter("build_dataset")
        root = request.iteration_dir / "artifacts" / "dataset" / "slow_loop"
        (root / "datasets").mkdir(parents=True, exist_ok=True)
        dataset_build_id = f"{request.iteration_id}-dataset"
        _write_json(
            root / "dataset_manifest.json",
            {
                "dataset_build_id": dataset_build_id,
                "official_split": {"name": "train", "train_task_ids": list(TRAIN_IDS)},
                "policy_lineage": {
                    "iteration": request.iteration,
                    "model_revision": request.model_revision,
                    "adapter_revision": request.adapter_revision,
                },
            },
        )
        for kind in ("sel", "act", "write", "maint"):
            (root / "datasets" / f"{kind}.jsonl").write_text("", encoding="utf-8")
        return StageResult(
            artifacts={"dataset": root},
            metadata={"dataset_build_id": dataset_build_id},
        )

    def audit_dataset(self, request: IterationRequest, record: dict[str, Any]) -> StageResult:
        self._enter("audit_dataset")
        report = request.iteration_dir / "artifacts" / "dataset_audit.json"
        _write_json(report, {"passed": True, "errors": []})
        return StageResult(
            artifacts={"dataset_audit": report},
            metadata={"passed": True},
        )

    def train(self, request: IterationRequest, record: dict[str, Any]) -> StageResult:
        self._enter("train")
        root = request.iteration_dir / "artifacts" / "training"
        checkpoint = root / "checkpoints" / "step-00000001"
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        _write_json(adapter / "adapter_config.json", {"r": 32, "lora_alpha": 64})
        if not self.incomplete_adapter:
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        child_revision = f"adapter-iteration-{request.iteration + 1}"
        _write_json(
            checkpoint / "checkpoint_manifest.json",
            {
                "adapter_path": "adapter",
                "adapter_revision": child_revision,
                "dataset_build_id": f"{request.iteration_id}-dataset",
                "source_lineage": {
                    "model_revision": request.model_revision,
                    "adapter_revision": request.adapter_revision,
                },
                "status": "checkpoint",
            },
        )
        _write_json(
            root / "training_manifest.json",
            {
                "adapter_revision": child_revision,
                "dataset_build_id": f"{request.iteration_id}-dataset",
                "latest_checkpoint": "checkpoints/step-00000001",
                "source_lineage": {
                    "model_revision": request.model_revision,
                    "adapter_revision": request.adapter_revision,
                },
                "status": "complete",
            },
        )
        return StageResult(
            artifacts={"training": root},
            metadata={
                "child_adapter_revision": child_revision,
                "latest_checkpoint": str(checkpoint),
            },
        )


class FourLoraFakeExecutor(FakeExecutor):
    def train(self, request: IterationRequest, record: dict[str, Any]) -> StageResult:
        self._enter("train")
        root = request.iteration_dir / "artifacts" / "training"
        checkpoints: dict[str, str] = {}
        revisions: dict[str, str] = {}
        for kind in ("sel", "act", "write", "maint"):
            checkpoint = root / kind / "checkpoints" / "step-00000001"
            adapter = checkpoint / "adapter"
            adapter.mkdir(parents=True, exist_ok=True)
            _write_json(adapter / "adapter_config.json", {"r": 32, "lora_alpha": 64})
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            revision = f"opd-{kind}-step-00000001"
            relative = checkpoint.relative_to(root).as_posix()
            checkpoints[kind] = relative
            revisions[kind] = revision
            _write_json(
                checkpoint / "checkpoint_manifest.json",
                {
                    "adapter_path": "adapter",
                    "adapter_revision": revision,
                    "dataset_build_id": f"{request.iteration_id}-dataset",
                    "opd_kind": kind,
                    "source_lineage": {
                        "model_revision": request.model_revision,
                        "adapter_revision": request.adapter_revision,
                    },
                    "status": "checkpoint",
                },
            )
        bundle_revision = "opd-four-lora-test"
        _write_json(
            root / "training_manifest.json",
            {
                "schema_version": 2,
                "adapter_revision": bundle_revision,
                "adapter_bundle_revision": bundle_revision,
                "adapter_checkpoints": checkpoints,
                "adapter_revisions": revisions,
                "dataset_build_id": f"{request.iteration_id}-dataset",
                "source_lineage": {
                    "model_revision": request.model_revision,
                    "adapter_revision": request.adapter_revision,
                },
                "status": "complete",
            },
        )
        return StageResult(
            artifacts={"training": root},
            metadata={
                "adapter_checkpoints": checkpoints,
                "child_adapter_revision": bundle_revision,
                "latest_checkpoint": str(root),
            },
        )


def _request(tmp_path: Path, **changes: Any) -> IterationRequest:
    snapshots = tmp_path / "history" / "snapshots"
    _write_snapshot(snapshots, "memory-0")
    config_path = tmp_path / "config.yaml"
    if not config_path.exists():
        config_path.write_text("pipeline:\n  iteration_task_count: 2\n", encoding="utf-8")
    request = IterationRequest(
        iteration_id="iteration-0000",
        iteration=0,
        iteration_dir=tmp_path / "iterations" / "iteration-0000",
        project_root=tmp_path,
        config_path=config_path,
        model_revision="model-revision-a",
        adapter_revision="zero-lora",
        parent_checkpoint=None,
        parent_iteration_dir=None,
        input_memory_snapshot_id="memory-0",
        memory_snapshots_dir=snapshots,
        task_ids=("0", "1"),
        official_train_task_ids=TRAIN_IDS,
        completed_train_tasks_before=0,
        qwen_base_url="http://127.0.0.1:8000/v1",
    )
    return replace(request, **changes)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_snapshot(root: Path, snapshot_id: str) -> None:
    path = root / snapshot_id
    path.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for tier in ("trajectory", "tip", "skill", "tool"):
        payload = (json.dumps({"tier": tier}, sort_keys=True) + "\n").encode()
        (path / f"{tier}_memory.json").write_bytes(payload)
        files[f"{tier}_memory.json"] = hashlib.sha256(payload).hexdigest()
    _write_json(
        path / "manifest.json",
        {
            "schema_version": 1,
            "memory_snapshot_id": snapshot_id,
            "counts": {tier: 0 for tier in ("trajectory", "tip", "skill", "tool")},
            "files": files,
        },
    )


def test_iteration_runs_stages_in_order_and_promotes_complete_lineage(tmp_path: Path) -> None:
    executor = FakeExecutor()

    result = run_iteration(_request(tmp_path), executor)

    assert executor.calls == ["rollout", "build_dataset", "audit_dataset", "train"]
    assert result.state is IterationState.PROMOTED
    promotion = result.promotion_manifest
    assert promotion["parent"]["adapter_revision"] == "zero-lora"
    assert promotion["child"]["adapter_revision"] == "adapter-iteration-1"
    assert promotion["memory"]["input_snapshot_id"] == "memory-0"
    assert promotion["memory"]["output_snapshot_id"] == "memory-1"
    assert promotion["task_ids"] == ["0", "1"]
    assert promotion["completed_train_tasks_after"] == 2
    record = load_iteration_record(result.iteration_dir)
    assert record["state"] == "promoted"
    assert all(
        artifact["sha256"]
        for stage in record["stages"].values()
        for artifact in stage["artifacts"].values()
    )


def test_iteration_promotes_four_lora_bundle(tmp_path: Path) -> None:
    result = run_iteration(_request(tmp_path), FourLoraFakeExecutor())

    assert result.state is IterationState.PROMOTED
    child = result.promotion_manifest["child"]
    assert child["adapter_revision"] == "opd-four-lora-test"
    assert Path(child["checkpoint"]).name == "training"
    assert set(child["adapter_checkpoints"]) == {"sel", "act", "write", "maint"}
    assert all(Path(path).is_dir() for path in child["adapter_checkpoints"].values())


def test_iteration_can_pause_after_dataset_and_resume_for_single_gpu_training(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    first = FakeExecutor()

    paused = run_iteration(
        request,
        first,
        stop_after=IterationState.DATASET_COMPLETE,
    )

    assert paused.state is IterationState.DATASET_COMPLETE
    assert paused.promotion_manifest is None
    assert first.calls == ["rollout", "build_dataset", "audit_dataset"]

    resumed = FakeExecutor()
    complete = run_iteration(request, resumed)

    assert complete.state is IterationState.PROMOTED
    assert resumed.calls == ["train"]


@pytest.mark.parametrize(
    ("failure", "expected_state", "completed_calls"),
    [
        ("rollout", "created", []),
        ("build_dataset", "rollout_complete", ["rollout"]),
        (
            "audit_dataset",
            "attribution_complete",
            ["rollout", "build_dataset"],
        ),
        (
            "train",
            "dataset_complete",
            ["rollout", "build_dataset", "audit_dataset"],
        ),
    ],
)
def test_iteration_resumes_without_rerunning_completed_stages(
    tmp_path: Path,
    failure: str,
    expected_state: str,
    completed_calls: list[str],
) -> None:
    request = _request(tmp_path)
    failing = FakeExecutor(fail_on=failure)
    with pytest.raises(RuntimeError, match="forced"):
        run_iteration(request, failing)
    assert load_iteration_record(request.iteration_dir)["state"] == expected_state

    resumed = FakeExecutor()
    result = run_iteration(request, resumed)

    assert result.state is IterationState.PROMOTED
    assert all(call not in resumed.calls for call in completed_calls)


def test_resume_rejects_modified_completed_artifact(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(RuntimeError):
        run_iteration(request, FakeExecutor(fail_on="build_dataset"))
    (request.iteration_dir / "artifacts" / "rollout" / "manifest.json").write_text(
        json.dumps({"task_ids": ["test-1"]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_iteration(request, FakeExecutor())


def test_resume_rejects_changed_iteration_config(tmp_path: Path) -> None:
    request = _request(tmp_path)
    with pytest.raises(RuntimeError):
        run_iteration(request, FakeExecutor(fail_on="build_dataset"))
    request.config_path.write_text(
        "pipeline:\n  iteration_task_count: 3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="persisted identity"):
        run_iteration(request, FakeExecutor())


def test_incomplete_adapter_is_never_promoted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValueError, match="adapter weight"):
        run_iteration(request, FakeExecutor(incomplete_adapter=True))

    assert load_iteration_record(request.iteration_dir)["state"] == "dataset_complete"
    assert not (request.iteration_dir / "promotion_manifest.json").exists()


def test_incomplete_memory_snapshot_is_never_recorded_or_promoted(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ValueError, match="memory snapshot"):
        run_iteration(request, FakeExecutor(incomplete_memory=True))

    assert load_iteration_record(request.iteration_dir)["state"] == "created"
    assert not (request.iteration_dir / "promotion_manifest.json").exists()


def test_two_iterations_form_a_strict_parent_child_chain(tmp_path: Path) -> None:
    first = run_iteration(_request(tmp_path), FakeExecutor())
    first_promotion = first.promotion_manifest
    second_request = _request(
        tmp_path,
        iteration_id="iteration-0001",
        iteration=1,
        iteration_dir=tmp_path / "iterations" / "iteration-0001",
        adapter_revision=first_promotion["child"]["adapter_revision"],
        parent_checkpoint=Path(first_promotion["child"]["checkpoint"]),
        parent_iteration_dir=first.iteration_dir,
        input_memory_snapshot_id=first_promotion["memory"]["output_snapshot_id"],
        task_ids=("2", "3"),
        completed_train_tasks_before=first_promotion["completed_train_tasks_after"],
    )

    second = run_iteration(second_request, FakeExecutor())

    assert second.promotion_manifest["parent"]["iteration_id"] == "iteration-0000"
    assert second.promotion_manifest["parent"]["adapter_revision"] == (
        first.promotion_manifest["child"]["adapter_revision"]
    )
    assert second.promotion_manifest["memory"]["input_snapshot_id"] == "memory-1"


def test_child_iteration_rejects_stale_parent_lineage(tmp_path: Path) -> None:
    first = run_iteration(_request(tmp_path), FakeExecutor())
    stale = _request(
        tmp_path,
        iteration_id="iteration-0001",
        iteration=1,
        iteration_dir=tmp_path / "iterations" / "iteration-0001",
        adapter_revision="wrong-adapter",
        parent_checkpoint=Path(first.promotion_manifest["child"]["checkpoint"]),
        parent_iteration_dir=first.iteration_dir,
        input_memory_snapshot_id="memory-1",
        completed_train_tasks_before=2,
    )

    with pytest.raises(ValueError, match="parent adapter revision"):
        run_iteration(stale, FakeExecutor())


def test_iteration_rejects_non_train_task_before_creating_state(tmp_path: Path) -> None:
    request = _request(tmp_path, task_ids=("0", "test-1"))

    with pytest.raises(ValueError, match="official train split"):
        run_iteration(request, FakeExecutor())

    assert not request.iteration_dir.exists()
