from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Protocol

from tau3_retail_evolver.memory.json_store import write_bytes_atomic
from tau3_retail_evolver.memory.locking import reentrant_process_lock
from tau3_retail_evolver.pipeline.sampling import OPD_KINDS, assert_train_only_artifacts
from tau3_retail_evolver.pipeline.state import (
    IterationState,
    IterationStateStore,
)


_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class IterationRequest:
    iteration_id: str
    iteration: int
    iteration_dir: Path
    project_root: Path
    config_path: Path
    model_revision: str
    adapter_revision: str
    parent_checkpoint: Path | None
    parent_iteration_dir: Path | None
    input_memory_snapshot_id: str
    memory_snapshots_dir: Path
    task_ids: tuple[str, ...]
    official_train_task_ids: tuple[str, ...]
    completed_train_tasks_before: int
    qwen_base_url: str | None = None


@dataclass(frozen=True, slots=True)
class StageResult:
    artifacts: Mapping[str, Path]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class IterationResult:
    iteration_dir: Path
    state: IterationState
    promotion_manifest: Mapping[str, Any] | None


class IterationExecutor(Protocol):
    def rollout(self, request: IterationRequest) -> StageResult: ...

    def build_dataset(
        self, request: IterationRequest, record: dict[str, Any]
    ) -> StageResult: ...

    def audit_dataset(
        self, request: IterationRequest, record: dict[str, Any]
    ) -> StageResult: ...

    def train(
        self, request: IterationRequest, record: dict[str, Any]
    ) -> StageResult: ...


def run_iteration(
    request: IterationRequest,
    executor: IterationExecutor,
    *,
    stop_after: IterationState | None = None,
) -> IterationResult:
    _validate_request(request)
    if stop_after is IterationState.CREATED:
        raise ValueError("stop_after must name a completed iteration stage")
    iteration_dir = Path(request.iteration_dir).resolve()
    lock = reentrant_process_lock(iteration_dir, namespace="stage7-iteration")
    with lock:
        _validate_parent(request)
        store = IterationStateStore(iteration_dir)
        identity = _request_identity(request)
        if store.path.exists():
            record = store.load_verified()
            if record["identity"] != identity:
                raise ValueError("iteration request does not match persisted identity")
        else:
            iteration_dir.mkdir(parents=True, exist_ok=False)
            record = store.create(identity)

        if stop_after is not None and record["state"] == stop_after:
            return _iteration_result(iteration_dir, record)

        if record["state"] == IterationState.CREATED:
            result = executor.rollout(request)
            _validate_rollout_result(request, result)
            _validate_stage_artifacts(request, result)
            record = store.advance(
                record,
                target=IterationState.ROLLOUT_COMPLETE,
                stage_name="rollout",
                artifacts=result.artifacts,
                metadata=result.metadata,
            )
            if stop_after is IterationState.ROLLOUT_COMPLETE:
                return _iteration_result(iteration_dir, record)

        if record["state"] == IterationState.ROLLOUT_COMPLETE:
            result = executor.build_dataset(request, record)
            _validate_dataset_result(request, result)
            _validate_stage_artifacts(request, result)
            record = store.advance(
                record,
                target=IterationState.ATTRIBUTION_COMPLETE,
                stage_name="attribution",
                artifacts=result.artifacts,
                metadata=result.metadata,
            )
            if stop_after is IterationState.ATTRIBUTION_COMPLETE:
                return _iteration_result(iteration_dir, record)

        if record["state"] == IterationState.ATTRIBUTION_COMPLETE:
            result = executor.audit_dataset(request, record)
            if result.metadata.get("passed") is not True:
                raise ValueError("independent OPD dataset audit did not pass")
            _validate_stage_artifacts(request, result)
            record = store.advance(
                record,
                target=IterationState.DATASET_COMPLETE,
                stage_name="dataset",
                artifacts=result.artifacts,
                metadata=result.metadata,
            )
            if stop_after is IterationState.DATASET_COMPLETE:
                return _iteration_result(iteration_dir, record)

        if record["state"] == IterationState.DATASET_COMPLETE:
            result = executor.train(request, record)
            training = _validate_training_result(request, record, result)
            _validate_stage_artifacts(request, result)
            record = store.advance(
                record,
                target=IterationState.TRAINING_COMPLETE,
                stage_name="training",
                artifacts=result.artifacts,
                metadata=training,
            )
            if stop_after is IterationState.TRAINING_COMPLETE:
                return _iteration_result(iteration_dir, record)

        if record["state"] == IterationState.TRAINING_COMPLETE:
            promotion_path, promotion = _publish_promotion(request, record)
            _validate_stage_artifacts(
                request,
                StageResult(artifacts={"promotion_manifest": promotion_path}, metadata={}),
            )
            record = store.advance(
                record,
                target=IterationState.PROMOTED,
                stage_name="promotion",
                artifacts={"promotion_manifest": promotion_path},
                metadata={
                    "child_adapter_revision": promotion["child"]["adapter_revision"],
                    "output_memory_snapshot_id": promotion["memory"]["output_snapshot_id"],
                },
            )

        if record["state"] != IterationState.PROMOTED:
            raise RuntimeError(f"iteration stopped in unexpected state: {record['state']}")
        return _iteration_result(iteration_dir, record)


def _iteration_result(
    iteration_dir: Path,
    record: Mapping[str, Any],
) -> IterationResult:
    state = IterationState(record["state"])
    promotion = (
        _read_json(iteration_dir / "promotion_manifest.json")
        if state is IterationState.PROMOTED
        else None
    )
    return IterationResult(
        iteration_dir=iteration_dir,
        state=state,
        promotion_manifest=promotion,
    )


def _validate_request(request: IterationRequest) -> None:
    if not isinstance(request, IterationRequest):
        raise TypeError("request must be an IterationRequest")
    if not _SAFE_ID.fullmatch(request.iteration_id):
        raise ValueError("iteration_id must be a lowercase safe slug")
    if request.iteration < 0:
        raise ValueError("iteration must be non-negative")
    if request.completed_train_tasks_before < 0:
        raise ValueError("completed_train_tasks_before must be non-negative")
    for name, value in (
        ("model_revision", request.model_revision),
        ("adapter_revision", request.adapter_revision),
        ("input_memory_snapshot_id", request.input_memory_snapshot_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be blank")
    if not request.task_ids or len(request.task_ids) != len(set(request.task_ids)):
        raise ValueError("iteration task IDs must be nonempty and unique")
    train_ids = set(request.official_train_task_ids)
    unknown = [task_id for task_id in request.task_ids if task_id not in train_ids]
    if unknown:
        raise ValueError(
            "requested task IDs are not in the official train split: " + ", ".join(unknown)
        )
    _validate_memory_snapshot(request, request.input_memory_snapshot_id)


def _validate_parent(request: IterationRequest) -> None:
    if request.parent_iteration_dir is None:
        if request.iteration != 0:
            raise ValueError("nonzero iteration requires a promoted parent iteration")
        return
    parent_store = IterationStateStore(request.parent_iteration_dir)
    parent_record = parent_store.load_verified()
    if parent_record.get("state") != IterationState.PROMOTED:
        raise ValueError("parent iteration state is not promoted")
    promotion = _read_json(Path(request.parent_iteration_dir) / "promotion_manifest.json")
    if promotion.get("status") != "promoted":
        raise ValueError("parent iteration is not promoted")
    if promotion.get("iteration") != request.iteration - 1:
        raise ValueError("parent iteration number is not contiguous")
    child = promotion.get("child")
    memory = promotion.get("memory")
    if not isinstance(child, Mapping) or not isinstance(memory, Mapping):
        raise ValueError("parent promotion lineage is incomplete")
    if child.get("model_revision") != request.model_revision:
        raise ValueError("parent model revision does not match child request")
    if child.get("adapter_revision") != request.adapter_revision:
        raise ValueError("parent adapter revision does not match child request")
    if memory.get("output_snapshot_id") != request.input_memory_snapshot_id:
        raise ValueError("parent memory snapshot does not match child request")
    if promotion.get("completed_train_tasks_after") != request.completed_train_tasks_before:
        raise ValueError("parent completed task count does not match child request")
    if request.parent_checkpoint is None:
        raise ValueError("child iteration requires parent_checkpoint")
    if Path(str(child.get("checkpoint"))).resolve() != Path(request.parent_checkpoint).resolve():
        raise ValueError("parent checkpoint does not match child request")


def _request_identity(request: IterationRequest) -> dict[str, Any]:
    config_path = Path(request.config_path).resolve()
    try:
        config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"unable to hash iteration config: {config_path}") from error
    train_ids_payload = json.dumps(
        list(request.official_train_task_ids),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "adapter_revision": request.adapter_revision,
        "completed_train_tasks_before": request.completed_train_tasks_before,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "input_memory_snapshot_id": request.input_memory_snapshot_id,
        "iteration": request.iteration,
        "iteration_id": request.iteration_id,
        "model_revision": request.model_revision,
        "memory_snapshots_dir": str(Path(request.memory_snapshots_dir).resolve()),
        "parent_checkpoint": (
            str(Path(request.parent_checkpoint).resolve())
            if request.parent_checkpoint is not None
            else None
        ),
        "parent_iteration_dir": (
            str(Path(request.parent_iteration_dir).resolve())
            if request.parent_iteration_dir is not None
            else None
        ),
        "project_root": str(Path(request.project_root).resolve()),
        "qwen_base_url": request.qwen_base_url,
        "task_ids": list(request.task_ids),
        "train_task_ids_sha256": hashlib.sha256(train_ids_payload).hexdigest(),
    }


def _validate_rollout_result(request: IterationRequest, result: StageResult) -> None:
    metadata = result.metadata
    if metadata.get("input_memory_snapshot_id") != request.input_memory_snapshot_id:
        raise ValueError("rollout input memory snapshot does not match iteration")
    output_snapshot = metadata.get("output_memory_snapshot_id")
    if not isinstance(output_snapshot, str) or not output_snapshot:
        raise ValueError("rollout output memory snapshot is missing")
    _validate_memory_snapshot(request, output_snapshot)
    expected_completed = request.completed_train_tasks_before + len(request.task_ids)
    if metadata.get("completed_train_tasks_after") != expected_completed:
        raise ValueError("rollout completed task count does not match iteration")


def _validate_dataset_result(request: IterationRequest, result: StageResult) -> None:
    dataset = result.artifacts.get("dataset")
    if dataset is None:
        raise ValueError("dataset stage did not return a dataset artifact")
    manifest = _read_json(Path(dataset) / "dataset_manifest.json")
    lineage = manifest.get("policy_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("dataset policy lineage is missing")
    expected = {
        "iteration": request.iteration,
        "model_revision": request.model_revision,
        "adapter_revision": request.adapter_revision,
    }
    if any(lineage.get(key) != value for key, value in expected.items()):
        raise ValueError("dataset policy lineage does not match iteration")
    if manifest.get("dataset_build_id") != result.metadata.get("dataset_build_id"):
        raise ValueError("dataset build ID does not match stage metadata")


def _validate_training_result(
    request: IterationRequest,
    record: Mapping[str, Any],
    result: StageResult,
) -> dict[str, Any]:
    training = result.artifacts.get("training")
    if training is None:
        raise ValueError("training stage did not return a training artifact")
    training_root = Path(training).resolve()
    manifest = _read_json(training_root / "training_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("training manifest is not complete")
    dataset_build_id = record["stages"]["attribution"]["metadata"]["dataset_build_id"]
    if manifest.get("dataset_build_id") != dataset_build_id:
        raise ValueError("training dataset lineage does not match iteration")
    expected_source = {
        "model_revision": request.model_revision,
        "adapter_revision": request.adapter_revision,
    }
    if manifest.get("source_lineage") != expected_source:
        raise ValueError("training source lineage does not match iteration")
    if manifest.get("schema_version") == 2:
        return _validate_four_lora_training_result(
            training_root=training_root,
            manifest=manifest,
            expected_source=expected_source,
            dataset_build_id=dataset_build_id,
            result=result,
        )
    relative_checkpoint = manifest.get("latest_checkpoint")
    if not isinstance(relative_checkpoint, str) or not relative_checkpoint:
        raise ValueError("training latest checkpoint is missing")
    checkpoint = (training_root / relative_checkpoint).resolve()
    try:
        checkpoint.relative_to(training_root)
    except ValueError as error:
        raise ValueError("training checkpoint escapes training output") from error
    checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
    if checkpoint_manifest.get("status") != "checkpoint":
        raise ValueError("latest checkpoint is not published")
    if checkpoint_manifest.get("dataset_build_id") != dataset_build_id:
        raise ValueError("checkpoint dataset lineage does not match iteration")
    if checkpoint_manifest.get("source_lineage") != expected_source:
        raise ValueError("checkpoint source lineage does not match iteration")
    child_revision = checkpoint_manifest.get("adapter_revision")
    if not isinstance(child_revision, str) or not child_revision:
        raise ValueError("checkpoint adapter revision is missing")
    if manifest.get("adapter_revision") != child_revision:
        raise ValueError("training and checkpoint adapter revisions do not match")
    adapter_relative = checkpoint_manifest.get("adapter_path")
    if not isinstance(adapter_relative, str) or not adapter_relative:
        raise ValueError("checkpoint adapter path is missing")
    adapter = (checkpoint / adapter_relative).resolve()
    try:
        adapter.relative_to(checkpoint)
    except ValueError as error:
        raise ValueError("checkpoint adapter path escapes checkpoint") from error
    if not (adapter / "adapter_config.json").is_file():
        raise ValueError("checkpoint adapter config is missing")
    weights = (
        adapter / "adapter_model.safetensors",
        adapter / "adapter_model.bin",
    )
    if sum(path.is_file() for path in weights) != 1:
        raise ValueError("checkpoint must contain exactly one adapter weight file")
    metadata_checkpoint = result.metadata.get("latest_checkpoint")
    if not isinstance(metadata_checkpoint, str) or Path(metadata_checkpoint).resolve() != checkpoint:
        raise ValueError("training stage checkpoint metadata does not match manifest")
    if result.metadata.get("child_adapter_revision") != child_revision:
        raise ValueError("training stage adapter revision does not match manifest")
    return {
        "child_adapter_revision": child_revision,
        "dataset_build_id": dataset_build_id,
        "latest_checkpoint": str(checkpoint),
    }


def _validate_four_lora_training_result(
    *,
    training_root: Path,
    manifest: Mapping[str, Any],
    expected_source: Mapping[str, str],
    dataset_build_id: str,
    result: StageResult,
) -> dict[str, Any]:
    bundle_revision = manifest.get("adapter_bundle_revision")
    if (
        not isinstance(bundle_revision, str)
        or not bundle_revision
        or manifest.get("adapter_revision") != bundle_revision
    ):
        raise ValueError("four-LoRA bundle revision is invalid")
    raw_checkpoints = manifest.get("adapter_checkpoints")
    raw_revisions = manifest.get("adapter_revisions")
    if (
        not isinstance(raw_checkpoints, Mapping)
        or set(raw_checkpoints) != set(OPD_KINDS)
        or not isinstance(raw_revisions, Mapping)
        or set(raw_revisions) != set(OPD_KINDS)
    ):
        raise ValueError("four-LoRA bundle must contain exactly four adapters")

    checkpoints: dict[str, str] = {}
    for kind in OPD_KINDS:
        relative = raw_checkpoints[kind]
        revision = raw_revisions[kind]
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{kind} LoRA checkpoint path is invalid")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{kind} LoRA revision is invalid")
        checkpoint = (training_root / relative).resolve()
        try:
            checkpoint.relative_to(training_root)
        except ValueError as error:
            raise ValueError(f"{kind} LoRA checkpoint escapes training output") from error
        checkpoint_manifest = _read_json(checkpoint / "checkpoint_manifest.json")
        if checkpoint_manifest.get("status") != "checkpoint":
            raise ValueError(f"{kind} LoRA checkpoint is not published")
        if checkpoint_manifest.get("dataset_build_id") != dataset_build_id:
            raise ValueError(f"{kind} LoRA dataset lineage does not match iteration")
        if checkpoint_manifest.get("source_lineage") != expected_source:
            raise ValueError(f"{kind} LoRA source lineage does not match iteration")
        if checkpoint_manifest.get("opd_kind") != kind:
            raise ValueError(f"{kind} LoRA checkpoint kind mismatch")
        if checkpoint_manifest.get("adapter_revision") != revision:
            raise ValueError(f"{kind} LoRA checkpoint revision mismatch")
        adapter_relative = checkpoint_manifest.get("adapter_path")
        if not isinstance(adapter_relative, str) or not adapter_relative:
            raise ValueError(f"{kind} LoRA adapter path is missing")
        adapter = (checkpoint / adapter_relative).resolve()
        try:
            adapter.relative_to(checkpoint)
        except ValueError as error:
            raise ValueError(f"{kind} LoRA adapter path escapes checkpoint") from error
        if not (adapter / "adapter_config.json").is_file():
            raise ValueError(f"{kind} LoRA adapter config is missing")
        weights = (
            adapter / "adapter_model.safetensors",
            adapter / "adapter_model.bin",
        )
        if sum(path.is_file() for path in weights) != 1:
            raise ValueError(
                f"{kind} LoRA checkpoint must contain exactly one adapter weight file"
            )
        checkpoints[kind] = str(checkpoint)

    metadata_checkpoint = result.metadata.get("latest_checkpoint")
    if (
        not isinstance(metadata_checkpoint, str)
        or Path(metadata_checkpoint).resolve() != training_root
    ):
        raise ValueError("training stage bundle checkpoint metadata does not match manifest")
    if result.metadata.get("child_adapter_revision") != bundle_revision:
        raise ValueError("training stage bundle revision does not match manifest")
    metadata_adapters = result.metadata.get("adapter_checkpoints")
    if metadata_adapters != raw_checkpoints:
        raise ValueError("training stage adapter checkpoint metadata does not match manifest")
    return {
        "adapter_checkpoints": checkpoints,
        "child_adapter_revision": bundle_revision,
        "dataset_build_id": dataset_build_id,
        "latest_checkpoint": str(training_root),
    }


def _publish_promotion(
    request: IterationRequest,
    record: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    rollout = record["stages"]["rollout"]["metadata"]
    training = record["stages"]["training"]["metadata"]
    parent_iteration_id: str | None = None
    if request.parent_iteration_dir is not None:
        parent = _read_json(Path(request.parent_iteration_dir) / "promotion_manifest.json")
        parent_iteration_id = parent["iteration_id"]
    child = {
        "model_revision": request.model_revision,
        "checkpoint": training["latest_checkpoint"],
        "adapter_revision": training["child_adapter_revision"],
    }
    if "adapter_checkpoints" in training:
        child["adapter_checkpoints"] = training["adapter_checkpoints"]
    promotion = {
        "schema_version": 1,
        "status": "promoted",
        "iteration_id": request.iteration_id,
        "iteration": request.iteration,
        "task_ids": list(request.task_ids),
        "completed_train_tasks_before": request.completed_train_tasks_before,
        "completed_train_tasks_after": rollout["completed_train_tasks_after"],
        "dataset_build_id": training["dataset_build_id"],
        "parent": {
            "iteration_id": parent_iteration_id,
            "checkpoint": (
                str(Path(request.parent_checkpoint).resolve())
                if request.parent_checkpoint is not None
                else None
            ),
            "adapter_revision": request.adapter_revision,
        },
        "child": child,
        "memory": {
            "input_snapshot_id": request.input_memory_snapshot_id,
            "output_snapshot_id": rollout["output_memory_snapshot_id"],
        },
    }
    path = Path(request.iteration_dir).resolve() / "promotion_manifest.json"
    if path.exists():
        existing = _read_json(path)
        if existing != promotion:
            raise ValueError("existing promotion manifest does not match iteration")
        return path, existing
    payload = (
        json.dumps(
            promotion,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, payload)
    return path, promotion


def _validate_stage_artifacts(request: IterationRequest, result: StageResult) -> None:
    if not result.artifacts:
        raise ValueError("iteration stage must publish at least one artifact")
    iteration_dir = Path(request.iteration_dir).resolve()
    for path in result.artifacts.values():
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(iteration_dir)
        except ValueError as error:
            raise ValueError(f"stage artifact escapes iteration directory: {resolved}") from error
        if not resolved.exists():
            raise ValueError(f"stage artifact does not exist: {resolved}")
    assert_train_only_artifacts(
        iteration_dir,
        train_task_ids=request.official_train_task_ids,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read iteration JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"iteration JSON artifact must be an object: {path}")
    return value


def _validate_memory_snapshot(request: IterationRequest, snapshot_id: str) -> None:
    if not _SAFE_ID.fullmatch(snapshot_id):
        raise ValueError("memory snapshot ID is invalid")
    root = Path(request.memory_snapshots_dir).resolve()
    snapshot = (root / snapshot_id).resolve()
    try:
        snapshot.relative_to(root)
    except ValueError as error:
        raise ValueError("memory snapshot path escapes snapshot repository") from error
    if not snapshot.is_dir() or not (snapshot / "manifest.json").is_file():
        raise ValueError(f"memory snapshot is missing or incomplete: {snapshot}")
    manifest = _read_json(snapshot / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("memory snapshot schema mismatch")
    if manifest.get("memory_snapshot_id") != snapshot_id:
        raise ValueError("memory snapshot manifest ID mismatch")
    expected_names = {
        "trajectory_memory.json",
        "tip_memory.json",
        "skill_memory.json",
        "tool_memory.json",
    }
    files = manifest.get("files")
    counts = manifest.get("counts")
    if not isinstance(files, Mapping) or set(files) != expected_names:
        raise ValueError("memory snapshot file manifest is incomplete")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"trajectory", "tip", "skill", "tool"}
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError("memory snapshot counts are invalid")
    actual_names = {path.name for path in snapshot.glob("*.json")}
    if actual_names != expected_names | {"manifest.json"}:
        raise ValueError("memory snapshot directory contents are incomplete")
    for name, expected_hash in files.items():
        path = snapshot / name
        if (
            not isinstance(expected_hash, str)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash
        ):
            raise ValueError(f"memory snapshot file hash mismatch: {path}")
