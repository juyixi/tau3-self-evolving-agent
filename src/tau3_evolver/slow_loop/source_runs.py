from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from tau3_evolver.artifacts.maintenance import (
    MAINTENANCE_RECORD_SCHEMA_VERSION,
    maintenance_record_sha256,
)
from tau3_evolver.persistence.jsonl import iter_jsonl_objects
from tau3_evolver.persistence.layout import training_memory_root


ZERO_IMPACT_ADAPTER_REVISION = "zero-impact-init-v1"


def reject_evaluation_artifact_for_training(path: Path) -> Path:
    resolved = Path(path).resolve()
    lowered = tuple(part.casefold() for part in resolved.parts)
    if any(
        lowered[index : index + 2] == ("history", "evaluations")
        for index in range(len(lowered) - 1)
    ):
        raise ValueError(
            f"training input is inside the evaluation quarantine: {resolved}"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class SourceRun:
    path: Path
    run_id: str
    run: Mapping[str, Any]
    episodes: tuple[Mapping[str, Any], ...]
    episodes_path: Path
    run_sha256: str
    episodes_sha256: str
    task_offset: int

    @property
    def runtime_revision(self) -> str:
        return _runtime_revision(self.run["runtime"])

    @property
    def adapter_revision(self) -> str:
        checkpoint = self.run["policy"].get("checkpoint")
        return str(checkpoint or ZERO_IMPACT_ADAPTER_REVISION)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(str(episode["task_id"]) for episode in self.episodes)

    @property
    def summary(self) -> Mapping[str, Any]:
        completed = tuple(
            episode for episode in self.episodes if episode["status"] == "completed"
        )
        failed = tuple(
            episode for episode in self.episodes if episode["status"] == "failed"
        )
        memory = self.run["memory"]
        maintenance = memory.get("maintenance")
        if isinstance(maintenance, Mapping):
            records = maintenance.get("records", ())
            completed_before = maintenance["completed_train_tasks_before"]
            completed_after = maintenance["completed_train_tasks_after"]
        else:
            records = ()
            completed_before = self.task_offset
            completed_after = self.task_offset + len(self.episodes)
        return MappingProxyType(
            {
                "episode_count": len(completed),
                "successful_task_ids": tuple(item["task_id"] for item in completed),
                "failed_task_ids": tuple(item["task_id"] for item in failed),
                "input_memory_snapshot_id": memory["input_snapshot_id"],
                "output_memory_snapshot_id": memory["output_snapshot_id"],
                "total_terminal_reward": sum(
                    float(item["outcome"]["final_reward"]) for item in completed
                ),
                "maintenance_rounds_executed": tuple(
                    record["maintenance_round"] for record in records
                ),
                "completed_train_tasks_before": completed_before,
                "completed_train_tasks_after": completed_after,
            }
        )

    @property
    def maintenance_records(self) -> tuple[Mapping[str, Any], ...]:
        maintenance = self.run["memory"].get("maintenance")
        if not isinstance(maintenance, Mapping):
            return ()
        return tuple(maintenance.get("records", ()))


@dataclass(frozen=True, slots=True)
class SourceRunSet:
    runs: tuple[SourceRun, ...]
    benchmark: str
    memory_generation: int
    model_revision: str
    checkpoint: str | None
    runtime_revision: str
    split_hash: str
    memory_namespace: str
    task_scope: Literal["full", "debug"]
    project_root: Path

    @property
    def adapter_revision(self) -> str:
        return str(self.checkpoint or ZERO_IMPACT_ADAPTER_REVISION)


def load_source_runs(
    paths: Sequence[Path],
    *,
    benchmark: str,
    official_train_task_ids: Sequence[str],
    split_hash: str,
    project_root: Path,
    task_scope: Literal["full", "debug"] = "full",
) -> SourceRunSet:
    if not paths:
        raise ValueError("at least one source run path is required")
    resolved_paths = tuple(reject_evaluation_artifact_for_training(path) for path in paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("source run paths must be unique")

    official = frozenset(str(task_id) for task_id in official_train_task_ids)
    loaded: list[SourceRun] = []
    task_offset = 0
    for path in resolved_paths:
        run = _load_source_run(
            path,
            benchmark=benchmark,
            official_train_task_ids=official,
            split_hash=split_hash,
            project_root=project_root.resolve(),
            task_offset=task_offset,
            task_scope=task_scope,
        )
        loaded.append(run)
        task_offset += len(run.episodes)
    runs = tuple(loaded)
    _validate_run_set(runs)
    first = runs[0].run
    first_memory = first["memory"]
    memory_namespace = str(
        first_memory.get("destination_namespace") or benchmark
    )
    return SourceRunSet(
        runs=runs,
        benchmark=benchmark,
        memory_generation=int(first["memory"]["generation"]),
        model_revision=str(first["policy"]["model_revision"]),
        checkpoint=first["policy"].get("checkpoint"),
        runtime_revision=_runtime_revision(first["runtime"]),
        split_hash=str(first["execution"]["split_hash"]),
        memory_namespace=memory_namespace,
        task_scope=task_scope,
        project_root=project_root.resolve(),
    )


def _load_source_run(
    path: Path,
    *,
    benchmark: str,
    official_train_task_ids: frozenset[str],
    split_hash: str,
    project_root: Path,
    task_offset: int,
    task_scope: Literal["full", "debug"],
) -> SourceRun:
    if not path.is_dir():
        raise ValueError(f"source run directory does not exist: {path}")
    entries = {child.name for child in path.iterdir()}
    if entries != {"run.json", "episodes.jsonl"}:
        raise ValueError(
            f"source run must contain exactly run.json and episodes.jsonl: {path}"
        )
    run_path = path / "run.json"
    episodes_path = path / "episodes.jsonl"
    run = _read_json_object(run_path, "source run record")
    episodes = tuple(iter_jsonl_objects(episodes_path))

    _validate_run(
        run,
        path=path,
        benchmark=benchmark,
        official_train_task_ids=official_train_task_ids,
        split_hash=split_hash,
        episodes=episodes,
        episodes_path=episodes_path,
        task_scope=task_scope,
    )
    _validate_snapshots(run, project_root=project_root)
    return SourceRun(
        path=path,
        run_id=str(run["run_id"]),
        run=_freeze_mapping(run),
        episodes=tuple(_freeze_mapping(item) for item in episodes),
        episodes_path=episodes_path,
        run_sha256=_sha256(run_path),
        episodes_sha256=_sha256(episodes_path),
        task_offset=task_offset,
    )


def _validate_run(
    run: dict[str, Any],
    *,
    path: Path,
    benchmark: str,
    official_train_task_ids: frozenset[str],
    split_hash: str,
    episodes: tuple[dict[str, Any], ...],
    episodes_path: Path,
    task_scope: Literal["full", "debug"],
) -> None:
    if run.get("schema_version") != 1:
        raise ValueError(f"source run schema must be 1: {path}")
    run_id = _nonblank(run, "run_id", "source run")
    if path.name != run_id:
        raise ValueError(f"source run ID does not match directory: {path}")
    execution = _mapping(run, "execution", "source run")
    if execution.get("benchmark") != benchmark:
        raise ValueError(f"source run benchmark must be {benchmark}: {path}")
    if execution.get("mode") != "train" or execution.get("split") != "train":
        raise ValueError(f"source run must be a train execution: {path}")
    actual_scope = execution.get("task_scope", "full")
    if actual_scope != task_scope:
        if actual_scope == "debug" and task_scope == "full":
            raise ValueError(f"debug runs cannot be used as Slow Loop sources: {path}")
        raise ValueError(
            f"source run task scope must be {task_scope!r}: {path}"
        )
    if execution.get("split_hash") != split_hash:
        raise ValueError(f"source run split hash does not match benchmark: {path}")
    if run.get("status") != "completed":
        raise ValueError(f"source run must have completed without failures: {path}")

    policy = _mapping(run, "policy", "source run")
    _nonblank(policy, "model_revision", "source policy")
    checkpoint = policy.get("checkpoint")
    if checkpoint is not None and (
        not isinstance(checkpoint, str) or not checkpoint.strip()
    ):
        raise ValueError(f"source checkpoint is invalid: {path}")
    _mapping(run, "runtime", "source run")
    memory = _mapping(run, "memory", "source run")
    if memory.get("enabled") is not True:
        raise ValueError(f"source run must enable Memory: {path}")
    if not _is_nonnegative_int(memory.get("generation")):
        raise ValueError(f"source Memory generation is invalid: {path}")
    source_namespace = _nonblank(memory, "source_namespace", "source Memory")
    destination_namespace = memory.get("destination_namespace", benchmark)
    expected_destination = (
        f"{benchmark}-debug" if task_scope == "debug" else benchmark
    )
    if destination_namespace != expected_destination:
        raise ValueError(
            f"source destination Memory namespace must be "
            f"{expected_destination!r}: {path}"
        )
    if memory.get("input_snapshot_id") is None:
        raise ValueError(f"source run has no input Memory snapshot: {path}")
    if memory.get("output_snapshot_id") is None:
        raise ValueError(f"source run has no output Memory snapshot: {path}")
    if memory.get("cross_domain") is not (
        source_namespace != destination_namespace
    ):
        raise ValueError(f"source run cross-domain flag is inconsistent: {path}")
    _validate_maintenance(memory, episode_count=len(episodes), path=path)

    _validate_episode_artifact(run, episodes, episodes_path=episodes_path)
    task_ids = [episode.get("task_id") for episode in episodes]
    if (
        not task_ids
        or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError(f"source task IDs must be non-empty and unique: {path}")
    outside = set(task_ids) - official_train_task_ids
    if outside:
        raise ValueError(f"source run contains non-train task IDs: {sorted(outside)}")
    if execution.get("planned_task_count") != len(episodes):
        raise ValueError(f"source planned task count is inconsistent: {path}")
    for episode in episodes:
        _validate_episode(episode, run_id=run_id)
    _validate_summary(run, episodes)


def _validate_episode_artifact(
    run: Mapping[str, Any],
    episodes: tuple[dict[str, Any], ...],
    *,
    episodes_path: Path,
) -> None:
    artifacts = _mapping(run, "artifacts", "source run")
    metadata = _mapping(artifacts, "episodes", "source artifacts")
    payload = episodes_path.read_bytes()
    expected = {
        "path": "episodes.jsonl",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": len(episodes),
    }
    if dict(metadata) != expected:
        raise ValueError("source episode artifact metadata mismatch")


def _validate_episode(episode: Mapping[str, Any], *, run_id: str) -> None:
    if episode.get("schema_version") != 1:
        raise ValueError(f"source episode schema must be 1: {run_id}")
    task_id = _nonblank(episode, "task_id", "source episode")
    if episode.get("status") != "completed":
        raise ValueError(f"source episode is not completed: {run_id}/{task_id}")
    _nonblank(episode, "task_group", "source episode")
    if not _is_nonnegative_int(episode.get("seed")):
        raise ValueError(f"source episode seed is invalid: {run_id}/{task_id}")
    task = _mapping(episode, "task", "source episode")
    if not isinstance(task.get("tools"), list):
        raise ValueError(f"source episode tools are invalid: {run_id}/{task_id}")
    trajectory = episode.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"source episode trajectory is empty: {run_id}/{task_id}")
    outcome = _mapping(episode, "outcome", "source episode")
    reward = outcome.get("final_reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(reward)
    ):
        raise ValueError(f"source episode reward is invalid: {run_id}/{task_id}")
    if outcome.get("steps") != len(trajectory):
        raise ValueError(f"source episode step count is invalid: {run_id}/{task_id}")
    memory = _mapping(episode, "memory", "source episode")
    if memory.get("enabled") is not True:
        raise ValueError(f"source episode has Memory disabled: {run_id}/{task_id}")
    _mapping(memory, "retrieval", "source episode Memory")
    if not isinstance(memory.get("selected_memory_ids"), list):
        raise ValueError(f"source episode selection is invalid: {run_id}/{task_id}")
    if not isinstance(memory.get("writes"), list):
        raise ValueError(f"source episode writes are invalid: {run_id}/{task_id}")


def _validate_summary(run: Mapping[str, Any], episodes: tuple[dict[str, Any], ...]) -> None:
    summary = _mapping(run, "summary", "source run")
    metrics = _mapping(summary, "metrics", "source summary")
    rewards = [float(episode["outcome"]["final_reward"]) for episode in episodes]
    expected = {
        "task_count": len(episodes),
        "completed_count": len(episodes),
        "failure_count": 0,
        "mean_reward": sum(rewards) / len(rewards),
        "pass_rate": sum(reward > 0 for reward in rewards) / len(rewards),
    }
    if dict(metrics) != expected:
        raise ValueError("source run metrics do not match episodes")


def _validate_snapshots(run: Mapping[str, Any], *, project_root: Path) -> None:
    execution = run["execution"]
    memory = run["memory"]
    input_root = training_memory_root(memory["source_namespace"], root=project_root)
    output_namespace = memory.get("destination_namespace", execution["benchmark"])
    output_root = training_memory_root(output_namespace, root=project_root)
    _require_snapshot(input_root, memory["input_snapshot_id"])
    _require_snapshot(output_root, memory["output_snapshot_id"])
    maintenance = memory.get("maintenance")
    if isinstance(maintenance, Mapping):
        for record in maintenance.get("records", ()):
            _require_snapshot(output_root, record["memory_snapshot_id"])


def _validate_maintenance(
    memory: Mapping[str, Any],
    *,
    episode_count: int,
    path: Path,
) -> None:
    maintenance = memory.get("maintenance")
    if maintenance is None:
        return
    if not isinstance(maintenance, Mapping) or set(maintenance) != {
        "period",
        "completed_train_tasks_before",
        "completed_train_tasks_after",
        "records",
        "failures",
    }:
        raise ValueError(f"source maintenance summary is invalid: {path}")
    period = maintenance.get("period")
    before = maintenance.get("completed_train_tasks_before")
    after = maintenance.get("completed_train_tasks_after")
    records = maintenance.get("records")
    failures = maintenance.get("failures")
    if type(period) is not int or period <= 0:
        raise ValueError(f"source maintenance period is invalid: {path}")
    if (
        not _is_nonnegative_int(before)
        or not _is_nonnegative_int(after)
        or after - before != episode_count
    ):
        raise ValueError(f"source maintenance task range is invalid: {path}")
    if not isinstance(records, list) or not isinstance(failures, list):
        raise ValueError(f"source maintenance records are invalid: {path}")
    if failures:
        raise ValueError(f"completed source run contains maintenance failures: {path}")

    rounds: list[int] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "schema_version",
            "maintenance_round",
            "trigger_task_index",
            "period",
            "memory_snapshot_id",
            "diagnostics",
            "commands",
            "looked_up_ids",
            "created_ids",
            "updated_ids",
            "record_sha256",
        }:
            raise ValueError(f"source maintenance record is invalid: {path}")
        round_number = record.get("maintenance_round")
        trigger = record.get("trigger_task_index")
        if (
            record.get("schema_version") != MAINTENANCE_RECORD_SCHEMA_VERSION
            or type(round_number) is not int
            or round_number <= 0
            or type(trigger) is not int
            or trigger != after
            or record.get("period") != period
            or round_number > trigger // period
            or record.get("record_sha256") != maintenance_record_sha256(record)
        ):
            raise ValueError(f"source maintenance provenance is invalid: {path}")
        rounds.append(round_number)
    if rounds != sorted(set(rounds)):
        raise ValueError(f"source maintenance rounds are invalid: {path}")


def _validate_run_set(runs: tuple[SourceRun, ...]) -> None:
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("source runs contain duplicate run ID")
    task_ids = [task_id for run in runs for task_id in run.task_ids]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("source runs contain duplicate task IDs")
    lineages = {
        (
            run.run["execution"]["benchmark"],
            run.run["memory"]["generation"],
            run.run["policy"]["model_revision"],
            run.run["policy"].get("checkpoint"),
            json.dumps(dict(run.run["runtime"]), sort_keys=True),
            run.run["execution"]["split_hash"],
            run.run["execution"].get("task_scope", "full"),
            run.run["memory"].get(
                "destination_namespace", run.run["execution"]["benchmark"]
            ),
        )
        for run in runs
    }
    if len(lineages) != 1:
        raise ValueError("source runs must share one policy and runtime lineage")


def _require_snapshot(memory_root: Path, snapshot_id: str) -> None:
    snapshots_root = (memory_root / "snapshots").resolve()
    snapshot_path = (snapshots_root / snapshot_id).resolve()
    if snapshot_path.parent != snapshots_root:
        raise ValueError(f"Memory snapshot ID escapes snapshot root: {snapshot_id}")
    if not snapshot_path.is_dir():
        raise ValueError(f"Memory snapshot directory does not exist: {snapshot_id}")


def _runtime_revision(origin: Mapping[str, Any]) -> str:
    for key in ("git_commit", "package_version", "source_root"):
        value = origin.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("source runtime origin contains no usable revision")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{label} does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(nested) for key, nested in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    resolved = value.get(key)
    if not isinstance(resolved, Mapping):
        raise ValueError(f"{label} {key} must be an object")
    return resolved


def _nonblank(value: Mapping[str, Any], key: str, label: str) -> str:
    resolved = value.get(key)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{label} {key} must be a non-blank string")
    return resolved


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0
