from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.eval.guard import (
    reject_evaluation_artifact_for_training,
)
from tau3_retail_evolver.io.jsonl import iter_jsonl_objects


_TASK_GROUP = re.compile(r"^retail-actions-v1:[0-9a-f]{64}$")
_FAILURE_EVENTS = frozenset(
    {"EpisodeFailed", "MemoryWriteFailed", "MaintenanceFailed", "MemoryDisabled"}
)
_EPISODE_PREFIX = ("EpisodeStarted", "MemoryCandidatesRetrieved", "MemorySelected")
_EPISODE_SUFFIX = ("EpisodeFinished", "MemoryWriteProposed", "MemoryWriteCommitted")
_MAINTENANCE_EVENTS = frozenset(
    {"MaintenanceStarted", "MaintenanceProposed", "MaintenanceCommitted"}
)


@dataclass(frozen=True, slots=True)
class SourceRun:
    path: Path
    run_id: str
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]
    events_path: Path
    manifest_sha256: str
    summary_sha256: str
    events_sha256: str


@dataclass(frozen=True, slots=True)
class SourceRunSet:
    runs: tuple[SourceRun, ...]
    iteration: int
    model_revision: str
    adapter_revision: str | None
    tau2_commit: str
    split_hash: str
    memory_agent_id: str


def load_source_runs(
    paths: Sequence[Path],
    *,
    catalog: RetailTaskCatalog,
    memory_root: Path,
) -> SourceRunSet:
    if not paths:
        raise ValueError("at least one source run path is required")
    resolved_paths = tuple(Path(path).resolve() for path in paths)
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("source run paths must be unique")

    memory_root = memory_root.resolve()
    loaded = tuple(
        _load_source_run(path, catalog=catalog, memory_root=memory_root)
        for path in resolved_paths
    )
    runs = tuple(sorted(loaded, key=_completed_before))
    _validate_run_set(runs, catalog=catalog, memory_root=memory_root)
    first = runs[0].manifest
    return SourceRunSet(
        runs=runs,
        iteration=first["iteration"],
        model_revision=first["model_revision"],
        adapter_revision=first["adapter_revision"],
        tau2_commit=first["tau2_commit"],
        split_hash=first["split_hash"],
        memory_agent_id=first["rollout_options"]["memory_agent_id"],
    )


def _load_source_run(
    path: Path,
    *,
    catalog: RetailTaskCatalog,
    memory_root: Path,
) -> SourceRun:
    _reject_quarantine_path(path)
    if not path.is_dir():
        raise ValueError(f"source run directory does not exist: {path}")
    manifest_path = path / "manifest.json"
    summary_path = path / "fast_loop_summary.json"
    events_path = path / "rollouts" / "events.jsonl"
    manifest = _read_json_object(manifest_path, "source manifest")
    summary = _read_json_object(summary_path, "source summary")
    events = tuple(iter_jsonl_objects(events_path))

    _validate_manifest(manifest, path=path, catalog=catalog, memory_root=memory_root)
    _validate_summary(summary, manifest=manifest)
    _validate_events(events, manifest=manifest, summary=summary, memory_root=memory_root)
    return SourceRun(
        path=path,
        run_id=manifest["run_id"],
        manifest=_freeze_mapping(manifest),
        summary=_freeze_mapping(summary),
        events_path=events_path,
        manifest_sha256=_sha256(manifest_path),
        summary_sha256=_sha256(summary_path),
        events_sha256=_sha256(events_path),
    )


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    path: Path,
    catalog: RetailTaskCatalog,
    memory_root: Path,
) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError(f"source manifest schema must be 2: {path}")
    run_id = _nonblank(manifest, "run_id", "source manifest")
    if path.name != run_id:
        raise ValueError(f"source manifest run ID does not match directory: {path}")
    if manifest.get("split") != "train":
        raise ValueError(f"source run must use the train split: {path}")
    if not _is_nonnegative_int(manifest.get("iteration")):
        raise ValueError(f"source manifest iteration is invalid: {path}")
    _nonblank(manifest, "model_revision", "source manifest")
    adapter_revision = manifest.get("adapter_revision")
    if adapter_revision is not None and (
        not isinstance(adapter_revision, str) or not adapter_revision.strip()
    ):
        raise ValueError(f"source manifest adapter revision is invalid: {path}")
    _nonblank(manifest, "tau2_commit", "source manifest")
    split_hash = _nonblank(manifest, "split_hash", "source manifest")
    if getattr(catalog, "split_sha256", split_hash) != split_hash:
        raise ValueError(f"source manifest split hash does not match catalog: {path}")
    if not _is_nonnegative_int(manifest.get("seed")):
        raise ValueError(f"source manifest seed is invalid: {path}")

    task_ids = manifest.get("task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
        or len(task_ids) != len(set(task_ids))
    ):
        raise ValueError(f"source manifest task IDs must be non-empty and unique: {path}")
    official_train = set(catalog.task_ids("train"))
    non_train = set(task_ids) - official_train
    if non_train:
        raise ValueError(f"source manifest contains non-train task IDs: {sorted(non_train)}")

    environment = manifest.get("environment_options")
    if not isinstance(environment, Mapping) or environment.get("domain") != "retail":
        raise ValueError(f"source manifest domain must be retail: {path}")
    rollout = manifest.get("rollout_options")
    if not isinstance(rollout, Mapping) or rollout.get("memory_enabled") is not True:
        raise ValueError(f"source run must be memory-enabled: {path}")
    agent_id = _nonblank(rollout, "memory_agent_id", "source rollout options")
    if memory_root.name != "memory" or memory_root.parent.name != agent_id:
        raise ValueError("source Memory agent namespace does not match memory_root")
    _nonblank(manifest, "memory_snapshot_id", "source manifest")


def _validate_summary(summary: dict[str, Any], *, manifest: dict[str, Any]) -> None:
    run_id = manifest["run_id"]
    if summary.get("run_id") != run_id:
        raise ValueError(f"source summary run ID mismatch: {run_id}")
    if summary.get("memory_enabled") is not True:
        raise ValueError(f"source summary must be memory-enabled: {run_id}")
    task_ids = manifest["task_ids"]
    count = summary.get("episode_count")
    before = summary.get("completed_train_tasks_before")
    after = summary.get("completed_train_tasks_after")
    if not _is_nonnegative_int(count) or count != len(task_ids):
        raise ValueError(f"source summary episode count mismatch: {run_id}")
    if not _is_nonnegative_int(before) or not _is_nonnegative_int(after):
        raise ValueError(f"source summary completed-task range is invalid: {run_id}")
    if after - before != count:
        raise ValueError(f"source summary completed-task range mismatch: {run_id}")
    if summary.get("input_memory_snapshot_id") != manifest["memory_snapshot_id"]:
        raise ValueError(f"source summary input snapshot mismatch: {run_id}")
    _nonblank(summary, "output_memory_snapshot_id", "source summary")
    if summary.get("successful_task_ids") != task_ids:
        raise ValueError(f"source summary successful task IDs mismatch: {run_id}")
    reward = summary.get("total_terminal_reward")
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(reward)
    ):
        raise ValueError(f"source summary total reward is invalid: {run_id}")
    maintenance_rounds = summary.get("maintenance_rounds_executed")
    if not isinstance(maintenance_rounds, list) or any(
        not _is_nonnegative_int(round_number) for round_number in maintenance_rounds
    ):
        raise ValueError(f"source summary maintenance rounds are invalid: {run_id}")


def _validate_events(
    events: tuple[dict[str, Any], ...],
    *,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    memory_root: Path,
) -> None:
    run_id = manifest["run_id"]
    if not events:
        raise ValueError(f"source event stream is empty: {run_id}")
    task_ids = manifest["task_ids"]
    events_by_task: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in task_ids}
    first_episode_order: list[str] = []
    rewards: list[float] = []
    referenced_snapshots = {
        manifest["memory_snapshot_id"],
        summary["output_memory_snapshot_id"],
    }
    for event in events:
        event_type = event.get("event_type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError(f"source event type is invalid: {run_id}")
        if event_type in _FAILURE_EVENTS:
            raise ValueError(f"source event stream contains failure event {event_type}: {run_id}")
        if event.get("schema_version") != 2:
            raise ValueError(f"source event schema must be 2: {run_id}")
        expected = {
            "run_id": run_id,
            "iteration": manifest["iteration"],
            "split": "train",
            "mode": "learn",
            "model_revision": manifest["model_revision"],
            "adapter_revision": manifest["adapter_revision"],
            "seed": manifest["seed"],
        }
        if any(event.get(key) != value for key, value in expected.items()):
            raise ValueError(f"source event provenance mismatch: {run_id}")
        snapshot_id = event.get("memory_snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError(f"source event snapshot provenance is invalid: {run_id}")
        referenced_snapshots.add(snapshot_id)
        task_id = event.get("task_id")
        if isinstance(task_id, str) and task_id.startswith("maintenance-round-"):
            if event_type not in _MAINTENANCE_EVENTS:
                raise ValueError(f"source maintenance event type is invalid: {run_id}")
            continue
        if task_id not in events_by_task:
            raise ValueError(f"source event references undeclared task ID: {run_id}")
        task_group = event.get("task_group")
        if not isinstance(task_group, str) or not _TASK_GROUP.fullmatch(task_group):
            raise ValueError(f"source event task group is invalid: {run_id}")
        if event_type == "EpisodeStarted":
            first_episode_order.append(task_id)
        if event_type == "EpisodeFinished":
            reward = event.get("final_reward")
            if not isinstance(reward, (int, float)) or isinstance(reward, bool):
                raise ValueError(f"source terminal reward is invalid: {run_id}")
            rewards.append(float(reward))
        events_by_task[task_id].append(event)

    if first_episode_order != task_ids:
        raise ValueError(f"source episode order does not match manifest: {run_id}")
    if events_by_task[task_ids[0]][0]["memory_snapshot_id"] != manifest[
        "memory_snapshot_id"
    ]:
        raise ValueError(f"source first episode snapshot mismatch: {run_id}")
    for task_id, task_events in events_by_task.items():
        _validate_episode_lifecycle(task_events, run_id=run_id, task_id=task_id)
    if len(rewards) != len(task_ids) or not math.isclose(
        sum(rewards), float(summary["total_terminal_reward"]), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"source summary reward does not match events: {run_id}")
    for snapshot_id in referenced_snapshots:
        _require_snapshot(memory_root, snapshot_id)


def _validate_episode_lifecycle(
    events: list[dict[str, Any]],
    *,
    run_id: str,
    task_id: str,
) -> None:
    event_types = tuple(event.get("event_type") for event in events)
    if event_types[:3] != _EPISODE_PREFIX or event_types[-3:] != _EPISODE_SUFFIX:
        raise ValueError(f"source episode lifecycle is incomplete: {run_id}/{task_id}")
    turns = event_types[3:-3]
    if not turns or len(turns) % 2:
        raise ValueError(f"source episode lifecycle has invalid turns: {run_id}/{task_id}")
    for index in range(0, len(turns), 2):
        if turns[index : index + 2] != ("DecisionMade", "EnvironmentStepped"):
            raise ValueError(f"source episode lifecycle has invalid turns: {run_id}/{task_id}")
        turn = index // 2
        if events[index + 3].get("turn") != turn or events[index + 4].get("turn") != turn:
            raise ValueError(f"source episode lifecycle turn mismatch: {run_id}/{task_id}")
    finished = events[-3]
    if finished.get("steps") != len(turns) // 2:
        raise ValueError(f"source episode lifecycle step count mismatch: {run_id}/{task_id}")
    if len({event["memory_snapshot_id"] for event in events}) != 1:
        raise ValueError(f"source episode snapshot provenance mismatch: {run_id}/{task_id}")
    if len({event["task_group"] for event in events}) != 1:
        raise ValueError(f"source episode task group mismatch: {run_id}/{task_id}")


def _validate_run_set(
    runs: tuple[SourceRun, ...],
    *,
    catalog: RetailTaskCatalog,
    memory_root: Path,
) -> None:
    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("source runs contain duplicate run ID")
    lineages = {
        (
            run.manifest["iteration"],
            run.manifest["model_revision"],
            run.manifest["adapter_revision"],
            run.manifest["tau2_commit"],
            run.manifest["split_hash"],
            run.manifest["environment_options"]["domain"],
            run.manifest["rollout_options"]["memory_agent_id"],
        )
        for run in runs
    }
    if len(lineages) != 1:
        raise ValueError("source runs must share one on-policy policy lineage")
    all_task_ids = [task_id for run in runs for task_id in run.manifest["task_ids"]]
    official_train = set(catalog.task_ids("train"))
    if not set(all_task_ids) <= official_train:
        raise ValueError("source runs contain task IDs outside the official train split")

    for previous, current in zip(runs, runs[1:], strict=False):
        if previous.summary["completed_train_tasks_after"] != current.summary[
            "completed_train_tasks_before"
        ]:
            raise ValueError("source run task range continuity is broken")
        if previous.summary["output_memory_snapshot_id"] != current.manifest[
            "memory_snapshot_id"
        ]:
            raise ValueError("source run snapshot continuity is broken")
    for run in runs:
        _require_snapshot(memory_root, run.manifest["memory_snapshot_id"])
        _require_snapshot(memory_root, run.summary["output_memory_snapshot_id"])


def _require_snapshot(memory_root: Path, snapshot_id: str) -> None:
    snapshots_root = (memory_root / "snapshots").resolve()
    snapshot_path = (snapshots_root / snapshot_id).resolve()
    if snapshot_path.parent != snapshots_root:
        raise ValueError(f"memory snapshot ID escapes snapshot root: {snapshot_id}")
    if not snapshot_path.is_dir():
        raise ValueError(f"memory snapshot directory does not exist: {snapshot_id}")


def _completed_before(run: SourceRun) -> int:
    return run.summary["completed_train_tasks_before"]


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


def _nonblank(value: Mapping[str, Any], key: str, label: str) -> str:
    resolved = value.get(key)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{label} {key} must be a non-blank string")
    return resolved


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _reject_quarantine_path(path: Path) -> None:
    reject_evaluation_artifact_for_training(path)
