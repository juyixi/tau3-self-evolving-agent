from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Literal
import uuid

from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.slow_loop.evidence import build_evidence
from tau3_retail_evolver.slow_loop.source_runs import load_source_runs
from tau3_retail_evolver.slow_loop.task_grouping import RETAIL_TASK_GROUP


_SAFE_BUILD_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_TERMINAL_EVENTS = frozenset({"EpisodeFinished", "EpisodeFailed", "TaskFailed"})
_FAILURE_EVENTS = frozenset(
    {
        "EpisodeFailed",
        "TaskFailed",
        "MemoryWriteFailed",
        "MaintenanceFailed",
        "MaintenanceTaskFailed",
    }
)
_MAINTENANCE_LIFECYCLE = (
    "MaintenanceStarted",
    "MaintenanceProposed",
    "MaintenanceCommitted",
)
_SAFE_REPLAY_FIELDS = (
    "memory_id",
    "tier",
    "tier_schema_version",
    "payload",
    "content",
    "source_task_ids",
    "created_round",
)


@dataclass(frozen=True, slots=True)
class CanonicalizeRequest:
    source_run_paths: tuple[Path, ...]
    maintenance_event_paths: tuple[Path, ...]
    output_root: Path
    build_id: str
    final_memory_snapshot_id: str
    maintenance_period: int
    expected_seeds: tuple[int, ...]
    catalog: RetailTaskCatalog
    memory_root: Path
    deep_validate: bool = True


@dataclass(frozen=True, slots=True)
class CanonicalizeResult:
    root: Path
    index_path: Path
    source_run_paths: tuple[Path, ...]
    index: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _SourceRun:
    path: Path
    run_id: str
    manifest: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    raw_lines: tuple[str, ...]
    priority: int
    manifest_sha256: str
    events_sha256: str

    @property
    def seed(self) -> int:
        return self.manifest["seed"]


@dataclass(frozen=True, slots=True)
class _EventSource:
    path: Path
    source_id: str
    rows: tuple[dict[str, Any], ...]
    raw_lines: tuple[str, ...]
    priority: int
    events_sha256: str


@dataclass(frozen=True, slots=True)
class _TaskCandidate:
    source: _SourceRun
    task_id: str
    start_line: int
    end_line: int
    rows: tuple[dict[str, Any], ...]
    outcome: Literal["complete", "failed"]
    reason: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            str(self.source.path),
            self.start_line,
            self.end_line,
            self.outcome,
        )


@dataclass(frozen=True, slots=True)
class _MaintenanceCandidate:
    source: _EventSource
    maintenance_round: int
    start_line: int
    end_line: int
    rows: tuple[dict[str, Any], ...]
    outcome: Literal["complete", "failed"]
    reason: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            str(self.source.path),
            self.start_line,
            self.end_line,
            self.outcome,
        )


@dataclass(frozen=True, slots=True)
class _CanonicalRunPlan:
    source: _SourceRun
    selected: tuple[_TaskCandidate, ...]
    completed_before: int
    completed_after: int
    input_snapshot_id: str
    output_snapshot_id: str
    maintenance_after: Mapping[int, tuple[_MaintenanceCandidate, ...]]


def canonicalize_opd_sources(request: CanonicalizeRequest) -> CanonicalizeResult:
    if not isinstance(request, CanonicalizeRequest):
        raise TypeError("request must be a CanonicalizeRequest")
    build_id = _validate_build_id(request.build_id)
    if type(request.maintenance_period) is not int or request.maintenance_period <= 0:
        raise ValueError("maintenance_period must be a positive integer")
    if (
        not request.expected_seeds
        or len(request.expected_seeds) != len(set(request.expected_seeds))
        or any(type(seed) is not int or seed < 0 for seed in request.expected_seeds)
    ):
        raise ValueError("expected_seeds must contain unique non-negative integers")
    if not request.source_run_paths:
        raise ValueError("at least one source run is required")

    output_root = Path(request.output_root).resolve()
    final_root = output_root / build_id
    if final_root.exists():
        raise FileExistsError(f"refusing to overwrite canonical source: {final_root}")
    memory_root = Path(request.memory_root).resolve()
    _require_snapshot(memory_root, request.final_memory_snapshot_id)

    sources = _load_sources(request.source_run_paths)
    _validate_lineage(sources, expected_seeds=request.expected_seeds)
    anchors = _anchor_task_orders(sources, expected_seeds=request.expected_seeds)
    candidates = tuple(
        candidate
        for source in sources
        for candidate in _task_candidates(source)
    )
    selected, candidate_rows = _select_task_candidates(
        candidates,
        anchors=anchors,
        sources=sources,
    )
    event_sources = _event_sources(
        sources,
        request.maintenance_event_paths,
    )
    maintenance_candidates = tuple(
        candidate
        for source in event_sources
        for candidate in _maintenance_candidates(source)
    )
    selected_maintenance, maintenance_rows = _select_maintenance_candidates(
        maintenance_candidates,
        total_tasks=sum(len(task_ids) for task_ids in anchors.values()),
        maintenance_period=request.maintenance_period,
    )
    plans = _plan_runs(
        sources,
        selected=selected,
        anchors=anchors,
        selected_maintenance=selected_maintenance,
        maintenance_period=request.maintenance_period,
        final_memory_snapshot_id=request.final_memory_snapshot_id,
    )
    _require_plan_snapshots(plans, memory_root=memory_root)

    output_root.mkdir(parents=True, exist_ok=True)
    temp_root = output_root / f".{build_id}.tmp-{uuid.uuid4().hex}"
    try:
        temp_root.mkdir(parents=False, exist_ok=False)
        run_rows, provenance = _write_canonical_runs(
            temp_root,
            plans=plans,
            maintenance_period=request.maintenance_period,
        )
        temp_source_paths = tuple(temp_root / plan.source.run_id for plan in plans)
        loaded = load_source_runs(
            temp_source_paths,
            catalog=request.catalog,
            memory_root=memory_root,
        )
        deep_validation: dict[str, Any] = {
            "source_loader_passed": True,
            "evidence_builder_passed": False,
        }
        if request.deep_validate:
            ledger = build_evidence(loaded, memory_root=memory_root)
            if any(episode.task_group != RETAIL_TASK_GROUP for episode in ledger.episodes):
                raise ValueError("canonical evidence contains a non-retail-v2 task group")
            expected_episodes = sum(
                candidate.outcome == "complete" for candidate in selected.values()
            )
            if len(ledger.episodes) != expected_episodes:
                raise ValueError("canonical evidence episode count mismatch")
            if len(ledger.maintenance) != len(selected_maintenance):
                raise ValueError("canonical evidence maintenance count mismatch")
            deep_validation.update(
                {
                    "evidence_builder_passed": True,
                    "evidence_episode_count": len(ledger.episodes),
                    "evidence_maintenance_count": len(ledger.maintenance),
                    "canonical_task_group": RETAIL_TASK_GROUP,
                }
            )

        index = _build_index(
            request,
            sources=sources,
            anchors=anchors,
            plans=plans,
            candidate_rows=candidate_rows,
            maintenance_rows=maintenance_rows,
            selected=selected,
            selected_maintenance=selected_maintenance,
            run_rows=run_rows,
            provenance=provenance,
            deep_validation=deep_validation,
        )
        index_path = temp_root / "canonical_index.json"
        _write_json(index_path, index)
        temp_root.rename(final_root)
        _fsync_directory(output_root)
    except BaseException:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        raise

    final_index_path = final_root / "canonical_index.json"
    final_index = _read_json_object(final_index_path, "canonical index")
    return CanonicalizeResult(
        root=final_root,
        index_path=final_index_path,
        source_run_paths=tuple(final_root / plan.source.run_id for plan in plans),
        index=final_index,
    )


def _load_sources(paths: Sequence[Path]) -> tuple[_SourceRun, ...]:
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(resolved) != len(set(resolved)):
        raise ValueError("source run paths must be unique")
    loaded: list[_SourceRun] = []
    for priority, path in enumerate(resolved):
        manifest_path = path / "manifest.json"
        events_path = path / "rollouts" / "events.jsonl"
        manifest = _read_json_object(manifest_path, "source manifest")
        raw_lines = tuple(events_path.read_text(encoding="utf-8").splitlines())
        rows = tuple(json.loads(line) for line in raw_lines if line.strip())
        if len(rows) != len(raw_lines):
            raise ValueError(f"source event stream contains blank lines: {events_path}")
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"source run ID is invalid: {path}")
        if path.name != run_id:
            raise ValueError(f"source run ID does not match directory: {path}")
        if manifest.get("schema_version") != 2 or manifest.get("split") != "train":
            raise ValueError(f"source run must be a schema-2 train run: {path}")
        seed = manifest.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError(f"source seed is invalid: {path}")
        task_ids = manifest.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or not all(isinstance(task_id, str) and task_id for task_id in task_ids)
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError(f"source task IDs are invalid: {path}")
        loaded.append(
            _SourceRun(
                path=path,
                run_id=run_id,
                manifest=manifest,
                rows=rows,
                raw_lines=raw_lines,
                priority=priority,
                manifest_sha256=_sha256(manifest_path),
                events_sha256=_sha256(events_path),
            )
        )
    return tuple(loaded)


def _validate_lineage(
    sources: Sequence[_SourceRun],
    *,
    expected_seeds: Sequence[int],
) -> None:
    lineages = {
        (
            source.manifest.get("iteration"),
            source.manifest.get("model_revision"),
            source.manifest.get("adapter_revision"),
            source.manifest.get("tau2_commit"),
            source.manifest.get("split_hash"),
            source.manifest.get("environment_options", {}).get("domain"),
            source.manifest.get("rollout_options", {}).get("memory_agent_id"),
        )
        for source in sources
    }
    if len(lineages) != 1:
        raise ValueError("source runs must share one on-policy policy lineage")
    actual_seeds = {source.seed for source in sources}
    if actual_seeds != set(expected_seeds):
        raise ValueError(
            f"source seeds mismatch: expected {sorted(expected_seeds)}, "
            f"got {sorted(actual_seeds)}"
        )
    if any(source.manifest.get("rollout_options", {}).get("memory_enabled") is not True for source in sources):
        raise ValueError("all source runs must be memory-enabled")


def _anchor_task_orders(
    sources: Sequence[_SourceRun],
    *,
    expected_seeds: Sequence[int],
) -> dict[int, tuple[str, ...]]:
    anchors: dict[int, tuple[str, ...]] = {}
    for seed in expected_seeds:
        matching = [source for source in sources if source.seed == seed]
        anchor = max(
            matching,
            key=lambda source: (
                len(source.manifest["task_ids"]),
                -source.priority,
            ),
        )
        task_ids = tuple(anchor.manifest["task_ids"])
        anchor_set = set(task_ids)
        for source in matching:
            if not set(source.manifest["task_ids"]) <= anchor_set:
                raise ValueError(
                    f"source run contains task outside seed-{seed} anchor: {source.run_id}"
                )
        anchors[seed] = task_ids
    return anchors


def _task_candidates(source: _SourceRun) -> tuple[_TaskCandidate, ...]:
    candidates: list[_TaskCandidate] = []
    cursor = 0
    while cursor < len(source.rows):
        event = source.rows[cursor]
        event_type = event.get("event_type")
        if event_type == "EpisodeStarted":
            end = _episode_candidate_end(source.rows, cursor)
            rows = source.rows[cursor:end]
            task_id = event.get("task_id")
            if isinstance(task_id, str) and task_id:
                complete, reason = _classify_episode(rows)
                candidates.append(
                    _TaskCandidate(
                        source=source,
                        task_id=task_id,
                        start_line=cursor + 1,
                        end_line=end,
                        rows=rows,
                        outcome="complete" if complete else "failed",
                        reason=reason,
                    )
                )
            cursor = max(end, cursor + 1)
            continue
        if event_type in {"TaskFailed", "EpisodeFailed"}:
            task_id = event.get("task_id")
            if isinstance(task_id, str) and not task_id.startswith("maintenance-round-"):
                candidates.append(
                    _TaskCandidate(
                        source=source,
                        task_id=task_id,
                        start_line=cursor + 1,
                        end_line=cursor + 1,
                        rows=(event,),
                        outcome="failed",
                        reason=_failure_reason(event),
                    )
                )
        cursor += 1
    return tuple(candidates)


def _episode_candidate_end(rows: Sequence[dict[str, Any]], start: int) -> int:
    task_id = rows[start].get("task_id")
    cursor = start + 1
    while cursor < len(rows):
        event = rows[cursor]
        event_type = event.get("event_type")
        if event_type in {"EpisodeStarted", "TaskFailed", "MaintenanceStarted"}:
            break
        if event.get("task_id") != task_id:
            break
        cursor += 1
        if event_type in {"MemoryWriteCommitted", "EpisodeFailed"}:
            break
    return cursor


def _classify_episode(rows: Sequence[dict[str, Any]]) -> tuple[bool, str]:
    event_types = tuple(event.get("event_type") for event in rows)
    if len(event_types) < 8:
        return False, _failure_reason(rows[-1])
    if event_types[:3] != (
        "EpisodeStarted",
        "MemoryCandidatesRetrieved",
        "MemorySelected",
    ):
        return False, "invalid_retrieval_lifecycle"
    if event_types[-3:] != (
        "EpisodeFinished",
        "MemoryWriteProposed",
        "MemoryWriteCommitted",
    ):
        return False, _failure_reason(rows[-1])
    turns = event_types[3:-3]
    if not turns or len(turns) % 2:
        return False, "invalid_turn_lifecycle"
    for offset in range(0, len(turns), 2):
        if turns[offset : offset + 2] != ("DecisionMade", "EnvironmentStepped"):
            return False, "invalid_turn_lifecycle"
    task_ids = {event.get("task_id") for event in rows}
    snapshots = {event.get("memory_snapshot_id") for event in rows}
    if len(task_ids) != 1 or len(snapshots) != 1:
        return False, "mixed_episode_provenance"
    return True, "complete_lifecycle"


def _failure_reason(event: Mapping[str, Any]) -> str:
    rendered = json.dumps(event, sort_keys=True, ensure_ascii=False).casefold()
    if "tokenizer" in rendered:
        return "tokenizer_infrastructure_failure"
    if "http 400" in rendered or "http400" in rendered or "badrequest" in rendered:
        return "http_400_infrastructure_failure"
    if "context" in rendered and (
        "length" in rendered or "window" in rendered or "maximum" in rendered
    ):
        return "context_infrastructure_failure"
    event_type = event.get("event_type")
    if event_type == "TaskFailed":
        return "task_failed"
    if event_type == "EpisodeFailed":
        return "episode_failed"
    if event_type in _FAILURE_EVENTS:
        return str(event_type).casefold()
    return "incomplete_lifecycle"


def _select_task_candidates(
    candidates: Sequence[_TaskCandidate],
    *,
    anchors: Mapping[int, Sequence[str]],
    sources: Sequence[_SourceRun],
) -> tuple[
    dict[tuple[int, str], _TaskCandidate],
    tuple[dict[str, Any], ...],
]:
    by_key: dict[tuple[int, str], list[_TaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_key[(candidate.source.seed, candidate.task_id)].append(candidate)
    selected: dict[tuple[int, str], _TaskCandidate] = {}
    for seed, task_ids in anchors.items():
        for task_id in task_ids:
            key = (seed, task_id)
            available = sorted(
                by_key.get(key, ()),
                key=lambda item: (
                    item.source.priority,
                    item.start_line,
                    item.end_line,
                ),
            )
            complete = [item for item in available if item.outcome == "complete"]
            if complete:
                selected[key] = complete[0]
            elif available:
                selected[key] = available[0]
            else:
                raise ValueError(f"logical train task has no terminal evidence: {key}")

    selected_keys = {candidate.key for candidate in selected.values()}
    rows: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.source.seed,
            item.task_id,
            item.source.priority,
            item.start_line,
        ),
    ):
        logical_key = (candidate.source.seed, candidate.task_id)
        chosen = candidate.key in selected_keys
        if chosen and candidate.outcome == "complete":
            decision = "selected_complete"
        elif chosen:
            decision = "selected_failure_marker"
        elif candidate.outcome == "complete":
            decision = "duplicate_complete"
        else:
            decision = candidate.reason
        rows.append(
            {
                "logical_key": [candidate.source.seed, candidate.task_id],
                "source_run_id": candidate.source.run_id,
                "source_events_path": str(
                    candidate.source.path / "rollouts" / "events.jsonl"
                ),
                "source_event_start": candidate.start_line,
                "source_event_end": candidate.end_line,
                "source_event_sha256": _event_block_hash(candidate.rows),
                "candidate_outcome": candidate.outcome,
                "selected": chosen,
                "decision": decision,
            }
        )
    expected = {
        (seed, task_id)
        for seed, task_ids in anchors.items()
        for task_id in task_ids
    }
    if set(selected) != expected:
        raise ValueError("canonical task selection does not cover all logical keys")
    selected_by_source = defaultdict(list)
    for candidate in selected.values():
        selected_by_source[candidate.source.run_id].append(candidate)
    if any(
        len({candidate.task_id for candidate in values}) != len(values)
        for values in selected_by_source.values()
    ):
        raise ValueError("canonical source run contains a duplicate selected task")
    source_ids = {source.run_id for source in sources}
    if not set(selected_by_source) <= source_ids:
        raise ValueError("canonical selection references an unknown source run")
    return selected, tuple(rows)


def _event_sources(
    sources: Sequence[_SourceRun],
    maintenance_paths: Sequence[Path],
) -> tuple[_EventSource, ...]:
    values: list[_EventSource] = [
        _EventSource(
            path=source.path / "rollouts" / "events.jsonl",
            source_id=source.run_id,
            rows=source.rows,
            raw_lines=source.raw_lines,
            priority=source.priority,
            events_sha256=source.events_sha256,
        )
        for source in sources
    ]
    seen = {value.path.resolve() for value in values}
    base_priority = len(values)
    for offset, raw_path in enumerate(maintenance_paths):
        path = Path(raw_path).resolve()
        if path in seen:
            continue
        raw_lines = tuple(path.read_text(encoding="utf-8").splitlines())
        rows = tuple(json.loads(line) for line in raw_lines if line.strip())
        if len(rows) != len(raw_lines):
            raise ValueError(f"maintenance event stream contains blank lines: {path}")
        values.append(
            _EventSource(
                path=path,
                source_id=path.parent.parent.name,
                rows=rows,
                raw_lines=raw_lines,
                priority=base_priority + offset,
                events_sha256=_sha256(path),
            )
        )
        seen.add(path)
    return tuple(values)


def _maintenance_candidates(
    source: _EventSource,
) -> tuple[_MaintenanceCandidate, ...]:
    candidates: list[_MaintenanceCandidate] = []
    cursor = 0
    while cursor < len(source.rows):
        event = source.rows[cursor]
        event_type = event.get("event_type")
        maintenance_round = event.get("maintenance_round")
        if event_type == "MaintenanceStarted" and type(maintenance_round) is int:
            block = source.rows[cursor : cursor + 3]
            if tuple(item.get("event_type") for item in block) == _MAINTENANCE_LIFECYCLE:
                candidates.append(
                    _MaintenanceCandidate(
                        source=source,
                        maintenance_round=maintenance_round,
                        start_line=cursor + 1,
                        end_line=cursor + 3,
                        rows=tuple(block),
                        outcome="complete",
                        reason="complete_lifecycle",
                    )
                )
                cursor += 3
                continue
            end = min(cursor + 2, len(source.rows))
            candidates.append(
                _MaintenanceCandidate(
                    source=source,
                    maintenance_round=maintenance_round,
                    start_line=cursor + 1,
                    end_line=end,
                    rows=tuple(source.rows[cursor:end]),
                    outcome="failed",
                    reason="failed_maintenance",
                )
            )
        elif event_type in {"MaintenanceFailed", "MaintenanceTaskFailed"} and type(
            maintenance_round
        ) is int:
            candidates.append(
                _MaintenanceCandidate(
                    source=source,
                    maintenance_round=maintenance_round,
                    start_line=cursor + 1,
                    end_line=cursor + 1,
                    rows=(event,),
                    outcome="failed",
                    reason="failed_maintenance",
                )
            )
        cursor += 1
    return tuple(candidates)


def _select_maintenance_candidates(
    candidates: Sequence[_MaintenanceCandidate],
    *,
    total_tasks: int,
    maintenance_period: int,
) -> tuple[
    dict[int, _MaintenanceCandidate],
    tuple[dict[str, Any], ...],
]:
    expected_rounds = tuple(range(1, total_tasks // maintenance_period + 1))
    by_round: dict[int, list[_MaintenanceCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_round[candidate.maintenance_round].append(candidate)
    selected: dict[int, _MaintenanceCandidate] = {}
    for round_number in expected_rounds:
        complete = sorted(
            (
                candidate
                for candidate in by_round.get(round_number, ())
                if candidate.outcome == "complete"
            ),
            key=lambda item: (
                item.source.priority,
                item.start_line,
                item.end_line,
            ),
        )
        if not complete:
            raise ValueError(
                f"maintenance round {round_number} has no complete lifecycle"
            )
        selected[round_number] = complete[0]

    selected_keys = {candidate.key for candidate in selected.values()}
    rows: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.maintenance_round,
            item.source.priority,
            item.start_line,
        ),
    ):
        chosen = candidate.key in selected_keys
        rows.append(
            {
                "maintenance_round": candidate.maintenance_round,
                "source_id": candidate.source.source_id,
                "source_events_path": str(candidate.source.path),
                "source_event_start": candidate.start_line,
                "source_event_end": candidate.end_line,
                "source_event_sha256": _event_block_hash(candidate.rows),
                "candidate_outcome": candidate.outcome,
                "selected": chosen,
                "decision": (
                    "selected_complete"
                    if chosen
                    else (
                        "duplicate_complete"
                        if candidate.outcome == "complete"
                        else candidate.reason
                    )
                ),
            }
        )
    return selected, tuple(rows)


def _plan_runs(
    sources: Sequence[_SourceRun],
    *,
    selected: Mapping[tuple[int, str], _TaskCandidate],
    anchors: Mapping[int, Sequence[str]],
    selected_maintenance: Mapping[int, _MaintenanceCandidate],
    maintenance_period: int,
    final_memory_snapshot_id: str,
) -> tuple[_CanonicalRunPlan, ...]:
    selected_by_source: dict[str, list[_TaskCandidate]] = defaultdict(list)
    for candidate in selected.values():
        selected_by_source[candidate.source.run_id].append(candidate)

    ordered_sources: list[_SourceRun] = []
    for seed in anchors:
        ordered_sources.extend(
            source
            for source in sources
            if source.seed == seed and source.run_id in selected_by_source
        )
    if len({source.run_id for source in ordered_sources}) != len(ordered_sources):
        raise ValueError("canonical source run IDs must be unique")

    ordered_selected: list[tuple[_SourceRun, tuple[_TaskCandidate, ...]]] = []
    for source in ordered_sources:
        items = tuple(
            sorted(
                selected_by_source[source.run_id],
                key=lambda item: (item.start_line, item.end_line),
            )
        )
        ordered_selected.append((source, items))

    total_tasks = sum(len(items) for _, items in ordered_selected)
    if total_tasks != sum(len(task_ids) for task_ids in anchors.values()):
        raise ValueError("canonical run plan task count mismatch")

    maintenance_after: dict[str, dict[int, list[_MaintenanceCandidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    cumulative = 0
    for source, items in ordered_selected:
        for local_index, _ in enumerate(items, start=1):
            cumulative += 1
            if cumulative % maintenance_period == 0:
                round_number = cumulative // maintenance_period
                maintenance_after[source.run_id][local_index].append(
                    selected_maintenance[round_number]
                )

    input_snapshots = [
        _candidate_snapshot(items[0])
        for _, items in ordered_selected
    ]
    plans: list[_CanonicalRunPlan] = []
    completed_before = 0
    for index, (source, items) in enumerate(ordered_selected):
        output_snapshot = (
            input_snapshots[index + 1]
            if index + 1 < len(input_snapshots)
            else final_memory_snapshot_id
        )
        completed_after = completed_before + len(items)
        plans.append(
            _CanonicalRunPlan(
                source=source,
                selected=items,
                completed_before=completed_before,
                completed_after=completed_after,
                input_snapshot_id=input_snapshots[index],
                output_snapshot_id=output_snapshot,
                maintenance_after={
                    local_index: tuple(values)
                    for local_index, values in maintenance_after[source.run_id].items()
                },
            )
        )
        completed_before = completed_after
    return tuple(plans)


def _candidate_snapshot(candidate: _TaskCandidate) -> str:
    for event in candidate.rows:
        snapshot_id = event.get("memory_snapshot_id")
        if isinstance(snapshot_id, str) and snapshot_id:
            return snapshot_id
    snapshot_id = candidate.source.manifest.get("memory_snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError(
            f"task candidate has no snapshot provenance: "
            f"{candidate.source.run_id}/{candidate.task_id}"
        )
    return snapshot_id


def _require_plan_snapshots(
    plans: Sequence[_CanonicalRunPlan],
    *,
    memory_root: Path,
) -> None:
    snapshots = {
        plan.input_snapshot_id
        for plan in plans
    } | {plan.output_snapshot_id for plan in plans}
    for plan in plans:
        for candidate in plan.selected:
            snapshots.update(
                event["memory_snapshot_id"]
                for event in candidate.rows
                if isinstance(event.get("memory_snapshot_id"), str)
            )
        for candidates in plan.maintenance_after.values():
            for candidate in candidates:
                snapshots.update(
                    event["memory_snapshot_id"]
                    for event in candidate.rows
                    if isinstance(event.get("memory_snapshot_id"), str)
                )
    for snapshot_id in snapshots:
        _require_snapshot(memory_root, snapshot_id)


def _write_canonical_runs(
    root: Path,
    *,
    plans: Sequence[_CanonicalRunPlan],
    maintenance_period: int,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    run_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for plan in plans:
        run_dir = root / plan.source.run_id
        events_path = run_dir / "rollouts" / "events.jsonl"
        events: list[dict[str, Any]] = []
        task_outcomes: list[tuple[str, str, float]] = []
        maintenance_rounds: list[int] = []
        for local_index, candidate in enumerate(plan.selected, start=1):
            start_line = len(events) + 1
            if candidate.outcome == "complete":
                block, event_normalizations = _normalize_episode(candidate)
                final_reward = float(block[-3]["final_reward"])
                terminal_type = "EpisodeFinished"
            else:
                block = [_failure_marker(candidate)]
                event_normalizations = ()
                final_reward = 0.0
                terminal_type = "TaskFailed"
            events.extend(block)
            end_line = len(events)
            task_outcomes.append(
                (candidate.task_id, terminal_type, final_reward)
            )
            provenance.append(
                {
                    "logical_key": [plan.source.seed, candidate.task_id],
                    "status": (
                        "included_episode"
                        if candidate.outcome == "complete"
                        else "excluded_failure"
                    ),
                    "source_run_id": candidate.source.run_id,
                    "source_events_path": str(
                        candidate.source.path / "rollouts" / "events.jsonl"
                    ),
                    "source_event_start": candidate.start_line,
                    "source_event_end": candidate.end_line,
                    "source_event_sha256": _event_block_hash(candidate.rows),
                    "canonical_run_id": plan.source.run_id,
                    "canonical_event_start": start_line,
                    "canonical_event_end": end_line,
                    "canonical_event_sha256": _event_block_hash(block),
                    "reason": candidate.reason,
                    "event_normalizations": _json_copy(event_normalizations),
                }
            )
            for maintenance in plan.maintenance_after.get(local_index, ()):
                trigger = maintenance.maintenance_round * maintenance_period
                normalized = _normalize_maintenance(
                    maintenance,
                    plan=plan,
                    trigger_task_index=trigger,
                    maintenance_period=maintenance_period,
                )
                maintenance_start = len(events) + 1
                events.extend(normalized)
                maintenance_end = len(events)
                maintenance_rounds.append(maintenance.maintenance_round)
                provenance.append(
                    {
                        "maintenance_round": maintenance.maintenance_round,
                        "status": "included_maintenance",
                        "source_id": maintenance.source.source_id,
                        "source_events_path": str(maintenance.source.path),
                        "source_event_start": maintenance.start_line,
                        "source_event_end": maintenance.end_line,
                        "source_event_sha256": _event_block_hash(maintenance.rows),
                        "canonical_run_id": plan.source.run_id,
                        "canonical_event_start": maintenance_start,
                        "canonical_event_end": maintenance_end,
                        "canonical_event_sha256": _event_block_hash(normalized),
                        "normalized_trigger_task_index": trigger,
                    }
                )

        manifest = _json_copy(plan.source.manifest)
        manifest.update(
            {
                "run_id": plan.source.run_id,
                "seed": plan.source.seed,
                "split": "train",
                "task_ids": [task_id for task_id, _, _ in task_outcomes],
                "memory_snapshot_id": plan.input_snapshot_id,
                "canonical_source": True,
            }
        )
        rollout = manifest.get("rollout_options")
        if isinstance(rollout, dict):
            rollout["task_order_seed"] = plan.source.seed
            rollout["memory_enabled"] = True

        failed_task_ids = [
            task_id
            for task_id, terminal_type, _ in task_outcomes
            if terminal_type == "TaskFailed"
        ]
        successful_task_ids = [
            task_id
            for task_id, terminal_type, _ in task_outcomes
            if terminal_type == "EpisodeFinished"
        ]
        summary = {
            "schema_version": 2,
            "run_id": plan.source.run_id,
            "iteration": manifest["iteration"],
            "memory_enabled": True,
            "attempted_task_count": len(task_outcomes),
            "episode_count": len(successful_task_ids),
            "failed_task_count": len(failed_task_ids),
            "failed_task_ids": failed_task_ids,
            "successful_task_ids": successful_task_ids,
            "total_terminal_reward": math.fsum(
                reward
                for _, terminal_type, reward in task_outcomes
                if terminal_type == "EpisodeFinished"
            ),
            "completed_train_tasks_before": plan.completed_before,
            "completed_train_tasks_after": plan.completed_after,
            "input_memory_snapshot_id": plan.input_snapshot_id,
            "output_memory_snapshot_id": plan.output_snapshot_id,
            "maintenance_rounds_executed": maintenance_rounds,
            "canonical_source": True,
        }
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(run_dir / "fast_loop_summary.json", summary)
        _write_jsonl(events_path, events)
        run_rows.append(
            {
                "run_id": plan.source.run_id,
                "seed": plan.source.seed,
                "task_count": len(task_outcomes),
                "episode_count": len(successful_task_ids),
                "failed_task_count": len(failed_task_ids),
                "maintenance_rounds": maintenance_rounds,
                "completed_train_tasks_before": plan.completed_before,
                "completed_train_tasks_after": plan.completed_after,
                "input_memory_snapshot_id": plan.input_snapshot_id,
                "output_memory_snapshot_id": plan.output_snapshot_id,
                "manifest_sha256": _sha256(run_dir / "manifest.json"),
                "summary_sha256": _sha256(run_dir / "fast_loop_summary.json"),
                "events_sha256": _sha256(events_path),
                "events_line_count": len(events),
            }
        )
    return tuple(run_rows), tuple(provenance)


def _normalize_episode(
    candidate: _TaskCandidate,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    events = [_json_copy(event) for event in candidate.rows]
    proposed = events[-2]
    committed = events[-1]
    proposals = proposed.get("proposals")
    written_ids = committed.get("written_memory_ids")
    replayed_ids = committed.get("replayed_memory_ids")
    if (
        not isinstance(proposals, list)
        or not isinstance(written_ids, list)
        or not isinstance(replayed_ids, list)
        or not all(isinstance(item, dict) for item in proposals)
    ):
        return events, ()

    proposal_ids = [item.get("memory_id") for item in proposals]
    if not all(isinstance(memory_id, str) and memory_id for memory_id in proposal_ids):
        return events, ()
    if not all(isinstance(memory_id, str) and memory_id for memory_id in written_ids):
        return events, ()
    if not all(isinstance(memory_id, str) and memory_id for memory_id in replayed_ids):
        return events, ()
    if written_ids != proposal_ids:
        raise ValueError(
            f"source write proposal/commit mismatch for task {candidate.task_id}"
        )

    proposal_counts: dict[str, int] = defaultdict(int)
    replay_counts: dict[str, int] = defaultdict(int)
    for memory_id in proposal_ids:
        proposal_counts[memory_id] += 1
    for memory_id in replayed_ids:
        replay_counts[memory_id] += 1
    if not set(replay_counts) <= set(proposal_counts):
        raise ValueError(
            f"source replay references an unknown proposal for task {candidate.task_id}"
        )

    retained: dict[str, dict[str, Any]] = {}
    normalized_proposals: list[dict[str, Any]] = []
    dropped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        memory_id = proposal["memory_id"]
        first = retained.get(memory_id)
        if first is None:
            retained[memory_id] = proposal
            normalized_proposals.append(proposal)
            continue
        if any(first.get(field) != proposal.get(field) for field in _SAFE_REPLAY_FIELDS):
            raise ValueError(
                f"conflicting duplicate write proposal for task "
                f"{candidate.task_id}: {memory_id}"
            )
        dropped[memory_id].append(proposal)

    if not dropped:
        return events, ()

    normalized_replayed: list[str] = []
    normalizations: list[dict[str, Any]] = []
    for proposal in normalized_proposals:
        memory_id = proposal["memory_id"]
        source_count = proposal_counts[memory_id]
        replay_count = replay_counts.get(memory_id, 0)
        if replay_count == source_count:
            normalized_replayed.append(memory_id)
        elif replay_count != source_count - 1:
            raise ValueError(
                f"ambiguous duplicate replay history for task "
                f"{candidate.task_id}: {memory_id}"
            )
        if memory_id not in dropped:
            continue
        variants = dropped[memory_id]
        differing_fields = sorted(
            {
                field
                for variant in variants
                for field in set(proposal) | set(variant)
                if proposal.get(field) != variant.get(field)
            }
        )
        normalizations.append(
            {
                "policy": "runtime_safe_replay_keep_first",
                "memory_id": memory_id,
                "source_proposal_count": source_count,
                "canonical_proposal_count": 1,
                "source_replayed_count": replay_count,
                "canonical_replayed_count": int(replay_count == source_count),
                "retained_proposal_sha256": _event_block_hash((proposal,)),
                "dropped_proposal_sha256": [
                    _event_block_hash((variant,)) for variant in variants
                ],
                "differing_non_identity_fields": differing_fields,
                "source_proposal_event_line": candidate.end_line - 1,
                "source_commit_event_line": candidate.end_line,
            }
        )

    proposed["proposals"] = normalized_proposals
    committed["written_memory_ids"] = [
        proposal["memory_id"] for proposal in normalized_proposals
    ]
    committed["replayed_memory_ids"] = normalized_replayed
    return events, tuple(normalizations)


def _failure_marker(candidate: _TaskCandidate) -> dict[str, Any]:
    source_event = candidate.rows[-1]
    snapshot_id = _candidate_snapshot(candidate)
    task_group = source_event.get("task_group")
    if not isinstance(task_group, str) or not task_group:
        task_group = RETAIL_TASK_GROUP
    return {
        "schema_version": 2,
        "event_type": "TaskFailed",
        "run_id": candidate.source.run_id,
        "iteration": candidate.source.manifest["iteration"],
        "split": "train",
        "mode": "learn",
        "model_revision": candidate.source.manifest["model_revision"],
        "adapter_revision": candidate.source.manifest["adapter_revision"],
        "seed": candidate.source.seed,
        "task_id": candidate.task_id,
        "task_group": task_group,
        "memory_snapshot_id": snapshot_id,
        "canonical_exclusion_reason": candidate.reason,
        "source_event_type": source_event.get("event_type"),
        "source_event_sha256": _event_block_hash(candidate.rows),
    }


def _normalize_maintenance(
    candidate: _MaintenanceCandidate,
    *,
    plan: _CanonicalRunPlan,
    trigger_task_index: int,
    maintenance_period: int,
) -> list[dict[str, Any]]:
    events = [_json_copy(event) for event in candidate.rows]
    for event in events:
        event.update(
            {
                "schema_version": 2,
                "run_id": plan.source.run_id,
                "iteration": plan.source.manifest["iteration"],
                "split": "train",
                "mode": "learn",
                "model_revision": plan.source.manifest["model_revision"],
                "adapter_revision": plan.source.manifest["adapter_revision"],
                "seed": plan.source.seed,
            }
        )
    events[0]["completed_train_tasks"] = trigger_task_index
    events[0]["period"] = maintenance_period
    return events


def _build_index(
    request: CanonicalizeRequest,
    *,
    sources: Sequence[_SourceRun],
    anchors: Mapping[int, Sequence[str]],
    plans: Sequence[_CanonicalRunPlan],
    candidate_rows: Sequence[Mapping[str, Any]],
    maintenance_rows: Sequence[Mapping[str, Any]],
    selected: Mapping[tuple[int, str], _TaskCandidate],
    selected_maintenance: Mapping[int, _MaintenanceCandidate],
    run_rows: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    deep_validation: Mapping[str, Any],
) -> dict[str, Any]:
    complete_count = sum(
        candidate.outcome == "complete" for candidate in selected.values()
    )
    failed_count = len(selected) - complete_count
    duplicate_keys = sum(
        count > 1
        for count in _logical_candidate_counts(candidate_rows).values()
    )
    normalizations = [
        normalization
        for row in provenance
        for normalization in row.get("event_normalizations", ())
    ]
    source_rows = [
        {
            "run_id": source.run_id,
            "seed": source.seed,
            "path": str(source.path),
            "manifest_sha256": source.manifest_sha256,
            "events_sha256": source.events_sha256,
            "event_count": len(source.rows),
        }
        for source in sources
    ]
    maintenance_source_rows = [
        {
            "path": str(path),
            "events_sha256": _sha256(Path(path)),
        }
        for path in request.maintenance_event_paths
    ]
    return {
        "schema_version": 1,
        "artifact_type": "opd_canonical_source_index",
        "build_id": request.build_id,
        "policy": {
            "logical_key": ["seed", "task_id"],
            "complete_episode_preferred": True,
            "duplicate_policy": "first_complete_by_declared_source_priority",
            "failed_tasks_are_training_examples": False,
            "maintenance_policy": "first_complete_by_round",
            "duplicate_memory_write_policy": "runtime_safe_replay_keep_first",
            "canonical_task_group": RETAIL_TASK_GROUP,
        },
        "expected": {
            "seeds": list(request.expected_seeds),
            "task_ids_by_seed": {
                str(seed): list(task_ids) for seed, task_ids in anchors.items()
            },
            "logical_task_count": sum(len(task_ids) for task_ids in anchors.values()),
            "maintenance_period": request.maintenance_period,
            "maintenance_rounds": sorted(selected_maintenance),
            "final_memory_snapshot_id": request.final_memory_snapshot_id,
        },
        "coverage": {
            "logical_task_count": len(selected),
            "included_episode_count": complete_count,
            "excluded_failure_count": failed_count,
            "duplicate_logical_key_count": duplicate_keys,
            "selected_maintenance_count": len(selected_maintenance),
            "failed_maintenance_candidate_count": sum(
                row.get("candidate_outcome") == "failed"
                for row in maintenance_rows
            ),
        },
        "source_runs": source_rows,
        "maintenance_sources": maintenance_source_rows,
        "canonical_runs": _json_copy(run_rows),
        "task_candidates": _json_copy(candidate_rows),
        "maintenance_candidates": _json_copy(maintenance_rows),
        "provenance": _json_copy(provenance),
        "normalization": {
            "duplicate_memory_write_group_count": len(normalizations),
            "removed_proposal_count": sum(
                normalization["source_proposal_count"] - 1
                for normalization in normalizations
            ),
            "entries": _json_copy(normalizations),
        },
        "validation": {
            "source_split": "train",
            "test_leakage_detected": False,
            "logical_keys_unique": len(selected)
            == len(
                {
                    (seed, task_id)
                    for seed, task_ids in anchors.items()
                    for task_id in task_ids
                }
            ),
            "snapshot_chain": [
                plans[0].input_snapshot_id,
                *(plan.output_snapshot_id for plan in plans),
            ],
            **_json_copy(deep_validation),
        },
    }


def _logical_candidate_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], int]:
    values: dict[tuple[int, str], int] = defaultdict(int)
    for row in rows:
        logical_key = row.get("logical_key")
        if (
            isinstance(logical_key, list)
            and len(logical_key) == 2
            and type(logical_key[0]) is int
            and isinstance(logical_key[1], str)
        ):
            values[(logical_key[0], logical_key[1])] += 1
    return dict(values)


def _event_block_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(rows),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_snapshot(memory_root: Path, snapshot_id: str) -> None:
    snapshots_root = (memory_root / "snapshots").resolve()
    path = (snapshots_root / snapshot_id).resolve()
    if path.parent != snapshots_root or not path.is_dir():
        raise ValueError(f"memory snapshot does not exist: {snapshot_id}")


def _validate_build_id(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_BUILD_ID.fullmatch(value):
        raise ValueError("build_id must contain lowercase letters, digits, '_' or '-'")
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("xb") as destination:
        destination.write(payload.encode("utf-8"))
        destination.flush()
        os.fsync(destination.fileno())
    _fsync_directory(path.parent)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        for row in rows:
            destination.write(
                (
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            )
        destination.flush()
        os.fsync(destination.fileno())
    _fsync_directory(path.parent)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
