from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from tau3_retail_evolver.envs.task_catalog import (
    OFFICIAL_SPLIT_COUNTS,
    OFFICIAL_SPLIT_SHA256,
)
from tau3_retail_evolver.io.jsonl import iter_jsonl_objects
from tau3_retail_evolver.slow_loop.attribution import MemoryScore, compute_memory_scores
from tau3_retail_evolver.slow_loop.evidence import (
    EpisodeEvidence,
    EvidenceLedger,
    MaintenanceEvidence,
)
from tau3_retail_evolver.slow_loop.examples import (
    ONLINE_SAMPLING_CONTRACT,
    OPDExample,
    audit_example_boundaries,
)
from tau3_retail_evolver.slow_loop.leakage import audit_artifact_payload


_EXPECTED_ARTIFACTS = (
    "evidence/episodes.jsonl",
    "attribution/memory_scores.jsonl",
    "datasets/sel.jsonl",
    "datasets/act.jsonl",
    "datasets/write.jsonl",
    "datasets/maint.jsonl",
)


class _AuditModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AuditError(_AuditModel):
    code: str
    message: str
    artifact: str | None = None


class AuditReport(_AuditModel):
    audit_schema_version: Literal[1] = 1
    dataset_build_id: str | None
    passed: bool
    checked_artifacts: tuple[str, ...]
    errors: tuple[AuditError, ...]


def audit_dataset(path: Path) -> AuditReport:
    root = Path(path).resolve()
    errors: list[AuditError] = []
    manifest = _read_manifest(root, errors)
    if manifest is None:
        return _report(None, (), errors)

    build_id = manifest.get("dataset_build_id")
    if not isinstance(build_id, str) or not build_id:
        _error(errors, "manifest_invalid", "dataset_build_id is missing")
        build_id = None
    if manifest.get("dataset_schema_version") != 1:
        _error(errors, "manifest_invalid", "dataset schema version must be 1")

    official = manifest.get("official_split")
    train_ids: set[str] = set()
    if not isinstance(official, dict):
        _error(errors, "non_train_source", "official split metadata is missing")
    else:
        if official.get("sha256") != OFFICIAL_SPLIT_SHA256:
            _error(errors, "non_train_source", "official retail split hash mismatch")
        raw_train_ids = official.get("train_task_ids")
        if (
            not isinstance(raw_train_ids, list)
            or not all(isinstance(task_id, str) and task_id for task_id in raw_train_ids)
            or len(raw_train_ids) != len(set(raw_train_ids))
            or len(raw_train_ids) != OFFICIAL_SPLIT_COUNTS["train"]
        ):
            _error(errors, "non_train_source", "official train task ID set is invalid")
        else:
            train_ids = set(raw_train_ids)
            expected_ids_hash = _canonical_hash(raw_train_ids)
            if official.get("train_task_ids_sha256") != expected_ids_hash:
                _error(errors, "non_train_source", "official train task ID hash mismatch")

    source_task_ids = _source_task_ids(manifest, train_ids, errors)
    _audit_source_lineage(manifest, errors)
    artifact_rows = _read_artifacts(root, manifest, errors)

    evidence_rows = artifact_rows.get("evidence/episodes.jsonl", ())
    episodes, maintenance = _audit_evidence(
        evidence_rows,
        manifest=manifest,
        train_ids=train_ids,
        source_task_ids=source_task_ids,
        errors=errors,
    )
    scores = _audit_scores(
        artifact_rows.get("attribution/memory_scores.jsonl", ()),
        episodes=episodes,
        maintenance=maintenance,
        manifest=manifest,
        errors=errors,
    )
    _audit_examples(
        artifact_rows,
        episodes=episodes,
        maintenance=maintenance,
        scores=scores,
        errors=errors,
    )
    _audit_manifest_counts(
        manifest,
        episodes=episodes,
        maintenance=maintenance,
        scores=scores,
        artifact_rows=artifact_rows,
        errors=errors,
    )
    return _report(build_id, tuple(sorted(artifact_rows)), errors)


def _read_manifest(root: Path, errors: list[AuditError]) -> dict[str, Any] | None:
    path = root / "dataset_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _error(errors, "manifest_invalid", f"unable to read dataset manifest: {error}")
        return None
    if not isinstance(value, dict):
        _error(errors, "manifest_invalid", "dataset manifest must be an object")
        return None
    try:
        audit_artifact_payload(value)
    except ValueError as error:
        _error(errors, "artifact_forbidden_content", str(error), "dataset_manifest.json")
    return value


def _source_task_ids(
    manifest: dict[str, Any],
    train_ids: set[str],
    errors: list[AuditError],
) -> set[str]:
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list) or not source_runs:
        _error(errors, "manifest_invalid", "source runs are missing")
        return set()
    task_ids: list[str] = []
    run_ids: list[str] = []
    for source in source_runs:
        if not isinstance(source, dict):
            _error(errors, "manifest_invalid", "source run entry must be an object")
            continue
        run_id = source.get("run_id")
        raw_task_ids = source.get("task_ids")
        if not isinstance(run_id, str) or not run_id:
            _error(errors, "manifest_invalid", "source run ID is invalid")
        else:
            run_ids.append(run_id)
        if not isinstance(raw_task_ids, list) or not all(
            isinstance(task_id, str) and task_id for task_id in raw_task_ids
        ):
            _error(errors, "manifest_invalid", "source run task IDs are invalid")
            continue
        task_ids.extend(raw_task_ids)
    if len(run_ids) != len(set(run_ids)) or len(task_ids) != len(set(task_ids)):
        _error(errors, "duplicate_source_identity", "source run or task identity is duplicated")
    if train_ids and not set(task_ids) <= train_ids:
        _error(errors, "non_train_source", "source run contains a non-train task")
    return set(task_ids)


def _read_artifacts(
    root: Path,
    manifest: dict[str, Any],
    errors: list[AuditError],
) -> dict[str, tuple[dict[str, Any], ...]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        _error(errors, "manifest_invalid", "artifact manifest is missing")
        artifacts = {}
    if set(artifacts) != set(_EXPECTED_ARTIFACTS):
        _error(errors, "manifest_invalid", "artifact set is not canonical")

    rows_by_path: dict[str, tuple[dict[str, Any], ...]] = {}
    for relative in _EXPECTED_ARTIFACTS:
        path = root / relative
        metadata = artifacts.get(relative)
        if not isinstance(metadata, dict):
            _error(errors, "artifact_missing", "artifact metadata is missing", relative)
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            _error(errors, "artifact_missing", f"unable to read artifact: {error}", relative)
            continue
        actual_hash = hashlib.sha256(payload).hexdigest()
        if metadata.get("sha256") != actual_hash:
            _error(errors, "artifact_hash_mismatch", "artifact SHA256 mismatch", relative)
        try:
            rows = tuple(iter_jsonl_objects(path))
        except ValueError as error:
            _error(errors, "artifact_jsonl_invalid", str(error), relative)
            continue
        for row in rows:
            try:
                audit_artifact_payload(row)
            except ValueError as error:
                _error(errors, "artifact_forbidden_content", str(error), relative)
        if metadata.get("line_count") != len(rows):
            _error(errors, "artifact_line_count_mismatch", "artifact line count mismatch", relative)
        rows_by_path[relative] = rows
    return rows_by_path


def _audit_source_lineage(manifest: dict[str, Any], errors: list[AuditError]) -> None:
    source_runs = manifest.get("source_runs")
    memory = manifest.get("memory")
    if not isinstance(source_runs, list) or not source_runs or not isinstance(memory, dict):
        _error(errors, "source_lineage_invalid", "source lineage metadata is missing")
        return
    valid = all(isinstance(source, dict) for source in source_runs)
    if not valid:
        _error(errors, "source_lineage_invalid", "source lineage entry is invalid")
        return
    for previous, current in zip(source_runs, source_runs[1:], strict=False):
        if previous.get("completed_train_tasks_after") != current.get(
            "completed_train_tasks_before"
        ):
            _error(errors, "source_lineage_invalid", "source task ranges are not continuous")
        if previous.get("output_memory_snapshot_id") != current.get(
            "input_memory_snapshot_id"
        ):
            _error(errors, "source_lineage_invalid", "source snapshots are not continuous")
    expected_chain = [
        source_runs[0].get("input_memory_snapshot_id"),
        *(source.get("output_memory_snapshot_id") for source in source_runs),
    ]
    if memory.get("snapshot_chain") != expected_chain or any(
        not isinstance(snapshot_id, str) or not snapshot_id for snapshot_id in expected_chain
    ):
        _error(errors, "source_lineage_invalid", "manifest snapshot chain mismatch")


def _audit_evidence(
    rows: tuple[dict[str, Any], ...],
    *,
    manifest: dict[str, Any],
    train_ids: set[str],
    source_task_ids: set[str],
    errors: list[AuditError],
) -> tuple[dict[str, EpisodeEvidence], dict[str, MaintenanceEvidence]]:
    episodes: dict[str, EpisodeEvidence] = {}
    maintenance: dict[str, MaintenanceEvidence] = {}
    lineage = manifest.get("policy_lineage")
    lineage = lineage if isinstance(lineage, dict) else {}
    for row in rows:
        if "episode_id" in row:
            try:
                episode = EpisodeEvidence.model_validate(row)
            except ValidationError as error:
                _error(errors, "evidence_schema_invalid", str(error), "evidence/episodes.jsonl")
                continue
            if episode.episode_id in episodes:
                _error(errors, "duplicate_episode_id", episode.episode_id, "evidence/episodes.jsonl")
                continue
            episodes[episode.episode_id] = episode
            if episode.task_id not in train_ids or episode.task_id not in source_task_ids:
                _error(errors, "non_train_source", episode.task_id, "evidence/episodes.jsonl")
            expected = (
                lineage.get("iteration"),
                lineage.get("model_revision"),
                lineage.get("adapter_revision"),
                lineage.get("tau2_commit"),
                manifest.get("official_split", {}).get("sha256")
                if isinstance(manifest.get("official_split"), dict)
                else None,
            )
            actual = (
                episode.iteration,
                episode.model_revision,
                episode.adapter_revision,
                episode.tau2_commit,
                episode.split_hash,
            )
            if actual != expected:
                _error(errors, "evidence_lineage_mismatch", episode.episode_id)
        elif "maintenance_id" in row:
            try:
                item = MaintenanceEvidence.model_validate(row)
            except ValidationError as error:
                _error(errors, "evidence_schema_invalid", str(error), "evidence/episodes.jsonl")
                continue
            if item.maintenance_id in maintenance:
                _error(errors, "duplicate_maintenance_id", item.maintenance_id)
                continue
            maintenance[item.maintenance_id] = item
        else:
            _error(errors, "evidence_schema_invalid", "unknown evidence row type")
    if {episode.task_id for episode in episodes.values()} != source_task_ids:
        _error(errors, "evidence_lineage_mismatch", "evidence task set differs from source runs")
    source_run_ids = {
        source.get("run_id")
        for source in manifest.get("source_runs", [])
        if isinstance(source, dict)
    }
    if {episode.run_id for episode in episodes.values()} != source_run_ids:
        _error(errors, "evidence_lineage_mismatch", "evidence run set differs from source runs")
    snapshot_ids = set(
        manifest.get("memory", {}).get("snapshot_chain", [])
        if isinstance(manifest.get("memory"), dict)
        else []
    )
    for item in maintenance.values():
        if not set(item.prior_episode_ids) <= episodes.keys():
            _error(errors, "evidence_lineage_mismatch", item.maintenance_id)
        if item.run_id not in source_run_ids or item.memory_snapshot_id not in snapshot_ids:
            _error(errors, "evidence_lineage_mismatch", item.maintenance_id)
    return episodes, maintenance


def _audit_scores(
    rows: tuple[dict[str, Any], ...],
    *,
    episodes: dict[str, EpisodeEvidence],
    maintenance: dict[str, MaintenanceEvidence],
    manifest: dict[str, Any],
    errors: list[AuditError],
) -> dict[str, MemoryScore]:
    scores: dict[str, MemoryScore] = {}
    episode_order = {episode_id: index for index, episode_id in enumerate(episodes)}
    for row in rows:
        try:
            score = MemoryScore.model_validate(row)
        except ValidationError as error:
            _error(errors, "attribution_schema_invalid", str(error))
            continue
        if score.memory_id in scores:
            _error(errors, "duplicate_memory_score", score.memory_id)
            continue
        scores[score.memory_id] = score
        if not set(score.source_episode_ids) <= episodes.keys():
            _error(errors, "attribution_provenance_invalid", score.memory_id)
        if score.creator_episode_id is not None:
            creator_index = episode_order.get(score.creator_episode_id)
            if creator_index is None or any(
                episode_order.get(source_id, -1) <= creator_index
                for source_id in score.source_episode_ids
            ):
                _error(errors, "attribution_chronology_invalid", score.memory_id)
        for group in score.groups:
            if not set(group.source_episode_ids) <= set(score.source_episode_ids):
                _error(errors, "attribution_provenance_invalid", score.memory_id)
    lineage = manifest.get("policy_lineage")
    split = manifest.get("official_split")
    memory = manifest.get("memory")
    config = manifest.get("resolved_config")
    source_runs = manifest.get("source_runs")
    if not all(
        isinstance(value, dict) for value in (lineage, split, memory, config)
    ) or not isinstance(source_runs, list):
        _error(errors, "attribution_recompute_mismatch", "recompute metadata is missing")
        return scores
    try:
        ledger = EvidenceLedger(
            iteration=lineage["iteration"],
            model_revision=lineage["model_revision"],
            adapter_revision=lineage.get("adapter_revision"),
            tau2_commit=lineage["tau2_commit"],
            split_hash=split["sha256"],
            memory_agent_id=memory["agent_id"],
            source_run_ids=tuple(source["run_id"] for source in source_runs),
            episodes=tuple(episodes.values()),
            maintenance=tuple(maintenance.values()),
        )
        expected = compute_memory_scores(
            ledger,
            tier_priors=config["tier_priors"],
            score_threshold=config["score_threshold"],
        )
    except (KeyError, TypeError, ValueError) as error:
        _error(errors, "attribution_recompute_mismatch", str(error))
        return scores
    expected_rows = [
        item.model_dump(mode="json") for item in sorted(expected, key=lambda item: item.memory_id)
    ]
    actual_rows = [
        item.model_dump(mode="json") for item in sorted(scores.values(), key=lambda item: item.memory_id)
    ]
    if actual_rows != expected_rows:
        _error(errors, "attribution_recompute_mismatch", "stored scores differ from Eq.11-Eq.12 recomputation")
    return scores


def _audit_manifest_counts(
    manifest: dict[str, Any],
    *,
    episodes: dict[str, EpisodeEvidence],
    maintenance: dict[str, MaintenanceEvidence],
    scores: dict[str, MemoryScore],
    artifact_rows: dict[str, tuple[dict[str, Any], ...]],
    errors: list[AuditError],
) -> None:
    actual = {
        "evidence_episodes": len(episodes),
        "evidence_maintenance": len(maintenance),
        "memory_scores": len(scores),
        **{
            kind: len(artifact_rows.get(f"datasets/{kind}.jsonl", ()))
            for kind in ("sel", "act", "write", "maint")
        },
    }
    if manifest.get("counts") != actual:
        _error(errors, "manifest_count_mismatch", "manifest counts differ from artifacts")


def _audit_examples(
    artifact_rows: dict[str, tuple[dict[str, Any], ...]],
    *,
    episodes: dict[str, EpisodeEvidence],
    maintenance: dict[str, MaintenanceEvidence],
    scores: dict[str, MemoryScore],
    errors: list[AuditError],
) -> None:
    seen: set[str] = set()
    for kind in ("sel", "act", "write", "maint"):
        relative = f"datasets/{kind}.jsonl"
        for row in artifact_rows.get(relative, ()):
            try:
                example = OPDExample.model_validate(row)
            except ValidationError as error:
                _error(errors, "example_schema_invalid", str(error), relative)
                continue
            if example.example_id in seen:
                _error(errors, "duplicate_example_id", example.example_id, relative)
            seen.add(example.example_id)
            if example.kind != kind:
                _error(errors, "example_kind_mismatch", example.example_id, relative)
            if example.sampling_contract != ONLINE_SAMPLING_CONTRACT:
                _error(
                    errors,
                    "missing_online_sampling_contract",
                    example.example_id,
                    relative,
                )
            try:
                audit_example_boundaries(example)
            except (TypeError, ValueError) as error:
                code = (
                    "missing_online_sampling_contract"
                    if "online sampling contract" in str(error)
                    else "public_privileged_leak"
                )
                _error(errors, code, str(error), relative)
            episode_id = example.provenance.get("episode_id")
            maintenance_id = example.provenance.get("maintenance_id")
            if episode_id is not None and episode_id not in episodes:
                _error(errors, "example_provenance_invalid", example.example_id, relative)
            if maintenance_id is not None and maintenance_id not in maintenance:
                _error(errors, "example_provenance_invalid", example.example_id, relative)
            _audit_example_memory_refs(example, scores, errors, relative)


def _audit_example_memory_refs(
    example: OPDExample,
    scores: dict[str, MemoryScore],
    errors: list[AuditError],
    artifact: str,
) -> None:
    privileged = example.privileged_hindsight
    rows: list[dict[str, Any]] = []
    for key in ("candidate_scores", "valuable_selected_memories", "written_memory_scores", "memory_diagnostics"):
        value = privileged.get(key, [])
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    for row in rows:
        memory_id = row.get("memory_id")
        if not isinstance(memory_id, str):
            continue
        score = scores.get(memory_id)
        if score is None:
            _error(errors, "example_provenance_invalid", memory_id, artifact)
            continue
        expected = {
            "tier": score.tier.value,
            "value": score.value,
            "attribution": score.attribution,
            "gamma": score.confidence,
            "confidence": score.confidence,
            "status": score.status,
        }
        if any(field in row and row[field] != value for field, value in expected.items()):
            _error(errors, "example_score_mismatch", memory_id, artifact)


def _report(
    build_id: str | None,
    checked: tuple[str, ...],
    errors: list[AuditError],
) -> AuditReport:
    unique: dict[tuple[str, str, str | None], AuditError] = {}
    for error in errors:
        unique[(error.code, error.message, error.artifact)] = error
    ordered = tuple(
        unique[key]
        for key in sorted(unique, key=lambda item: (item[0], item[2] or "", item[1]))
    )
    return AuditReport(
        dataset_build_id=build_id,
        passed=not ordered,
        checked_artifacts=checked,
        errors=ordered,
    )


def _error(
    errors: list[AuditError],
    code: str,
    message: str,
    artifact: str | None = None,
) -> None:
    errors.append(AuditError(code=code, message=message, artifact=artifact))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
