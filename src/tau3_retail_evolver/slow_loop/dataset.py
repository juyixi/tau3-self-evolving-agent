from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
import uuid

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.memory.paths import project_root as default_project_root
from tau3_retail_evolver.memory.paths import training_memory_root
from tau3_retail_evolver.slow_loop.attribution import MemoryScore, compute_memory_scores
from tau3_retail_evolver.slow_loop.audit import audit_dataset
from tau3_retail_evolver.slow_loop.evidence import EvidenceLedger, build_evidence
from tau3_retail_evolver.slow_loop.examples import (
    OPDExample,
    build_action_examples,
    build_maintenance_examples,
    build_selection_examples,
    build_writing_examples,
)
from tau3_retail_evolver.slow_loop.source_runs import load_source_runs
from tau3_retail_evolver.slow_loop.task_grouping import (
    GROUPING_REVISION,
    RetailTaskGroups,
)


DATASET_SCHEMA_VERSION = 1
ATTRIBUTION_REVISION = "opd-evolver-paper-eq11-eq12-v1"
_SAFE_BUILD_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_KINDS = ("sel", "act", "write", "maint")


@dataclass(frozen=True, slots=True)
class DatasetBuildRequest:
    source_run_paths: tuple[Path, ...]
    dataset_build_id: str
    output_root: Path
    config_path: Path
    project_root: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetBuildResult:
    dataset_dir: Path
    manifest: Mapping[str, Any]
    audit_report: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _MaterializedDataset:
    ledger: EvidenceLedger
    scores: tuple[MemoryScore, ...]
    examples: Mapping[str, tuple[OPDExample, ...]]
    source_runs: tuple[Mapping[str, Any], ...]
    official_train_task_ids: tuple[str, ...]
    snapshot_chain: tuple[str, ...]
    resolved_config: Mapping[str, Any]
    build_code_revision: str


def build_opd_dataset(request: DatasetBuildRequest) -> DatasetBuildResult:
    if not isinstance(request, DatasetBuildRequest):
        raise TypeError("request must be a DatasetBuildRequest")
    build_id = _validate_build_id(request.dataset_build_id)
    project = (request.project_root or default_project_root()).resolve()
    output_root = _resolve_from_project(request.output_root, project)
    final_build_root = output_root / build_id
    final_dataset_dir = final_build_root / "slow_loop"
    if final_build_root.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset build: {final_build_root}")

    materialized = _materialize(request)
    output_root.mkdir(parents=True, exist_ok=True)
    temp_build_root = output_root / f".{build_id}.tmp-{uuid.uuid4().hex}"
    temp_dataset_dir = temp_build_root / "slow_loop"
    try:
        temp_dataset_dir.mkdir(parents=True, exist_ok=False)
        artifacts = _write_artifacts(temp_dataset_dir, materialized)
        manifest = _build_manifest(build_id, materialized, artifacts)
        _write_json(temp_dataset_dir / "dataset_manifest.json", manifest)
        report = audit_dataset(temp_dataset_dir)
        _write_json(
            temp_dataset_dir / "audit_report.json",
            report.model_dump(mode="json"),
        )
        if not report.passed:
            codes = ", ".join(error.code for error in report.errors)
            raise ValueError(f"dataset audit failed before publication: {codes}")
        try:
            temp_build_root.rename(final_build_root)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite existing dataset build: {final_build_root}"
            ) from error
    except BaseException:
        if temp_build_root.exists():
            shutil.rmtree(temp_build_root, ignore_errors=True)
        raise

    return DatasetBuildResult(
        dataset_dir=final_dataset_dir,
        manifest=_json_copy(manifest),
        audit_report=report.model_dump(mode="json"),
    )


def _materialize(request: DatasetBuildRequest) -> _MaterializedDataset:
    project = (request.project_root or default_project_root()).resolve()
    config_path = _resolve_from_project(request.config_path, project)
    config = load_config(config_path)
    if config.memory.enabled is not True:
        raise ValueError("Stage 5 requires memory.enabled=true")

    tau2_path = _resolve_from_project(config.tau2.repo_path, project)
    runtime = Tau2Runtime.inspect_metadata(tau2_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.retail_tasks_path,
        runtime.retail_split_path,
    )
    catalog.require_official_compatibility()
    memory_root = training_memory_root(config.memory.agent_id, root=project)
    source_paths = tuple(
        _resolve_from_project(path, project) for path in request.source_run_paths
    )
    source_set = load_source_runs(
        source_paths,
        catalog=catalog,
        memory_root=memory_root,
    )
    if source_set.tau2_commit.casefold() != runtime.git_commit.casefold():
        raise ValueError("source Tau2 revision does not match configured checkout")
    ledger = build_evidence(source_set, memory_root=memory_root)

    task_ids = tuple(
        task_id
        for source in source_set.runs
        for task_id in source.manifest["task_ids"]
    )
    task_groups = RetailTaskGroups.from_file(
        runtime.retail_tasks_path,
        task_ids=task_ids,
    )
    for episode in ledger.episodes:
        if episode.task_group != task_groups.signature_for(episode.task_id):
            raise ValueError(f"source task group mismatch: {episode.episode_id}")

    scores = compute_memory_scores(
        ledger,
        tier_priors=config.slow_loop.tier_priors,
        score_threshold=config.memory.score_threshold,
    )
    examples = {
        "sel": build_selection_examples(
            ledger,
            scores,
            score_threshold=config.memory.score_threshold,
        ),
        "act": build_action_examples(
            ledger,
            scores,
            score_threshold=config.memory.score_threshold,
            teacher_memory_cap=config.memory.teacher_memory_cap,
        ),
        "write": build_writing_examples(
            ledger,
            scores,
            score_threshold=config.memory.score_threshold,
        ),
        "maint": build_maintenance_examples(
            ledger,
            scores,
            teacher_memory_cap=config.memory.teacher_memory_cap,
            redundancy_threshold=config.slow_loop.redundancy_threshold,
            max_redundancy_pairs=config.slow_loop.max_redundancy_pairs,
        ),
    }
    source_rows = tuple(
        {
            "run_id": source.run_id,
            "task_ids": list(source.manifest["task_ids"]),
            "manifest_sha256": source.manifest_sha256,
            "events_sha256": source.events_sha256,
            "summary_sha256": source.summary_sha256,
            "input_memory_snapshot_id": source.manifest["memory_snapshot_id"],
            "output_memory_snapshot_id": source.summary["output_memory_snapshot_id"],
            "completed_train_tasks_before": source.summary[
                "completed_train_tasks_before"
            ],
            "completed_train_tasks_after": source.summary[
                "completed_train_tasks_after"
            ],
        }
        for source in source_set.runs
    )
    snapshot_chain = (
        source_rows[0]["input_memory_snapshot_id"],
        *(row["output_memory_snapshot_id"] for row in source_rows),
    )
    return _MaterializedDataset(
        ledger=ledger,
        scores=scores,
        examples=examples,
        source_runs=source_rows,
        official_train_task_ids=tuple(catalog.task_ids("train")),
        snapshot_chain=tuple(snapshot_chain),
        resolved_config={
            "tier_priors": dict(config.slow_loop.tier_priors),
            "score_threshold": config.memory.score_threshold,
            "teacher_memory_cap": config.memory.teacher_memory_cap,
            "redundancy_threshold": config.slow_loop.redundancy_threshold,
            "max_redundancy_pairs": config.slow_loop.max_redundancy_pairs,
            "maintenance_period": config.memory.maintenance_period,
            "embedding_model": config.memory.embedding_model,
        },
        build_code_revision=_git_revision(project),
    )


def _write_artifacts(
    root: Path,
    materialized: _MaterializedDataset,
) -> dict[str, dict[str, Any]]:
    evidence_rows: Sequence[Any] = (
        *materialized.ledger.episodes,
        *materialized.ledger.maintenance,
    )
    rows_by_path: dict[str, Sequence[Any]] = {
        "evidence/episodes.jsonl": evidence_rows,
        "attribution/memory_scores.jsonl": tuple(
            sorted(materialized.scores, key=lambda score: score.memory_id)
        ),
    }
    for kind in _KINDS:
        rows_by_path[f"datasets/{kind}.jsonl"] = tuple(
            sorted(materialized.examples.get(kind, ()), key=lambda item: item.example_id)
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for relative in sorted(rows_by_path):
        path = root / relative
        line_count = _write_jsonl(path, rows_by_path[relative])
        artifacts[relative] = {
            "line_count": line_count,
            "sha256": _sha256(path),
        }
    return artifacts


def _build_manifest(
    build_id: str,
    materialized: _MaterializedDataset,
    artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ledger = materialized.ledger
    counts = {
        "evidence_episodes": len(ledger.episodes),
        "evidence_maintenance": len(ledger.maintenance),
        "memory_scores": len(materialized.scores),
        **{kind: len(materialized.examples.get(kind, ())) for kind in _KINDS},
    }
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "dataset_build_id": build_id,
        "build_code_revision": materialized.build_code_revision,
        "policy_lineage": {
            "iteration": ledger.iteration,
            "model_revision": ledger.model_revision,
            "adapter_revision": ledger.adapter_revision,
            "tau2_commit": ledger.tau2_commit,
        },
        "official_split": {
            "name": "train",
            "sha256": ledger.split_hash,
            "train_task_ids": list(materialized.official_train_task_ids),
            "train_task_ids_sha256": _canonical_value_hash(
                list(materialized.official_train_task_ids)
            ),
        },
        "memory": {
            "agent_id": ledger.memory_agent_id,
            "snapshot_chain": list(materialized.snapshot_chain),
        },
        "revisions": {
            "task_grouping": GROUPING_REVISION,
            "attribution": ATTRIBUTION_REVISION,
        },
        "resolved_config": _json_copy(materialized.resolved_config),
        "source_runs": _json_copy(materialized.source_runs),
        "counts": counts,
        "skip_reasons": _skip_reasons(materialized),
        "artifacts": _json_copy(artifacts),
    }


def _skip_reasons(materialized: _MaterializedDataset) -> dict[str, int]:
    score_by_id = {score.memory_id: score for score in materialized.scores}
    threshold = float(materialized.resolved_config["score_threshold"])
    no_scored_candidate = 0
    no_qualified_selected = 0
    no_future_write = 0
    for episode in materialized.ledger.episodes:
        candidate_scores = [score_by_id[item.memory_id] for item in episode.candidates]
        if not any(
            score.value is not None and abs(score.value) >= threshold
            for score in candidate_scores
        ):
            no_scored_candidate += 1
        if not any(
            score_by_id[memory_id].value is not None
            and score_by_id[memory_id].value >= threshold
            for memory_id in episode.selected_memory_ids
        ):
            no_qualified_selected += 1
        if episode.committed_new_memory_ids and not any(
            score_by_id[memory_id].value is not None
            and abs(score_by_id[memory_id].value) >= threshold
            for memory_id in episode.committed_new_memory_ids
        ):
            no_future_write += 1
    return {
        "insufficient_selected_control": sum(
            score.status == "insufficient_evidence" for score in materialized.scores
        ),
        "no_scored_candidate": no_scored_candidate,
        "no_successful_same_group_trajectory": _no_successful_group_count(materialized),
        "no_qualified_selected_memory": no_qualified_selected,
        "no_future_write_evidence": no_future_write,
        "no_committed_maintenance": int(not materialized.ledger.maintenance),
    }


def _no_successful_group_count(materialized: _MaterializedDataset) -> int:
    successful_groups = {
        episode.task_group
        for episode in materialized.ledger.episodes
        if episode.final_reward == 1.0
    }
    return sum(
        episode.task_group not in successful_groups
        for episode in materialized.ledger.episodes
    )


def _write_jsonl(path: Path, rows: Sequence[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        for row in rows:
            destination.write(_canonical_json_bytes(_json_value(row)))
    return len(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(_canonical_json_bytes(value))
        destination.flush()
        os.fsync(destination.fileno())


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return _json_copy(value)


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=lambda item: list(item) if isinstance(item, tuple) else dict(item),
        )
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_value_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_build_id(value: str) -> str:
    if not isinstance(value, str) or not _SAFE_BUILD_ID.fullmatch(value):
        raise ValueError("dataset_build_id must be a lowercase safe slug")
    return value


def _resolve_from_project(path: Path, project: Path) -> Path:
    candidate = Path(path).expanduser()
    return (project / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _git_revision(project: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError("unable to resolve dataset build code revision") from error
    revision = result.stdout.strip()
    if result.returncode != 0 or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise RuntimeError("project root is not a usable Git checkout")
    return revision.casefold()
