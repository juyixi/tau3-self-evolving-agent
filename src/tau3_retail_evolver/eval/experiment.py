from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import random
from typing import Any

from tau3_retail_evolver.eval.metrics import (
    compare_evaluation_reports,
    read_evaluation_json,
    write_evaluation_json,
)
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.types import MEMORY_TIERS, MemoryTier


EXPERIMENT_SCHEMA_VERSION = 1
EXPERIMENT_REPORT_TYPE = "tau3-retail-stage8-experiment"
DEFAULT_TRAIN_PASSES = 3
DEFAULT_TEST_TRIALS = 1
TRAIN_TASK_COUNT = 74
TEST_TASK_COUNT = 40

BASE_NO_MEMORY = "base_no_memory"
BASE_WITH_MEMORY = "base_with_memory"
OPD_WITH_MEMORY = "opd_with_memory"
OPD_NO_MEMORY = "opd_no_memory"
EXPERIMENT_ORDER = (
    BASE_NO_MEMORY,
    BASE_WITH_MEMORY,
    OPD_WITH_MEMORY,
    OPD_NO_MEMORY,
)

_CONTRASTS = (
    ("memory_gain_base", BASE_WITH_MEMORY, BASE_NO_MEMORY),
    ("opd_gain_with_memory", OPD_WITH_MEMORY, BASE_WITH_MEMORY),
    ("opd_internalization", OPD_NO_MEMORY, BASE_NO_MEMORY),
    ("memory_gain_after_opd", OPD_WITH_MEMORY, OPD_NO_MEMORY),
    ("full_system_gain", OPD_WITH_MEMORY, BASE_NO_MEMORY),
)
_OPD_KINDS = ("sel", "act", "write", "maint")


def build_stage8_experiment_report(
    *,
    experiment_id: str,
    evaluation_reports: Mapping[str, Mapping[str, Any]],
    train_run_dirs: Sequence[Path],
    dataset_dir: Path,
    training_dir: Path,
    memory_snapshot_path: Path,
    expected_train_passes: int = DEFAULT_TRAIN_PASSES,
    expected_test_tasks: int = TEST_TASK_COUNT,
    expected_test_trials: int = DEFAULT_TEST_TRIALS,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if not experiment_id.strip():
        raise ValueError("experiment ID must not be blank")
    if type(expected_train_passes) is not int or expected_train_passes < 1:
        raise ValueError("expected train passes must be positive")
    if type(expected_test_tasks) is not int or expected_test_tasks < 1:
        raise ValueError("expected test tasks must be positive")
    if type(expected_test_trials) is not int or expected_test_trials < 1:
        raise ValueError("expected test trials must be positive")
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("bootstrap samples must be at least 100")

    reports = _validate_evaluation_matrix(
        evaluation_reports,
        memory_snapshot_path=memory_snapshot_path,
        expected_test_tasks=expected_test_tasks,
        expected_test_trials=expected_test_trials,
    )
    comparison = compare_evaluation_reports(
        reports,
        baseline_label=BASE_NO_MEMORY,
    )
    train_summary = _summarize_train_passes(
        train_run_dirs,
        memory_snapshot_path=memory_snapshot_path,
        expected_train_passes=expected_train_passes,
    )
    dataset_summary = _summarize_dataset(dataset_dir)
    training_summary = _summarize_training(training_dir)
    _validate_training_lineage(
        reports,
        dataset_summary=dataset_summary,
        training_summary=training_summary,
        train_summary=train_summary,
    )

    evaluation_cells = {
        label: _evaluation_cell(reports[label])
        for label in EXPERIMENT_ORDER
    }
    contrasts = {
        name: _paired_contrast(
            reports[target],
            reports[control],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, (name, target, control) in enumerate(_CONTRASTS)
    }
    interaction = _interaction_contrast(reports)

    report = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "report_type": EXPERIMENT_REPORT_TYPE,
        "experiment_id": experiment_id,
        "design": {
            "matrix": "base_or_opd_checkpoint_x_no_memory_or_frozen_memory",
            "train_passes": expected_train_passes,
            "train_tasks_per_pass": train_summary["tasks_per_pass"],
            "test_tasks": expected_test_tasks,
            "test_trials": expected_test_trials,
            "memory_protocol": "test_static",
            "primary_metric": "pass_at_1",
        },
        "evaluation": {
            "cells": evaluation_cells,
            "contrasts": contrasts,
            "interaction": interaction,
            "controlled_comparison": comparison,
        },
        "fast_loop": train_summary,
        "opd_dataset": dataset_summary,
        "opd_training": training_summary,
        "artifacts": {
            "memory_snapshot_path": str(Path(memory_snapshot_path).resolve()),
            "memory_snapshot_id": Path(memory_snapshot_path).resolve().name,
            "train_run_dirs": [
                str(Path(path).resolve()) for path in train_run_dirs
            ],
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "training_dir": str(Path(training_dir).resolve()),
        },
    }
    _require_json_safe(report)
    return report


def write_stage8_experiment_report(
    path: Path,
    report: Mapping[str, Any],
) -> None:
    write_evaluation_json(path, report)


def load_labeled_evaluation_reports(
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    return {
        label: read_evaluation_json(path)
        for label, path in paths.items()
    }


def _validate_evaluation_matrix(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    memory_snapshot_path: Path,
    expected_test_tasks: int,
    expected_test_trials: int,
) -> dict[str, Mapping[str, Any]]:
    if set(reports) != set(EXPERIMENT_ORDER):
        raise ValueError(
            "Stage 8 evaluation reports must contain exactly: "
            + ", ".join(EXPERIMENT_ORDER)
        )
    ordered = {label: reports[label] for label in EXPERIMENT_ORDER}
    for label, report in ordered.items():
        if report.get("report_type") != "tau3-retail-evaluation":
            raise ValueError(f"{label} is not a Retail evaluation report")

    base_none = ordered[BASE_NO_MEMORY]["provenance"]
    base_memory = ordered[BASE_WITH_MEMORY]["provenance"]
    opd_memory = ordered[OPD_WITH_MEMORY]["provenance"]
    opd_none = ordered[OPD_NO_MEMORY]["provenance"]

    _require_cell(
        BASE_NO_MEMORY,
        base_none,
        protocol="no_memory",
        trained=False,
        memory_snapshot_id=None,
    )
    snapshot = ReadOnlyMemoryRepository(
        Path(memory_snapshot_path).resolve()
    )
    snapshot_id = snapshot.memory_snapshot_id
    snapshot_counts = {
        tier.value: len(snapshot.list(tier=tier))
        for tier in MemoryTier
    }
    if not sum(snapshot_counts.values()):
        raise ValueError("Stage 8 frozen Memory snapshot must not be empty")
    _require_cell(
        BASE_WITH_MEMORY,
        base_memory,
        protocol="test_static",
        trained=False,
        memory_snapshot_id=snapshot_id,
    )
    _require_cell(
        OPD_WITH_MEMORY,
        opd_memory,
        protocol="test_static",
        trained=True,
        memory_snapshot_id=snapshot_id,
    )
    _require_cell(
        OPD_NO_MEMORY,
        opd_none,
        protocol="no_memory",
        trained=True,
        memory_snapshot_id=None,
    )
    if base_memory.get("memory_counts") != snapshot_counts:
        raise ValueError("base_with_memory counts do not match the frozen snapshot")
    if opd_memory.get("memory_counts") != snapshot_counts:
        raise ValueError("opd_with_memory counts do not match the frozen snapshot")
    for field in ("checkpoint", "adapter_revision"):
        if opd_memory[field] != opd_none[field]:
            raise ValueError(f"trained evaluation cells use different {field}")
    compare_evaluation_reports(ordered, baseline_label=BASE_NO_MEMORY)
    _validate_evaluation_coverage(
        ordered,
        expected_test_tasks=expected_test_tasks,
        expected_test_trials=expected_test_trials,
    )
    return ordered


def _require_cell(
    label: str,
    provenance: Mapping[str, Any],
    *,
    protocol: str,
    trained: bool,
    memory_snapshot_id: str | None,
) -> None:
    if provenance.get("protocol") != protocol:
        raise ValueError(f"{label} must use protocol {protocol}")
    checkpoint = provenance.get("checkpoint")
    adapter = provenance.get("adapter_revision")
    if trained and (
        not isinstance(checkpoint, str)
        or not checkpoint.strip()
        or not isinstance(adapter, str)
        or not adapter.strip()
    ):
        raise ValueError(f"{label} must use an OPD checkpoint and adapter")
    if not trained and (checkpoint is not None or adapter is not None):
        raise ValueError(f"{label} must use the bare base model")
    if provenance.get("memory_snapshot_id") != memory_snapshot_id:
        raise ValueError(f"{label} uses the wrong Memory snapshot")
    output_snapshot_ids = provenance.get("output_memory_snapshot_ids")
    if not isinstance(output_snapshot_ids, list) or not output_snapshot_ids or any(
        snapshot_id != memory_snapshot_id
        for snapshot_id in output_snapshot_ids
    ):
        raise ValueError(f"{label} has invalid output Memory snapshots")
    counts = provenance.get("memory_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(MEMORY_TIERS)
        or any(type(value) is not int or value < 0 for value in counts.values())
    ):
        raise ValueError(f"{label} has invalid Memory counts")
    if memory_snapshot_id is None and any(counts.values()):
        raise ValueError(f"{label} must not expose Memory items")


def _validate_evaluation_coverage(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    expected_test_tasks: int,
    expected_test_trials: int,
) -> None:
    expected_episodes = expected_test_tasks * expected_test_trials
    for label, report in reports.items():
        output_snapshot_ids = report["provenance"].get("output_memory_snapshot_ids")
        if len(output_snapshot_ids) != expected_test_trials:
            raise ValueError(
                f"{label} output Memory snapshot count must be "
                f"{expected_test_trials}"
            )
        summary = report["summary"]
        expected = {
            "task_count": expected_test_tasks,
            "trial_count": expected_test_trials,
            "episode_count": expected_episodes,
            "completed_count": expected_episodes,
            "token_usage_episode_count": expected_episodes,
        }
        for field, value in expected.items():
            if summary.get(field) != value:
                raise ValueError(
                    f"{label} {field} must be {value} for a complete Stage 8 report"
                )
        mean_tokens = summary.get("mean_agent_tokens")
        if (
            not isinstance(mean_tokens, (int, float))
            or isinstance(mean_tokens, bool)
            or not math.isfinite(float(mean_tokens))
        ):
            raise ValueError(f"{label} mean Agent tokens are missing")


def _evaluation_cell(report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report["provenance"]
    summary = report["summary"]
    return {
        "run_id": provenance["run_id"],
        "protocol": provenance["protocol"],
        "checkpoint": provenance["checkpoint"],
        "adapter_revision": provenance["adapter_revision"],
        "memory_snapshot_id": provenance["memory_snapshot_id"],
        "pass_at_1": summary["pass_at_1"],
        "mean_reward": summary["mean_reward"],
        "mean_agent_tokens": summary["mean_agent_tokens"],
        "mean_agent_tokens_successful": summary[
            "mean_agent_tokens_successful"
        ],
        "token_usage_episode_count": summary["token_usage_episode_count"],
        "memory_item_count": summary["memory_item_count"],
        "memory_counts": summary["memory_counts"],
        "memory_reuse_coverage": summary["memory_reuse_coverage"],
        "unique_reused_memory_count": summary[
            "unique_reused_memory_count"
        ],
        "mean_selected_memories": summary["mean_selected_memories"],
        "parse_error_rate": summary["parse_error_rate"],
        "mean_steps": summary["mean_steps"],
    }


def _paired_contrast(
    target: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    target_by_task = _task_episode_values(target)
    control_by_task = _task_episode_values(control)
    if target_by_task.keys() != control_by_task.keys():
        raise ValueError("paired evaluation task sets differ")
    task_ids = tuple(target_by_task)
    pass_deltas = [
        _mean(target_by_task[task_id]["success"])
        - _mean(control_by_task[task_id]["success"])
        for task_id in task_ids
    ]
    token_deltas = [
        _mean(target_by_task[task_id]["tokens"])
        - _mean(control_by_task[task_id]["tokens"])
        for task_id in task_ids
        if target_by_task[task_id]["tokens"]
        and control_by_task[task_id]["tokens"]
    ]
    low, high = _bootstrap_mean_interval(
        pass_deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "pass_at_1_delta": _mean(pass_deltas),
        "pass_at_1_ci95": [low, high],
        "paired_task_count": len(pass_deltas),
        "mean_agent_tokens_delta": (
            _mean(token_deltas) if token_deltas else None
        ),
    }


def _interaction_contrast(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    values = {
        label: float(reports[label]["summary"]["pass_at_1"])
        for label in EXPERIMENT_ORDER
    }
    return {
        "definition": "(C-D)-(B-A)",
        "pass_at_1": (
            values[OPD_WITH_MEMORY]
            - values[OPD_NO_MEMORY]
            - values[BASE_WITH_MEMORY]
            + values[BASE_NO_MEMORY]
        ),
    }


def _task_episode_values(
    report: Mapping[str, Any],
) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = {}
    for task_id in report["provenance"]["task_ids"]:
        values[task_id] = {"success": [], "tokens": []}
    for episode in report["episodes"]:
        task = values[episode["task_id"]]
        task["success"].append(1.0 if episode["success"] else 0.0)
        if episode["agent_total_tokens"] is not None:
            task["tokens"].append(float(episode["agent_total_tokens"]))
    return values


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    rng = random.Random(seed)
    size = len(values)
    estimates = sorted(
        _mean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    )
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def _summarize_train_passes(
    run_dirs: Sequence[Path],
    *,
    memory_snapshot_path: Path,
    expected_train_passes: int,
) -> dict[str, Any]:
    paths = tuple(Path(path).resolve() for path in run_dirs)
    if len(paths) != expected_train_passes:
        raise ValueError(
            f"Stage 8 requires exactly {expected_train_passes} train passes"
        )
    final_snapshot_path = Path(memory_snapshot_path).resolve()
    snapshot_parent = final_snapshot_path.parent
    manifests = [_read_json(path / "manifest.json") for path in paths]
    summaries = [
        _read_json(path / "fast_loop_summary.json") for path in paths
    ]
    events_by_pass = [
        _read_jsonl(path / "rollouts" / "events.jsonl") for path in paths
    ]

    task_sets = []
    pass_rows = []
    previous_completed_after: int | None = None
    for index, (manifest, summary, events) in enumerate(
        zip(manifests, summaries, events_by_pass, strict=True),
        start=1,
    ):
        task_ids = manifest.get("task_ids")
        if (
            manifest.get("split") != "train"
            or not isinstance(task_ids, list)
            or len(task_ids) != TRAIN_TASK_COUNT
            or len(task_ids) != len(set(task_ids))
        ):
            raise ValueError(
                f"train pass {index} must contain all {TRAIN_TASK_COUNT} train tasks"
            )
        task_sets.append(set(task_ids))
        failed_task_ids = summary.get("failed_task_ids", [])
        if (
            not isinstance(failed_task_ids, list)
            or len(failed_task_ids) != len(set(failed_task_ids))
            or not set(failed_task_ids) <= set(task_ids)
        ):
            raise ValueError(f"train pass {index} failed task IDs are invalid")
        failed_task_set = set(failed_task_ids)
        successful_task_ids = [
            task_id for task_id in task_ids if task_id not in failed_task_set
        ]
        attempted_count = summary.get(
            "attempted_task_count",
            TRAIN_TASK_COUNT,
        )
        failed_count = summary.get(
            "failed_task_count",
            len(failed_task_ids),
        )
        if (
            summary.get("run_id") != manifest.get("run_id")
            or attempted_count != TRAIN_TASK_COUNT
            or summary.get("episode_count") != len(successful_task_ids)
            or failed_count != len(failed_task_ids)
            or summary.get("memory_enabled") is not True
        ):
            raise ValueError(f"train pass {index} task outcome counts are invalid")
        recorded_successes = summary.get("successful_task_ids")
        if (
            recorded_successes is not None
            and recorded_successes != successful_task_ids
        ):
            raise ValueError(f"train pass {index} successful task IDs are invalid")
        completed_before = summary.get("completed_train_tasks_before")
        completed_after = summary.get("completed_train_tasks_after")
        if (
            type(completed_before) is not int
            or type(completed_after) is not int
            or completed_before < 0
            or completed_after != completed_before + TRAIN_TASK_COUNT
            or (
                previous_completed_after is not None
                and completed_before != previous_completed_after
            )
        ):
            raise ValueError(f"train pass {index} completed-task range is invalid")
        previous_completed_after = completed_after
        if summary.get("input_memory_snapshot_id") != manifest.get(
            "memory_snapshot_id"
        ):
            raise ValueError(f"train pass {index} input snapshot is invalid")
        if index > 1 and (
            summaries[index - 2].get("output_memory_snapshot_id")
            != manifest.get("memory_snapshot_id")
        ):
            raise ValueError("train pass Memory snapshot chain is broken")
        finished = [
            event for event in events
            if event.get("event_type") == "EpisodeFinished"
        ]
        failed = [
            event for event in events
            if event.get("event_type") == "TaskFailed"
        ]
        outcomes = [
            event for event in events
            if event.get("event_type") in {"EpisodeFinished", "TaskFailed"}
        ]
        if (
            [event.get("task_id") for event in outcomes] != task_ids
            or [event.get("task_id") for event in finished]
            != successful_task_ids
            or [event.get("task_id") for event in failed] != failed_task_ids
        ):
            raise ValueError(
                f"train pass {index} does not contain one outcome per task"
            )
        token_count = summary.get("token_usage_episode_count")
        mean_tokens = summary.get("mean_agent_tokens")
        if (
            token_count != len(finished)
            or (
                finished
                and (
                    not isinstance(mean_tokens, (int, float))
                    or isinstance(mean_tokens, bool)
                    or not math.isfinite(float(mean_tokens))
                )
            )
            or (not finished and mean_tokens is not None)
        ):
            raise ValueError(f"train pass {index} Agent token usage is incomplete")
        input_counts = _snapshot_counts(
            snapshot_parent,
            manifest.get("memory_snapshot_id"),
        )
        output_counts = _snapshot_counts(
            snapshot_parent,
            summary.get("output_memory_snapshot_id"),
        )
        if summary.get("input_memory_counts") != input_counts:
            raise ValueError(f"train pass {index} input Memory counts are invalid")
        if summary.get("output_memory_counts") != output_counts:
            raise ValueError(f"train pass {index} output Memory counts are invalid")
        pass_rows.append(
            {
                "pass_index": index,
                "run_id": manifest["run_id"],
                "seed": manifest.get("seed"),
                "task_order": list(task_ids),
                "attempted_task_count": TRAIN_TASK_COUNT,
                "episode_count": len(finished),
                "failed_task_count": len(failed),
                "failed_task_ids": list(failed_task_ids),
                "pass_at_1": (
                    sum(
                        1.0
                        for event in finished
                        if event.get("final_reward") == 1.0
                    )
                    / TRAIN_TASK_COUNT
                ),
                "mean_agent_tokens": (
                    float(mean_tokens) if mean_tokens is not None else None
                ),
                "input_memory_snapshot_id": manifest.get(
                    "memory_snapshot_id"
                ),
                "output_memory_snapshot_id": summary.get(
                    "output_memory_snapshot_id"
                ),
                "input_memory_counts": input_counts,
                "output_memory_counts": output_counts,
                "maintenance_rounds": list(
                    summary.get("maintenance_rounds_executed", [])
                ),
            }
        )
    if any(task_set != task_sets[0] for task_set in task_sets[1:]):
        raise ValueError("train passes do not use the same official task set")
    for field in (
        "iteration",
        "model_revision",
        "adapter_revision",
        "tau2_commit",
        "split_hash",
    ):
        values = {manifest.get(field) for manifest in manifests}
        if len(values) != 1:
            raise ValueError(f"train pass {field} differs")
    model_revision = manifests[0].get("model_revision")
    adapter_revision = manifests[0].get("adapter_revision")
    if not isinstance(model_revision, str) or not model_revision.strip():
        raise ValueError("train pass model revision is invalid")
    if adapter_revision is not None and (
        not isinstance(adapter_revision, str) or not adapter_revision.strip()
    ):
        raise ValueError("train pass adapter revision is invalid")
    seeds = [manifest.get("seed") for manifest in manifests]
    if (
        any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("train pass seeds must be distinct non-negative integers")
    if summaries[-1].get("output_memory_snapshot_id") != final_snapshot_path.name:
        raise ValueError("final train pass does not produce the evaluation snapshot")

    reuse = _train_memory_reuse(
        events_by_pass,
        snapshot_parent=snapshot_parent,
    )
    final_counts = _snapshot_counts(
        snapshot_parent,
        final_snapshot_path.name,
    )
    return {
        "pass_count": len(paths),
        "tasks_per_pass": TRAIN_TASK_COUNT,
        "attempted_episode_count": TRAIN_TASK_COUNT * len(paths),
        "episode_count": sum(row["episode_count"] for row in pass_rows),
        "failed_task_count": sum(
            row["failed_task_count"] for row in pass_rows
        ),
        "passes": pass_rows,
        "snapshot_chain": [
            manifests[0]["memory_snapshot_id"],
            *[
                summary["output_memory_snapshot_id"]
                for summary in summaries
            ],
        ],
        "policy_lineage": {
            "iteration": manifests[0].get("iteration"),
            "model_revision": model_revision,
            "adapter_revision": adapter_revision,
            "tau2_commit": manifests[0].get("tau2_commit"),
            "split_hash": manifests[0].get("split_hash"),
        },
        "final_memory_snapshot_id": final_snapshot_path.name,
        "final_memory_counts": final_counts,
        "final_memory_item_count": sum(final_counts.values()),
        **reuse,
    }


def _train_memory_reuse(
    events_by_pass: Sequence[Sequence[Mapping[str, Any]]],
    *,
    snapshot_parent: Path,
) -> dict[str, Any]:
    eligible: set[str] = set()
    selected: set[str] = set()
    selection_count = 0
    current_active: set[str] | None = None
    active_by_snapshot: dict[str, set[str]] = {}
    for events in events_by_pass:
        for event in events:
            event_type = event.get("event_type")
            if event_type == "EpisodeStarted":
                snapshot_id = event.get("memory_snapshot_id")
                if not isinstance(snapshot_id, str) or not snapshot_id:
                    raise ValueError(
                        "EpisodeStarted is missing its Memory snapshot ID"
                    )
                if snapshot_id not in active_by_snapshot:
                    repository = ReadOnlyMemoryRepository(
                        snapshot_parent / snapshot_id
                    )
                    active_by_snapshot[snapshot_id] = {
                        item.id for item in repository.list()
                    }
                current_active = active_by_snapshot[snapshot_id]
                eligible.update(current_active)
            elif event_type == "MemorySelected":
                ids = _string_list(event.get("selected_memory_ids"))
                if current_active is None or not set(ids).issubset(
                    current_active
                ):
                    raise ValueError(
                        "MemorySelected contains an item unavailable to the episode"
                    )
                selected.update(ids)
                selection_count += len(ids)
    reused = selected & eligible
    return {
        "eligible_memory_count": len(eligible),
        "unique_reused_memory_count": len(reused),
        "memory_selection_count": selection_count,
        "memory_reuse_coverage": (
            len(reused) / len(eligible) if eligible else None
        ),
    }


def _summarize_dataset(dataset_dir: Path) -> dict[str, Any]:
    root = Path(dataset_dir).resolve()
    manifest = _read_json(root / "dataset_manifest.json")
    audit = _read_json(root / "audit_report.json")
    build_id = manifest.get("dataset_build_id")
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("OPD dataset build ID is missing")
    if (
        audit.get("passed") is not True
        or audit.get("dataset_build_id") != build_id
    ):
        raise ValueError("OPD dataset audit did not pass")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("OPD dataset manifest counts are missing")
    kind_counts = {
        kind: _nonnegative_int(counts.get(kind), f"dataset {kind} count")
        for kind in _OPD_KINDS
    }
    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("run_id"), str)
        or not row["run_id"]
        for row in source_runs
    ):
        raise ValueError("OPD dataset source runs are invalid")
    source_run_ids = [row["run_id"] for row in source_runs]
    if len(source_run_ids) != len(set(source_run_ids)):
        raise ValueError("OPD dataset source runs contain duplicates")
    policy_lineage = manifest.get("policy_lineage")
    memory = manifest.get("memory")
    official_split = manifest.get("official_split")
    if (
        not isinstance(policy_lineage, Mapping)
        or not isinstance(memory, Mapping)
        or not isinstance(official_split, Mapping)
    ):
        raise ValueError("OPD dataset lineage is incomplete")
    snapshot_chain = memory.get("snapshot_chain")
    if not isinstance(snapshot_chain, list) or any(
        not isinstance(snapshot_id, str) or not snapshot_id
        for snapshot_id in snapshot_chain
    ):
        raise ValueError("OPD dataset Memory snapshot chain is invalid")
    return {
        "dataset_build_id": build_id,
        "audit_passed": True,
        "example_count": sum(kind_counts.values()),
        "kind_counts": kind_counts,
        "evidence_episode_count": _nonnegative_int(
            counts.get("evidence_episodes"),
            "dataset evidence episode count",
        ),
        "evidence_maintenance_count": _nonnegative_int(
            counts.get("evidence_maintenance"),
            "dataset evidence maintenance count",
        ),
        "memory_score_count": _nonnegative_int(
            counts.get("memory_scores"),
            "dataset Memory score count",
        ),
        "skip_reasons": dict(manifest.get("skip_reasons", {})),
        "source_run_ids": source_run_ids,
        "policy_lineage": dict(policy_lineage),
        "official_split": dict(official_split),
        "memory_snapshot_chain": list(snapshot_chain),
    }


def _summarize_training(training_dir: Path) -> dict[str, Any]:
    root = Path(training_dir).resolve()
    manifest = _read_json(root / "training_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("OPD training manifest is not complete")
    dataset_build_id = manifest.get("dataset_build_id")
    latest_checkpoint = manifest.get("latest_checkpoint")
    adapter_revision = manifest.get("adapter_revision")
    completed_examples = _nonnegative_int(
        manifest.get("completed_examples"),
        "training completed example count",
    )
    total_examples = _nonnegative_int(
        manifest.get("total_examples"),
        "training total example count",
    )
    optimizer_steps = _nonnegative_int(
        manifest.get("optimizer_steps"),
        "training optimizer step count",
    )
    if (
        not isinstance(dataset_build_id, str)
        or not dataset_build_id
        or not isinstance(latest_checkpoint, str)
        or not latest_checkpoint
        or not isinstance(adapter_revision, str)
        or not adapter_revision
        or completed_examples == 0
        or completed_examples != total_examples
        or optimizer_steps == 0
    ):
        raise ValueError("OPD training completion lineage is invalid")
    checkpoint_path = (root / latest_checkpoint).resolve()
    try:
        checkpoint_path.relative_to(root)
    except ValueError as error:
        raise ValueError("OPD training checkpoint escapes the training directory") from error
    if not checkpoint_path.is_dir():
        raise ValueError("OPD training checkpoint directory is missing")
    checkpoint_manifest = _read_json(
        checkpoint_path / "checkpoint_manifest.json"
    )
    source_lineage = manifest.get("source_lineage")
    if not isinstance(source_lineage, Mapping):
        raise ValueError("OPD training source lineage is missing")
    if (
        checkpoint_manifest.get("status") != "checkpoint"
        or checkpoint_manifest.get("dataset_build_id") != dataset_build_id
        or checkpoint_manifest.get("adapter_revision") != adapter_revision
        or checkpoint_manifest.get("source_lineage") != source_lineage
    ):
        raise ValueError("OPD checkpoint lineage does not match training")
    metrics = _read_jsonl(root / "training_metrics.jsonl")
    generations = _read_jsonl(root / "training_generations.jsonl")
    if (
        len(metrics) != completed_examples
        or len(generations) != completed_examples
    ):
        raise ValueError("OPD training logs are incomplete")
    forward_kl_rows = []
    by_kind: dict[str, list[float]] = defaultdict(list)
    for row in metrics:
        nested = row.get("metrics")
        value = nested.get("forward_kl") if isinstance(nested, Mapping) else None
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("training forward KL metric is invalid")
        numeric = float(value)
        forward_kl_rows.append(
            {
                "sequence_index": row.get("sequence_index"),
                "epoch": row.get("epoch"),
                "kind": row.get("kind"),
                "forward_kl": numeric,
            }
        )
        by_kind[str(row.get("kind"))].append(numeric)
    response_token_counts = []
    for row in generations:
        response_ids = row.get("response_ids")
        if not isinstance(response_ids, list) or any(
            type(token_id) is not int for token_id in response_ids
        ):
            raise ValueError("training generation response IDs are invalid")
        response_token_counts.append(len(response_ids))
    return {
        "status": manifest["status"],
        "dataset_build_id": dataset_build_id,
        "latest_checkpoint": latest_checkpoint,
        "latest_checkpoint_path": str(checkpoint_path),
        "checkpoint_revision": adapter_revision,
        "completed_examples": completed_examples,
        "total_examples": total_examples,
        "optimizer_steps": optimizer_steps,
        "forward_kl_mean": (
            _mean(row["forward_kl"] for row in forward_kl_rows)
            if forward_kl_rows
            else None
        ),
        "forward_kl_by_kind": {
            kind: _mean(values) for kind, values in sorted(by_kind.items())
        },
        "forward_kl_curve": _downsample(forward_kl_rows, limit=240),
        "generated_response_token_count": sum(response_token_counts),
        "mean_generated_response_tokens": (
            _mean(response_token_counts) if response_token_counts else None
        ),
        "metric_row_count": len(metrics),
        "training_config": dict(manifest.get("training_config", {})),
        "source_lineage": dict(source_lineage),
    }


def _validate_training_lineage(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    dataset_summary: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    train_summary: Mapping[str, Any],
) -> None:
    expected_runs = [
        row["run_id"] for row in train_summary["passes"]
    ]
    if dataset_summary["source_run_ids"] != expected_runs:
        raise ValueError("OPD dataset does not consume the three train passes")
    if dataset_summary["memory_snapshot_chain"] != train_summary["snapshot_chain"]:
        raise ValueError("OPD dataset Memory lineage does not match train passes")
    train_lineage = train_summary["policy_lineage"]
    dataset_lineage = dataset_summary["policy_lineage"]
    for field in ("iteration", "model_revision", "adapter_revision", "tau2_commit"):
        if dataset_lineage.get(field) != train_lineage.get(field):
            raise ValueError(
                f"OPD dataset {field} lineage does not match train passes"
            )
    official_split = dataset_summary["official_split"]
    if (
        official_split.get("name") != "train"
        or official_split.get("sha256") != train_lineage["split_hash"]
    ):
        raise ValueError("OPD dataset split lineage does not match train passes")
    if training_summary["dataset_build_id"] != dataset_summary["dataset_build_id"]:
        raise ValueError("OPD training consumes a different dataset build")
    expected_source_lineage = {
        "model_revision": train_lineage["model_revision"],
        "adapter_revision": train_lineage["adapter_revision"],
    }
    if training_summary["source_lineage"] != expected_source_lineage:
        raise ValueError("OPD training source policy does not match train passes")
    if (
        reports[BASE_NO_MEMORY]["provenance"]["model_revision"]
        != train_lineage["model_revision"]
    ):
        raise ValueError("base evaluation model does not match train rollout model")
    trained = reports[OPD_WITH_MEMORY]["provenance"]
    checkpoint = trained["checkpoint"]
    if not isinstance(checkpoint, str):
        raise ValueError("trained checkpoint lineage is missing")
    if Path(checkpoint).resolve() != Path(
        training_summary["latest_checkpoint_path"]
    ):
        raise ValueError("evaluation checkpoint is not the completed OPD checkpoint")
    if trained["adapter_revision"] != training_summary["checkpoint_revision"]:
        raise ValueError("evaluation adapter revision is not the completed OPD adapter")


def _snapshot_counts(parent: Path, snapshot_id: Any) -> dict[str, int]:
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("Memory snapshot ID is missing")
    repository = ReadOnlyMemoryRepository(parent / snapshot_id)
    return {
        tier.value: len(repository.list(tier=tier))
        for tier in MemoryTier
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid JSONL artifact: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSONL artifact: {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"JSONL row must be an object: {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _string_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError("Memory event IDs must be a list of nonblank strings")
    return tuple(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _downsample(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return [dict(row) for row in rows]
    indices = {
        round(index * (len(rows) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [dict(rows[index]) for index in sorted(indices)]


def _mean(values: Sequence[float] | Any) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return math.fsum(materialized) / len(materialized)


def _require_json_safe(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Stage 8 experiment report is not JSON safe") from error
