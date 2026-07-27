from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from tau3_retail_evolver.eval.guard import EvaluationProtocol
from tau3_retail_evolver.eval.runner import EvaluationRunResult, TrialEpisode
from tau3_retail_evolver.fast_loop.runner import EpisodeResult
from tau3_retail_evolver.memory.json_store import write_bytes_atomic
from tau3_retail_evolver.memory.types import MEMORY_TIERS
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


REPORT_SCHEMA_VERSION = 2
REPORT_TYPE = "tau3-retail-evaluation"
COMPARISON_TYPE = "tau3-retail-evaluation-comparison"


@dataclass(frozen=True, slots=True)
class EvaluationProvenance:
    run_id: str
    protocol: EvaluationProtocol
    official_base_reproduction: bool
    split: str
    checkpoint: str | None
    base_model: str
    model_revision: str
    adapter_revision: str | None
    tau2_commit: str
    split_hash: str
    task_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    user_simulator_config: Mapping[str, Any]
    nl_evaluator: Mapping[str, Any]
    memory_snapshot_id: str | None
    max_episode_steps: int
    model_serving_contract: Mapping[str, Any]
    capabilities: Mapping[str, bool]
    memory_counts: Mapping[str, int] | None = None
    domain: str = "retail"

    def __post_init__(self) -> None:
        for field, value in (
            ("run_id", self.run_id),
            ("base_model", self.base_model),
            ("model_revision", self.model_revision),
            ("tau2_commit", self.tau2_commit),
            ("split_hash", self.split_hash),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must not be blank")
        if not isinstance(self.protocol, EvaluationProtocol):
            raise TypeError("protocol must be an EvaluationProtocol")
        if self.domain not in {"retail", "airline"}:
            raise ValueError("evaluation provenance domain must be retail or airline")
        if self.split not in {"test", "base"}:
            raise ValueError("evaluation provenance split must be test or base")
        if self.official_base_reproduction != (self.split == "base"):
            raise ValueError("official base reproduction marker does not match split")
        if not self.task_ids or len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("task IDs must be non-empty and unique")
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(type(seed) is not int or seed < 0 for seed in self.seeds)
        ):
            raise ValueError("seeds must be non-negative and unique")
        if type(self.max_episode_steps) is not int or self.max_episode_steps <= 0:
            raise ValueError("max episode steps must be positive")
        if (
            self.protocol is EvaluationProtocol.TEST_STATIC
            and not self.memory_snapshot_id
        ):
            raise ValueError("test_static provenance requires a Memory snapshot")
        if (
            self.protocol is not EvaluationProtocol.TEST_STATIC
            and self.memory_snapshot_id is not None
        ):
            raise ValueError(
                f"{self.protocol.value} provenance must not use a source Memory snapshot"
            )
        counts = dict(self.memory_counts or {})
        if counts and set(counts) != set(MEMORY_TIERS):
            raise ValueError("memory counts must contain every Memory tier")
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("memory counts must be non-negative integers")
        if (
            self.protocol is not EvaluationProtocol.TEST_STATIC
            and any(counts.values())
        ):
            raise ValueError("only test_static may have source Memory items")

    def as_dict(
        self,
        *,
        output_memory_snapshot_ids: Sequence[str | None],
    ) -> dict[str, Any]:
        return sanitize_artifact_data(
            {
                "run_id": self.run_id,
                "domain": self.domain,
                "protocol": self.protocol.value,
                "official_base_reproduction": self.official_base_reproduction,
                "split": self.split,
                "checkpoint": self.checkpoint,
                "base_model": self.base_model,
                "model_revision": self.model_revision,
                "adapter_revision": self.adapter_revision,
                "tau2_commit": self.tau2_commit,
                "split_hash": self.split_hash,
                "task_ids": list(self.task_ids),
                "task_order": list(self.task_ids),
                "seeds": list(self.seeds),
                "num_trials": len(self.seeds),
                "user_simulator_config": dict(self.user_simulator_config),
                "nl_evaluator": dict(self.nl_evaluator),
                "memory_snapshot_id": self.memory_snapshot_id,
                "memory_counts": dict(self.memory_counts or {}),
                "output_memory_snapshot_ids": list(output_memory_snapshot_ids),
                "max_episode_steps": self.max_episode_steps,
                "model_serving_contract": dict(self.model_serving_contract),
                "capabilities": dict(self.capabilities),
            }
        )


def build_evaluation_report(
    provenance: EvaluationProvenance,
    run: EvaluationRunResult,
) -> dict[str, Any]:
    _validate_episode_order(provenance, run)
    episode_rows = [_episode_row(episode) for episode in run.episodes]
    total_steps = sum(row["steps"] for row in episode_rows)
    environment_parse_error_count = sum(
        row["environment_parse_error_count"] for row in episode_rows
    )
    response_count = sum(row["response_count"] for row in episode_rows)
    response_parse_error_count = sum(
        row["response_parse_error_count"] for row in episode_rows
    )
    parse_error_count = (
        environment_parse_error_count + response_parse_error_count
    )
    parse_error_opportunities = total_steps + response_count
    failure_counts = Counter(
        category
        for row in episode_rows
        for category in row["failure_categories"]
    )
    episode_count = len(episode_rows)
    token_rows = [
        row for row in episode_rows if row["agent_total_tokens"] is not None
    ]
    successful_token_rows = [row for row in token_rows if row["success"]]
    selected_memory_ids = {
        memory_id
        for row in episode_rows
        for memory_id in row["selected_memory_ids"]
    }
    memory_counts = dict(provenance.memory_counts or {})
    memory_item_count = sum(memory_counts.values())
    memory_selection_count = sum(
        len(row["selected_memory_ids"]) for row in episode_rows
    )
    task_results = [
        _task_result(task_id, episode_rows)
        for task_id in provenance.task_ids
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": _evaluation_report_type(provenance.domain),
        "provenance": provenance.as_dict(
            output_memory_snapshot_ids=run.output_memory_snapshot_ids,
        ),
        "summary": {
            "task_count": len(provenance.task_ids),
            "trial_count": len(provenance.seeds),
            "episode_count": episode_count,
            "completed_count": sum(row["completed"] for row in episode_rows),
            "mean_reward": _mean(row["reward"] for row in episode_rows),
            "success_rate": _mean(
                1.0 if row["success"] else 0.0 for row in episode_rows
            ),
            "pass_at_1": _mean(
                1.0 if row["success"] else 0.0 for row in episode_rows
            ),
            "token_usage_episode_count": len(token_rows),
            "total_agent_prompt_tokens": sum(
                row["agent_prompt_tokens"] for row in token_rows
            ),
            "total_agent_completion_tokens": sum(
                row["agent_completion_tokens"] for row in token_rows
            ),
            "total_agent_tokens": sum(
                row["agent_total_tokens"] for row in token_rows
            ),
            "mean_agent_tokens": (
                _mean(row["agent_total_tokens"] for row in token_rows)
                if token_rows
                else None
            ),
            "mean_agent_tokens_successful": (
                _mean(
                    row["agent_total_tokens"]
                    for row in successful_token_rows
                )
                if successful_token_rows
                else None
            ),
            "memory_counts": memory_counts,
            "memory_item_count": memory_item_count,
            "memory_selection_count": memory_selection_count,
            "unique_reused_memory_count": len(selected_memory_ids),
            "memory_reuse_coverage": (
                len(selected_memory_ids) / memory_item_count
                if memory_item_count
                else None
            ),
            "mean_selected_memories": (
                memory_selection_count / episode_count if episode_count else 0.0
            ),
            "total_steps": total_steps,
            "mean_steps": _mean(row["steps"] for row in episode_rows),
            "environment_parse_error_count": environment_parse_error_count,
            "environment_parse_error_rate": (
                environment_parse_error_count / total_steps
                if total_steps
                else 0.0
            ),
            "response_count": response_count,
            "response_parse_error_count": response_parse_error_count,
            "response_parse_error_rate": (
                response_parse_error_count / response_count
                if response_count
                else 0.0
            ),
            "parse_error_count": parse_error_count,
            "parse_error_rate": (
                parse_error_count / parse_error_opportunities
                if parse_error_opportunities
                else 0.0
            ),
            "failure_categories": {
                category: failure_counts[category]
                for category in sorted(failure_counts)
            },
            "maintenance_rounds_by_trial": [
                list(rounds) for rounds in run.maintenance_rounds_by_trial
            ],
        },
        "task_results": task_results,
        "episodes": episode_rows,
    }
    _require_json_safe(report)
    return report


def classify_failure(result: EpisodeResult) -> tuple[str, ...]:
    if _is_success(result.final_reward):
        return ()

    categories: set[str] = set()
    reward_info = result.terminal_evaluation
    breakdown = reward_info.get("reward_breakdown")
    if isinstance(breakdown, Mapping):
        for component, reward in breakdown.items():
            if _is_failed_component(reward):
                categories.add(_component_category(str(component)))

    db_check = reward_info.get("db_check")
    if isinstance(db_check, Mapping) and db_check.get("db_match") is False:
        categories.add("db")
    if _has_failed_check(reward_info.get("env_assertions"), "met"):
        categories.add("environment_assertion")
    if _has_failed_check(reward_info.get("action_checks"), "action_match"):
        categories.add("action")
    if _has_failed_check(reward_info.get("nl_assertions"), "met"):
        categories.add("nl_assertion")
    if _has_failed_check(reward_info.get("communicate_checks"), "met"):
        categories.add("communicate")
    if result.parse_error_count or result.response_parse_error_count:
        categories.add("parse_error")
    if result.project_truncated:
        categories.add("max_steps")
    if not result.completed:
        categories.add("incomplete")
    if not categories:
        categories.add("other")
    order = {
        "db": 0,
        "environment_assertion": 1,
        "action": 2,
        "communicate": 3,
        "nl_assertion": 4,
        "parse_error": 5,
        "max_steps": 6,
        "incomplete": 7,
        "other": 8,
    }
    return tuple(sorted(categories, key=lambda category: (order.get(category, 9), category)))


def compare_evaluation_reports(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    baseline_label: str,
) -> dict[str, Any]:
    if len(reports) < 2:
        raise ValueError("at least two evaluation reports are required")
    if baseline_label not in reports:
        raise ValueError("baseline label is not present in reports")
    normalized = {
        label: _validated_report(report, label=label)
        for label, report in reports.items()
    }
    baseline = normalized[baseline_label]
    controls = _comparison_controls(baseline)
    for label, report in normalized.items():
        candidate = _comparison_controls(report)
        for key, expected in controls.items():
            if candidate[key] != expected:
                raise ValueError(
                    f"comparison control {key} differs for report {label}"
                )

    baseline_summary = baseline["summary"]
    rows = []
    for label, report in normalized.items():
        provenance = report["provenance"]
        summary = report["summary"]
        rows.append(
            {
                "label": label,
                "run_id": provenance["run_id"],
                "protocol": provenance["protocol"],
                "checkpoint": provenance["checkpoint"],
                "adapter_revision": provenance["adapter_revision"],
                "memory_snapshot_id": provenance["memory_snapshot_id"],
                "mean_reward": summary["mean_reward"],
                "mean_reward_delta": (
                    summary["mean_reward"] - baseline_summary["mean_reward"]
                ),
                "success_rate": summary["success_rate"],
                "pass_at_1": summary["pass_at_1"],
                "success_rate_delta": (
                    summary["success_rate"] - baseline_summary["success_rate"]
                ),
                "completed_count": summary["completed_count"],
                "parse_error_rate": summary["parse_error_rate"],
                "mean_agent_tokens": summary.get("mean_agent_tokens"),
                "memory_item_count": summary.get("memory_item_count", 0),
                "memory_reuse_coverage": summary.get("memory_reuse_coverage"),
            }
        )
    comparison = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": _comparison_report_type(
            str(baseline["provenance"].get("domain", "retail"))
        ),
        "baseline_label": baseline_label,
        "controls": controls,
        "rows": rows,
    }
    _require_json_safe(comparison)
    return comparison


def write_evaluation_json(path: Path, payload: Mapping[str, Any]) -> None:
    _require_json_safe(payload)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, encoded)


def read_evaluation_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid evaluation JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"evaluation JSON must be an object: {path}")
    return payload


def _validate_episode_order(
    provenance: EvaluationProvenance,
    run: EvaluationRunResult,
) -> None:
    expected = [
        (trial_index, seed, task_id)
        for trial_index, seed in enumerate(provenance.seeds)
        for task_id in provenance.task_ids
    ]
    actual = [
        (episode.trial_index, episode.seed, episode.result.task_id)
        for episode in run.episodes
    ]
    if actual != expected:
        raise ValueError("evaluation episode order does not match task order and seeds")
    if len(run.maintenance_rounds_by_trial) != len(provenance.seeds):
        raise ValueError("maintenance trial count does not match seeds")
    if len(run.output_memory_snapshot_ids) != len(provenance.seeds):
        raise ValueError("output Memory snapshot count does not match seeds")


def _episode_row(episode: TrialEpisode) -> dict[str, Any]:
    result = episode.result
    if (
        not isinstance(result.final_reward, (int, float))
        or isinstance(result.final_reward, bool)
        or not math.isfinite(float(result.final_reward))
    ):
        raise ValueError("episode reward must be finite")
    if result.steps < 0 or not 0 <= result.parse_error_count <= result.steps:
        raise ValueError("episode step or parse-error count is invalid")
    if (
        result.response_count < 0
        or not 0
        <= result.response_parse_error_count
        <= result.response_count
    ):
        raise ValueError("episode response or response parse-error count is invalid")
    total_parse_errors = (
        result.parse_error_count + result.response_parse_error_count
    )
    if (result.agent_prompt_tokens is None) != (
        result.agent_completion_tokens is None
    ):
        raise ValueError("episode token usage must be complete or entirely absent")
    for value in (result.agent_prompt_tokens, result.agent_completion_tokens):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("episode token usage must be non-negative")
    total_agent_tokens = (
        result.agent_prompt_tokens + result.agent_completion_tokens
        if result.agent_prompt_tokens is not None
        and result.agent_completion_tokens is not None
        else None
    )
    parse_error_opportunities = result.steps + result.response_count
    return sanitize_artifact_data(
        {
            "task_id": result.task_id,
            "trial_index": episode.trial_index,
            "seed": episode.seed,
            "reward": float(result.final_reward),
            "success": _is_success(result.final_reward),
            "completed": result.completed,
            "steps": result.steps,
            "agent_prompt_tokens": result.agent_prompt_tokens,
            "agent_completion_tokens": result.agent_completion_tokens,
            "agent_total_tokens": total_agent_tokens,
            "selected_memory_ids": list(result.selected_memory_ids),
            "written_memory_ids": list(result.written_memory_ids),
            "environment_parse_error_count": result.parse_error_count,
            "environment_parse_error_rate": (
                result.parse_error_count / result.steps if result.steps else 0.0
            ),
            "response_count": result.response_count,
            "response_parse_error_count": result.response_parse_error_count,
            "response_parse_error_rate": (
                result.response_parse_error_count / result.response_count
                if result.response_count
                else 0.0
            ),
            "parse_error_count": total_parse_errors,
            "parse_error_rate": (
                total_parse_errors / parse_error_opportunities
                if parse_error_opportunities
                else 0.0
            ),
            "truncated": result.truncated,
            "project_truncated": result.project_truncated,
            "failure_categories": list(classify_failure(result)),
            "reward_info": dict(result.terminal_evaluation),
        }
    )


def _task_result(
    task_id: str,
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = [episode for episode in episodes if episode["task_id"] == task_id]
    total_steps = sum(episode["steps"] for episode in selected)
    response_count = sum(episode["response_count"] for episode in selected)
    parse_errors = sum(episode["parse_error_count"] for episode in selected)
    token_rows = [
        episode for episode in selected
        if episode["agent_total_tokens"] is not None
    ]
    parse_error_opportunities = total_steps + response_count
    return {
        "task_id": task_id,
        "completed_count": sum(episode["completed"] for episode in selected),
        "mean_reward": _mean(episode["reward"] for episode in selected),
        "success_rate": _mean(
            1.0 if episode["success"] else 0.0 for episode in selected
        ),
        "mean_steps": _mean(episode["steps"] for episode in selected),
        "mean_agent_tokens": (
            _mean(episode["agent_total_tokens"] for episode in token_rows)
            if token_rows
            else None
        ),
        "parse_error_rate": (
            parse_errors / parse_error_opportunities
            if parse_error_opportunities
            else 0.0
        ),
    }


def _comparison_controls(report: Mapping[str, Any]) -> dict[str, Any]:
    provenance = report["provenance"]
    controls = {
        key: provenance[key]
        for key in (
            "official_base_reproduction",
            "split",
            "base_model",
            "model_revision",
            "tau2_commit",
            "split_hash",
            "task_ids",
            "task_order",
            "seeds",
            "num_trials",
            "user_simulator_config",
            "nl_evaluator",
            "max_episode_steps",
            "model_serving_contract",
        )
    }
    controls["domain"] = provenance.get("domain", "retail")
    return controls


def _validated_report(
    report: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    provenance = report.get("provenance")
    domain = (
        str(provenance.get("domain", "retail"))
        if isinstance(provenance, Mapping)
        else ""
    )
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or domain not in {"retail", "airline"}
        or report.get("report_type") != _evaluation_report_type(domain)
        or not isinstance(provenance, Mapping)
        or not isinstance(report.get("summary"), Mapping)
    ):
        raise ValueError(f"invalid evaluation report: {label}")
    for metric in ("mean_reward", "success_rate", "parse_error_rate"):
        value = report["summary"].get(metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"invalid evaluation metric {metric}: {label}")
    return report


def _evaluation_report_type(domain: str) -> str:
    return f"tau3-{domain}-evaluation"


def _comparison_report_type(domain: str) -> str:
    return f"tau3-{domain}-evaluation-comparison"


def _component_category(component: str) -> str:
    normalized = "".join(character for character in component.casefold() if character.isalnum())
    aliases = {
        "db": "db",
        "database": "db",
        "env": "environment_assertion",
        "environment": "environment_assertion",
        "action": "action",
        "communicate": "communicate",
        "communication": "communicate",
        "nl": "nl_assertion",
    }
    return aliases.get(normalized, normalized or "other")


def _is_failed_component(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) < 1.0
    )


def _has_failed_check(value: Any, boolean_field: str) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    for check in value:
        if not isinstance(check, Mapping):
            continue
        if check.get(boolean_field) is False:
            return True
        for reward_field in ("reward", "action_reward", "communicate_reward"):
            if _is_failed_component(check.get(reward_field)):
                return True
    return False


def _is_success(reward: float) -> bool:
    return math.isclose(float(reward), 1.0, rel_tol=1e-12, abs_tol=1e-12)


def _mean(values: Any) -> float:
    collected = [float(value) for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _require_json_safe(value: Any) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("evaluation artifact must be JSON safe") from error
