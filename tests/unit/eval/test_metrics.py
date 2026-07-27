from __future__ import annotations

from dataclasses import replace

import pytest

from tau3_retail_evolver.eval.guard import EvaluationProtocol
from tau3_retail_evolver.eval.metrics import (
    EvaluationProvenance,
    build_evaluation_report,
    classify_failure,
    compare_evaluation_reports,
)
from tau3_retail_evolver.eval.runner import EvaluationRunResult, TrialEpisode
from tau3_retail_evolver.fast_loop.runner import EpisodeResult


def _episode(
    task_id: str,
    reward: float,
    *,
    steps: int = 2,
    parse_errors: int = 0,
    completed: bool = True,
    project_truncated: bool = False,
) -> EpisodeResult:
    reward_info = {
        "reward": reward,
        "reward_breakdown": {
            "DB": reward,
            "COMMUNICATE": 1.0,
        },
        "nl_assertions": [
            {
                "nl_assertion": "The requested outcome is communicated.",
                "met": reward == 1.0,
                "justification": "official evaluator output",
            }
        ],
    }
    return EpisodeResult(
        task_id=task_id,
        final_reward=reward,
        steps=steps,
        terminal_evaluation=reward_info,
        simulation_result={"termination_reason": "agent_stop"},
        selected_memory_ids=(),
        written_memory_ids=(),
        truncated=project_truncated,
        parse_error_count=parse_errors,
        completed=completed,
        project_truncated=project_truncated,
    )


def _provenance(
    *,
    protocol: EvaluationProtocol = EvaluationProtocol.TEST_STATIC,
    split_hash: str = "split-hash",
) -> EvaluationProvenance:
    return EvaluationProvenance(
        run_id=f"eval-{protocol.value}",
        protocol=protocol,
        official_base_reproduction=False,
        split="test",
        checkpoint="runs/iterations/iteration-0001/checkpoints/step-10",
        base_model="Qwen/Qwen3.5-9B",
        model_revision="qwen-sha",
        adapter_revision="adapter-sha",
        tau2_commit="tau2-sha",
        split_hash=split_hash,
        task_ids=("75", "76"),
        seeds=(42, 43),
        user_simulator_config={
            "model": "deepseek/deepseek-v4-pro",
            "base_url": "https://api.example.invalid",
        },
        nl_evaluator={
            "model": "openrouter/openai/gpt-4.1",
            "temperature": 0.0,
        },
        memory_snapshot_id=(
            "memory-sha"
            if protocol is EvaluationProtocol.TEST_STATIC
            else None
        ),
        max_episode_steps=40,
        model_serving_contract={"max_tokens": 8192},
        capabilities={
            "optimizer_create": False,
            "attribution": False,
            "dataset_write": False,
            "checkpoint_write": False,
            "train_memory_write": False,
            "memory_write": False,
        },
    )


def _run_result() -> EvaluationRunResult:
    return EvaluationRunResult(
        episodes=(
            TrialEpisode(0, 42, _episode("75", 1.0, steps=2)),
            TrialEpisode(
                0,
                42,
                _episode("76", 0.0, steps=3, parse_errors=1),
            ),
            TrialEpisode(1, 43, _episode("75", 0.5, steps=4)),
            TrialEpisode(1, 43, _episode("76", 1.0, steps=1)),
        ),
        maintenance_rounds_by_trial=((), ()),
        output_memory_snapshot_ids=("memory-sha", "memory-sha"),
    )


def test_report_preserves_official_rewards_and_aggregates_metrics() -> None:
    report = build_evaluation_report(_provenance(), _run_result())

    assert report["schema_version"] == 2
    assert report["report_type"] == "tau3-retail-evaluation"
    assert report["summary"] == {
        "task_count": 2,
        "trial_count": 2,
        "episode_count": 4,
        "completed_count": 4,
        "mean_reward": pytest.approx(0.625),
        "success_rate": pytest.approx(0.5),
        "pass_at_1": pytest.approx(0.5),
        "token_usage_episode_count": 0,
        "total_agent_prompt_tokens": 0,
        "total_agent_completion_tokens": 0,
        "total_agent_tokens": 0,
        "mean_agent_tokens": None,
        "mean_agent_tokens_successful": None,
        "memory_counts": {},
        "memory_item_count": 0,
        "memory_selection_count": 0,
        "unique_reused_memory_count": 0,
        "memory_reuse_coverage": None,
        "mean_selected_memories": pytest.approx(0.0),
        "total_steps": 10,
        "mean_steps": pytest.approx(2.5),
        "environment_parse_error_count": 1,
        "environment_parse_error_rate": pytest.approx(0.1),
        "response_count": 0,
        "response_parse_error_count": 0,
        "response_parse_error_rate": pytest.approx(0.0),
        "parse_error_count": 1,
        "parse_error_rate": pytest.approx(0.1),
        "failure_categories": {
            "db": 2,
            "nl_assertion": 2,
            "parse_error": 1,
        },
        "maintenance_rounds_by_trial": [[], []],
    }
    failed = report["episodes"][1]
    assert failed["reward"] == 0.0
    assert failed["reward_info"]["reward_breakdown"]["DB"] == 0.0
    assert failed["failure_categories"] == [
        "db",
        "nl_assertion",
        "parse_error",
    ]
    assert report["task_results"] == [
        {
            "task_id": "75",
            "completed_count": 2,
            "mean_reward": pytest.approx(0.75),
            "success_rate": pytest.approx(0.5),
            "mean_steps": pytest.approx(3.0),
            "mean_agent_tokens": None,
            "parse_error_rate": pytest.approx(0.0),
        },
        {
            "task_id": "76",
            "completed_count": 2,
            "mean_reward": pytest.approx(0.5),
            "success_rate": pytest.approx(0.5),
            "mean_steps": pytest.approx(2.0),
            "mean_agent_tokens": None,
            "parse_error_rate": pytest.approx(0.25),
        },
    ]


def test_airline_report_and_comparison_types_are_domain_specific() -> None:
    provenance = replace(_provenance(), domain="airline")
    report = build_evaluation_report(provenance, _run_result())

    assert report["report_type"] == "tau3-airline-evaluation"
    assert report["provenance"]["domain"] == "airline"
    comparison = compare_evaluation_reports(
        {"base": report, "candidate": report},
        baseline_label="base",
    )
    assert comparison["report_type"] == "tau3-airline-evaluation-comparison"
    assert comparison["controls"]["domain"] == "airline"
    assert report["provenance"]["task_order"] == ["75", "76"]
    assert report["provenance"]["seeds"] == [42, 43]
    assert report["provenance"]["num_trials"] == 2
    assert report["provenance"]["output_memory_snapshot_ids"] == [
        "memory-sha",
        "memory-sha",
    ]


def test_failure_classification_marks_incomplete_and_max_steps() -> None:
    result = _episode(
        "75",
        0.0,
        parse_errors=2,
        completed=False,
        project_truncated=True,
    )

    assert classify_failure(result) == (
        "db",
        "nl_assertion",
        "parse_error",
        "max_steps",
        "incomplete",
    )


def test_success_has_no_failure_category_even_with_nonfatal_parse_error() -> None:
    assert classify_failure(_episode("75", 1.0, parse_errors=1)) == ()


def test_report_separates_environment_and_response_parse_errors() -> None:
    result = replace(
        _episode("75", 0.0, steps=2, parse_errors=1),
        response_count=3,
        response_parse_error_count=1,
    )
    provenance = replace(
        _provenance(),
        task_ids=("75",),
        seeds=(42,),
    )
    run = EvaluationRunResult(
        episodes=(TrialEpisode(0, 42, result),),
        maintenance_rounds_by_trial=((),),
        output_memory_snapshot_ids=("memory-sha",),
    )

    report = build_evaluation_report(provenance, run)

    assert report["summary"]["environment_parse_error_rate"] == pytest.approx(0.5)
    assert report["summary"]["response_parse_error_rate"] == pytest.approx(1 / 3)
    assert report["summary"]["parse_error_rate"] == pytest.approx(2 / 5)


def test_report_aggregates_agent_tokens_and_memory_reuse() -> None:
    result = replace(
        _episode("75", 1.0),
        selected_memory_ids=("memory-a", "memory-b"),
        agent_prompt_tokens=90,
        agent_completion_tokens=30,
    )
    provenance = replace(
        _provenance(),
        task_ids=("75",),
        seeds=(42,),
        memory_counts={
            "trajectory": 0,
            "tip": 2,
            "skill": 1,
            "tool": 0,
        },
    )
    run = EvaluationRunResult(
        episodes=(TrialEpisode(0, 42, result),),
        maintenance_rounds_by_trial=((),),
        output_memory_snapshot_ids=("memory-sha",),
    )

    report = build_evaluation_report(provenance, run)

    assert report["summary"]["pass_at_1"] == 1.0
    assert report["summary"]["mean_agent_tokens"] == 120
    assert report["summary"]["mean_agent_tokens_successful"] == 120
    assert report["summary"]["memory_item_count"] == 3
    assert report["summary"]["unique_reused_memory_count"] == 2
    assert report["summary"]["memory_reuse_coverage"] == pytest.approx(2 / 3)


def test_report_rejects_missing_or_out_of_order_trials() -> None:
    run = _run_result()
    reordered = replace(
        run,
        episodes=(run.episodes[1], run.episodes[0], *run.episodes[2:]),
    )

    with pytest.raises(ValueError, match="episode order"):
        build_evaluation_report(_provenance(), reordered)


def test_comparison_accepts_protocol_as_treatment_and_reports_deltas() -> None:
    static = build_evaluation_report(_provenance(), _run_result())
    no_memory_provenance = replace(
        _provenance(protocol=EvaluationProtocol.NO_MEMORY),
        memory_snapshot_id=None,
        adapter_revision=None,
        checkpoint=None,
    )
    no_memory_run = replace(
        _run_result(),
        output_memory_snapshot_ids=(None, None),
    )
    no_memory = build_evaluation_report(no_memory_provenance, no_memory_run)

    comparison = compare_evaluation_reports(
        {
            "base_qwen": no_memory,
            "trained_static": static,
        },
        baseline_label="base_qwen",
    )

    assert comparison["baseline_label"] == "base_qwen"
    assert comparison["controls"]["tau2_commit"] == "tau2-sha"
    assert comparison["rows"][0]["label"] == "base_qwen"
    assert comparison["rows"][1]["protocol"] == "test_static"
    assert comparison["rows"][1]["mean_reward_delta"] == pytest.approx(0.0)


def test_comparison_rejects_changed_control_variables() -> None:
    baseline = build_evaluation_report(_provenance(), _run_result())
    mismatched = build_evaluation_report(
        _provenance(split_hash="other-split"),
        _run_result(),
    )

    with pytest.raises(ValueError, match="split_hash"):
        compare_evaluation_reports(
            {"baseline": baseline, "changed": mismatched},
            baseline_label="baseline",
        )
