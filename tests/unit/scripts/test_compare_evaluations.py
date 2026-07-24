from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import compare_evaluations


def _report(run_id: str, reward: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_type": "tau3-retail-evaluation",
        "provenance": {
            "run_id": run_id,
            "protocol": "no_memory",
            "official_base_reproduction": False,
            "split": "test",
            "checkpoint": None,
            "base_model": "Qwen/Qwen3.5-9B",
            "model_revision": "qwen-sha",
            "adapter_revision": None,
            "tau2_commit": "tau2-sha",
            "split_hash": "split-sha",
            "task_ids": ["75"],
            "task_order": ["75"],
            "seeds": [42],
            "num_trials": 1,
            "user_simulator_config": {"model": "deepseek"},
            "nl_evaluator": {"model": "gpt-4.1"},
            "memory_snapshot_id": None,
            "output_memory_snapshot_ids": [None],
            "max_episode_steps": 40,
            "model_serving_contract": {"max_tokens": 8192},
            "capabilities": {},
        },
        "summary": {
            "mean_reward": reward,
            "success_rate": reward,
            "completed_count": 1,
            "parse_error_rate": 0.0,
        },
        "task_results": [],
        "episodes": [],
    }


def test_compare_cli_writes_controlled_comparison(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(_report("base", 0.25)), encoding="utf-8")
    candidate.write_text(json.dumps(_report("candidate", 0.75)), encoding="utf-8")

    result = compare_evaluations.main(
        [
            "--report",
            f"base={baseline}",
            "--report",
            f"candidate={candidate}",
            "--baseline-label",
            "base",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text("utf-8"))
    assert result == 0
    assert payload["rows"][1]["mean_reward_delta"] == pytest.approx(0.5)
    assert json.loads(capsys.readouterr().out)["output"] == str(output)


def test_report_argument_rejects_duplicate_labels(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report("base", 1.0)), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        compare_evaluations._load_labeled_reports(
            [f"same={path}", f"same={path}"]
        )
