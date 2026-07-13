from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ENV = (
    "QWEN_BASE_URL",
    "QWEN_MODEL_REVISION",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
)
MISSING_ENV = tuple(name for name in REQUIRED_ENV if not os.environ.get(name))

pytestmark = [
    pytest.mark.tau2_integration,
    pytest.mark.skipif(
        os.environ.get("RUN_FAST_LOOP_TAU2_INTEGRATION") != "1" or bool(MISSING_ENV),
        reason=(
            "set RUN_FAST_LOOP_TAU2_INTEGRATION=1 and provide the Qwen endpoint "
            "(QWEN_BASE_URL, QWEN_MODEL_REVISION), OpenRouter evaluator credential "
            "(OPENROUTER_API_KEY), and default DeepSeek simulator credential "
            "(DEEPSEEK_API_KEY)"
        ),
    ),
]


def test_real_fast_loop_collects_five_official_train_episodes(tmp_path: Path) -> None:
    config_path = WORKTREE_ROOT / "configs" / "default.yaml"
    config = load_config(config_path)
    runtime = Tau2Runtime.inspect_metadata(config.tau2.repo_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.retail_tasks_path,
        runtime.retail_split_path,
    )
    catalog.require_official_compatibility()
    task_ids = catalog.task_ids("train")[:5]
    run_id = "pytest-fast-loop-five"
    output_root = tmp_path / "runs"
    command = [
        sys.executable,
        "-m",
        "scripts.run_fast_loop",
        "--config",
        str(config_path),
        "--split",
        "train",
        "--run-id",
        run_id,
        "--output-root",
        str(output_root),
        "--project-root",
        str(tmp_path / "project"),
        "--qwen-base-url",
        os.environ["QWEN_BASE_URL"],
        "--model-revision",
        os.environ["QWEN_MODEL_REVISION"],
    ]
    for task_id in task_ids:
        command.extend(("--task-id", task_id))

    result = subprocess.run(
        command,
        cwd=WORKTREE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )

    assert result.returncode == 0, result.stderr
    run_path = output_root / run_id
    manifest = json.loads((run_path / "manifest.json").read_text("utf-8"))
    summary = json.loads((run_path / "fast_loop_summary.json").read_text("utf-8"))
    events = [
        json.loads(line)
        for line in (run_path / "rollouts" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    terminal_events = [event for event in events if event["event_type"] == "EpisodeFinished"]
    assert json.loads(result.stdout) == summary
    assert summary["episode_count"] == 5
    assert summary["successful_task_ids"] == list(task_ids)
    assert len(terminal_events) == 5
    assert manifest["tau2_commit"] == runtime.git_commit
    assert manifest["split_hash"] == catalog.split_sha256
    assert manifest["task_ids"] == list(task_ids)
    assert manifest["memory_snapshot_id"] == summary["input_memory_snapshot_id"]

    required_lifecycle = {
        "EpisodeStarted",
        "MemoryCandidatesRetrieved",
        "MemorySelected",
        "DecisionMade",
        "EnvironmentStepped",
        "EpisodeFinished",
        "MemoryWriteProposed",
        "MemoryWriteCommitted",
    }
    for task_id in task_ids:
        task_events = [event for event in events if event["task_id"] == task_id]
        ordered_types = [event["event_type"] for event in task_events]
        event_types = set(ordered_types)
        assert required_lifecycle <= event_types
        lifecycle_positions = [
            ordered_types.index(event_type)
            for event_type in (
                "EpisodeStarted",
                "MemoryCandidatesRetrieved",
                "MemorySelected",
                "DecisionMade",
                "EnvironmentStepped",
                "EpisodeFinished",
                "MemoryWriteProposed",
                "MemoryWriteCommitted",
            )
        ]
        assert lifecycle_positions == sorted(lifecycle_positions)
        assert all(
            event["memory_snapshot_id"] == summary["input_memory_snapshot_id"]
            for event in task_events
        )
        finished = next(event for event in task_events if event["event_type"] == "EpisodeFinished")
        assert isinstance(finished["final_reward"], (int, float))
        assert isinstance(finished["terminal_evaluation"], dict)
        assert isinstance(finished["simulation_result"], dict)
        proposed = next(
            event for event in task_events if event["event_type"] == "MemoryWriteProposed"
        )
        assert all(
            proposal["source_task_ids"] == [task_id]
            for proposal in proposed["proposals"]
        )

    canonical_written_ids = {
        memory_id
        for event in events
        if event["event_type"] == "MemoryWriteCommitted"
        for memory_id in event["written_memory_ids"]
        if memory_id not in set(event["replayed_memory_ids"])
    }
    maintenance_changed_ids = {
        memory_id
        for event in events
        if event["event_type"] == "MaintenanceCommitted"
        for field in ("created_ids", "updated_ids")
        for memory_id in event[field]
    }
    if canonical_written_ids or maintenance_changed_ids:
        assert summary["output_memory_snapshot_id"] != summary["input_memory_snapshot_id"]
    else:
        assert summary["output_memory_snapshot_id"] == summary["input_memory_snapshot_id"]

    memory_root = tmp_path / "project" / "history" / "agents" / "retail" / "memory"
    memory_items = [
        item
        for path in memory_root.glob("*_memory.json")
        for item in json.loads(path.read_text("utf-8"))["items"]
    ]
    assert all(set(item["source_task_ids"]) <= set(task_ids) for item in memory_items)
