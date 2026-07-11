from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = [
    pytest.mark.tau2_integration,
    pytest.mark.skipif(
        os.environ.get("RUN_TAU2_INTEGRATION") != "1",
        reason="set RUN_TAU2_INTEGRATION=1 to run against the real Tau2 checkout",
    ),
]


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COMMIT = "1901a301961cbbe3fd11f3e84a2a376530c759e3"
EXPECTED_SPLIT_HASH = "235237983dd826c6c16989e90797e9d58f8ed52059020c9079e60069288147eb"
EXPECTED_USER_SIMULATOR_CONFIG = {
    "solo_mode": False,
    "user_llm": "deepseek/deepseek-v4-pro",
    "user_llm_args": {
        "thinking": {"type": "disabled"},
        "temperature": 0.0,
        "max_tokens": 8192,
    },
}


def run_smoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scripts.check_tau2_retail", *args],
        cwd=WORKTREE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_real_tau2_retail_inspect_reports_the_pinned_train_task() -> None:
    result = run_smoke("--split", "train", "--task-id", "0", "--inspect")

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload == {
        "mode": "inspect",
        "status": "ok",
        "tau2_commit": EXPECTED_COMMIT,
        "tau2_package_version": "1.0.0",
        "split": "train",
        "split_hash": EXPECTED_SPLIT_HASH,
        "task_id": "0",
    }


def test_real_tau2_retail_reset_and_close_emit_machine_readable_summary() -> None:
    result = run_smoke("--split", "train", "--task-id", "0")

    payload = json.loads(result.stdout)
    if payload["status"] == "blocked":
        assert result.returncode == 2
        assert "Retail domain does not support solo mode" in payload["block_reason"]
        assert payload["tool_count"] is None
        assert payload["policy_sha256"] is None
        assert payload["initial_observation_length"] is None
        assert payload["user_simulator_config"] == EXPECTED_USER_SIMULATOR_CONFIG
        pytest.skip(payload["block_reason"])

    assert result.returncode == 0, result.stderr
    assert payload["mode"] == "reset_close"
    assert payload["status"] == "ok"
    assert payload["tau2_commit"] == EXPECTED_COMMIT
    assert payload["split_hash"] == EXPECTED_SPLIT_HASH
    assert payload["split"] == "train"
    assert payload["task_id"] == "0"
    assert payload["tool_count"] > 0
    assert len(payload["policy_sha256"]) == 64
    assert payload["initial_observation_length"] > 0
    assert payload["user_simulator_config"] == EXPECTED_USER_SIMULATOR_CONFIG
