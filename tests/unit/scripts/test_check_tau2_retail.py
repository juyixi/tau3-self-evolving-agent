from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import check_tau2_retail


class EmptyObservationEnvironment:
    user_simulator_config = {
        "solo_mode": False,
        "user_llm": "resolved-user-model",
        "user_llm_args": {"temperature": 0.0},
    }

    def __init__(self, task_id: str, config: Any) -> None:
        self.task_id = task_id

    def __enter__(self) -> EmptyObservationEnvironment:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def reset(self, seed: int) -> SimpleNamespace:
        return SimpleNamespace(
            observation="",
            info={
                "tools": ["find_order"],
                "policy": "retail policy",
                "simulation_run": {"api_key": "secret-value"},
            },
        )


def test_empty_initial_observation_is_blocked_without_leaking_simulation_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = SimpleNamespace(
        tau2=SimpleNamespace(
            repo_path=Path("external/tau2-bench"),
            solo_mode=False,
            user_llm="configured-user-model",
            user_llm_args={},
        ),
        training=SimpleNamespace(seed=42),
    )
    runtime = SimpleNamespace(
        git_commit="pinned-commit",
        package_version="1.0.0",
        retail_tasks_path=Path("tasks.json"),
        retail_split_path=Path("split_tasks.json"),
    )
    catalog = SimpleNamespace(
        split_sha256="split-hash",
        task_ids=lambda split: ("0",),
    )
    monkeypatch.setattr(check_tau2_retail, "load_config", lambda path: config)
    monkeypatch.setattr(
        check_tau2_retail,
        "Tau2Runtime",
        SimpleNamespace(inspect=lambda repo_path: runtime),
    )
    monkeypatch.setattr(
        check_tau2_retail,
        "RetailTaskCatalog",
        SimpleNamespace(from_files=lambda tasks_path, split_path: catalog),
    )
    monkeypatch.setattr(check_tau2_retail, "Tau2RetailEnv", EmptyObservationEnvironment)

    returncode = check_tau2_retail.main(["--split", "train", "--task-id", "0"])

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert returncode == 2
    assert payload["status"] == "blocked"
    assert "empty initial observation" in payload["block_reason"]
    assert payload["tool_count"] is None
    assert payload["policy_sha256"] is None
    assert payload["initial_observation_length"] is None
    assert payload["user_simulator_config"] == {
        "solo_mode": False,
        "user_llm": "resolved-user-model",
        "user_llm_args": {"temperature": 0.0},
    }
    assert "secret-value" not in stdout
