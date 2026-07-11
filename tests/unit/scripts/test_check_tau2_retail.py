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

    def __init__(self, task_id: str, config: Any, gym_factory: Any = None) -> None:
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


@pytest.mark.parametrize(
    ("summary", "expected_status"),
    (
        (
            lambda config: check_tau2_retail._reset_summary(
                "Welcome",
                {"tools": ["find_order"], "policy": "retail policy"},
                config,
            ),
            "ok",
        ),
        (
            lambda config: check_tau2_retail._blocked_reset_summary(
                "simulator unavailable", config
            ),
            "blocked",
        ),
    ),
)
def test_summaries_redact_nested_user_simulator_secrets(
    summary: Any, expected_status: str
) -> None:
    user_simulator_config = {
        "solo_mode": False,
        "user_llm": "resolved-user-model",
        "user_llm_args": {
            "api_key": "api-secret",
            "nested": [
                {"Authorization": "Bearer token-secret"},
                {"client_secret": "client-secret-value"},
                {"temperature": 0.0},
            ],
        },
        "credentials": "top-level-secret",
        "unexpected": "must-not-be-output",
    }

    payload = summary(user_simulator_config)

    assert payload.get("status", "ok") == expected_status
    assert payload["user_simulator_config"] == {
        "solo_mode": False,
        "user_llm": "resolved-user-model",
        "user_llm_args": {
            "api_key": "[REDACTED]",
            "nested": [
                {"Authorization": "[REDACTED]"},
                {"client_secret": "[REDACTED]"},
                {"temperature": 0.0},
            ],
        },
    }


@pytest.mark.parametrize("tools", ("find_order", b"find_order", bytearray(b"find_order")))
def test_reset_summary_rejects_string_like_tools(tools: Any) -> None:
    with pytest.raises(RuntimeError, match="sequence of tools"):
        check_tau2_retail._reset_summary(
            "Welcome",
            {"tools": tools, "policy": "retail policy"},
            EmptyObservationEnvironment.user_simulator_config,
        )


@pytest.mark.parametrize("contract_failure", ("pin mismatch", "split mismatch"))
def test_runtime_contract_failure_happens_before_gym_construction(
    contract_failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        tau2=SimpleNamespace(repo_path=Path("external/tau2-bench")),
        training=SimpleNamespace(seed=42),
    )
    runtime = SimpleNamespace(
        repo_path=Path("external/tau2-bench"),
        git_commit="a" * 40,
        package_version="1.0.0",
        retail_tasks_path=Path("tasks.json"),
        retail_split_path=Path("split_tasks.json"),
    )

    class RuntimeAPI:
        inspect = staticmethod(lambda repo_path: runtime)

        @staticmethod
        def require_pinned_commit(fingerprint: Any) -> None:
            if contract_failure == "pin mismatch":
                raise RuntimeError(contract_failure)

        @staticmethod
        def load_verified_gym_factory(repo_path: Path) -> Any:
            raise AssertionError("Gym factory must not load after contract failure")

    class Catalog:
        split_sha256 = "split-hash"

        def require_official_compatibility(self) -> None:
            if contract_failure == "split mismatch":
                raise RuntimeError(contract_failure)

        def task_ids(self, split: str) -> tuple[str, ...]:
            return ("0",)

    def construct_environment(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Gym must not be constructed after contract failure")

    monkeypatch.setattr(check_tau2_retail, "load_config", lambda path: config)
    monkeypatch.setattr(check_tau2_retail, "Tau2Runtime", RuntimeAPI)
    monkeypatch.setattr(
        check_tau2_retail,
        "RetailTaskCatalog",
        SimpleNamespace(from_files=lambda tasks_path, split_path: Catalog()),
    )
    monkeypatch.setattr(check_tau2_retail, "Tau2RetailEnv", construct_environment)

    with pytest.raises(RuntimeError, match=contract_failure):
        check_tau2_retail.main(["--split", "train", "--task-id", "0"])


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
        repo_path=Path("external/tau2-bench"),
        git_commit="pinned-commit",
        package_version="1.0.0",
        retail_tasks_path=Path("tasks.json"),
        retail_split_path=Path("split_tasks.json"),
    )
    catalog = SimpleNamespace(
        split_sha256="split-hash",
        task_ids=lambda split: ("0",),
        require_official_compatibility=lambda: None,
    )
    monkeypatch.setattr(check_tau2_retail, "load_config", lambda path: config)
    monkeypatch.setattr(
        check_tau2_retail,
        "Tau2Runtime",
        SimpleNamespace(
            inspect=lambda repo_path: runtime,
            require_pinned_commit=lambda fingerprint: None,
            load_verified_gym_factory=lambda repo_path: object(),
        ),
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
