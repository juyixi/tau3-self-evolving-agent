from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tau3_retail_evolver.envs.runtime as runtime_module
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
            "provider_settings": {
                "api_key": "api-secret",
                "apiKey": "camel-api-secret",
                "api_token": "api-token-secret",
                "access_token": "access-token-secret",
                "auth_token": "auth-token-secret",
                "Authorization": "Bearer token-secret",
                "client_secret": "client-secret-value",
                "password": "password-secret",
                "credentials": "credentials-secret",
                "private_key": "private-key-secret",
                "access_key": "access-key-secret",
            },
            "nested": [
                {
                    "max_tokens": 512,
                    "tokenizer": "tau-tokenizer",
                    "token_count": 12,
                    "max_completion_tokens": 256,
                    "temperature": 0.0,
                },
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
            "provider_settings": {
                "api_key": "[REDACTED]",
                "apiKey": "[REDACTED]",
                "api_token": "[REDACTED]",
                "access_token": "[REDACTED]",
                "auth_token": "[REDACTED]",
                "Authorization": "[REDACTED]",
                "client_secret": "[REDACTED]",
                "password": "[REDACTED]",
                "credentials": "[REDACTED]",
                "private_key": "[REDACTED]",
                "access_key": "[REDACTED]",
            },
            "nested": [
                {
                    "max_tokens": 512,
                    "tokenizer": "tau-tokenizer",
                    "token_count": 12,
                    "max_completion_tokens": 256,
                    "temperature": 0.0,
                },
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


@pytest.mark.parametrize(
    ("contract_failure", "message"),
    (("pin mismatch", "pin mismatch"), ("split mismatch", "split count mismatch")),
)
def test_runtime_contract_failure_happens_before_gym_construction(
    contract_failure: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "external" / "tau2-bench"
    retail_root = checkout / "data" / "tau2" / "domains" / "retail"
    retail_root.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        "[project]\nname = 'tau2-bench'\nversion = '1.0.0'\n", encoding="utf-8"
    )
    (retail_root / "tasks.json").write_text(
        json.dumps([{"id": "0"}]), encoding="utf-8"
    )
    (retail_root / "split_tasks.json").write_text(
        json.dumps({"train": ["0"], "test": [], "base": ["0"]}),
        encoding="utf-8",
    )
    checkout.with_suffix(".commit").write_text(
        ("b" if contract_failure == "pin mismatch" else "a") * 40,
        encoding="utf-8",
    )
    config = SimpleNamespace(
        tau2=SimpleNamespace(repo_path=checkout),
        training=SimpleNamespace(seed=42),
    )

    def construct_environment(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Gym must not be constructed after contract failure")

    def import_tau2(name: str) -> None:
        raise AssertionError(f"Tau2 import must not run after contract failure: {name}")

    monkeypatch.setattr(check_tau2_retail, "load_config", lambda path: config)
    monkeypatch.setattr(runtime_module, "_git_commit", lambda path: "a" * 40)
    monkeypatch.setattr(runtime_module.importlib, "import_module", import_tau2)
    monkeypatch.setattr(check_tau2_retail, "Tau2RetailEnv", construct_environment)

    with pytest.raises(RuntimeError, match=message):
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
            inspect_metadata=lambda repo_path: runtime,
            require_pinned_commit=lambda fingerprint: None,
            load_verified_gym_factory=lambda repo_path: object(),
            probe_gym=lambda repo_path: None,
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
