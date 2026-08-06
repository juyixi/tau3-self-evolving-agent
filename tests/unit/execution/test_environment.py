from pathlib import Path

import pytest

from tau3_evolver.benchmarks import benchmark_registry
from tau3_evolver.config import load_config
from tau3_evolver.execution.environment import (
    OnlineCredentialError,
    load_project_environment,
    preflight_online_credentials,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_loads_project_env_without_overriding_process_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# deployment secrets\n"
        "export DEEPSEEK_API_KEY='file-secret'\n"
        "EXISTING=from-file\n",
        encoding="utf-8",
    )
    environ = {"EXISTING": "from-process"}

    result = load_project_environment(root=tmp_path, environ=environ)

    assert result.path == env_file.resolve()
    assert result.loaded_names == ("DEEPSEEK_API_KEY",)
    assert environ == {
        "DEEPSEEK_API_KEY": "file-secret",
        "EXISTING": "from-process",
    }


def test_missing_project_env_is_a_noop(tmp_path: Path) -> None:
    environ: dict[str, str] = {}

    result = load_project_environment(root=tmp_path, environ=environ)

    assert result.path == (tmp_path / ".env").resolve()
    assert result.loaded_names == ()
    assert environ == {}


def test_rejects_ambiguous_env_without_echoing_its_value(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=literal-test-secret\n"
        "DEEPSEEK_API_KEY=second-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        load_project_environment(root=tmp_path, environ={})

    assert "DEEPSEEK_API_KEY" in str(error.value)
    assert "literal-test-secret" not in str(error.value)
    assert "second-secret" not in str(error.value)


def test_preflight_covers_user_and_nl_assertion_consumers() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    requirements = benchmark_registry.resolve("retail").credential_requirements(
        config
    )

    with pytest.raises(OnlineCredentialError) as error:
        preflight_online_credentials(
            requirements,
            environ={},
            env_path=PROJECT_ROOT / ".env",
        )

    message = str(error.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "Tau2 user simulator" in message
    assert "Tau2 NL assertion evaluator" in message

    preflight_online_credentials(
        requirements,
        environ={"DEEPSEEK_API_KEY": "literal-test-secret"},
    )
