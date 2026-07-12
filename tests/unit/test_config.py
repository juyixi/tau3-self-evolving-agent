from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from tau3_retail_evolver.config import ProjectConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_requires_python_312_through_313() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["requires-python"] == ">=3.12,<3.14"


def test_default_config_has_the_required_retail_environment() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    assert config.tau2.repo_path == Path("external/tau2-bench")
    assert config.tau2.domain == "retail"
    assert config.tau2.train_split == "train"
    assert config.tau2.eval_split == "test"
    assert not hasattr(config.tau2, "dev")
    assert config.model.base_model == "Qwen/Qwen3.5-9B"
    assert config.lora.use_peft is True
    assert config.lora.lora_r == 32
    assert config.lora.lora_alpha == 64
    assert config.lora.lora_dropout == 0.05
    assert config.rollout.temperature == 1.0
    assert config.rollout.top_p == 0.95
    assert config.rollout.max_episode_steps == 40
    assert config.memory.model_dump() == {
        "tiers": ("trajectory", "tip", "skill", "tool"),
        "retrieve_top_k": 50,
        "teacher_memory_cap": 20,
        "score_threshold": 0.01,
        "maintenance_period": 30,
    }
    assert config.evaluation.nl_assertions.model == "openrouter/openai/gpt-4.1"
    assert config.evaluation.nl_assertions.model_args == {"temperature": 0.0}
    assert config.evaluation.nl_assertions.api_key_env == "OPENROUTER_API_KEY"


def test_project_config_defaults_the_nl_assertion_evaluator() -> None:
    config = ProjectConfig.model_validate(
        {
            "tau2": {
                "repo_path": "external/tau2-bench",
                "domain": "retail",
                "train_split": "train",
                "eval_split": "test",
                "user_llm": "Qwen/Qwen3.5-4B",
            },
            "model": {"base_model": "Qwen/Qwen3.5-9B"},
        }
    )

    assert config.evaluation.nl_assertions.model == "openrouter/openai/gpt-4.1"
    assert config.evaluation.nl_assertions.model_args == {"temperature": 0.0}
    assert config.evaluation.nl_assertions.api_key_env == "OPENROUTER_API_KEY"


def test_load_config_applies_typed_overrides() -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        overrides=(
            "rollout.max_episode_steps=24",
            "training.seed=7",
            "evaluation.nl_assertions.model=openrouter/anthropic/claude-sonnet-4",
            "evaluation.nl_assertions.model_args.temperature=0.25",
            "evaluation.nl_assertions.api_key_env=TEST_OPENROUTER_API_KEY",
        ),
    )

    assert config.rollout.max_episode_steps == 24
    assert config.training.seed == 7
    assert config.evaluation.nl_assertions.model == "openrouter/anthropic/claude-sonnet-4"
    assert config.evaluation.nl_assertions.model_args == {"temperature": 0.25}
    assert config.evaluation.nl_assertions.api_key_env == "TEST_OPENROUTER_API_KEY"


def test_user_simulator_may_differ_from_agent_model() -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        overrides=("tau2.user_llm=Qwen/Qwen3.5-4B",),
    )

    assert config.tau2.user_llm == "Qwen/Qwen3.5-4B"
    assert config.model.base_model == "Qwen/Qwen3.5-9B"


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ("lora.use_peft=false", "use_peft"),
        ("tau2.train_split=base", "train_split"),
    ),
)
def test_load_config_rejects_disallowed_training_settings(
    override: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(PROJECT_ROOT / "configs" / "default.yaml", overrides=(override,))


def test_load_config_rejects_blank_nl_assertion_model(tmp_path: Path) -> None:
    config_path = _write_temporary_config(
        tmp_path,
        """
evaluation:
  nl_assertions:
    model: "   "
    model_args: {}
    api_key_env: OPENROUTER_API_KEY
""",
    )

    with pytest.raises(ValueError, match="model"):
        load_config(config_path)


def test_load_config_rejects_invalid_nl_assertion_api_key_env(tmp_path: Path) -> None:
    config_path = _write_temporary_config(
        tmp_path,
        """
evaluation:
  nl_assertions:
    model: openrouter/openai/gpt-4.1
    model_args: {}
    api_key_env: 1INVALID_ENV
""",
    )

    with pytest.raises(ValueError, match="api_key_env"):
        load_config(config_path)


def test_load_config_rejects_nested_credential_model_args(tmp_path: Path) -> None:
    config_path = _write_temporary_config(
        tmp_path,
        """
evaluation:
  nl_assertions:
    model: openrouter/openai/gpt-4.1
    model_args:
      retry:
        api-key: should-not-be-configured
    api_key_env: OPENROUTER_API_KEY
""",
    )

    with pytest.raises(ValueError, match="model_args"):
        load_config(config_path)


@pytest.mark.parametrize(
    "credential_key",
    (
        "access_token",
        "client_secret",
        "auth_token",
        "api_token",
        "private_key",
        "access_key",
    ),
)
def test_load_config_rejects_nested_compound_credential_model_args(
    tmp_path: Path, credential_key: str
) -> None:
    config_path = _write_temporary_config(
        tmp_path,
        f"""
evaluation:
  nl_assertions:
    model: openrouter/openai/gpt-4.1
    model_args:
      retry:
        {credential_key}: should-not-be-configured
    api_key_env: OPENROUTER_API_KEY
""",
    )

    with pytest.raises(ValueError, match="model_args"):
        load_config(config_path)


def _write_temporary_config(tmp_path: Path, evaluation_yaml: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
        + evaluation_yaml,
        encoding="utf-8",
    )
    return config_path
