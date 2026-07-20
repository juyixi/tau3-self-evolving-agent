from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from tau3_retail_evolver.config import ProjectConfig, SlowLoopConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_requires_python_312_through_313() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["requires-python"] == ">=3.12,<3.14"


def test_training_dependencies_are_optional() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    training_dependencies = project["project"]["optional-dependencies"]["training"]

    assert {dependency.split(">=", 1)[0] for dependency in training_dependencies} == {
        "accelerate",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    }
    assert "transformers>=5.2" in training_dependencies
    assert "peft>=0.19" in training_dependencies


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
    assert config.memory.agent_id == "retail"
    assert config.memory.enabled is True
    assert config.memory.model_dump() == {
        "enabled": True,
        "agent_id": "retail",
        "tiers": ("trajectory", "tip", "skill", "tool"),
        "retrieve_top_k": 50,
        "teacher_memory_cap": 20,
        "score_threshold": 0.01,
        "maintenance_period": 30,
        "embedding_provider": "local",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_device": "cuda",
        "embedding_dtype": "float16",
        "embedding_max_length": 2048,
        "embedding_batch_size": 16,
        "embedding_cache": True,
    }
    assert config.slow_loop.tier_priors == {
        "trajectory": 0.9,
        "tip": 0.8,
        "skill": 1.0,
        "tool": 1.2,
    }
    assert config.slow_loop.redundancy_threshold == 0.90
    assert config.slow_loop.max_redundancy_pairs == 50
    assert config.training.dtype == "bfloat16"
    assert config.training.max_sequence_length == 8192
    assert config.training.loss_type == "forward_kl"
    assert config.training.target_modules == "all-linear"
    assert config.evaluation.nl_assertions.model == "openrouter/openai/gpt-4.1"
    assert config.evaluation.nl_assertions.model_args == {"temperature": 0.0}
    assert config.evaluation.nl_assertions.api_key_env == "OPENROUTER_API_KEY"


@pytest.mark.parametrize(
    "tier_priors",
    [
        {"trajectory": 0.9, "tip": 0.8, "skill": 1.0},
        {
            "trajectory": 0.9,
            "tip": 0.8,
            "skill": 1.0,
            "tool": 1.2,
            "other": 1.0,
        },
    ],
)
def test_slow_loop_tier_priors_require_exact_keys(
    tier_priors: dict[str, float],
) -> None:
    with pytest.raises(ValidationError, match="tier_priors"):
        SlowLoopConfig(tier_priors=tier_priors)


def test_memory_config_accepts_yaml_false(tmp_path: Path) -> None:
    config = load_config(_write_config_with_memory_enabled(tmp_path, "false"))

    assert config.memory.enabled is False


@pytest.mark.parametrize("enabled_yaml", ('"false"', "0"))
def test_memory_config_rejects_non_strict_enabled_values_from_yaml(
    tmp_path: Path, enabled_yaml: str
) -> None:
    with pytest.raises(ValidationError):
        load_config(_write_config_with_memory_enabled(tmp_path, enabled_yaml))


@pytest.mark.parametrize(
    "agent_id",
    ("", ".", "..", "RETAIL", "retail/other", r"retail\\other", "零售"),
)
def test_memory_config_rejects_unsafe_agent_id(agent_id: str) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        ProjectConfig.model_validate(
            {
                "tau2": {
                    "repo_path": "external/tau2-bench",
                    "domain": "retail",
                    "train_split": "train",
                    "eval_split": "test",
                    "user_llm": "test-user",
                },
                "model": {"base_model": "Qwen/Qwen3.5-9B"},
                "memory": {"agent_id": agent_id},
            }
        )


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


def test_validation_error_does_not_echo_rejected_model_args_credentials() -> None:
    sentinel = "secret-sentinel"

    with pytest.raises(ValueError) as error:
        ProjectConfig.model_validate(
            {
                "tau2": {
                    "repo_path": "external/tau2-bench",
                    "domain": "retail",
                    "train_split": "train",
                    "eval_split": "test",
                    "user_llm": "Qwen/Qwen3.5-4B",
                },
                "model": {"base_model": "Qwen/Qwen3.5-9B"},
                "evaluation": {
                    "nl_assertions": {
                        "model_args": {"access_token": sentinel},
                    }
                },
            }
        )

    assert "model_args" in str(error.value)
    assert sentinel not in str(error.value)


@pytest.mark.parametrize(
    "credential_key",
    (
        "authorization",
        "openrouter_api_key",
        "credentials",
        "passwords",
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


def _write_config_with_memory_enabled(tmp_path: Path, enabled_yaml: str) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (PROJECT_ROOT / "configs" / "default.yaml")
        .read_text(encoding="utf-8")
        .replace("  enabled: true", f"  enabled: {enabled_yaml}", 1),
        encoding="utf-8",
    )
    return config_path
