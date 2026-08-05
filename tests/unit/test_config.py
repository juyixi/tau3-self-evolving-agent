from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from tau3_evolver.config import ProjectConfig, SlowLoopConfig, load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_metadata_and_training_extra() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)

    assert project["project"]["name"] == "tau3-evolver"
    assert project["project"]["requires-python"] == ">=3.12,<3.14"
    assert "tau3" in project["project"]["scripts"]
    assert "transformers>=5.2" in project["project"]["optional-dependencies"]["training"]


def test_default_config_contains_runtime_not_task_routing() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    assert config.tau2.repo_path == Path("external/tau2-bench")
    assert not hasattr(config.tau2, "domain")
    assert not hasattr(config.tau2, "train_split")
    assert not hasattr(config.memory, "enabled")
    assert not hasattr(config.memory, "agent_id")
    assert config.execution.max_concurrency == 3
    assert config.execution.seed == 42
    assert not hasattr(config, "pipeline")


def test_cli_overrides_are_typed() -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        overrides=("rollout.max_episode_steps=24", "execution.max_concurrency=5"),
    )

    assert config.rollout.max_episode_steps == 24
    assert config.execution.max_concurrency == 5


def test_unknown_routing_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="domain"):
        ProjectConfig.model_validate(
            {
                "tau2": {
                    "repo_path": "external/tau2-bench",
                    "user_llm": "test-user",
                    "domain": "retail",
                },
                "model": {"base_model": "Qwen/Qwen3.5-9B"},
            }
        )


@pytest.mark.parametrize(
    "tier_priors",
    (
        {"trajectory": 0.9, "tip": 0.8, "skill": 1.0},
        {
            "trajectory": 0.9,
            "tip": 0.8,
            "skill": 1.0,
            "tool": 1.2,
            "other": 1.0,
        },
    ),
)
def test_slow_loop_tier_priors_require_exact_keys(
    tier_priors: dict[str, float],
) -> None:
    with pytest.raises(ValidationError, match="tier_priors"):
        SlowLoopConfig(tier_priors=tier_priors)
