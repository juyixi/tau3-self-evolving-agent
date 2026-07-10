from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from tau3_retail_evolver.config import load_config


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


def test_load_config_applies_typed_overrides() -> None:
    config = load_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        overrides=("rollout.max_episode_steps=24", "training.seed=7"),
    )

    assert config.rollout.max_episode_steps == 24
    assert config.training.seed == 7


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
