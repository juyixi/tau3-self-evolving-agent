from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

import tau3_retail_evolver.envs.tau2_retail as tau2_retail
from tau3_retail_evolver.config import ProjectConfig
from tau3_retail_evolver.envs.factory import create_tau2_retail_env
from tau3_retail_evolver.envs.tau2_retail import Tau2RetailEnv


class FakeGymEnv:
    instances: list[FakeGymEnv] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.solo_mode = kwargs["solo_mode"]
        self.user_llm = kwargs["user_llm"]
        self.user_llm_args = kwargs["user_llm_args"] or {"temperature": 0.0}
        self.reset_result: tuple[str, Mapping[str, Any]] = (
            "Welcome to retail.",
            {
                "task": {"id": kwargs["task_id"]},
                "simulation_run": {"id": "simulation-1"},
                "tools": ["find_order"],
                "policy": {"name": "retail-policy"},
            },
        )
        self.step_results: list[tuple[str, float, bool, bool, Mapping[str, Any]]] = []
        self.close_calls = 0
        type(self).instances.append(self)

    def reset(self, *, seed: int) -> tuple[str, Mapping[str, Any]]:
        return self.reset_result

    def step(self, action: str) -> tuple[str, float, bool, bool, Mapping[str, Any]]:
        return self.step_results.pop(0)

    def close(self) -> None:
        self.close_calls += 1


class FalsyGymFactory:
    def __bool__(self) -> bool:
        return False

    def __call__(self, **kwargs: Any) -> FakeGymEnv:
        return FakeGymEnv(**kwargs)


@pytest.fixture(autouse=True)
def clear_fake_instances() -> None:
    FakeGymEnv.instances.clear()


@pytest.fixture
def config() -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "tau2": {
                "repo_path": "external/tau2-bench",
                "domain": "retail",
                "train_split": "train",
                "eval_split": "test",
                "user_llm": "Qwen/Qwen3.5-4B",
                "user_llm_args": {"temperature": 0.3},
                "solo_mode": True,
            },
            "model": {"base_model": "Qwen/Qwen3.5-9B"},
            "rollout": {"max_episode_steps": 2},
        }
    )


def test_constructs_gym_with_the_official_retail_arguments(config: ProjectConfig) -> None:
    Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)

    assert FakeGymEnv.instances[0].kwargs == {
        "domain": "retail",
        "task_id": "task-17",
        "max_steps": 2,
        "solo_mode": True,
        "user_llm": "Qwen/Qwen3.5-4B",
        "user_llm_args": {"temperature": 0.3},
        "all_messages_as_observation": True,
    }


def test_uses_an_injected_gym_factory_even_when_it_is_falsy(
    config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_lazy_import(**kwargs: Any) -> Any:
        raise AssertionError("lazy Tau2 loader must not be called")

    monkeypatch.setattr(tau2_retail, "_load_agent_gym_env", fail_lazy_import)

    environment = Tau2RetailEnv("task-17", config, gym_factory=FalsyGymFactory())

    assert isinstance(environment, Tau2RetailEnv)
    assert FakeGymEnv.instances[0].kwargs["task_id"] == "task-17"


def test_exposes_the_user_simulator_config_resolved_by_gym(config: ProjectConfig) -> None:
    config.tau2.user_llm_args = {}
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)

    assert environment.user_simulator_config == {
        "solo_mode": True,
        "user_llm": "Qwen/Qwen3.5-4B",
        "user_llm_args": {"temperature": 0.0},
    }


def test_user_simulator_config_is_a_defensive_copy(config: ProjectConfig) -> None:
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)

    resolved = environment.user_simulator_config
    resolved["user_llm_args"]["temperature"] = 0.9

    assert environment.user_simulator_config["user_llm_args"] == {"temperature": 0.3}


def test_reset_preserves_official_task_tools_and_policy_info(config: ProjectConfig) -> None:
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)
    gym = FakeGymEnv.instances[0]

    result = environment.reset(seed=123)

    assert result.observation == "Welcome to retail."
    assert result.info is gym.reset_result[1]
    assert result.info["task"] == {"id": "task-17"}
    assert result.info["tools"] == ["find_order"]
    assert result.info["policy"] == {"name": "retail-policy"}


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    ((False, False), (True, False), (False, True)),
)
def test_step_normalizes_the_official_five_value_result(
    config: ProjectConfig, terminated: bool, truncated: bool
) -> None:
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)
    gym = FakeGymEnv.instances[0]
    step_info = {"reward_info": {"score": 1.0}}
    gym.step_results.append(("Assistant response", 1.25, terminated, truncated, step_info))

    result = environment.step("search order")

    assert result.observation == "Assistant response"
    assert result.reward == 1.25
    assert result.terminated is terminated
    assert result.truncated is truncated
    assert result.done is (terminated or truncated)
    assert result.info is step_info


def test_step_preserves_parse_errors_and_official_reward_info(config: ProjectConfig) -> None:
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)
    gym = FakeGymEnv.instances[0]
    reward_info = {"total_reward": 0.4, "per_grader_rewards": {"policy": 0.4}}
    step_info = {"parse_error": "invalid action", "reward_info": reward_info}
    gym.step_results.append(("Parse error: invalid action", -0.2, False, False, step_info))

    result = environment.step("not a valid tool call")

    assert result.observation == "Parse error: invalid action"
    assert result.info is step_info
    assert result.info["reward_info"] is reward_info


def test_step_truncates_at_the_project_episode_limit(config: ProjectConfig) -> None:
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)
    gym = FakeGymEnv.instances[0]
    gym.step_results.extend(
        [
            ("first", 0.0, False, False, {}),
            ("second", 0.0, False, False, {}),
        ]
    )

    first = environment.step("first action")
    second = environment.step("second action")

    assert not first.done
    assert not first.truncated
    assert second.done
    assert not second.terminated
    assert second.truncated


@pytest.mark.parametrize("operation", ("reset", "step"))
def test_environment_exceptions_include_training_context_and_preserve_the_cause(
    config: ProjectConfig, operation: str
) -> None:
    environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)
    gym = FakeGymEnv.instances[0]
    original_error = ValueError("upstream failure")

    if operation == "reset":
        def fail_reset(*, seed: int) -> tuple[str, Mapping[str, Any]]:
            raise original_error

        gym.reset = fail_reset
        call = lambda: environment.reset(seed=123)
        expected_step = 0
    else:
        def fail_step(action: str) -> tuple[str, float, bool, bool, Mapping[str, Any]]:
            raise original_error

        gym.step = fail_step
        call = lambda: environment.step("search order")
        expected_step = 1

    with pytest.raises(RuntimeError) as error:
        call()

    assert error.value.__cause__ is original_error
    assert "train" in str(error.value)
    assert "task-17" in str(error.value)
    assert f"step {expected_step}" in str(error.value)
    assert "upstream failure" in str(error.value)


def test_close_is_idempotent_and_context_manager_closes_once(config: ProjectConfig) -> None:
    with Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv) as environment:
        environment.close()
        environment.close()

    assert FakeGymEnv.instances[0].close_calls == 1


def test_factory_creates_a_train_tau2_retail_environment(config: ProjectConfig) -> None:
    environment = create_tau2_retail_env("task-17", config, gym_factory=FakeGymEnv)

    assert isinstance(environment, Tau2RetailEnv)
