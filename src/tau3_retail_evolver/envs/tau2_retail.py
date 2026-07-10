from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from tau3_retail_evolver.config import ProjectConfig
from tau3_retail_evolver.envs.base import ResetResult, StepResult


GymFactory = Callable[..., Any]


class Tau2RetailEnv:
    """Adapt Tau2's Gymnasium environment to the project rollout contract."""

    def __init__(
        self,
        task_id: str,
        config: ProjectConfig,
        gym_factory: GymFactory | None = None,
    ) -> None:
        self._task_id = task_id
        self._split = config.tau2.train_split
        self._max_episode_steps = config.rollout.max_episode_steps
        self._episode_step = 0
        self._closed = False
        factory = gym_factory or _load_agent_gym_env
        try:
            self._gym = factory(
                domain=config.tau2.domain,
                task_id=task_id,
                max_steps=self._max_episode_steps,
                solo_mode=config.tau2.solo_mode,
                user_llm=config.tau2.user_llm,
                user_llm_args=config.tau2.user_llm_args,
                all_messages_as_observation=True,
            )
        except Exception as error:
            raise self._contextual_error("construct", self._episode_step, error) from error

    def reset(self, seed: int) -> ResetResult:
        try:
            observation, info = self._gym.reset(seed=seed)
        except Exception as error:
            raise self._contextual_error("reset", self._episode_step, error) from error
        self._episode_step = 0
        return ResetResult(observation=observation, info=info)

    def step(self, action: str) -> StepResult:
        next_step = self._episode_step + 1
        try:
            observation, reward, terminated, truncated, info = self._gym.step(action)
        except Exception as error:
            raise self._contextual_error("step", next_step, error) from error

        self._episode_step = next_step
        truncated = truncated or next_step >= self._max_episode_steps
        return StepResult(
            observation=observation,
            reward=reward,
            done=terminated or truncated,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._gym.close()
        except Exception as error:
            raise self._contextual_error("close", self._episode_step, error) from error

    def __enter__(self) -> Tau2RetailEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _contextual_error(self, operation: str, episode_step: int, error: Exception) -> RuntimeError:
        return RuntimeError(
            f"Tau2 retail {operation} failed for split {self._split}, task {self._task_id}, "
            f"episode step {episode_step}: {error}"
        )


def _load_agent_gym_env(**kwargs: Any) -> Any:
    from tau2.gym.gym_agent import AgentGymEnv

    return AgentGymEnv(**kwargs)
