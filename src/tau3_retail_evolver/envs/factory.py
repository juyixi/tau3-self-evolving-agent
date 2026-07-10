from __future__ import annotations

from tau3_retail_evolver.config import ProjectConfig
from tau3_retail_evolver.envs.tau2_retail import GymFactory, Tau2RetailEnv


def create_tau2_retail_env(
    task_id: str,
    config: ProjectConfig,
    gym_factory: GymFactory | None = None,
) -> Tau2RetailEnv:
    return Tau2RetailEnv(task_id, config, gym_factory=gym_factory)
