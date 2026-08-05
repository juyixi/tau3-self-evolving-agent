from __future__ import annotations

from dataclasses import dataclass

from tau3_evolver.execution.results import BatchResult


@dataclass(frozen=True, slots=True)
class RewardMetrics:
    task_count: int
    completed_count: int
    failure_count: int
    mean_reward: float
    pass_rate: float


def compute_reward_metrics(result: BatchResult) -> RewardMetrics:
    rewards = [episode.final_reward for episode in result.episodes]
    task_count = len(result.episodes) + len(result.failures)
    return RewardMetrics(
        task_count=task_count,
        completed_count=len(result.episodes),
        failure_count=len(result.failures),
        mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        pass_rate=(
            sum(reward > 0 for reward in rewards) / task_count if task_count else 0.0
        ),
    )
