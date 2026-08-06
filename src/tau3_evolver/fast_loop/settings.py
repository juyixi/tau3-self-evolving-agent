from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FastLoopConfig:
    retrieve_top_k: int = 50
    max_episode_steps: int = 40
    memory_enabled: bool = True
    max_new_tips_per_episode: int = 2
    max_new_skills_per_episode: int = 1
    max_new_tools_per_episode: int = 1
    max_new_trajectories_per_episode: int = 1
    maintenance_tip_capacity: int = 240
    maintenance_similarity_threshold: float = 0.92
    maintenance_priority_pair_limit: int = 24
    retrieval_mmr_lambda_tip: float = 0.65
    retrieval_mmr_lambda_skill: float = 0.80
    retrieval_mmr_lambda_tool: float = 0.85
    retrieval_mmr_lambda_trajectory: float = 0.75
    retrieval_global_mmr_lambda: float = 0.75
    retrieval_quota_tip: int = 18
    retrieval_quota_skill: int = 18
    retrieval_quota_tool: int = 6
    retrieval_quota_trajectory: int = 4
    selection_max_total: int = 20
    selection_max_tip: int = 7
    selection_max_skill: int = 8
    selection_max_tool: int = 3
    selection_max_trajectory: int = 2

    def __post_init__(self) -> None:
        if type(self.memory_enabled) is not bool:
            raise ValueError("memory_enabled must be a bool")
        if self.retrieve_top_k < 1 or self.max_episode_steps < 1:
            raise ValueError("fast-loop limits must be positive")
        if any(
            type(limit) is not int or limit < 0
            for limit in (
                self.max_new_tips_per_episode,
                self.max_new_skills_per_episode,
                self.max_new_tools_per_episode,
                self.max_new_trajectories_per_episode,
            )
        ):
            raise ValueError("memory write quotas must be non-negative integers")
        if self.maintenance_tip_capacity < 1:
            raise ValueError("maintenance tip capacity must be positive")
        if not -1.0 <= self.maintenance_similarity_threshold <= 1.0:
            raise ValueError(
                "maintenance similarity threshold must be between -1 and 1"
            )
        if self.maintenance_priority_pair_limit < 0:
            raise ValueError("maintenance priority pair limit must be non-negative")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                *self.retrieval_mmr_lambdas().values(),
                self.retrieval_global_mmr_lambda,
            )
        ):
            raise ValueError("retrieval MMR lambdas must be between 0 and 1")
        if any(value < 0 for value in self.retrieval_tier_quotas().values()):
            raise ValueError("retrieval tier quotas must be non-negative")
        if self.selection_max_total < 1 or any(
            value < 0 for value in self.selection_tier_limits().values()
        ):
            raise ValueError(
                "selection limits must be non-negative with a positive total"
            )

    def write_quota_for(self, tier: str) -> int:
        return {
            "tip": self.max_new_tips_per_episode,
            "skill": self.max_new_skills_per_episode,
            "tool": self.max_new_tools_per_episode,
            "trajectory": self.max_new_trajectories_per_episode,
        }[tier]

    def retrieval_tier_quotas(self) -> dict[str, int]:
        return {
            "tip": self.retrieval_quota_tip,
            "skill": self.retrieval_quota_skill,
            "tool": self.retrieval_quota_tool,
            "trajectory": self.retrieval_quota_trajectory,
        }

    def retrieval_mmr_lambdas(self) -> dict[str, float]:
        return {
            "tip": self.retrieval_mmr_lambda_tip,
            "skill": self.retrieval_mmr_lambda_skill,
            "tool": self.retrieval_mmr_lambda_tool,
            "trajectory": self.retrieval_mmr_lambda_trajectory,
        }

    def selection_tier_limits(self) -> dict[str, int]:
        return {
            "tip": self.selection_max_tip,
            "skill": self.selection_max_skill,
            "tool": self.selection_max_tool,
            "trajectory": self.selection_max_trajectory,
        }


__all__ = ["FastLoopConfig"]
