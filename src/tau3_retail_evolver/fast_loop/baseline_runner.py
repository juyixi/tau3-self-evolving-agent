from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.events import RunContext
from tau3_retail_evolver.models.policy import DecisionRequest, Policy
from tau3_retail_evolver.runs.manifest import sanitize_artifact_data


class BaselineEnvironment(Protocol):
    def reset(self, *, seed: int) -> ResetResult: ...

    def step(self, action: str) -> StepResult: ...

    def close(self) -> None: ...


EnvironmentFactory = Callable[[str], BaselineEnvironment]


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    task_id: str
    final_reward: float
    steps: int
    reward_info: Mapping[str, Any]
    evaluator_details: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    episodes: tuple[EpisodeSummary, ...]

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def total_reward(self) -> float:
        return sum(episode.final_reward for episode in self.episodes)


def run_baseline(
    tasks: Sequence[str],
    env_factory: EnvironmentFactory,
    policy: Policy,
    run_context: RunContext,
) -> RolloutSummary:
    """Run deterministic no-memory episodes and emit their canonical event stream."""
    if run_context.adapter_revision is not None or run_context.memory_snapshot_id is not None:
        raise ValueError("baseline runs must not use an adapter or memory snapshot")

    episodes = [
        _run_episode(task_id, env_factory, policy, run_context)
        for task_id in tasks
    ]
    return RolloutSummary(episodes=tuple(episodes))


def _run_episode(
    task_id: str,
    env_factory: EnvironmentFactory,
    policy: Policy,
    run_context: RunContext,
) -> EpisodeSummary:
    environment = env_factory(task_id)
    try:
        reset = environment.reset(seed=run_context.seed)
        _emit(
            run_context,
            run_context.event(
                "EpisodeStarted",
                task_id,
                observation=reset.observation,
                reset_info=dict(reset.info),
            )
        )

        turn = 0
        while True:
            # Tau2 provides the entire transcript in each observation, so history stays empty.
            decision = policy.generate(
                DecisionRequest(
                    observation=reset.observation,
                    reset_info=reset.info,
                    temperature=run_context.temperature,
                    top_p=run_context.top_p,
                    history=(),
                )
            )
            _emit(
                run_context,
                run_context.event(
                    "DecisionMade",
                    task_id,
                    turn=turn,
                    observation=reset.observation,
                    raw_output=decision.raw_output,
                    parsed_action=decision.parsed_action,
                    sampling_params=dict(decision.sampling_params),
                    latency_s=decision.latency_s,
                )
            )
            step = environment.step(decision.parsed_action)
            _emit(
                run_context,
                run_context.event(
                    "EnvironmentStepped",
                    task_id,
                    turn=turn,
                    action=decision.parsed_action,
                    observation=step.observation,
                    reward=step.reward,
                    done=step.done,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    info=dict(step.info),
                )
            )
            if step.done:
                reward_info = step.info.get("reward_info", {})
                evaluator_details = step.info.get(
                    "evaluator_details", step.info.get("evaluator", {})
                )
                _emit(
                    run_context,
                    run_context.event(
                        "EpisodeFinished",
                        task_id,
                        steps=turn + 1,
                        final_reward=step.reward,
                        reward_info=reward_info,
                        evaluator_details=evaluator_details,
                        final_info=dict(step.info),
                    )
                )
                return EpisodeSummary(
                    task_id=task_id,
                    final_reward=step.reward,
                    steps=turn + 1,
                    reward_info=reward_info,
                    evaluator_details=evaluator_details,
                )
            reset = ResetResult(observation=step.observation, info=reset.info)
            turn += 1
    finally:
        environment.close()


def _emit(run_context: RunContext, event: dict[str, Any]) -> None:
    run_context.event_writer.append(sanitize_artifact_data(event))
