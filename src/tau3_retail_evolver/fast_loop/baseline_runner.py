from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Any, Protocol

from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.baseline_prompt import build_baseline_prompt
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
    terminal_evaluation: Mapping[str, Any]
    simulation_result: Mapping[str, Any]


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
    episode_summary: EpisodeSummary | None = None
    try:
        reset = environment.reset(seed=run_context.seed)
        decision_reset_info = _decision_reset_info(reset.info, task_id)
        prompt = build_baseline_prompt(reset.observation, decision_reset_info)
        _emit(
            run_context,
            run_context.event(
                "EpisodeStarted",
                task_id,
                observation=reset.observation,
                policy=prompt.messages[0]["content"],
                tool_schemas=list(prompt.tools),
                tool_count=len(prompt.tools),
            )
        )

        turn = 0
        while True:
            # Tau2 provides the entire transcript in each observation, so history stays empty.
            decision = policy.generate(
                DecisionRequest(
                    observation=reset.observation,
                    reset_info=decision_reset_info,
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
                    public_info=_public_step_info(step.info),
                )
            )
            if step.done:
                terminal_evaluation = _terminal_json_mapping(step.info, "reward_info", task_id)
                simulation_result = _terminal_json_mapping(step.info, "simulation_run", task_id)
                _emit(
                    run_context,
                    run_context.event(
                        "EpisodeFinished",
                        task_id,
                        steps=turn + 1,
                        final_reward=step.reward,
                        terminal_evaluation=terminal_evaluation,
                        simulation_result=simulation_result,
                    )
                )
                episode_summary = EpisodeSummary(
                    task_id=task_id,
                    final_reward=step.reward,
                    steps=turn + 1,
                    terminal_evaluation=terminal_evaluation,
                    simulation_result=simulation_result,
                )
                break
            reset = ResetResult(observation=step.observation, info=reset.info)
            turn += 1
    except BaseException as error:
        _close_after_failure(environment, error)
        raise
    environment.close()
    if episode_summary is None:
        raise RuntimeError(f"Tau2 episode ended without a terminal result for task {task_id}")
    return episode_summary


def _emit(run_context: RunContext, event: dict[str, Any]) -> None:
    run_context.event_writer.append(sanitize_artifact_data(event))


def _decision_reset_info(info: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    if "policy" not in info or "tools" not in info:
        raise RuntimeError(f"Tau2 reset info is missing public policy or tools for task {task_id}")
    return {"policy": info["policy"], "tools": info["tools"]}


def _public_step_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return sanitize_artifact_data(
        {key: info[key] for key in ("parse_error",) if key in info}
    )


def _terminal_json_mapping(
    info: Mapping[str, Any], field: str, task_id: str
) -> Mapping[str, Any]:
    value = info.get(field)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"terminal {field} JSON is invalid for task {task_id}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"terminal {field} must be a JSON object for task {task_id}")
    try:
        encoded = json.dumps(sanitize_artifact_data(value), allow_nan=False)
        parsed = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"terminal {field} is not JSON safe for task {task_id}") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"terminal {field} must be a JSON object for task {task_id}")
    return parsed


def _close_after_failure(environment: BaselineEnvironment, error: BaseException) -> None:
    try:
        environment.close()
    except Exception as cleanup_error:
        error.add_note(f"Tau2 cleanup also failed: {cleanup_error}")
