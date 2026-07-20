from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.baseline_runner import run_baseline
from tau3_retail_evolver.fast_loop.events import RunContext
from tau3_retail_evolver.models.policy import DecisionResponse
from tests.support.policy import ScriptedPolicy


@dataclass
class FakeEnvironment:
    reset_result: ResetResult
    step_results: list[StepResult]
    closed: bool = False
    reset_seed: int | None = None
    actions: list[str] | None = None
    close_error: Exception | None = None

    def __post_init__(self) -> None:
        self.actions = []

    def reset(self, *, seed: int) -> ResetResult:
        self.reset_seed = seed
        return self.reset_result

    def step(self, action: str) -> StepResult:
        assert self.actions is not None
        self.actions.append(action)
        return self.step_results.pop(0)

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class EventCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class FailingEventWriter:
    def append(self, event: dict[str, Any]) -> None:
        raise TypeError("event must be JSON serializable")


def _response(action: str) -> DecisionResponse:
    return DecisionResponse(
        raw_output=f"raw {action}",
        parsed_action=action,
        sampling_params={"temperature": 1.0, "top_p": 0.95},
        latency_s=0.125,
    )


def test_runs_two_tasks_with_canonical_auditable_event_order_and_closes_environments() -> None:
    environments = {
        "task-1": FakeEnvironment(
            reset_result=ResetResult(
                observation="task one observation",
                info={"policy": "retail policy", "tools": []},
            ),
            step_results=[
                StepResult(
                    observation="task one complete",
                    reward=0.75,
                    done=True,
                    terminated=True,
                    truncated=False,
                    info={
                        "reward_info": '{"total_reward":0.75,"policy":0.75}',
                        "simulation_run": '{"id":"simulation-1","completed":true}',
                    },
                )
            ],
        ),
        "task-2": FakeEnvironment(
            reset_result=ResetResult(
                observation="task two observation",
                info={"policy": "retail policy", "tools": []},
            ),
            step_results=[
                StepResult(
                    observation="task two in progress",
                    reward=0.1,
                    done=False,
                    terminated=False,
                    truncated=False,
                    info={},
                ),
                StepResult(
                    observation="task two complete",
                    reward=1.0,
                    done=True,
                    terminated=True,
                    truncated=False,
                    info={
                        "reward_info": '{"total_reward":1.0,"policy":1.0}',
                        "simulation_run": '{"id":"simulation-2","completed":true}',
                    },
                ),
            ],
        ),
    }
    events = EventCollector()
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="Qwen/Qwen3.5-9B@revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=events,
        task_groups={"task-1": "returns", "task-2": "orders"},
    )
    policy = ScriptedPolicy([_response("lookup"), _response("search"), _response("complete")])

    summary = run_baseline(
        ("task-1", "task-2"),
        lambda task_id: environments[task_id],
        policy,
        context,
    )

    assert summary.episode_count == 2
    assert summary.total_reward == 1.75
    assert [episode.task_id for episode in summary.episodes] == ["task-1", "task-2"]
    assert [episode.final_reward for episode in summary.episodes] == [0.75, 1.0]
    assert [event["event_type"] for event in events.events] == [
        "EpisodeStarted",
        "DecisionMade",
        "EnvironmentStepped",
        "EpisodeFinished",
        "EpisodeStarted",
        "DecisionMade",
        "EnvironmentStepped",
        "DecisionMade",
        "EnvironmentStepped",
        "EpisodeFinished",
    ]
    for event in events.events:
        assert event["schema_version"] == 2
        assert {
            "run_id": "baseline-001",
            "iteration": 0,
            "split": "train",
            "model_revision": "Qwen/Qwen3.5-9B@revision-a",
            "adapter_revision": None,
            "memory_snapshot_id": None,
            "seed": 17,
        }.items() <= event.items()
        assert event["task_id"] in {"task-1", "task-2"}
        assert event["task_group"] in {"returns", "orders"}

    finished = [event for event in events.events if event["event_type"] == "EpisodeFinished"]
    assert finished[0]["final_reward"] == 0.75
    assert finished[0]["terminal_evaluation"] == {"total_reward": 0.75, "policy": 0.75}
    assert finished[0]["simulation_result"] == {"id": "simulation-1", "completed": True}
    assert finished[1]["final_reward"] == 1.0
    assert finished[1]["terminal_evaluation"] == {"total_reward": 1.0, "policy": 1.0}
    assert finished[1]["simulation_result"] == {"id": "simulation-2", "completed": True}
    assert "reward_info" not in finished[0]
    assert "evaluator_details" not in finished[0]
    assert all(environment.closed for environment in environments.values())
    assert [request.history for request in policy.requests] == [(), (), ()]


def test_closes_environment_when_policy_generation_fails() -> None:
    environment = FakeEnvironment(
        reset_result=ResetResult(observation="observation", info={"policy": "policy", "tools": []}),
        step_results=[],
    )
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=EventCollector(),
    )
    policy = ScriptedPolicy([])

    try:
        run_baseline(("task-1",), lambda task_id: environment, policy, context)
    except RuntimeError as error:
        assert "no remaining responses" in str(error)
    else:
        raise AssertionError("expected policy failure")

    assert environment.closed


def test_sanitizes_credentials_in_environment_metadata_before_emitting_events() -> None:
    environment = FakeEnvironment(
        reset_result=ResetResult(
            observation="observation",
            info={"policy": "policy", "tools": [], "provider": {"api_key": "secret-value"}},
        ),
        step_results=[
            StepResult(
                observation="complete",
                reward=1.0,
                done=True,
                terminated=True,
                truncated=False,
                info={
                    "reward_info": '{"score":1.0}',
                    "simulation_run": '{"id":"simulation-1"}',
                    "evaluator": {"access_token": "secret-value"},
                },
            )
        ],
    )
    events = EventCollector()
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=events,
    )

    run_baseline(("task-1",), lambda task_id: environment, ScriptedPolicy([_response("stop")]), context)

    rendered = repr(events.events)
    assert "secret-value" not in rendered
    assert "evaluator" not in rendered


def test_closes_environment_when_event_serialization_fails() -> None:
    environment = FakeEnvironment(
        reset_result=ResetResult(observation="observation", info={"policy": "policy", "tools": []}),
        step_results=[],
    )
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=FailingEventWriter(),
    )

    with pytest.raises(TypeError, match="JSON serializable"):
        run_baseline(("task-1",), lambda task_id: environment, ScriptedPolicy([]), context)

    assert environment.closed


def test_emits_only_allowlisted_public_context_and_never_passes_task_metadata_to_policy() -> None:
    environment = FakeEnvironment(
        reset_result=ResetResult(
            observation="Welcome",
            info={
                "policy": "public policy",
                "tools": [{"type": "function", "function": {"name": "lookup"}}],
                "Task": {"id": "secret-task"},
                "evaluation_criteria": {"hidden": True},
                "runtime": {"api_key": "never-record"},
            },
        ),
        step_results=[
            StepResult(
                observation="Complete",
                reward=1.0,
                done=True,
                terminated=True,
                truncated=False,
                info={
                    "parse_error": "none",
                    "evaluation_criteria": {"hidden": True},
                    "reward_info": '{"total_reward":1.0,"nested":{"all":true}}',
                    "simulation_run": '{"run_id":"sim-1"}',
                },
            )
        ],
    )
    events = EventCollector()
    policy = ScriptedPolicy([_response("lookup")])
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=events,
    )

    run_baseline(("task-1",), lambda task_id: environment, policy, context)

    started, _, stepped, finished = events.events
    assert started["policy"] == "public policy"
    assert started["tool_count"] == 1
    assert started["tool_schemas"] == [{"type": "function", "function": {"name": "lookup"}}]
    assert "reset_info" not in started
    assert "Task" not in repr(started)
    assert "evaluation_criteria" not in repr(started)
    assert stepped["public_info"] == {"parse_error": "none"}
    assert "info" not in stepped
    assert finished["terminal_evaluation"] == {"total_reward": 1.0, "nested": {"all": True}}
    assert finished["simulation_result"] == {"run_id": "sim-1"}
    assert set(policy.requests[0].reset_info) == {"policy", "tools"}


def test_rejects_malformed_terminal_reward_json_with_task_context() -> None:
    environment = FakeEnvironment(
        reset_result=ResetResult(observation="Welcome", info={"policy": "policy", "tools": []}),
        step_results=[
            StepResult(
                observation="Complete",
                reward=1.0,
                done=True,
                terminated=True,
                truncated=False,
                info={"reward_info": "not-json", "simulation_run": "{}"},
            )
        ],
    )
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=EventCollector(),
    )

    with pytest.raises(RuntimeError, match="terminal reward_info JSON.*task-1"):
        run_baseline(("task-1",), lambda task_id: environment, ScriptedPolicy([_response("stop")]), context)

    assert environment.closed


def test_preserves_primary_policy_error_when_adapter_cleanup_also_fails() -> None:
    environment = FakeEnvironment(
        reset_result=ResetResult(observation="Welcome", info={"policy": "policy", "tools": []}),
        step_results=[],
        close_error=RuntimeError("cleanup failed"),
    )
    context = RunContext(
        run_id="baseline-001",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=EventCollector(),
    )

    with pytest.raises(RuntimeError, match="no remaining responses") as error:
        run_baseline(("task-1",), lambda task_id: environment, ScriptedPolicy([]), context)

    assert environment.closed
    assert any("cleanup failed" in note for note in error.value.__notes__)
