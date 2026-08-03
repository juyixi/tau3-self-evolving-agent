from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pytest

from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.eval.guard import EvaluationGuard, EvaluationProtocol
from tau3_retail_evolver.eval.runner import (
    run_evaluation_episode,
    run_evaluation_trials,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.runner import FastLoopConfig, LifecycleResponse
from tau3_retail_evolver.memory.paths import training_memory_root
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever


class EventCollector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class DeterministicEmbeddings:
    model_revision = "fake-embedding@1"
    dimension = 2

    def embed(self, _text: str) -> tuple[float, ...]:
        return (1.0, 0.0)

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [(1.0, 0.0) for _ in texts]


@dataclass
class FakeEnvironment:
    step_results: list[StepResult]
    reset_calls: int = 0
    close_calls: int = 0
    reset_seeds: list[int] = field(default_factory=list)

    def reset(self, *, seed: int) -> ResetResult:
        self.reset_calls += 1
        self.reset_seeds.append(seed)
        return ResetResult(
            observation="Customer asks for help",
            info={
                "policy": {"text": "Follow retail policy"},
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup_order"},
                    }
                ],
            },
        )

    def step(self, _action: str) -> StepResult:
        return self.step_results.pop(0)

    def close(self) -> None:
        self.close_calls += 1


class ScriptedPolicy:
    def __init__(
        self,
        outputs: list[str],
        repairs: list[str] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.repairs = list(repairs or [])
        self.prompts: list[Any] = []

    def generate(self, prompt: Any) -> LifecycleResponse:
        self.prompts.append(prompt)
        return LifecycleResponse(
            raw_output=self.outputs.pop(0),
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            latency_s=0.01,
        )

    def repair(
        self,
        prompt: Any,
        _raw_output: str,
        _error: str,
    ) -> LifecycleResponse:
        if not self.repairs:
            raise AssertionError("repair was not expected")
        self.prompts.append(prompt)
        return LifecycleResponse(
            raw_output=self.repairs.pop(0),
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            latency_s=0.01,
        )


class FailingOncePolicy:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, _prompt: Any) -> LifecycleResponse:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("first task failed")
        return LifecycleResponse(
            raw_output='{"action":"finish"}',
            sampling_params={"temperature": 0.7, "top_p": 0.9},
            latency_s=0.01,
        )

    def repair(
        self,
        _prompt: Any,
        _raw_output: str,
        _error: str,
    ) -> LifecycleResponse:
        raise AssertionError("repair was not expected")


def _terminal_step(
    reward: float = 1.0,
    *,
    parse_error: str | None = None,
) -> StepResult:
    return StepResult(
        observation="Done",
        reward=reward,
        done=True,
        terminated=True,
        truncated=False,
        info={
            "parse_error": parse_error,
            "reward_info": json.dumps(
                {
                    "reward": reward,
                    "reward_breakdown": {
                        "DB": reward,
                        "COMMUNICATE": reward,
                    },
                }
            ),
            "simulation_run": '{"termination_reason":"agent_stop"}',
        },
    )


def _context(events: EventCollector, **overrides: Any) -> RunContext:
    values = {
        "run_id": "eval-001",
        "iteration": 2,
        "split": "test",
        "model_revision": "qwen-revision",
        "adapter_revision": "adapter-2",
        "memory_snapshot_id": None,
        "seed": 42,
        "event_writer": events,
        "mode": RunMode.EVALUATE,
        "trial_index": 0,
    }
    values.update(overrides)
    return RunContext(**values)


def _training_snapshot(project_root: Path):
    repository = MemoryRepository(
        training_memory_root("retail", root=project_root)
    )
    item = repository.add(
        tier="tip",
        content="Verify the order before changing it.",
        source_task_ids=("0",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    return repository.snapshot(), item


def test_static_episode_retrieves_memory_without_writing(
    tmp_path: Path,
) -> None:
    snapshot, item = _training_snapshot(tmp_path)
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STATIC,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
        memory_snapshot_path=snapshot.path,
    )
    memory = guard.open_memory(trial_index=0)
    events = EventCollector()
    policy = ScriptedPolicy(
        [
            json.dumps({"memory_ids": [item.id]}),
            '{"action":"finish"}',
        ]
    )

    result = run_evaluation_episode(
        task_id="75",
        task_instruction="Resolve the retail request.",
        environment=FakeEnvironment([_terminal_step()]),
        policy=policy,
        memory=memory,
        retriever=Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(),
        context=_context(
            events,
            memory_snapshot_id=snapshot.memory_snapshot_id,
        ),
        guard=guard,
        trial_index=0,
    )

    event_types = [event["event_type"] for event in events.events]
    assert result.selected_memory_ids == (item.id,)
    assert result.written_memory_ids == ()
    assert [prompt.kind for prompt in policy.prompts] == ["selection", "action"]
    assert "MemoryWriteProposed" not in event_types
    assert "MemoryWriteCommitted" not in event_types
    assert len(memory.repository.list()) == 1


def test_streaming_episode_writes_only_to_its_trial_quarantine(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    memory = guard.open_memory(trial_index=0)
    assert memory.repository is not None
    snapshot_id = memory.repository.snapshot().memory_snapshot_id
    policy = ScriptedPolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            json.dumps(
                {
                    "memories": [
                            {
                                "tier": "tip",
                                "payload": {
                                    "guidance": "Check the order before changing it."
                                },
                                "retrieval_text": "order change check",
                                "metadata": {},
                            }
                    ]
                }
            ),
        ]
    )
    events = EventCollector()

    result = run_evaluation_episode(
        task_id="75",
        task_instruction="Resolve the retail request.",
        environment=FakeEnvironment([_terminal_step(parse_error="recovered")]),
        policy=policy,
        memory=memory,
        retriever=Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(),
        context=_context(events, memory_snapshot_id=snapshot_id),
        guard=guard,
        trial_index=0,
    )

    written_items = memory.repository.list()
    assert sorted(item.tier.value for item in written_items) == ["tip", "trajectory"]
    assert len(result.written_memory_ids) == 2
    assert result.parse_error_count == 1
    assert result.completed is True
    assert not training_memory_root("retail", root=tmp_path).exists()


def test_evaluation_counts_repaired_model_responses(
    tmp_path: Path,
) -> None:
    snapshot, _item = _training_snapshot(tmp_path)
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STATIC,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
        memory_snapshot_path=snapshot.path,
    )
    memory = guard.open_memory(trial_index=0)
    policy = ScriptedPolicy(
        ["not-json", '{"action":"finish"}'],
        repairs=['{"memory_ids":[]}'],
    )

    result = run_evaluation_episode(
        task_id="75",
        task_instruction="Resolve the retail request.",
        environment=FakeEnvironment([_terminal_step()]),
        policy=policy,
        memory=memory,
        retriever=Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(),
        context=_context(
            EventCollector(),
            memory_snapshot_id=snapshot.memory_snapshot_id,
        ),
        guard=guard,
        trial_index=0,
    )

    assert result.response_count == 2
    assert result.response_parse_error_count == 1


def test_evaluation_context_is_rejected_before_reset(tmp_path: Path) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.NO_MEMORY,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    memory = guard.open_memory(trial_index=0)
    environment = FakeEnvironment([_terminal_step()])

    with pytest.raises(ValueError, match="EVALUATE"):
        run_evaluation_episode(
            task_id="75",
            task_instruction="Resolve the retail request.",
            environment=environment,
            policy=ScriptedPolicy(['{"action":"finish"}']),
            memory=memory,
            retriever=None,
            config=FastLoopConfig(memory_enabled=False),
            context=_context(EventCollector(), mode=RunMode.LEARN),
            guard=guard,
            trial_index=0,
        )

    assert environment.reset_calls == 0


def test_trial_dependency_mismatch_is_rejected_before_quarantine_creation(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )

    with pytest.raises(ValueError, match="memory_enabled=true"):
        run_evaluation_trials(
            task_ids=("75",),
            seeds=(42,),
            env_factory=lambda _task_id: FakeEnvironment([_terminal_step()]),
            policy=ScriptedPolicy([]),
            guard=guard,
            retriever_factory=lambda _memory: Retriever(
                DeterministicEmbeddings()
            ),
            config=FastLoopConfig(memory_enabled=False),
            context=_context(EventCollector()),
            maintenance_period=30,
        )

    assert not guard.quarantine_root.exists()


def test_evaluation_trials_record_failed_episode_and_continue(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.NO_MEMORY,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    events = EventCollector()

    result = run_evaluation_trials(
        task_ids=("75", "76"),
        seeds=(42,),
        env_factory=lambda _task_id: FakeEnvironment([_terminal_step()]),
        policy=FailingOncePolicy(),
        guard=guard,
        retriever_factory=None,
        config=FastLoopConfig(memory_enabled=False),
        context=_context(events),
        maintenance_period=30,
    )

    assert [episode.result.task_id for episode in result.episodes] == ["75", "76"]
    assert result.episodes[0].result.final_reward == 0.0
    assert result.episodes[0].result.completed is False
    assert result.episodes[0].result.terminal_evaluation["evaluation_error"] == {
        "type": "RuntimeError",
        "message": "operation failed",
    }
    assert result.episodes[1].result.final_reward == 1.0
    assert result.episodes[1].result.completed is True
    assert [event["event_type"] for event in events.events] == [
        "EpisodeStarted",
        "MemoryDisabled",
        "EpisodeFailed",
        "EpisodeStarted",
        "MemoryDisabled",
        "DecisionMade",
        "EnvironmentStepped",
        "EpisodeFinished",
    ]


def test_streaming_trials_each_start_from_empty_memory(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-001",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    policy = ScriptedPolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
                '{"memories":[{"tier":"tip","payload":{"guidance":"Trial zero"},'
                '"metadata":{}}]}',
            '{"memory_ids":[]}',
            '{"action":"finish"}',
                '{"memories":[{"tier":"tip","payload":{"guidance":"Trial one"},'
                '"metadata":{}}]}',
        ]
    )

    result = run_evaluation_trials(
        task_ids=("75",),
        seeds=(42, 43),
        env_factory=lambda _task_id: FakeEnvironment([_terminal_step()]),
        policy=policy,
        guard=guard,
        retriever_factory=lambda _memory: Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(),
        context=_context(EventCollector()),
        maintenance_period=30,
    )

    assert len(result.episodes) == 2
    assert [episode.seed for episode in result.episodes] == [42, 43]
    assert [episode.trial_index for episode in result.episodes] == [0, 1]
    assert all(episode.result.selected_memory_ids == () for episode in result.episodes)
    assert len(set(result.output_memory_snapshot_ids)) == 2
    assert result.maintenance_rounds_by_trial == ((), ())


def test_streaming_executes_maintenance_at_task_thirty(
    tmp_path: Path,
) -> None:
    guard = EvaluationGuard(
        protocol=EvaluationProtocol.TEST_STREAMING,
        run_id="eval-thirty",
        agent_id="retail",
        project_root=tmp_path,
        split="test",
    )
    task_ids = tuple(str(task_id) for task_id in range(75, 105))
    outputs: list[str] = []
    for task_id in task_ids:
        outputs.extend(
            (
                '{"memory_ids":[]}',
                '{"action":"finish"}',
                json.dumps(
                    {
                        "memories": [
                                {
                                    "tier": "tip",
                                    "payload": {
                                        "guidance": (
                                            f"Experience from task {task_id}"
                                        )
                                    },
                                    "metadata": {},
                                }
                        ]
                    }
                ),
            )
        )
    outputs.append('{"commands":[]}')
    events = EventCollector()

    result = run_evaluation_trials(
        task_ids=task_ids,
        seeds=(42,),
        env_factory=lambda _task_id: FakeEnvironment([_terminal_step()]),
        policy=ScriptedPolicy(outputs),
        guard=guard,
        retriever_factory=lambda _memory: Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(),
        context=_context(events, run_id="eval-thirty"),
        maintenance_period=30,
    )

    assert result.maintenance_rounds_by_trial == ((1,),)
    assert [event["event_type"] for event in events.events].count(
        "MaintenanceCommitted"
    ) == 1
