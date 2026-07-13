from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pytest

from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.runner import (
    FastLoopConfig,
    LifecycleResponse,
    run_fast_loop_episode,
)
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
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

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, text: str) -> tuple[float, ...]:
        self.embedded.append(text)
        return (1.0, 0.0)

    def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
        self.embedded.extend(texts)
        return [(1.0, 0.0) for _ in texts]


@dataclass
class FakeEnvironment:
    reset_result: ResetResult
    step_results: list[StepResult] = field(default_factory=list)
    step_error: Exception | None = None
    close_error: Exception | None = None
    reset_calls: int = 0
    close_calls: int = 0
    actions: list[str] = field(default_factory=list)

    def reset(self, *, seed: int) -> ResetResult:
        self.reset_calls += 1
        self.reset_seed = seed
        return self.reset_result

    def step(self, action: str) -> StepResult:
        self.actions.append(action)
        if self.step_error is not None:
            raise self.step_error
        return self.step_results.pop(0)

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class ScriptedLifecyclePolicy:
    def __init__(
        self,
        outputs: list[str | BaseException],
        repairs: list[str | BaseException] | None = None,
    ) -> None:
        self.outputs = list(outputs)
        self.repairs = list(repairs or [])
        self.prompts: list[Any] = []
        self.repair_calls: list[tuple[Any, str, str]] = []

    def generate(self, prompt: Any) -> LifecycleResponse:
        self.prompts.append(prompt)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return _response(output)

    def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
        self.repair_calls.append((prompt, raw_output, error))
        output = self.repairs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return _response(output)


class FailOnSecondAddRepository(MemoryRepository):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.runner_add_calls = 0

    def add(self, **kwargs: Any):
        self.runner_add_calls += 1
        if self.runner_add_calls == 2:
            raise OSError("database password=super-secret")
        return super().add(**kwargs)


def _response(raw_output: str) -> LifecycleResponse:
    return LifecycleResponse(
        raw_output=raw_output,
        sampling_params={"temperature": 0.7, "top_p": 0.9},
        latency_s=0.01,
    )


def _context(events: EventCollector, **overrides: Any) -> RunContext:
    values = {
        "run_id": "learn-001",
        "iteration": 3,
        "split": "train",
        "model_revision": "Qwen@revision-a",
        "adapter_revision": "adapter-3",
        "memory_snapshot_id": "memory-2",
        "seed": 19,
        "event_writer": events,
        "mode": RunMode.LEARN,
    }
    values.update(overrides)
    return RunContext(**values)


def _reset() -> ResetResult:
    return ResetResult(
        observation="Customer asks for a refund",
        info={
            "policy": {"text": "Verify identity before refunds"},
            "tools": [{"type": "function", "function": {"name": "lookup_order"}}],
            "Task": {"id": "hidden-task"},
            "evaluation_criteria": {"secret": True},
        },
    )


def _terminal_step(reward: float = 0.8) -> StepResult:
    return StepResult(
        observation="Refund complete",
        reward=reward,
        done=True,
        terminated=True,
        truncated=False,
        info={
            "parse_error": None,
            "reward_info": '{"score":0.8,"details":{"policy":1}}',
            "simulation_run": '{"id":"sim-1","status":"complete"}',
            "evaluation_criteria": {"hidden": True},
        },
    )


def _run(
    *,
    repository: MemoryRepository,
    environment: FakeEnvironment,
    policy: ScriptedLifecyclePolicy,
    events: EventCollector | None = None,
    config: FastLoopConfig | None = None,
    context: RunContext | None = None,
):
    collector = events or EventCollector()
    return run_fast_loop_episode(
        task_id="provenance-task-923",
        task_instruction="Help the customer complete a refund",
        environment=environment,
        policy=policy,
        repository=repository,
        retriever=Retriever(DeterministicEmbeddings()),
        config=config or FastLoopConfig(),
        context=context or _context(collector),
    )


def test_happy_path_emits_canonical_evidence_and_persists_provenance(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    candidate = repository.add(
        tier="tip",
        content="Check the order before refunding",
        source_task_ids=("seed-task",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            json.dumps({"memory_ids": [candidate.id]}),
            json.dumps({"action": "lookup_order(order_id='123')"}),
            json.dumps(
                {
                    "memories": [
                        {
                            "tier": "tip",
                            "content": "Verify the order before issuing a refund",
                            "retrieval_text": "refund verification",
                            "metadata": {
                                "note": "learned",
                                "source_run_id": "model-override",
                                "attribution_score": 0.99,
                            },
                        }
                    ]
                }
            ),
        ]
    )
    events = EventCollector()
    embeddings = DeterministicEmbeddings()

    result = run_fast_loop_episode(
        task_id="provenance-task-923",
        task_instruction="Help the customer complete a refund",
        environment=environment,
        policy=policy,
        repository=repository,
        retriever=Retriever(embeddings),
        config=FastLoopConfig(retrieve_top_k=7, max_episode_steps=4),
        context=_context(events),
    )

    assert [event["event_type"] for event in events.events] == [
        "EpisodeStarted",
        "MemoryCandidatesRetrieved",
        "MemorySelected",
        "DecisionMade",
        "EnvironmentStepped",
        "EpisodeFinished",
        "MemoryWriteProposed",
        "MemoryWriteCommitted",
    ]
    candidates = events.events[1]["candidates"]
    selected = events.events[2]["selected"]
    assert {item["memory_id"] for item in selected} <= {
        item["memory_id"] for item in candidates
    }
    assert candidates == [
        {
            "memory_id": candidate.id,
            "memory_version": 1,
            "tier": "tip",
            "rank": 1,
            "similarity": pytest.approx(1.0),
        }
    ]
    assert events.events[2]["raw_output"] == json.dumps(
        {"memory_ids": [candidate.id]}
    )
    assert events.events[2]["repaired_output"] is None
    assert events.events[2]["error"] is None
    assert environment.actions == ["lookup_order(order_id='123')"]
    assert events.events[3]["parsed_action"] == environment.actions[0]
    assert events.events[4]["public_info"] == {"parse_error": None}
    assert result.final_reward == 0.8
    assert result.terminal_evaluation == {"score": 0.8, "details": {"policy": 1}}
    assert result.simulation_result == {"id": "sim-1", "status": "complete"}
    assert result.steps == 1
    assert result.selected_memory_ids == (candidate.id,)
    assert len(result.written_memory_ids) == 1
    assert result.truncated is False
    written = repository.get(result.written_memory_ids[0])
    assert written is not None
    assert written.source_task_ids == ("provenance-task-923",)
    assert written.created_round == 3
    assert written.retrieval_text == "refund verification"
    assert written.metadata == {
        "note": "learned",
        "source_run_id": "learn-001",
        "source_iteration": 3,
        "source_final_reward": 0.8,
        "selected_memory_ids": [candidate.id],
    }
    assert "attribution_score" not in repr(events.events)
    assert all("provenance-task-923" not in prompt.model_dump_json() for prompt in policy.prompts)
    assert "Help the customer complete a refund" in embeddings.embedded[-1]
    assert "lookup_order" in embeddings.embedded[-1]
    assert "Customer asks for a refund" in embeddings.embedded[-1]
    assert environment.reset_seed == 19
    assert environment.close_calls == 1


@pytest.mark.parametrize(
    ("split", "mode"),
    [
        ("test", RunMode.LEARN),
        ("train", RunMode.BASELINE),
        ("train", RunMode.EVALUATE),
    ],
)
def test_pre_reset_guards_reject_non_learning_contexts_without_side_effects(
    tmp_path: Path, split: str, mode: RunMode
) -> None:
    repository = MemoryRepository(tmp_path / f"memory-{split}-{mode}")
    environment = FakeEnvironment(_reset())
    policy = ScriptedLifecyclePolicy([])
    events = EventCollector()

    with pytest.raises(ValueError):
        _run(
            repository=repository,
            environment=environment,
            policy=policy,
            events=events,
            context=_context(events, split=split, mode=mode),
        )

    assert environment.reset_calls == 0
    assert environment.close_calls == 0
    assert policy.prompts == []
    assert repository.list() == []
    assert events.events == []


def test_pre_reset_guard_rejects_read_only_repository_without_side_effects(
    tmp_path: Path,
) -> None:
    mutable = MemoryRepository(tmp_path / "memory")
    read_only = ReadOnlyMemoryRepository(mutable.snapshot().path)
    environment = FakeEnvironment(_reset())
    policy = ScriptedLifecyclePolicy([])
    events = EventCollector()

    with pytest.raises(ValueError, match="mutable"):
        run_fast_loop_episode(
            task_id="provenance-task-923",
            task_instruction="Help with a refund",
            environment=environment,
            policy=policy,
            repository=read_only,
            retriever=Retriever(DeterministicEmbeddings()),
            config=FastLoopConfig(),
            context=_context(events),
        )

    assert environment.reset_calls == 0
    assert environment.close_calls == 0
    assert policy.prompts == []
    assert events.events == []


def test_selection_unknown_id_with_failed_repair_emits_failure(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    candidate = repository.add(
        tier="tip",
        content="Known memory",
        source_task_ids=("seed",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    events = EventCollector()
    environment = FakeEnvironment(_reset())
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":["unknown"]}'],
        ['{"memory_ids":["still-unknown"]}'],
    )

    with pytest.raises(ValueError, match="selection"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert candidate.id not in policy.repair_calls[0][1]
    assert len(policy.repair_calls) == 1
    assert [event["event_type"] for event in events.events] == [
        "EpisodeStarted",
        "MemoryCandidatesRetrieved",
        "EpisodeFailed",
    ]
    assert "still-unknown" not in repr(events.events[-1])
    assert environment.actions == []
    assert environment.close_calls == 1


def test_invalid_action_with_failed_repair_never_steps_environment(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset())
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":""}'],
        ['{"action":"   "}'],
    )

    with pytest.raises(ValueError, match="action"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert len(policy.repair_calls) == 1
    assert environment.actions == []
    assert events.events[-1]["event_type"] == "EpisodeFailed"
    assert environment.close_calls == 1


def test_policy_timeout_emits_sanitized_failure_and_closes(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset())
    policy = ScriptedLifecyclePolicy([TimeoutError("token=super-secret")])

    with pytest.raises(TimeoutError, match="super-secret"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert events.events[-1]["event_type"] == "EpisodeFailed"
    assert "super-secret" not in repr(events.events[-1])
    assert environment.close_calls == 1


def test_environment_error_preserves_prior_decision_evidence(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset(), step_error=RuntimeError("step failed"))
    policy = ScriptedLifecyclePolicy(['{"memory_ids":[]}', '{"action":"lookup"}'])

    with pytest.raises(RuntimeError, match="step failed"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert [event["event_type"] for event in events.events][-2:] == [
        "DecisionMade",
        "EpisodeFailed",
    ]
    assert environment.actions == ["lookup"]
    assert environment.close_calls == 1


def test_max_step_boundary_finishes_as_project_truncation(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(
        _reset(),
        [
            StepResult("still working", 0.1, False, False, False, {}),
            StepResult("limit reached", 0.25, False, False, False, {}),
        ],
    )
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"first"}',
            '{"action":"second"}',
            '{"memories":[]}',
        ]
    )

    result = _run(
        repository=repository,
        environment=environment,
        policy=policy,
        events=events,
        config=FastLoopConfig(max_episode_steps=2),
    )

    assert environment.actions == ["first", "second"]
    assert result.steps == 2
    assert result.final_reward == 0.25
    assert result.truncated is True
    assert result.terminal_evaluation == {}
    assert result.simulation_result == {}
    assert events.events[-3]["event_type"] == "EpisodeFinished"
    assert events.events[-3]["project_truncated"] is True
    assert environment.close_calls == 1


def test_official_environment_truncation_is_not_labeled_project_truncation(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    terminal = StepResult(
        "environment limit",
        0.4,
        True,
        False,
        True,
        {"reward_info": "{}", "simulation_run": "{}"},
    )
    environment = FakeEnvironment(_reset(), [terminal])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"continue"}',
            '{"memories":[]}',
        ]
    )

    result = _run(
        repository=repository,
        environment=environment,
        policy=policy,
        events=events,
    )

    finished = next(
        event for event in events.events if event["event_type"] == "EpisodeFinished"
    )
    assert result.truncated is True
    assert finished["truncated"] is True
    assert finished["project_truncated"] is False


def test_repository_partial_write_failure_reports_committed_ids_and_raises(
    tmp_path: Path,
) -> None:
    repository = FailOnSecondAddRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            json.dumps(
                {
                    "memories": [
                        {"tier": "tip", "content": "First committed"},
                        {"tier": "skill", "content": "Second fails"},
                    ]
                }
            ),
        ]
    )

    with pytest.raises(OSError, match="super-secret"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    failed = events.events[-1]
    assert failed["event_type"] == "MemoryWriteFailed"
    assert len(failed["committed_memory_ids"]) == 1
    assert "super-secret" not in repr(failed)
    assert not any(event["event_type"] == "MemoryWriteCommitted" for event in events.events)
    assert environment.close_calls == 1


def test_duplicate_write_is_accepted_only_as_safe_stable_id_replay(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    existing = repository.add(
        tier="tip",
        content="  Verify   identity before refund  ",
        source_task_ids=("provenance-task-923",),
        created_round=1,
    )
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            '{"memories":[{"tier":"tip","content":"Verify identity before refund"}]}',
        ]
    )

    result = _run(repository=repository, environment=environment, policy=policy, events=events)

    assert result.written_memory_ids == (existing.id,)
    assert events.events[-1]["event_type"] == "MemoryWriteCommitted"
    assert events.events[-1]["replayed_memory_ids"] == [existing.id]
    assert len(repository.list()) == 1


def test_cleanup_failure_is_attached_to_primary_exception(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(
        _reset(),
        close_error=RuntimeError("cleanup failed"),
    )
    policy = ScriptedLifecyclePolicy([TimeoutError("policy timeout")])

    with pytest.raises(TimeoutError, match="policy timeout") as error:
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert environment.close_calls == 1
    assert any("cleanup failed" in note for note in error.value.__notes__)


@pytest.mark.parametrize("field", ["retrieve_top_k", "max_episode_steps"])
def test_config_requires_positive_limits(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        FastLoopConfig(**{field: 0})
