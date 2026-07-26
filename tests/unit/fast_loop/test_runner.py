from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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
from tau3_retail_evolver.memory.tier_contracts import (
    SkillPayload,
    SkillStep,
    TipPayload,
    ToolPayload,
    render_tier_payload,
)
from tau3_retail_evolver.memory.types import MemoryTier, stable_memory_id


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
    close_error: BaseException | None = None
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


class FailOnNthAddRepository(MemoryRepository):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.runner_add_calls = 0
        self.fail_on_add = 2

    def add(self, **kwargs: Any):
        self.runner_add_calls += 1
        if self.runner_add_calls == self.fail_on_add:
            raise OSError("database password=super-secret")
        return super().add(**kwargs)


class FailOnceOnGetRepository(MemoryRepository):
    fail_memory_id: str | None = None

    def get(self, memory_id: str):
        if memory_id == self.fail_memory_id:
            self.fail_memory_id = None
            raise OSError("lookup password=super-secret")
        return super().get(memory_id)


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
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_order",
                        "parameters": {
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"],
                        },
                    },
                }
            ],
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


def test_fast_loop_config_preserves_positional_limit_arguments() -> None:
    config = FastLoopConfig(7, 4)

    assert config.retrieve_top_k == 7
    assert config.max_episode_steps == 4
    assert config.memory_enabled is True


@pytest.mark.parametrize("memory_enabled", ("false", 0, 1, None))
def test_fast_loop_config_rejects_non_boolean_memory_enabled(memory_enabled: Any) -> None:
    with pytest.raises(ValueError, match="memory_enabled must be a bool"):
        FastLoopConfig(memory_enabled=memory_enabled)


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


def test_disabled_memory_bypasses_the_memory_lifecycle() -> None:
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(['{"action":"finish"}'])

    result = run_fast_loop_episode(
        task_id="provenance-task-923",
        task_instruction="Help the customer complete a refund",
        environment=environment,
        policy=policy,
        repository=None,
        retriever=None,
        config=FastLoopConfig(memory_enabled=False),
        context=_context(events),
    )

    event_types = {event["event_type"] for event in events.events}
    forbidden_memory_events = {
        "MemoryCandidatesRetrieved",
        "MemorySelected",
        "MemoryWriteProposed",
        "MemoryWriteCommitted",
        "MemoryWriteFailed",
    }
    assert [prompt.kind for prompt in policy.prompts] == ["action"]
    assert "memories" not in policy.prompts[0].payload
    assert result.selected_memory_ids == ()
    assert result.written_memory_ids == ()
    assert [event["event_type"] for event in events.events].count("MemoryDisabled") == 1
    memory_disabled = next(
        event for event in events.events if event["event_type"] == "MemoryDisabled"
    )
    assert memory_disabled["reason"] == "config"
    assert not forbidden_memory_events.intersection(event_types)


@pytest.mark.parametrize(
    ("memory_enabled", "repository", "retriever"),
    [
        (True, None, Retriever(DeterministicEmbeddings())),
        (True, MemoryRepository, None),
        (True, MemoryRepository, object()),
        (False, MemoryRepository, None),
        (False, None, Retriever(DeterministicEmbeddings())),
    ],
)
def test_memory_dependency_contract_fails_before_reset(
    tmp_path: Path,
    memory_enabled: bool,
    repository: MemoryRepository | type[MemoryRepository] | None,
    retriever: Any,
) -> None:
    events = EventCollector()
    environment = FakeEnvironment(_reset())
    policy = ScriptedLifecyclePolicy([])
    resolved_repository = (
        repository(tmp_path / "memory") if repository is MemoryRepository else repository
    )

    with pytest.raises(ValueError):
        run_fast_loop_episode(
            task_id="provenance-task-923",
            task_instruction="Help the customer complete a refund",
            environment=environment,
            policy=policy,
            repository=resolved_repository,
            retriever=retriever,
            config=FastLoopConfig(memory_enabled=memory_enabled),
            context=_context(events),
        )

    assert environment.reset_calls == 0
    assert environment.close_calls == 0
    assert policy.prompts == []
    assert events.events == []


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
            _write_output(
                _tip_write(
                    "Verify the order before issuing a refund",
                    retrieval_text="refund verification",
                    metadata={
                        "note": "learned",
                        "source_run_id": "model-override",
                    },
                )
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
    selection_output = json.dumps({"memory_ids": [candidate.id]})
    assert events.events[2]["selected_memory_ids"] == [candidate.id]
    assert events.events[2]["raw_output_sha256"] == hashlib.sha256(
        selection_output.encode("utf-8")
    ).hexdigest()
    assert events.events[2]["repaired_output_sha256"] is None
    assert events.events[2]["parse_failed"] is False
    assert events.events[2]["repair_used"] is False
    assert "raw_output" not in events.events[2]
    assert "repaired_output" not in events.events[2]
    assert "error" not in events.events[2]
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
    assert written.tier_schema_version == 2
    assert written.payload == {
        "condition": None,
        "guidance": "Verify the order before issuing a refund",
        "rationale": None,
        "scope": [],
    }
    assert written.content == "Guidance: Verify the order before issuing a refund"
    assert written.retrieval_text == "refund verification"
    assert written.metadata == {
        "classification_rule": "tip-contract-v2",
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


def test_selected_candidate_details_follow_teacher_preference_order(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    first = repository.add(
        tier="tip",
        content="First by retrieval score.",
        source_task_ids=("seed-task",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )


def _tip_write(
    guidance: str,
    *,
    retrieval_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tier": "tip",
        "payload": {"guidance": guidance},
    }
    if retrieval_text is not None:
        result["retrieval_text"] = retrieval_text
    if metadata is not None:
        result["metadata"] = metadata
    return result


def _skill_write(goal: str) -> dict[str, Any]:
    return {
        "tier": "skill",
        "payload": {
            "goal": goal,
            "steps": [
                {"order": 1, "instruction": "Inspect the current task state."},
                {"order": 2, "instruction": "Apply the verified next action."},
            ],
            "success_condition": "The requested operation is completed.",
        },
    }


def _tool_write(purpose: str) -> dict[str, Any]:
    return {
        "tier": "tool",
        "payload": {
            "tool_name": "lookup_order",
            "purpose": purpose,
            "preconditions": ["An order ID is available."],
            "argument_rules": {"order_id": "Use the exact customer order ID."},
            "expected_effect": "The current order state is returned.",
        },
    }


def _write_output(*memories: dict[str, Any]) -> str:
    return json.dumps({"memories": list(memories)})


def _add_v2_tip(
    repository: MemoryRepository,
    guidance: str,
    *,
    source_task_ids: tuple[str, ...] = ("provenance-task-923",),
    created_round: int = 0,
    **kwargs: Any,
):
    return _add_v2_memory(
        repository,
        _tip_write(guidance),
        source_task_ids=source_task_ids,
        created_round=created_round,
        **kwargs,
    )


def _add_v2_memory(
    repository: MemoryRepository,
    draft: dict[str, Any],
    *,
    source_task_ids: tuple[str, ...] = ("provenance-task-923",),
    created_round: int = 0,
    **kwargs: Any,
):
    tier = MemoryTier(draft["tier"])
    payload_type = {
        MemoryTier.TIP: TipPayload,
        MemoryTier.SKILL: SkillPayload,
        MemoryTier.TOOL: ToolPayload,
    }[tier]
    payload = payload_type.model_validate_json(json.dumps(draft["payload"]))
    return repository.add(
        tier=tier,
        tier_schema_version=2,
        payload=payload.model_dump(mode="json"),
        content=render_tier_payload(tier, payload),
        source_task_ids=source_task_ids,
        created_round=created_round,
        **kwargs,
    )
    second = repository.add(
        tier="skill",
        content="Second by retrieval score.",
        source_task_ids=("seed-task",),
        created_round=0,
        embedding=(0.8, 0.6),
        embedding_model_revision="fake-embedding@1",
    )
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            json.dumps({"memory_ids": [second.id, first.id]}),
            json.dumps({"action": "lookup_order(order_id='123')"}),
            json.dumps({"memories": []}),
        ]
    )
    events = EventCollector()

    run_fast_loop_episode(
        task_id="selection-order-task",
        task_instruction="Help the customer complete a refund",
        environment=environment,
        policy=policy,
        repository=repository,
        retriever=Retriever(DeterministicEmbeddings()),
        config=FastLoopConfig(retrieve_top_k=2),
        context=_context(events),
    )

    selected = next(
        event for event in events.events if event["event_type"] == "MemorySelected"
    )
    assert selected["selected_memory_ids"] == [second.id, first.id]
    assert [item["memory_id"] for item in selected["selected"]] == [second.id, first.id]


def test_write_prompt_uses_latest_nonblank_observation_after_terminal_empty_step(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    terminal = _terminal_step()
    terminal = StepResult(
        observation="",
        reward=terminal.reward,
        done=terminal.done,
        terminated=terminal.terminated,
        truncated=terminal.truncated,
        info=terminal.info,
    )
    environment = FakeEnvironment(_reset(), [terminal])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            '{"memories":[]}',
        ]
    )

    result = _run(
        repository=repository,
        environment=environment,
        policy=policy,
    )

    write_prompt = policy.prompts[-1]
    assert write_prompt.kind == "write"
    assert write_prompt.payload["observation"] == "Customer asks for a refund"
    assert (
        write_prompt.payload["trajectory_format"]
        == "final_observation_plus_actions_v1"
    )
    compact_step = write_prompt.payload["trajectory"][0]
    assert compact_step["action"] == "finish"
    assert compact_step["reward"] == 0.8
    assert compact_step["done"] is True
    assert "observation" not in compact_step
    assert "next_observation" not in compact_step
    assert result.final_reward == 0.8


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


def test_repaired_selection_records_hashes_without_persisting_invalid_text(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    candidate = repository.add(
        tier="tip",
        content="Known memory",
        source_task_ids=("seed",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    attribution_sentinel = "selection-attribution-sentinel"
    credential_sentinel = "selection-credential-sentinel"
    raw_output = json.dumps(
        {
            "memory_ids": ["unknown"],
            "attributionScore": attribution_sentinel,
            "apiToken": credential_sentinel,
        }
    )
    repaired_output = json.dumps({"memory_ids": [candidate.id]})
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [raw_output, '{"action":"finish"}', '{"memories":[]}'],
        [repaired_output],
    )

    _run(repository=repository, environment=environment, policy=policy, events=events)

    selected = next(
        event for event in events.events if event["event_type"] == "MemorySelected"
    )
    assert selected["selected_memory_ids"] == [candidate.id]
    assert selected["raw_output_sha256"] == hashlib.sha256(
        raw_output.encode("utf-8")
    ).hexdigest()
    assert selected["repaired_output_sha256"] == hashlib.sha256(
        repaired_output.encode("utf-8")
    ).hexdigest()
    assert selected["parse_failed"] is True
    assert selected["repair_used"] is True
    serialized_events = json.dumps(events.events)
    serialized_memory = json.dumps(
        [item.model_dump(mode="json") for item in repository.list()]
    )
    for sentinel in (attribution_sentinel, credential_sentinel, "unknown"):
        assert sentinel not in serialized_events
        assert sentinel not in serialized_memory


@pytest.mark.parametrize(
    "reset_info",
    [
        {
            "policy": {"nested": {"evaluatorMetadata": "forbidden-policy-value"}},
            "tools": [{"name": "lookup", "schema": {}}],
        },
        {
            "policy": {"text": "public"},
            "tools": [
                {
                    "name": "lookup",
                    "schema": {"ｔｅｓｔＴａｓｋＩｄ": "forbidden-tool-value"},
                }
            ],
        },
    ],
)
def test_public_boundary_rejects_nested_forbidden_reset_fields_before_side_effects(
    tmp_path: Path,
    reset_info: dict[str, Any],
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(
        ResetResult(observation="Public observation", info=reset_info)
    )
    policy = ScriptedLifecyclePolicy([])
    embeddings = DeterministicEmbeddings()

    with pytest.raises(ValueError, match="forbidden"):
        run_fast_loop_episode(
            task_id="safe-task",
            task_instruction="Public instruction",
            environment=environment,
            policy=policy,
            repository=repository,
            retriever=Retriever(embeddings),
            config=FastLoopConfig(),
            context=_context(events),
        )

    assert [event["event_type"] for event in events.events] == ["EpisodeFailed"]
    assert "forbidden" not in json.dumps(events.events)
    assert embeddings.embedded == []
    assert policy.prompts == []
    assert environment.close_calls == 1


def test_public_boundary_redacts_credentials_before_events_retrieval_and_policy(
    tmp_path: Path,
) -> None:
    credential_sentinel = "reset-credential-sentinel"
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    embeddings = DeterministicEmbeddings()
    environment = FakeEnvironment(
        ResetResult(
            observation="Public observation",
            info={
                "policy": {
                    "text": "Public policy",
                    "nested": {"apiToken": credential_sentinel},
                },
                "tools": [{"name": "lookup"}],
            },
        ),
        [_terminal_step()],
    )
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":"finish"}', '{"memories":[]}']
    )

    run_fast_loop_episode(
        task_id="safe-task",
        task_instruction="Public instruction",
        environment=environment,
        policy=policy,
        repository=repository,
        retriever=Retriever(embeddings),
        config=FastLoopConfig(),
        context=_context(events),
    )

    assert credential_sentinel not in json.dumps(events.events)
    assert all(credential_sentinel not in text for text in embeddings.embedded)
    assert all(credential_sentinel not in prompt.model_dump_json() for prompt in policy.prompts)
    assert "[REDACTED]" in json.dumps(events.events)


@pytest.mark.parametrize(
    "forbidden_key",
    ["ａｔｔｒｉｂｕｔｉｏｎ＿ｓｃｏｒｅ", "apiToken", "dbPassword", "refreshToken"],
)
def test_forbidden_write_metadata_receives_one_clean_repair_without_leakage(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    sentinel = "write-metadata-secret"
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    invalid_write = _write_output(
        _tip_write(
            "Clean public memory",
            metadata={"nested": {forbidden_key: sentinel}},
        )
    )
    repaired_write = _write_output(
        _tip_write("Clean public memory", metadata={"note": "public"})
    )
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":"finish"}', invalid_write],
        [repaired_write],
    )

    result = _run(
        repository=repository,
        environment=environment,
        policy=policy,
        events=events,
    )

    assert len(policy.repair_calls) == 1
    assert result.written_memory_ids
    assert sentinel not in json.dumps(events.events)
    assert sentinel not in json.dumps(
        [item.model_dump(mode="json") for item in repository.list()]
    )


@pytest.mark.parametrize("forbidden_key", ["dbPassword", "refreshToken"])
def test_credential_write_metadata_in_repair_stops_before_proposal_or_persistence(
    tmp_path: Path,
    forbidden_key: str,
) -> None:
    sentinel = "repaired-write-credential-sentinel"
    invalid_write = _write_output(
        _tip_write(
            "Must not persist",
            metadata={"nested": {forbidden_key: sentinel}},
        )
    )
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":"finish"}', invalid_write],
        [invalid_write],
    )

    with pytest.raises(ValueError, match="write decision after repair"):
        _run(
            repository=repository,
            environment=FakeEnvironment(_reset(), [_terminal_step()]),
            policy=policy,
            events=events,
        )

    assert len(policy.repair_calls) == 1
    assert not any(event["event_type"] == "MemoryWriteProposed" for event in events.events)
    assert repository.list() == []
    assert sentinel not in json.dumps(events.events)


def test_full_width_attribution_failed_repair_stops_before_proposal_or_persistence(
    tmp_path: Path,
) -> None:
    forbidden_key = "ａｔｔｒｉｂｕｔｉｏｎ＿ｓｃｏｒｅ"
    sentinel = "failed-repair-attribution-sentinel"
    invalid_write = _write_output(
        _tip_write("Must not persist", metadata={forbidden_key: sentinel})
    )
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":"finish"}', invalid_write],
        [invalid_write],
    )

    with pytest.raises(ValueError, match="write decision after repair"):
        _run(
            repository=repository,
            environment=environment,
            policy=policy,
            events=events,
        )

    assert len(policy.repair_calls) == 1
    assert not any(event["event_type"] == "MemoryWriteProposed" for event in events.events)
    assert repository.list() == []
    assert sentinel not in json.dumps(events.events)


def test_invalid_non_sensitive_write_after_repair_skips_memory_and_keeps_episode(
    tmp_path: Path,
) -> None:
    invalid_tool = _tool_write("Look up the order before deciding the next action.")
    invalid_tool["payload"]["argument_rules"] = {}
    invalid_write = _write_output(invalid_tool)
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":"finish"}', invalid_write],
        [invalid_write],
    )

    result = _run(
        repository=repository,
        environment=FakeEnvironment(_reset(), [_terminal_step()]),
        policy=policy,
        events=events,
    )

    proposed = next(
        event for event in events.events if event["event_type"] == "MemoryWriteProposed"
    )
    committed = next(
        event for event in events.events if event["event_type"] == "MemoryWriteCommitted"
    )
    assert len(policy.repair_calls) == 1
    assert result.final_reward == 0.8
    assert result.written_memory_ids == ()
    assert result.response_parse_error_count == 1
    assert proposed["proposals"] == []
    assert proposed["invalid_output_skipped"] is True
    assert committed["written_memory_ids"] == []
    assert committed["replayed_memory_ids"] == []
    assert repository.list() == []


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


def test_repaired_action_event_does_not_persist_raw_repair_or_error_text(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    sentinel = "sentinel-sensitive-attribution"
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            json.dumps(
                {
                    "action": "",
                    "attributionScore": sentinel,
                    "secret": sentinel,
                }
            ),
            '{"memories":[]}',
        ],
        [json.dumps({"action": 'lookup_order(order_id="123")'})],
    )

    _run(repository=repository, environment=environment, policy=policy, events=events)

    decision = next(
        event for event in events.events if event["event_type"] == "DecisionMade"
    )
    assert decision["parsed_action"] == 'lookup_order(order_id="123")'
    assert decision["repair_used"] is True
    assert {"raw_output", "repaired_output", "error"}.isdisjoint(decision)
    assert sentinel not in repr(events.events)


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


def test_inconsistent_terminal_flags_are_recorded_before_episode_failure(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(
        _reset(),
        [
            StepResult(
                observation="inconsistent terminal state",
                reward=0.35,
                done=False,
                terminated=True,
                truncated=False,
                info={"parse_error": "public parse detail"},
            )
        ],
    )
    policy = ScriptedLifecyclePolicy(
        ['{"memory_ids":[]}', '{"action":"lookup"}']
    )

    with pytest.raises(RuntimeError, match="terminal flags are inconsistent"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert [event["event_type"] for event in events.events][-3:] == [
        "DecisionMade",
        "EnvironmentStepped",
        "EpisodeFailed",
    ]
    stepped = events.events[-2]
    assert stepped["observation"] == "inconsistent terminal state"
    assert stepped["reward"] == 0.35
    assert stepped["done"] is False
    assert stepped["terminated"] is True
    assert stepped["truncated"] is False
    assert stepped["public_info"] == {"parse_error": "public parse detail"}


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
    repository = FailOnNthAddRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            _write_output(
                _tip_write("First committed"),
                _skill_write("Second fails"),
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


def test_partial_write_failure_separates_replays_from_new_commits(
    tmp_path: Path,
) -> None:
    repository = FailOnNthAddRepository(tmp_path / "memory")
    repository.fail_on_add = 99
    existing = _add_v2_tip(repository, "Existing replay")
    repository.runner_add_calls = 0
    repository.fail_on_add = 3
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            _write_output(
                _tip_write("Existing replay"),
                _skill_write("New committed"),
                _tool_write("Third fails"),
            ),
        ]
    )

    with pytest.raises(OSError, match="super-secret"):
        _run(repository=repository, environment=environment, policy=policy, events=events)

    failed = events.events[-1]
    new_item = next(
        item for item in repository.list() if item.tier is MemoryTier.SKILL
    )
    assert failed["event_type"] == "MemoryWriteFailed"
    assert failed["committed_memory_ids"] == [new_item.id]
    assert failed["replayed_memory_ids"] == [existing.id]


def test_replay_lookup_failure_preserves_prior_write_progress(
    tmp_path: Path,
) -> None:
    repository = FailOnceOnGetRepository(tmp_path / "memory")
    replay = _add_v2_tip(
        repository,
        "Existing replay",
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    lookup_failure = _add_v2_memory(
        repository,
        _tool_write("Lookup fails for this replay"),
        embedding=(1.0, 0.0),
        embedding_model_revision="fake-embedding@1",
    )
    repository.fail_memory_id = lookup_failure.id
    events = EventCollector()
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            _write_output(
                _tip_write("Existing replay"),
                _skill_write("New committed"),
                _tool_write("Lookup fails for this replay"),
            ),
        ]
    )

    with pytest.raises(OSError, match="super-secret"):
        _run(
            repository=repository,
            environment=FakeEnvironment(_reset(), [_terminal_step()]),
            policy=policy,
            events=events,
        )

    failed = events.events[-1]
    new_skill = SkillPayload.model_validate_json(
        json.dumps(_skill_write("New committed")["payload"])
    )
    new_memory_id = stable_memory_id(
        MemoryTier.SKILL,
        render_tier_payload(MemoryTier.SKILL, new_skill),
    )
    assert failed["event_type"] == "MemoryWriteFailed"
    assert failed["committed_memory_ids"] == [new_memory_id]
    assert failed["replayed_memory_ids"] == [replay.id]
    assert repository.get(new_memory_id) is not None
    assert "super-secret" not in json.dumps(events.events)


def test_duplicate_write_is_accepted_only_as_safe_stable_id_replay(tmp_path: Path) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    payload = TipPayload(guidance="Verify identity before refund")
    existing = repository.add(
        tier="tip",
        tier_schema_version=2,
        payload=payload.model_dump(mode="json"),
        content="  Guidance:   Verify identity before refund  ",
        source_task_ids=("provenance-task-923",),
        created_round=1,
    )
    events = EventCollector()
    environment = FakeEnvironment(_reset(), [_terminal_step()])
    policy = ScriptedLifecyclePolicy(
        [
            '{"memory_ids":[]}',
            '{"action":"finish"}',
            _write_output(_tip_write("Verify identity before refund")),
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


def test_cleanup_base_exception_is_a_note_on_the_primary_exception(
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events = EventCollector()
    environment = FakeEnvironment(
        _reset(),
        close_error=KeyboardInterrupt("cleanup interrupted"),
    )
    policy = ScriptedLifecyclePolicy([TimeoutError("policy timeout")])

    with pytest.raises(TimeoutError, match="policy timeout") as error:
        _run(repository=repository, environment=environment, policy=policy, events=events)

    assert environment.close_calls == 1
    assert any("cleanup interrupted" in note for note in error.value.__notes__)


@pytest.mark.parametrize("field", ["retrieve_top_k", "max_episode_steps"])
def test_config_requires_positive_limits(field: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        FastLoopConfig(**{field: 0})
