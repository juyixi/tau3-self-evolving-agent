from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tau3_retail_evolver.envs.runtime import Tau2RunDomainRuntime
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.runner import FastLoopConfig, LifecycleResponse
from tau3_retail_evolver.fast_loop.tau2_run_domain import (
    _finalize_simulation,
    run_tau2_fast_loop_batch,
)
from tau3_retail_evolver.memory.repository import MemoryRepository


class _Message:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)
        self.content = kwargs.get("content")
        self.tool_calls = kwargs.get("tool_calls")
        self.raw_data = kwargs.get("raw_data")


class _AssistantMessage(_Message):
    pass


class _UserMessage(_Message):
    pass


class _ToolMessage(_Message):
    pass


class _MultiToolMessage(_Message):
    pass


class _ToolCall:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _HalfDuplexAgent:
    def __init__(self, tools: list[Any], domain_policy: str) -> None:
        self.tools = tools
        self.domain_policy = domain_policy


class _Registry:
    def __init__(self) -> None:
        self._agent_factories: dict[str, Any] = {}

    def register_agent_factory(self, factory: Any, name: str) -> None:
        self._agent_factories[name] = factory


class _TextRunConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _Artifact:
    def __init__(self, reward: float = 1.0) -> None:
        self.reward = reward

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"reward": self.reward, "reward_breakdown": {"db": self.reward}}


class _Simulation:
    def __init__(
        self,
        task_id: str,
        seed: int,
        messages: list[Any],
        *,
        reward: float = 1.0,
        termination_reason: str = "agent_stop",
    ) -> None:
        self.task_id = task_id
        self.seed = seed
        self.messages = messages
        self.reward_info = _Artifact(reward)
        self.termination_reason = SimpleNamespace(value=termination_reason)
        self.info = {}

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"task_id": self.task_id, "seed": self.seed}


@dataclass
class _Writer:
    events: list[dict[str, Any]]

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class _Policy:
    def generate(self, prompt: Any) -> LifecycleResponse:
        assert prompt.kind == "action"
        return LifecycleResponse(
            raw_output='{"action":"###STOP###"}',
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            latency_s=0.01,
            prompt_tokens=3,
            completion_tokens=2,
        )

    def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
        raise AssertionError("repair should not be needed")


class _WritePolicy:
    def __init__(self, expected_polarity: str = "positive") -> None:
        self.expected_polarity = expected_polarity

    def generate(self, prompt: Any) -> LifecycleResponse:
        assert prompt.kind == "write"
        assert prompt.payload["memory_outcome"]["polarity"] == self.expected_polarity
        return LifecycleResponse(
            raw_output=json.dumps(
                {
                    "memories": [
                        {"tier": "tip", "payload": {"guidance": f"Tip {index}"}}
                        for index in range(3)
                    ]
                }
            ),
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            latency_s=0.01,
            prompt_tokens=4,
            completion_tokens=3,
        )

    def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
        raise AssertionError("repair should not be needed")


def test_run_domain_batch_uses_one_concurrent_domain_call_and_preserves_order() -> None:
    registry = _Registry()
    captured: dict[str, Any] = {}

    def run_domain(config: _TextRunConfig) -> Any:
        captured["config"] = config
        factory = registry._agent_factories[config.agent]
        simulations = []
        for index, task_id in enumerate(reversed(config.task_ids)):
            tool = SimpleNamespace(openai_schema={"type": "function", "function": {"name": "x"}})
            agent = factory([tool], "Public airline policy")
            user = _UserMessage(role="user", content=f"request {task_id}")
            state = agent.get_init_state()
            assistant, _state = agent.generate_next_message(user, state)
            simulations.append(
                _Simulation(str(task_id), 100 + index, [user, assistant])
            )
        return SimpleNamespace(simulations=simulations)

    runtime = Tau2RunDomainRuntime(
        run_domain=run_domain,
        text_run_config=_TextRunConfig,
        registry=registry,
        half_duplex_agent=_HalfDuplexAgent,
        assistant_message=_AssistantMessage,
        tool_call=_ToolCall,
        user_message=_UserMessage,
        tool_message=_ToolMessage,
        multi_tool_message=_MultiToolMessage,
    )
    writer = _Writer([])
    context = RunContext(
        run_id="airline-train",
        iteration=1,
        split="train",
        model_revision="model-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=42,
        event_writer=writer,
        mode=RunMode.LEARN,
        task_groups={"1": "airline-v2", "2": "airline-v2"},
        default_task_group="airline-v2:maintenance",
    )

    batch = run_tau2_fast_loop_batch(
        runtime=runtime,
        domain="airline",
        split="train",
        task_ids=("1", "2"),
        run_seed=7,
        max_concurrency=8,
        user_llm="user-model",
        user_llm_args={"temperature": 0.0},
        agent_model="agent-model",
        policy=_Policy(),
        repository=None,
        retriever=None,
        config=FastLoopConfig(
            retrieve_top_k=50,
            max_episode_steps=4,
            memory_enabled=False,
        ),
        context_factory=lambda _task_id, _seed: context,
        task_instruction="Resolve the airline request.",
        write_memory=False,
    )

    assert [episode.result.task_id for episode in batch.episodes] == ["1", "2"]
    assert batch.failures == ()
    assert captured["config"].domain == "airline"
    assert captured["config"].task_split_name == "train"
    assert captured["config"].max_concurrency == 2
    assert captured["config"].task_ids == ["1", "2"]
    assert captured["config"].save_to is None
    assert registry._agent_factories == {}
    assert [
        event["task_group"]
        for event in writer.events
        if event["event_type"] == "EpisodeFinished"
    ] == ["airline-v2", "airline-v2"]
    assert {event["seed"] for event in writer.events} == {42}


def test_run_domain_finalization_applies_latest_memory_write_quotas(
    tmp_path: Path,
) -> None:
    audit = {
        "raw_output_sha256": "a" * 64,
        "repaired_output_sha256": None,
        "parse_failed": False,
        "repair_used": False,
        "fallback_used": False,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0},
        "latency_s": 0.01,
        "prompt_tokens": 2,
        "completion_tokens": 1,
    }
    start = {
        "task_instruction": "Resolve the airline request.",
        "policy": "Public airline policy",
        "tools": [],
        "observation": "user: change my flight",
        "memory_enabled": True,
        "query_hash": "b" * 64,
        "retriever_revision": "embedding@1",
        "candidates": [],
        "selected": [],
        "selected_memory_ids": [],
        "selection_audit": audit,
    }
    assistant = _AssistantMessage(
        role="assistant",
        content="###STOP###",
        raw_data={
            "tau3_fast_loop": {
                "schema_version": 1,
                "turn": 0,
                "observation": "user: change my flight",
                "action": "###STOP###",
                "action_audit": audit,
                "start": start,
            }
        },
    )
    simulation = _Simulation(
        "1",
        101,
        [_UserMessage(role="user", content="change my flight"), assistant],
    )
    writer = _Writer([])
    context = RunContext(
        run_id="airline-train",
        iteration=1,
        split="train",
        model_revision="model-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=42,
        event_writer=writer,
        mode=RunMode.LEARN,
        task_groups={"1": "airline-v2"},
        default_task_group="airline-v2:maintenance",
    )
    repository = MemoryRepository(tmp_path / "memory")
    runtime = Tau2RunDomainRuntime(
        run_domain=lambda _config: None,
        text_run_config=_TextRunConfig,
        registry=_Registry(),
        half_duplex_agent=_HalfDuplexAgent,
        assistant_message=_AssistantMessage,
        tool_call=_ToolCall,
        user_message=_UserMessage,
        tool_message=_ToolMessage,
        multi_tool_message=_MultiToolMessage,
    )

    result = _finalize_simulation(
        runtime=runtime,
        simulation=simulation,
        policy=_WritePolicy(),
        repository=repository,
        config=FastLoopConfig(
            memory_enabled=True,
            max_new_tips_per_episode=1,
        ),
        context=context,
        write_memory=True,
    )

    assert len(result.written_memory_ids) == 1
    assert len(repository.list()) == 1
    proposed = next(
        event for event in writer.events if event["event_type"] == "MemoryWriteProposed"
    )
    assert proposed["dropped_by_tier"] == {"tip": 2}
    assert proposed["outcome_class"] == "success"
    assert proposed["polarity"] == "positive"


def test_run_domain_finalization_marks_failed_memory_as_caution(
    tmp_path: Path,
) -> None:
    audit = {
        "raw_output_sha256": "a" * 64,
        "repaired_output_sha256": None,
        "parse_failed": False,
        "repair_used": False,
        "fallback_used": False,
        "sampling_params": {"temperature": 0.0, "top_p": 1.0},
        "latency_s": 0.01,
        "prompt_tokens": 2,
        "completion_tokens": 1,
    }
    start = {
        "task_instruction": "Resolve the airline request.",
        "policy": "Public airline policy",
        "tools": [],
        "observation": "user: change my flight",
        "memory_enabled": True,
        "query_hash": "b" * 64,
        "retriever_revision": "embedding@1",
        "candidates": [],
        "selected": [],
        "selected_memory_ids": [],
        "selection_audit": audit,
    }
    assistant = _AssistantMessage(
        role="assistant",
        content="###STOP###",
        raw_data={
            "tau3_fast_loop": {
                "schema_version": 1,
                "turn": 0,
                "observation": "user: change my flight",
                "action": "###STOP###",
                "action_audit": audit,
                "start": start,
            }
        },
    )
    simulation = _Simulation(
        "1",
        101,
        [_UserMessage(role="user", content="change my flight"), assistant],
        reward=0.0,
    )
    writer = _Writer([])
    context = RunContext(
        run_id="airline-train",
        iteration=1,
        split="train",
        model_revision="model-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=42,
        event_writer=writer,
        mode=RunMode.LEARN,
        task_groups={"1": "airline-v2"},
        default_task_group="airline-v2:maintenance",
    )
    repository = MemoryRepository(tmp_path / "memory")
    runtime = Tau2RunDomainRuntime(
        run_domain=lambda _config: None,
        text_run_config=_TextRunConfig,
        registry=_Registry(),
        half_duplex_agent=_HalfDuplexAgent,
        assistant_message=_AssistantMessage,
        tool_call=_ToolCall,
        user_message=_UserMessage,
        tool_message=_ToolMessage,
        multi_tool_message=_MultiToolMessage,
    )

    result = _finalize_simulation(
        runtime=runtime,
        simulation=simulation,
        policy=_WritePolicy(expected_polarity="caution"),
        repository=repository,
        config=FastLoopConfig(memory_enabled=True, max_new_tips_per_episode=1),
        context=context,
        write_memory=True,
    )

    assert len(result.written_memory_ids) == 1
    memory = repository.list()[0]
    assert memory.metadata["source_final_reward"] == 0.0
    assert memory.metadata["outcome_class"] == "task_failure"
    assert memory.metadata["polarity"] == "caution"
    proposed = next(
        event for event in writer.events if event["event_type"] == "MemoryWriteProposed"
    )
    assert proposed["outcome_class"] == "task_failure"
    assert proposed["polarity"] == "caution"
