from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from tau3_evolver.benchmarks.tau2.agent import (
    create_tau3_agent_factory,
    filter_tool_memory_candidates,
)
from tau3_evolver.benchmarks.tau2.runtime import Tau2RuntimeBinding
from tau3_evolver.fast_loop.contracts import LifecycleResponse
from tau3_evolver.fast_loop.settings import FastLoopConfig
from tau3_evolver.memory.retrieval import MemoryCandidate
from tau3_evolver.memory.tier_contracts import ToolCallExample, ToolPayload, render_tier_payload
from tau3_evolver.memory.types import MemoryItem, MemoryTier


class _BaseAgent:
    def __init__(self, tools: list[Any], domain_policy: str) -> None:
        self.tools = tools
        self.domain_policy = domain_policy


class _Message:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)
        self.role = values.get("role", "message")
        self.content = values.get("content")
        self.tool_calls = values.get("tool_calls", [])


class _MultiToolMessage:
    def __init__(self, tool_messages: list[Any]) -> None:
        self.tool_messages = tool_messages


@dataclass
class _Tool:
    openai_schema: dict[str, Any]


class _Policy:
    def generate(self, prompt: Any) -> LifecycleResponse:
        assert prompt.kind == "action"
        return LifecycleResponse(
            raw_output='{"action":"lookup_order(order_id=\'1\')"}',
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            latency_s=0.1,
        )

    def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
        raise AssertionError((prompt, raw_output, error))


class _RepairPolicy:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def generate(self, prompt: Any) -> LifecycleResponse:
        assert prompt.kind == "action"
        return LifecycleResponse(
            raw_output='{"action":"unknown_tool(value=1)"}',
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            latency_s=0.1,
        )

    def repair(
        self,
        prompt: Any,
        raw_output: str,
        error: str,
    ) -> LifecycleResponse:
        assert prompt.kind == "action"
        assert raw_output == '{"action":"unknown_tool(value=1)"}'
        self.errors.append(error)
        return LifecycleResponse(
            raw_output='{"action":"lookup_order(order_id=\'1\')"}',
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            latency_s=0.1,
        )


def _runtime() -> Tau2RuntimeBinding:
    return Tau2RuntimeBinding(
        source_root=SimpleNamespace(),
        package_version=None,
        git_commit=None,
        registry=SimpleNamespace(),
        run_domain=lambda config: config,
        text_run_config_type=dict,
        task_type=object,
        half_duplex_agent_type=_BaseAgent,
        assistant_message_type=_Message,
        tool_call_type=_Message,
        tool_message_type=_Message,
        multi_tool_message_type=_MultiToolMessage,
    )


def _tools() -> list[_Tool]:
    return [
        _Tool(
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up an order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"],
                    },
                },
            }
        )
    ]


def test_factory_creates_independent_agent_state_and_tool_schema() -> None:
    factory = create_tau3_agent_factory(
        runtime=_runtime(),
        benchmark="retail",
        policy=_Policy(),
        repository=None,
        retriever=None,
        config=FastLoopConfig(memory_enabled=False),
        memory_source_namespace=None,
        cross_domain_memory=False,
    )
    first = factory(_tools(), "Retail policy")
    second = factory(_tools(), "Retail policy")
    first._public_tools[0]["function"]["description"] = "changed"

    first_state = first.get_init_state()
    second_state = second.get_init_state()
    assistant, updated = first.generate_next_message(
        _Message(role="user", content="Find order 1."), first_state
    )

    assert updated is first_state
    assert updated.turn == 1
    assert second_state.turn == 0
    assert second._public_tools[0]["function"]["description"] == "Look up an order."
    assert assistant.content is None
    assert assistant.tool_calls[0].name == "lookup_order"
    assert assistant.tool_calls[0].arguments == {"order_id": "1"}
    assert assistant.raw_data["tau3_agent"]["start"]["memory_enabled"] is False


def test_tau2_adapter_validates_actions_and_repairs_unknown_tools() -> None:
    policy = _RepairPolicy()
    factory = create_tau3_agent_factory(
        runtime=_runtime(),
        benchmark="retail",
        policy=policy,
        repository=None,
        retriever=None,
        config=FastLoopConfig(memory_enabled=False),
        memory_source_namespace=None,
        cross_domain_memory=False,
    )
    agent = factory(_tools(), "Retail policy")

    assistant, _ = agent.generate_next_message(
        _Message(role="user", content="Find order 1."),
        agent.get_init_state(),
    )

    assert len(policy.errors) == 1
    assert "not available" in policy.errors[0]
    assert assistant.tool_calls[0].name == "lookup_order"
    assert assistant.tool_calls[0].arguments == {"order_id": "1"}


def _tool_candidate(tool_name: str = "lookup_order") -> MemoryCandidate:
    payload = ToolPayload(
        tool_name=tool_name,
        purpose="Read the authoritative source record.",
        method="Call the tool once with the exact identifier.",
        preconditions=("The user supplied an identifier.",),
        argument_rules={"order_id": "Use the exact order identifier."},
        expected_effect="The source record is returned.",
        example=ToolCallExample(
            name=tool_name,
            arguments={"order_id": "123"},
        ),
    )
    item = MemoryItem(
        id=f"memory-{tool_name}",
        tier=MemoryTier.TOOL,
        tier_schema_version=2,
        payload=payload.model_dump(mode="json"),
        content=render_tier_payload(MemoryTier.TOOL, payload),
        retrieval_text=f"{tool_name} method",
        source_task_ids=("retail-task",),
        created_round=1,
        updated_round=1,
    )
    return MemoryCandidate(
        memory_id=item.id,
        memory_version=item.version,
        tier=item.tier,
        rank=1,
        similarity=0.9,
        retriever_revision="embedding-v1",
        query_hash="a" * 64,
        item=item,
    )


def test_cross_domain_tool_memory_with_unavailable_tool_is_filtered() -> None:
    compatible, filtered = filter_tool_memory_candidates(
        (_tool_candidate(),),
        (
            {
                "type": "function",
                "function": {
                    "name": "lookup_booking",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
    )

    assert compatible == []
    assert filtered == [
        {
            "memory_id": "memory-lookup_order",
            "source_tool": "lookup_order",
            "reason": "unavailable_tool",
        }
    ]


def test_cross_domain_tool_memory_with_incompatible_schema_is_filtered() -> None:
    compatible, filtered = filter_tool_memory_candidates(
        (_tool_candidate(),),
        (
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "parameters": {
                        "type": "object",
                        "properties": {"booking_id": {"type": "string"}},
                        "required": ["booking_id"],
                    },
                },
            },
        ),
    )

    assert compatible == []
    assert filtered[0]["reason"] == "incompatible_schema"
