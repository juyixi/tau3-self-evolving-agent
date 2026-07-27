from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pytest

from tau3_retail_evolver.models import openai_compatible
from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.decisions import (
    ActionDecision,
    MaintenanceDecision,
    SelectionDecision,
    WriteDecision,
    parse_decision,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.prompts import (
    LifecyclePrompt,
    build_action_prompt,
    build_maintenance_prompt,
    build_selection_prompt,
    build_write_prompt,
)
from tau3_retail_evolver.fast_loop.runner import FastLoopConfig, run_fast_loop_episode
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever
from tau3_retail_evolver.models.openai_compatible import (
    OpenAICompatibleFastLoopPolicy,
    OpenAICompatibleHttpClient,
    OpenAICompatibleQwenPolicy,
)
from tau3_retail_evolver.models.policy import DecisionRequest, DecisionResponse
from tests.support.policy import ScriptedPolicy


@dataclass
class FakeCompletion:
    content: str
    tool_call: str | None = None


class FakeClient:
    def __init__(self, completion: object) -> None:
        self.completion = completion
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.completion


class ScriptedClient:
    def __init__(self, completions: list[object]) -> None:
        self.completions = list(completions)
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.completions.pop(0)


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


SELECTION_SYSTEM = (
    'Return exactly one strict JSON object matching SelectionDecision: '
    '{"memory_ids":[...]}. Use only candidate IDs from the user payload. '
    "Positive memories describe proven strategies. Caution memories describe failure "
    "reflections and must only be selected when their warning is relevant. "
    'Do not use tools or include any other text.'
)
ACTION_SYSTEM = (
    "Choose exactly one Tau2 action using the official policy and public context. "
    "Use at most one provided tool call, or return a valid Tau2 text action. "
    "Treat positive memories as reusable strategies. Treat caution memories only as "
    "behavior to avoid or correct; never imitate their failed behavior or treat them "
    "as successful completion conditions. "
    "Do not include hidden data."
)
WRITE_SYSTEM = (
    "Return exactly one strict JSON object matching WriteDecision. Each memory must use "
    'one tier-specific payload: "tip" is one atomic condition or rule; "skill" is one '
    'reusable goal with at least two ordered steps; "tool" is usage knowledge for exactly '
    'one tool present in the supplied tool schemas; "trajectory" is one concrete observed '
    "episode case. Split mixed lessons into separate memories. Use this shape: "
    '{"memories":[{"tier":"tip","payload":{"condition":"optional condition",'
    '"guidance":"one atomic rule","rationale":"optional rationale","scope":[]},'
    '"retrieval_text":"optional retrieval query","metadata":{}}]}. '
    'Use {"memories":[]} when no durable lesson passes a tier definition. Content is '
    "rendered by the runtime and must not be returned. Attribution fields are not allowed. "
    "Follow memory_outcome exactly: successful outcomes may write positive memories; "
    "task failures may write only caution tips and failure trajectories. Failure memories "
    "must state what to avoid and the corrective behavior, never a success condition. "
    'When trajectory_format is "final_observation_plus_actions_v1", observation contains '
    "the complete cumulative transcript and trajectory contains its ordered action and "
    "outcome metadata without repeated observations. "
    "Do not use tools, Markdown fences, or include any other text."
)
MAINTENANCE_SYSTEM = (
    "You are the Memory maintenance controller. Return exactly one strict JSON object "
    'matching MaintenanceDecision: {"reviews":[...],"commands":[...]}. '
    "Apply these rules: "
    "1. Review priority candidates first. Every reviewed Memory ID appears exactly once "
    "with disposition keep, merge, or retire. "
    "2. Use keep when evidence is insufficient. Do not change a useful distinct Memory. "
    "3. Merge only redundant Memories from the same tier. Put all merged IDs in one merge "
    "command and mark those reviews as merge. Never also delete a merge source. "
    "4. Retire only clearly obsolete, incorrect, or redundant Memories. Mark deleted IDs "
    "as retire and provide a concrete reason. "
    "5. Commands may reference only IDs present in diagnostics. Do not repeat an ID across "
    "commands. Do not mix lookup commands with merge or delete commands. "
    "6. The runtime owns updated_round and typed payload fields; omit them. Return only "
    "semantic merge content or delete reasons. "
    "7. When requires_tip_reduction is true, reduce redundant tips toward tip_capacity. "
    'Otherwise commands may be empty. Use {"reviews":[],"commands":[]} only when there '
    "is genuinely nothing safe to review. "
    "Use only the supplied diagnostics and command schemas. Do not call tools, use Markdown, "
    "or include any text outside the JSON object."
)


def _decision_response_format(kind: str) -> dict[str, Any]:
    decision_type = {
        "selection": SelectionDecision,
        "write": WriteDecision,
        "maintenance": MaintenanceDecision,
    }[kind]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{kind}_decision",
            "schema": decision_type.model_json_schema(),
            "strict": True,
        },
    }


def _public_context() -> dict[str, object]:
    return {
        "task_instruction": "Help the customer update an order.",
        "policy": {"rule": "Verify the order first."},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "find_order",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                },
            }
        ],
        "observation": "Please check order 123.",
        "history": [{"role": "user", "content": "My name is Ada."}],
    }


def _lifecycle_prompts() -> dict[str, LifecyclePrompt]:
    context = _public_context()
    return {
        "selection": build_selection_prompt(
            **context,
            candidates=(
                {"id": "memory-1", "tier": "tip", "content": "Verify first.", "version": 1},
            ),
        ),
        "action": build_action_prompt(
            **context,
            memories=(
                {"id": "memory-1", "tier": "tip", "content": "Verify first.", "version": 1},
            ),
        ),
        "write": build_write_prompt(
            **context,
            trajectory=({"observation": "start", "action": "find_order"},),
            terminal_evaluation={"reward": 1.0},
        ),
        "maintenance": build_maintenance_prompt(
            diagnostics={
                tier: {"items": []}
                for tier in ("trajectory", "tip", "skill", "tool")
            }
        ),
    }


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _completion(content: str | None, *, tool_calls: object = None) -> dict[str, object]:
    message: dict[str, object] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _request() -> DecisionRequest:
    return DecisionRequest(
        observation="Where is order 123?",
        reset_info={
            "policy": "Use tools when needed.",
            "tools": [{"type": "function", "function": {"name": "find_order"}}],
        },
        temperature=1.0,
        top_p=0.95,
    )


def test_generates_text_and_records_sampling_parameters_and_latency() -> None:
    client = FakeClient(FakeCompletion(content="<think>Search first.</think>Your order is on the way."))
    policy = OpenAICompatibleQwenPolicy(
        client=client,
        tool_call_parser=lambda completion: completion.tool_call,
        clock=FakeClock(10.0, 10.25),
    )

    response = policy.generate(_request())

    assert client.calls == [
        {
            "messages": [
                {"role": "system", "content": "Use tools when needed."},
                {"role": "user", "content": "Where is order 123?"},
            ],
            "tools": [{"type": "function", "function": {"name": "find_order"}}],
            "temperature": 1.0,
            "top_p": 0.95,
        }
    ]
    assert response.raw_output == "<think>Search first.</think>Your order is on the way."
    assert response.parsed_action == "Your order is on the way."
    assert response.sampling_params == {"temperature": 1.0, "top_p": 0.95}
    assert response.latency_s == 0.25


def test_sends_public_prior_messages_to_the_completion_client() -> None:
    client = FakeClient(FakeCompletion(content="Your order is on the way."))
    policy = OpenAICompatibleQwenPolicy(client=client, clock=FakeClock(10.0, 10.1))
    request = DecisionRequest(
        observation="What is the latest status?",
        reset_info=_request().reset_info,
        temperature=1.0,
        top_p=0.95,
        history=({"role": "user", "content": "My order is 123."},),
    )

    policy.generate(request)

    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": "My order is 123."},
        {"role": "user", "content": "What is the latest status?"},
    ]


def test_prefers_the_official_qwen_tool_parser_result_over_message_text() -> None:
    client = FakeClient(
        FakeCompletion(content="I will look that up.", tool_call='find_order(order_id="123")')
    )
    parsed_completions: list[FakeCompletion] = []

    def parse_qwen_tool_call(completion: FakeCompletion) -> str | None:
        parsed_completions.append(completion)
        return completion.tool_call

    policy = OpenAICompatibleQwenPolicy(client=client, tool_call_parser=parse_qwen_tool_call, clock=FakeClock(1.0, 1.1))

    response = policy.generate(_request())

    assert parsed_completions == [client.completion]
    assert response.raw_output == "I will look that up."
    assert response.parsed_action == 'find_order(order_id="123")'


def test_default_parser_converts_one_structured_tool_call_and_preserves_raw_message() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "find_order", "arguments": '{"order_id":"123"}'},
            }
        ],
    }
    client = FakeClient({"choices": [{"message": message}]})
    policy = OpenAICompatibleQwenPolicy(client=client, clock=FakeClock(1.0, 1.1))

    response = policy.generate(_request())

    assert response.raw_output == json.dumps(message, sort_keys=True, separators=(",", ":"))
    assert response.parsed_action == '{"arguments":{"order_id":"123"},"name":"find_order"}'


@pytest.mark.parametrize(
    "message",
    (
        {"role": "assistant", "content": "Your order is on the way."},
        {"role": "assistant", "content": "Your order is on the way.", "tool_calls": None},
        {"role": "assistant", "content": "Your order is on the way.", "tool_calls": []},
    ),
)
def test_default_parser_treats_absent_null_or_empty_tool_calls_as_text(
    message: dict[str, Any],
) -> None:
    policy = OpenAICompatibleQwenPolicy(
        client=FakeClient({"choices": [{"message": message}]}), clock=FakeClock(1.0, 1.1)
    )

    response = policy.generate(_request())

    assert response.raw_output == "Your order is on the way."
    assert response.parsed_action == "Your order is on the way."


@pytest.mark.parametrize("tool_calls", (None, []))
def test_text_that_looks_like_malformed_tool_json_still_uses_the_codec(
    tool_calls: list[object] | None,
) -> None:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": '{"name":"find_order","arguments":"order_id=123"}',
        "tool_calls": tool_calls,
    }
    policy = OpenAICompatibleQwenPolicy(
        client=FakeClient({"choices": [{"message": message}]}), clock=FakeClock(1.0, 1.1)
    )

    with pytest.raises(ValueError, match="tool arguments"):
        policy.generate(_request())


def test_default_parser_prioritizes_one_structured_tool_call_when_content_is_present() -> None:
    message = {
        "role": "assistant",
        "content": "I will look that up.",
        "reasoning": "The order must be retrieved before I answer.",
        "tool_calls": [
            {"type": "function", "function": {"name": "find_order", "arguments": "{}"}}
        ],
    }
    policy = OpenAICompatibleQwenPolicy(
        client=FakeClient({"choices": [{"message": message}]}), clock=FakeClock(1.0, 1.1)
    )

    response = policy.generate(_request())

    assert response.parsed_action == '{"arguments":{},"name":"find_order"}'
    assert json.loads(response.raw_output) == message


def test_default_parser_rejects_multiple_structured_tool_calls() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"type": "function", "function": {"name": "find_order", "arguments": "{}"}},
            {"type": "function", "function": {"name": "find_order", "arguments": "{}"}},
        ],
    }
    policy = OpenAICompatibleQwenPolicy(
        client=FakeClient({"choices": [{"message": message}]}), clock=FakeClock(1.0, 1.1)
    )

    with pytest.raises(ValueError, match="exactly one"):
        policy.generate(_request())


@pytest.mark.parametrize(
    ("kind", "system_instruction", "output"),
    (
        ("selection", SELECTION_SYSTEM, '{"memory_ids":["memory-1"]}'),
        ("write", WRITE_SYSTEM, '{"memories":[]}'),
        ("maintenance", MAINTENANCE_SYSTEM, '{"reviews":[],"commands":[]}'),
    ),
)
def test_fast_loop_non_action_requests_use_exact_public_json_and_no_tools(
    kind: str,
    system_instruction: str,
    output: str,
) -> None:
    prompt = _lifecycle_prompts()[kind]
    client = FakeClient(_completion(output))
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=0.7,
        top_p=0.9,
        clock=FakeClock(3.0, 3.2),
    )

    response = policy.generate(prompt)

    public_request = dict(prompt.payload)
    if kind == "maintenance":
        public_request["command_schemas"] = list(prompt.command_schemas)
    expected_call = {
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": _canonical(public_request)},
        ],
        "tools": [],
        "temperature": 0.7,
        "top_p": 0.9,
        "response_format": _decision_response_format(kind),
    }
    if kind == "maintenance":
        expected_call["request_generation_settings"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    assert client.calls == [expected_call]
    assert response.raw_output == output
    assert response.sampling_params == {"temperature": 0.7, "top_p": 0.9}
    assert response.latency_s == pytest.approx(0.2)
    assert "task_id" not in client.calls[0]["messages"][1]["content"]
    assert "attribution" not in client.calls[0]["messages"][1]["content"]


def test_fast_loop_action_passes_exact_official_tools_and_converts_qwen_tool_call() -> None:
    prompt = _lifecycle_prompts()["action"]
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "find_order", "arguments": '{"order_id":"123"}'},
        }
    ]
    client = FakeClient(_completion(None, tool_calls=tool_calls))
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=0.6,
        top_p=0.8,
        clock=FakeClock(5.0, 4.5),
    )

    response = policy.generate(prompt)

    assert client.calls == [
        {
            "messages": [
                {"role": "system", "content": ACTION_SYSTEM},
                {"role": "user", "content": _canonical(prompt.payload)},
            ],
            "tools": prompt.payload["tools"],
            "temperature": 0.6,
            "top_p": 0.8,
            "response_format": None,
        }
    ]
    assert response.raw_output == (
        '{"action":"{\\"arguments\\":{\\"order_id\\":\\"123\\"},'
        '\\"name\\":\\"find_order\\"}"}'
    )
    assert response.sampling_params == {"temperature": 0.6, "top_p": 0.8}
    assert response.latency_s == 0.0


def test_fast_loop_policy_preserves_openai_token_usage() -> None:
    completion = _completion('{"memory_ids":[]}')
    completion["usage"] = {
        "prompt_tokens": 120,
        "completion_tokens": 8,
        "total_tokens": 128,
    }
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient(completion),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()["selection"])

    assert response.prompt_tokens == 120
    assert response.completion_tokens == 8


def test_fast_loop_action_converts_valid_assistant_text_to_canonical_action_json() -> None:
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient(_completion("I can help with that.")),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()["action"])

    assert response.raw_output == '{"action":"I can help with that."}'


def test_fast_loop_malformed_action_is_returned_invalid_for_runner_repair() -> None:
    malformed = '{"name":"find_order","arguments":"order_id=123"}'
    client = FakeClient(_completion(malformed))
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()["action"])
    parsed = parse_decision(response.raw_output, ActionDecision)

    assert json.loads(response.raw_output) == {"invalid_action_output": malformed}
    assert parsed.decision is None
    assert parsed.error is not None
    assert len(client.calls) == 1


def test_fast_loop_codec_failure_cannot_bypass_repair_with_action_decision_shell() -> None:
    nested_invalid_action = _canonical(
        {"action": _canonical({"name": "unknown", "arguments": {}})}
    )
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient(_completion(nested_invalid_action)),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()["action"])

    assert json.loads(response.raw_output) == {
        "invalid_action_output": nested_invalid_action
    }
    assert parse_decision(response.raw_output, ActionDecision).decision is None


@pytest.mark.parametrize(
    "message",
    (
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": None, "tool_calls": None},
        {"role": "assistant", "content": None, "tool_calls": []},
        {
            "role": "assistant",
            "content": '{"action":"mixed content"}',
            "tool_calls": {"function": {"name": "find_order", "arguments": "{}"}},
        },
        {
            "role": "assistant",
            "content": '{"action":"mixed content"}',
            "tool_calls": "malformed",
        },
    ),
)
def test_fast_loop_action_extraction_failures_return_parser_invalid_wrapper(
    message: dict[str, object],
) -> None:
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient({"choices": [{"message": message}]}),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()["action"])

    preserved = json.loads(response.raw_output)["invalid_action_output"]
    tool_calls = message.get("tool_calls")
    if tool_calls not in (None, []):
        assert preserved == _canonical(message)
    else:
        assert preserved == _canonical({"choices": [{"message": message}]})
    assert parse_decision(response.raw_output, ActionDecision).decision is None


def test_fast_loop_unserializable_action_completion_uses_bounded_safe_output() -> None:
    class LongRepresentation:
        def __repr__(self) -> str:
            return "invalid-structured-output-" + ("x" * 10_000)

    completion = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [LongRepresentation()],
                }
            }
        ]
    }
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient(completion),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()["action"])

    preserved = json.loads(response.raw_output)["invalid_action_output"]
    assert preserved.startswith("{'choices':")
    assert "invalid-structured-output" in preserved
    assert len(preserved) <= 4_096
    assert parse_decision(response.raw_output, ActionDecision).decision is None


@pytest.mark.parametrize("kind", ("selection", "write", "maintenance"))
def test_fast_loop_non_action_rejects_structured_tool_calls(kind: str) -> None:
    message = {
        "role": "assistant",
        "content": '{"commands":[]}',
        "tool_calls": [
            {"type": "function", "function": {"name": "hidden_tool", "arguments": "{}"}}
        ],
    }
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient({"choices": [{"message": message}]}),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()[kind])

    assert response.raw_output == _canonical(message)


@pytest.mark.parametrize("kind", ("selection", "write", "maintenance"))
@pytest.mark.parametrize(
    "tool_calls",
    (
        {"function": {"name": "hidden_tool", "arguments": "{}"}},
        "malformed-tool-call",
        7,
    ),
)
def test_fast_loop_non_action_rejects_any_nonempty_tool_calls_mixed_with_valid_content(
    kind: str,
    tool_calls: object,
) -> None:
    valid_content = {
        "selection": '{"memory_ids":[]}',
        "write": '{"memories":[]}',
        "maintenance": '{"commands":[]}',
    }[kind]
    message = {
        "role": "assistant",
        "content": valid_content,
        "tool_calls": tool_calls,
    }
    policy = OpenAICompatibleFastLoopPolicy(
        client=FakeClient({"choices": [{"message": message}]}),
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.generate(_lifecycle_prompts()[kind])

    assert response.raw_output == _canonical(message)
    assert response.raw_output != valid_content


def test_fast_loop_repair_sends_only_public_prompt_invalid_output_and_error() -> None:
    prompt = _lifecycle_prompts()["selection"]
    client = FakeClient(_completion('{"memory_ids":["memory-1"]}'))
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.repair(prompt, "not json", "invalid JSON: secret-free error")

    repair_request = {
        "prompt": prompt.model_dump(mode="json"),
        "invalid_output": "not json",
        "validation_error": "invalid JSON: secret-free error",
    }
    assert client.calls == [
        {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Repair the invalid selection response. " + SELECTION_SYSTEM
                    ),
                },
                {"role": "user", "content": _canonical(repair_request)},
            ],
            "tools": [],
            "temperature": 0.7,
            "top_p": 0.9,
            "response_format": _decision_response_format("selection"),
        }
    ]
    assert response.raw_output == '{"memory_ids":["memory-1"]}'
    serialized = client.calls[0]["messages"][1]["content"]
    assert "task_id" not in serialized
    assert "run_id" not in serialized
    assert "attribution" not in serialized


def test_fast_loop_action_repair_retains_tools_and_converts_structured_action() -> None:
    prompt = _lifecycle_prompts()["action"]
    client = FakeClient(
        _completion(
            None,
            tool_calls=[
                {
                    "type": "function",
                    "function": {"name": "find_order", "arguments": {"order_id": "123"}},
                }
            ],
        )
    )
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=0.7,
        top_p=0.9,
    )

    response = policy.repair(prompt, "bad action", "invalid action")

    assert client.calls[0]["tools"] is prompt.payload["tools"]
    assert response.raw_output == (
        '{"action":"{\\"arguments\\":{\\"order_id\\":\\"123\\"},'
        '\\"name\\":\\"find_order\\"}"}'
    )


def test_fast_loop_runner_repairs_codec_invalid_action_before_environment_step(
    tmp_path: Any,
) -> None:
    invalid_action = _canonical(
        {"action": _canonical({"name": "unknown", "arguments": {}})}
    )
    repaired_tool_call = [
        {
            "type": "function",
            "function": {"name": "find_order", "arguments": {"order_id": "123"}},
        }
    ]
    client = ScriptedClient(
        [
            _completion('{"memory_ids":[]}'),
            _completion(invalid_action),
            _completion(None, tool_calls=repaired_tool_call),
            _completion('{"memories":[]}'),
        ]
    )
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=0.7,
        top_p=0.9,
    )

    class OneStepEnvironment:
        def __init__(self) -> None:
            self.actions: list[str] = []

        def reset(self, *, seed: int) -> ResetResult:
            return ResetResult(
                observation="Please find order 123.",
                info={
                    "policy": {"rule": "Verify the order first."},
                    "tools": _public_context()["tools"],
                },
            )

        def step(self, action: str) -> StepResult:
            self.actions.append(action)
            return StepResult(
                observation="Order found.",
                reward=1.0,
                done=True,
                terminated=True,
                truncated=False,
                info={
                    "reward_info": '{"reward":1.0}',
                    "simulation_run": '{"status":"complete"}',
                },
            )

        def close(self) -> None:
            pass

    class EmptyEmbeddingProvider:
        model_revision = "empty@1"
        dimension = 2

        def embed(self, text: str) -> tuple[float, ...]:
            return (1.0, 0.0)

        def embed_batch(self, texts: list[str]) -> list[tuple[float, ...]]:
            return [(1.0, 0.0) for _ in texts]

    class EventCollector:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []

        def append(self, event: dict[str, Any]) -> None:
            self.events.append(event)

    environment = OneStepEnvironment()
    events = EventCollector()
    run_fast_loop_episode(
        task_id="runner-repair-test",
        task_instruction="Help the customer find an order.",
        environment=environment,
        policy=policy,
        repository=MemoryRepository(tmp_path / "memory"),
        retriever=Retriever(EmptyEmbeddingProvider()),
        config=FastLoopConfig(max_episode_steps=1),
        context=RunContext(
            run_id="task4a-review",
            iteration=1,
            split="train",
            model_revision="Qwen/Qwen3.5-9B",
            adapter_revision=None,
            memory_snapshot_id=None,
            seed=7,
            event_writer=events,
            mode=RunMode.LEARN,
        ),
    )

    expected_action = '{"arguments":{"order_id":"123"},"name":"find_order"}'
    assert environment.actions == [expected_action]
    action_calls = [call for call in client.calls if call["tools"]]
    assert len(action_calls) == 2
    repair_request = json.loads(action_calls[1]["messages"][1]["content"])
    invalid_wrapper = json.loads(repair_request["invalid_output"])
    assert invalid_wrapper == {"invalid_action_output": invalid_action}
    assert "unknown" in repair_request["invalid_output"]
    assert len(client.calls) == 4
    assert client.completions == []
    serialized_events = _canonical(events.events)
    assert "unknown" not in serialized_events
    assert "invalid_action_output" not in serialized_events


@pytest.mark.parametrize(
    ("temperature", "top_p"),
    ((float("nan"), 0.9), (0.7, float("inf")), (float("-inf"), 0.9)),
)
def test_fast_loop_rejects_nonfinite_sampling_values(
    temperature: float, top_p: float
) -> None:
    with pytest.raises(ValueError, match="finite"):
        OpenAICompatibleFastLoopPolicy(
            client=FakeClient(_completion("unused")),
            temperature=temperature,
            top_p=top_p,
        )


def test_fast_loop_sanitizes_client_exceptions_and_repr() -> None:
    secret = "api-key-and-raw-transport-body"

    class FailingClient:
        def create_chat_completion(self, **kwargs: Any) -> object:
            raise RuntimeError(secret)

    policy = OpenAICompatibleFastLoopPolicy(
        client=FailingClient(),
        temperature=0.7,
        top_p=0.9,
    )

    with pytest.raises(RuntimeError, match="fast-loop policy request failed") as error:
        policy.generate(_lifecycle_prompts()["selection"])

    assert secret not in repr(policy)
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


def test_http_client_posts_openai_compatible_request_with_generation_settings() -> None:
    requests: list[tuple[str, dict[str, str], bytes]] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        requests.append((url, headers, body))
        return 200, b'{"choices":[{"message":{"role":"assistant","content":"Done."}}]}'

    client = OpenAICompatibleHttpClient(
        base_url="https://qwen.example/v1/",
        model="Qwen/Qwen3.5-9B",
        api_key="test-api-key",
        max_tokens=256,
        generation_settings={"presence_penalty": 0.2},
        transport=transport,
    )

    response = client.create_chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        temperature=1.0,
        top_p=0.95,
        response_format={"type": "json_object"},
    )

    assert requests == [
        (
            "https://qwen.example/v1/chat/completions",
            {
                "Authorization": "Bearer test-api-key",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            b'{"max_tokens":256,"messages":[{"content":"Hello","role":"user"}],"model":"Qwen/Qwen3.5-9B","presence_penalty":0.2,"response_format":{"type":"json_object"},"temperature":1.0,"top_p":0.95}',
        )
    ]
    assert response == {"choices": [{"message": {"role": "assistant", "content": "Done."}}]}


def test_http_client_applies_request_generation_settings_last() -> None:
    requests: list[bytes] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        requests.append(body)
        return 200, b'{"choices":[{"message":{"role":"assistant","content":"Done."}}]}'

    client = OpenAICompatibleHttpClient(
        base_url="https://qwen.example/v1",
        model="Qwen/Qwen3.5-9B",
        api_key="test-api-key",
        generation_settings={
            "chat_template_kwargs": {"enable_thinking": True},
            "presence_penalty": 0.2,
        },
        transport=transport,
    )

    client.create_chat_completion(
        messages=[],
        tools=[],
        temperature=1.0,
        top_p=0.95,
        request_generation_settings={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )

    payload = json.loads(requests[0])
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["presence_penalty"] == 0.2


def test_http_client_includes_nonempty_tool_schemas() -> None:
    requests: list[bytes] = []

    def transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        requests.append(body)
        return 200, b'{"choices":[{"message":{"role":"assistant","content":"Done."}}]}'

    client = OpenAICompatibleHttpClient(
        base_url="https://qwen.example/v1",
        model="Qwen/Qwen3.5-9B",
        api_key="test-api-key",
        transport=transport,
    )
    tool = {"type": "function", "function": {"name": "lookup_order"}}

    client.create_chat_completion(
        messages=[],
        tools=[tool],
        temperature=1.0,
        top_p=0.95,
    )

    assert json.loads(requests[0])["tools"] == [tool]


def test_http_client_includes_truncated_error_body_for_non_success_status() -> None:
    client = OpenAICompatibleHttpClient(
        base_url="https://qwen.example/v1",
        model="Qwen/Qwen3.5-9B",
        api_key="test-api-key",
        transport=lambda _url, _headers, _body: (400, b"invalid request"),
    )

    with pytest.raises(RuntimeError, match="HTTP 400: invalid request"):
        client.create_chat_completion(
            messages=[],
            tools=[],
            temperature=1.0,
            top_p=0.95,
        )


def test_http_client_default_transport_forwards_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_timeouts: list[float] = []

    class Response:
        status = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"Done."}}]}'

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        observed_timeouts.append(timeout)
        return Response()

    monkeypatch.setattr(openai_compatible, "urlopen", fake_urlopen)
    client = OpenAICompatibleHttpClient(
        base_url="https://qwen.example/v1",
        model="Qwen/Qwen3.5-9B",
        api_key="test-api-key",
    )

    client.create_chat_completion(messages=[], tools=[], temperature=1.0, top_p=0.95)

    assert observed_timeouts == [120.0]


@pytest.mark.parametrize(
    "request_timeout_s",
    [0.0, -1.0, float("inf"), float("nan"), True, "120"],
)
def test_http_client_rejects_nonpositive_or_nonfinite_timeout(
    request_timeout_s: object,
) -> None:
    with pytest.raises(ValueError, match="request timeout"):
        OpenAICompatibleHttpClient(
            base_url="https://qwen.example/v1",
            model="Qwen/Qwen3.5-9B",
            api_key="test-api-key",
            request_timeout_s=request_timeout_s,  # type: ignore[arg-type]
        )


def test_http_client_does_not_expose_api_key_in_repr_or_errors() -> None:
    api_key = "super-secret-key"

    def failing_transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        raise OSError(api_key)

    client = OpenAICompatibleHttpClient(
        base_url="https://qwen.example/v1",
        model="Qwen/Qwen3.5-9B",
        api_key=api_key,
        transport=failing_transport,
    )

    with pytest.raises(RuntimeError) as error:
        client.create_chat_completion(messages=[], tools=[], temperature=1.0, top_p=0.95)

    assert api_key not in repr(client)
    assert api_key not in str(error.value)
    assert error.value.__cause__ is None


def test_scripted_policy_returns_supplied_responses_in_order() -> None:
    responses = (
        DecisionResponse("first", "first", {"temperature": 1.0, "top_p": 0.95}, 0.0),
        DecisionResponse("second", "second", {"temperature": 1.0, "top_p": 0.95}, 0.0),
    )
    policy = ScriptedPolicy(responses)

    assert policy.generate(_request()) == responses[0]
    assert policy.generate(_request()) == responses[1]
    assert policy.requests == (_request(), _request())
