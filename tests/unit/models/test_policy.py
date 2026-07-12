from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pytest

from tau3_retail_evolver.models.openai_compatible import (
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


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


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
    )

    assert requests == [
        (
            "https://qwen.example/v1/chat/completions",
            {
                "Authorization": "Bearer test-api-key",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            b'{"max_tokens":256,"messages":[{"content":"Hello","role":"user"}],"model":"Qwen/Qwen3.5-9B","presence_penalty":0.2,"temperature":1.0,"tools":[],"top_p":0.95}',
        )
    ]
    assert response == {"choices": [{"message": {"role": "assistant", "content": "Done."}}]}


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
