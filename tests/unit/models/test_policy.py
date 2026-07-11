from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tau3_retail_evolver.models.openai_compatible import OpenAICompatibleQwenPolicy
from tau3_retail_evolver.models.policy import DecisionRequest


@dataclass
class FakeCompletion:
    content: str
    tool_call: str | None = None


class FakeClient:
    def __init__(self, completion: FakeCompletion) -> None:
        self.completion = completion
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> FakeCompletion:
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


def test_prefers_the_official_qwen_tool_parser_result_over_message_text() -> None:
    client = FakeClient(
        FakeCompletion(content="I will look that up.", tool_call='find_order(order_id="123")')
    )
    parsed_completions: list[FakeCompletion] = []

    def parse_qwen_tool_call(completion: FakeCompletion) -> str | None:
        parsed_completions.append(completion)
        return completion.tool_call

    policy = OpenAICompatibleQwenPolicy(
        client=client,
        tool_call_parser=parse_qwen_tool_call,
        clock=FakeClock(1.0, 1.1),
    )

    response = policy.generate(_request())

    assert parsed_completions == [client.completion]
    assert response.raw_output == 'find_order(order_id="123")'
    assert response.parsed_action == 'find_order(order_id="123")'
