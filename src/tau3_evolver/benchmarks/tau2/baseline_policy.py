from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import monotonic
from typing import Any

from tau3_evolver.benchmarks.tau2.action_codec import Tau2ActionCodec
from tau3_evolver.benchmarks.tau2.tool_schemas import build_baseline_prompt
from tau3_evolver.models.openai_compatible import (
    OpenAICompatibleClient,
    QwenToolCallParser,
    completion_raw_output,
    completion_token_usage,
    parse_openai_qwen_tool_call,
)
from tau3_evolver.models.policy import DecisionRequest, DecisionResponse, Policy


class OpenAICompatibleQwenPolicy(Policy):
    """Generate legacy Tau2 baseline actions through a Qwen endpoint."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        tool_call_parser: QwenToolCallParser | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._client = client
        self._tool_call_parser = tool_call_parser
        self._clock = clock

    def generate(self, request: DecisionRequest) -> DecisionResponse:
        prompt = build_baseline_prompt(
            request.observation,
            request.reset_info,
            request.history,
        )
        started_at = self._clock()
        completion = self._client.create_chat_completion(
            messages=list(prompt.messages),
            tools=list(prompt.tools),
            temperature=request.temperature,
            top_p=request.top_p,
        )
        latency_s = self._clock() - started_at

        parser = self._tool_call_parser or parse_openai_qwen_tool_call
        tool_call = parser(completion)
        if tool_call is not None and not isinstance(tool_call, str):
            raise ValueError("Qwen tool-call parser must return text or None")
        raw_output = completion_raw_output(completion)
        parsed_action = Tau2ActionCodec.decode(
            tool_call or raw_output,
            _tool_names(prompt.tools),
        )
        prompt_tokens, completion_tokens = completion_token_usage(completion)
        return DecisionResponse(
            raw_output=raw_output,
            parsed_action=parsed_action,
            sampling_params={
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            latency_s=latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


__all__ = ["OpenAICompatibleQwenPolicy"]
