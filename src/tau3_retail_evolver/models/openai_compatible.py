from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import monotonic
from typing import Any, Protocol

from tau3_retail_evolver.fast_loop.action_codec import Tau2ActionCodec
from tau3_retail_evolver.fast_loop.baseline_prompt import build_baseline_prompt
from tau3_retail_evolver.models.policy import DecisionRequest, DecisionResponse, Policy


class OpenAICompatibleClient(Protocol):
    """Small adapter surface for an OpenAI-compatible Qwen endpoint client."""

    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tools: Sequence[Mapping[str, Any]],
        temperature: float,
        top_p: float,
    ) -> object: ...


QwenToolCallParser = Callable[[object], str | None]


class OpenAICompatibleQwenPolicy(Policy):
    """Generate Tau2 actions through an injected Qwen-compatible endpoint client.

    ``tool_call_parser`` is supplied by the serving integration and should use
    that server's official Qwen parser rather than reparsing tool syntax here.
    """

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        tool_call_parser: QwenToolCallParser,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._client = client
        self._tool_call_parser = tool_call_parser
        self._clock = clock

    def generate(self, request: DecisionRequest) -> DecisionResponse:
        prompt = build_baseline_prompt(request.observation, request.reset_info)
        started_at = self._clock()
        completion = self._client.create_chat_completion(
            messages=list(prompt.messages),
            tools=list(prompt.tools),
            temperature=request.temperature,
            top_p=request.top_p,
        )
        latency_s = self._clock() - started_at

        tool_call = self._tool_call_parser(completion)
        raw_output = _raw_output(completion, tool_call)
        parsed_action = Tau2ActionCodec.decode(raw_output, _tool_names(prompt.tools))
        return DecisionResponse(
            raw_output=raw_output,
            parsed_action=parsed_action,
            sampling_params={"temperature": request.temperature, "top_p": request.top_p},
            latency_s=latency_s,
        )


def _raw_output(completion: object, tool_call: str | None) -> str:
    if tool_call is not None:
        if not isinstance(tool_call, str):
            raise ValueError("Qwen tool-call parser must return text or None")
        return tool_call

    if isinstance(completion, Mapping):
        content = completion.get("content")
    else:
        content = getattr(completion, "content", None)
    if not isinstance(content, str):
        raise ValueError("Qwen completion has no text content")
    return content


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names
