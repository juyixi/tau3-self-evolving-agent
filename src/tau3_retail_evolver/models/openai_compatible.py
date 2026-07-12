from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from time import monotonic
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

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


HttpTransport = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]
QwenToolCallParser = Callable[[object], str | None]


class OpenAICompatibleHttpClient:
    """Concrete stdlib client for Qwen endpoints implementing OpenAI chat completions."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        max_tokens: int | None = None,
        generation_settings: Mapping[str, Any] | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        if not api_key:
            raise ValueError("api_key is required")
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._generation_settings = dict(generation_settings or {})
        reserved = {"model", "messages", "tools", "temperature", "top_p", "max_tokens"}
        if reserved.intersection(self._generation_settings):
            raise ValueError("generation settings must not override chat completion fields")
        self._transport = transport if transport is not None else _stdlib_transport

    def __repr__(self) -> str:
        return f"OpenAICompatibleHttpClient(model={self._model!r})"

    def create_chat_completion(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tools: Sequence[Mapping[str, Any]],
        temperature: float,
        top_p: float,
    ) -> object:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": temperature,
            "top_p": top_p,
            **self._generation_settings,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            status, response_body = self._transport(self._endpoint, headers, body)
        except Exception:
            raise RuntimeError("OpenAI-compatible endpoint request failed") from None
        if not 200 <= status < 300:
            raise RuntimeError(f"OpenAI-compatible endpoint returned HTTP {status}")
        try:
            response = json.loads(response_body)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("OpenAI-compatible endpoint returned invalid JSON") from error
        if not isinstance(response, Mapping):
            raise RuntimeError("OpenAI-compatible endpoint returned a non-object response")
        return dict(response)


class OpenAICompatibleQwenPolicy(Policy):
    """Generate Tau2 actions through an OpenAI-compatible Qwen endpoint client."""

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
        prompt = build_baseline_prompt(request.observation, request.reset_info, request.history)
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
        raw_output = _raw_output(completion)
        parsed_action = Tau2ActionCodec.decode(tool_call or raw_output, _tool_names(prompt.tools))
        return DecisionResponse(
            raw_output=raw_output,
            parsed_action=parsed_action,
            sampling_params={"temperature": request.temperature, "top_p": request.top_p},
            latency_s=latency_s,
        )


def parse_openai_qwen_tool_call(completion: object) -> str | None:
    """Consume the standard structured tool calls emitted by Qwen server parsers."""
    message = _assistant_message(completion)
    if message is None or "tool_calls" not in message:
        return None

    tool_calls = message["tool_calls"]
    if tool_calls is None:
        return None
    if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
        raise ValueError("structured tool calls must be a sequence")
    if not tool_calls:
        return None
    if len(tool_calls) != 1:
        raise ValueError("structured responses must contain exactly one tool call")

    tool_call = tool_calls[0]
    if not isinstance(tool_call, Mapping):
        raise ValueError("structured tool call must be an object")
    function = tool_call.get("function")
    if not isinstance(function, Mapping):
        raise ValueError("structured tool call must contain a function")
    name = function.get("name")
    if not isinstance(name, str):
        raise ValueError("structured tool call must contain a function name")
    arguments = _structured_arguments(function.get("arguments"))
    return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))


def _structured_arguments(value: Any) -> Mapping[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("structured tool call arguments must be JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("structured tool call arguments must be an object")
    return value


def _raw_output(completion: object) -> str:
    message = _assistant_message(completion)
    if message is not None and _has_structured_tool_calls(message):
        try:
            return json.dumps(message, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as error:
            raise ValueError("structured assistant message must be JSON serializable") from error
    if message is not None:
        content = message.get("content")
    elif isinstance(completion, Mapping):
        content = completion.get("content")
    else:
        content = getattr(completion, "content", None)
    if not isinstance(content, str):
        raise ValueError("Qwen completion has no text content")
    return content


def _has_structured_tool_calls(message: Mapping[str, Any]) -> bool:
    tool_calls = message.get("tool_calls")
    return (
        isinstance(tool_calls, Sequence)
        and not isinstance(tool_calls, (str, bytes))
        and bool(tool_calls)
    )


def _assistant_message(completion: object) -> Mapping[str, Any] | None:
    if not isinstance(completion, Mapping) or "choices" not in completion:
        return None
    choices = completion["choices"]
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices:
        raise ValueError("OpenAI-compatible response must contain a choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("OpenAI-compatible choice must be an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("OpenAI-compatible choice must contain an assistant message")
    return message


def _tool_names(tools: Sequence[Mapping[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        name = function.get("name") if isinstance(function, Mapping) else tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def _stdlib_transport(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request) as response:
            return response.status, response.read()
    except HTTPError as error:
        return error.code, error.read()
