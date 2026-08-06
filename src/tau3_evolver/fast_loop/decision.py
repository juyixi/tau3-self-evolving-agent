from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from tau3_evolver.fast_loop.contracts import (
    Decision,
    FastLoopPolicy,
    LifecycleResponse,
    parse_decision,
)
from tau3_evolver.fast_loop.prompts import LifecyclePrompt


DecisionT = TypeVar("DecisionT", bound=Decision)


def generate_decision(
    policy: FastLoopPolicy,
    prompt: LifecyclePrompt,
    decision_type: type[DecisionT],
    *,
    candidate_ids: Sequence[str] | None = None,
    validator: Callable[[DecisionT], Any] | None = None,
    invalid_fallback: Callable[[str], DecisionT] | None = None,
    label: str,
) -> tuple[DecisionT, dict[str, Any]]:
    response = policy.generate(prompt)
    responses = [response]
    result = parse_decision(
        response.raw_output,
        decision_type,
        validator=validator,
        candidate_ids=candidate_ids,
    )
    repaired_output: str | None = None
    initial_error = result.error
    if result.decision is None:
        repair = policy.repair(
            prompt,
            response.raw_output,
            result.error or "invalid output",
        )
        responses.append(repair)
        repaired_output = repair.raw_output
        result = parse_decision(
            repair.raw_output,
            decision_type,
            validator=validator,
            candidate_ids=candidate_ids,
        )
    fallback_used = False
    decision = result.decision
    if decision is None:
        terminal_error = result.error or "invalid output"
        if invalid_fallback is None:
            raise ValueError(f"invalid {label} decision after repair: {terminal_error}")
        decision = invalid_fallback(terminal_error)
        fallback_used = True
    prompt_tokens, completion_tokens = _combined_token_usage(responses)
    return decision, {
        "raw_output": response.raw_output,
        "repaired_output": repaired_output,
        "error": initial_error,
        "fallback_used": fallback_used,
        "sampling_params": dict(response.sampling_params),
        "latency_s": sum(item.latency_s for item in responses),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def accumulate_token_usage(
    prompt_total: int,
    completion_total: int,
    complete: bool,
    audit: Mapping[str, Any],
) -> tuple[int, int, bool]:
    prompt_tokens = audit["prompt_tokens"]
    completion_tokens = audit["completion_tokens"]
    if prompt_tokens is None or completion_tokens is None:
        return prompt_total, completion_total, False
    return (
        prompt_total + prompt_tokens,
        completion_total + completion_tokens,
        complete,
    )


def _combined_token_usage(
    responses: Sequence[LifecycleResponse],
) -> tuple[int | None, int | None]:
    if any(
        response.prompt_tokens is None or response.completion_tokens is None
        for response in responses
    ):
        return None, None
    return (
        sum(response.prompt_tokens or 0 for response in responses),
        sum(response.completion_tokens or 0 for response in responses),
    )


__all__ = ["accumulate_token_usage", "generate_decision"]
