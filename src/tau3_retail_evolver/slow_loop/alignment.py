from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any

from tau3_retail_evolver.slow_loop.examples import OPDExample

if TYPE_CHECKING:
    from torch import Tensor


_PUBLIC_INSTRUCTION = "Respond to the public input using the response schema."
_TEACHER_INSTRUCTION = (
    "Respond to the public input using the response schema. "
    "Use privileged hindsight only for teacher guidance."
)


@dataclass(frozen=True, slots=True)
class AlignedOPDBatch:
    student_input_ids: Tensor
    student_attention_mask: Tensor
    teacher_input_ids: Tensor
    teacher_attention_mask: Tensor
    student_response_positions: Tensor
    teacher_response_positions: Tensor


def render_public_prompt(example: OPDExample) -> str:
    """Render the student prompt as a deterministic JSON instruction."""
    _require_example(example)
    return _canonical_json(
        {
            "instruction": _PUBLIC_INSTRUCTION,
            "kind": example.kind,
            "public_input": example.public_input,
            "response_schema": example.response_schema,
        }
    )


def render_teacher_prompt(example: OPDExample) -> str:
    """Render the teacher prompt with the same public task plus hindsight."""
    _require_example(example)
    return _canonical_json(
        {
            "instruction": _TEACHER_INSTRUCTION,
            "kind": example.kind,
            "public_input": example.public_input,
            "privileged_hindsight": example.privileged_hindsight,
            "response_schema": example.response_schema,
        }
    )


def build_aligned_batch(
    example: OPDExample,
    tokenizer: Any,
    *,
    response_ids: Sequence[int],
    max_length: int,
) -> AlignedOPDBatch:
    """Encode paired OPD prompts while retaining the identical response suffix."""
    _require_example(example)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer")

    response_tokens = _response_tokens(response_ids)
    if len(response_tokens) > max_length:
        raise ValueError("response_ids exceed max_length")

    student_prompt_ids = _encode(tokenizer, render_public_prompt(example))
    teacher_prompt_ids = _encode(tokenizer, render_teacher_prompt(example))
    prompt_budget = max_length - len(response_tokens)

    return _make_batch(
        student_prompt_ids[-prompt_budget:] if prompt_budget else (),
        teacher_prompt_ids[-prompt_budget:] if prompt_budget else (),
        response_tokens,
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("OPD prompt content must be canonical JSON") from error


def _encode(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer(prompt, add_special_tokens=False)
    except TypeError as error:
        raise TypeError("tokenizer must accept text and add_special_tokens") from error
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError) as error:
        raise TypeError("tokenizer output must contain input_ids") from error
    return _flatten_token_ids(input_ids)


def _flatten_token_ids(input_ids: Any) -> tuple[int, ...]:
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes)):
        raise TypeError("tokenizer input_ids must be a token sequence")
    if len(input_ids) == 1 and isinstance(input_ids[0], Sequence):
        input_ids = input_ids[0]
    if any(isinstance(token_id, Sequence) for token_id in input_ids):
        raise TypeError("tokenizer input_ids must represent one prompt")
    try:
        return tuple(int(token_id) for token_id in input_ids)
    except (TypeError, ValueError) as error:
        raise TypeError("tokenizer input_ids must be integers") from error


def _response_tokens(response_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(response_ids, (str, bytes)):
        raise TypeError("response_ids must be a token sequence")
    try:
        return tuple(int(token_id) for token_id in response_ids)
    except (TypeError, ValueError) as error:
        raise TypeError("response_ids must be a token sequence") from error


def _make_batch(
    student_prompt_ids: Sequence[int],
    teacher_prompt_ids: Sequence[int],
    response_ids: Sequence[int],
) -> AlignedOPDBatch:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("build_aligned_batch requires the optional training dependency torch") from error

    student_input_ids = torch.tensor(
        [*student_prompt_ids, *response_ids], dtype=torch.long
    )
    teacher_input_ids = torch.tensor(
        [*teacher_prompt_ids, *response_ids], dtype=torch.long
    )
    student_response_positions = torch.arange(
        len(student_prompt_ids), len(student_input_ids), dtype=torch.long
    )
    teacher_response_positions = torch.arange(
        len(teacher_prompt_ids), len(teacher_input_ids), dtype=torch.long
    )
    return AlignedOPDBatch(
        student_input_ids=student_input_ids,
        student_attention_mask=torch.ones_like(student_input_ids),
        teacher_input_ids=teacher_input_ids,
        teacher_attention_mask=torch.ones_like(teacher_input_ids),
        student_response_positions=student_response_positions,
        teacher_response_positions=teacher_response_positions,
    )


def _require_example(example: OPDExample) -> None:
    if not isinstance(example, OPDExample):
        raise TypeError("example must be an OPDExample")
