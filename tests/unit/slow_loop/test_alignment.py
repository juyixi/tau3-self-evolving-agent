from __future__ import annotations

import json

import pytest
import torch

from tau3_retail_evolver.slow_loop.alignment import (
    build_aligned_batch,
    render_public_prompt,
    render_teacher_prompt,
)
from tau3_retail_evolver.slow_loop.examples import OPDExample


class ToyTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _example(kind: str = "act", *, public_input: dict[str, object] | None = None) -> OPDExample:
    return OPDExample(
        example_id=f"example-{kind}",
        kind=kind,
        public_input=public_input or {"observation": "public-only", "z": 1, "a": 2},
        privileged_hindsight={"score": 0.75, "note": "teacher-only"},
        response_schema={"type": "object", "required": ["action"]},
        sampling_contract={"mode": "online"},
        provenance={"run_id": "run-1"},
    )


@pytest.mark.parametrize("kind", ("sel", "act", "write", "maint"))
def test_build_aligned_batch_preserves_the_same_response_prefix_for_every_kind(
    kind: str,
) -> None:
    batch = build_aligned_batch(_example(kind), ToyTokenizer(), response_ids=(41, 42), max_length=512)

    assert batch.student_response_positions.shape == batch.teacher_response_positions.shape
    assert batch.student_input_ids[batch.student_response_positions].tolist() == [41, 42]
    assert batch.teacher_input_ids[batch.teacher_response_positions].tolist() == [41, 42]
    assert torch.equal(batch.student_attention_mask, torch.ones_like(batch.student_input_ids))
    assert torch.equal(batch.teacher_attention_mask, torch.ones_like(batch.teacher_input_ids))


def test_prompt_rendering_uses_canonical_json_and_keeps_hindsight_out_of_public_prompt() -> None:
    tokenizer = ToyTokenizer()
    example = _example(public_input={"z": 1, "a": {"y": 2, "b": 3}})

    public_prompt = render_public_prompt(example)
    teacher_prompt = render_teacher_prompt(example)

    assert public_prompt == render_public_prompt(example)
    assert public_prompt == json.dumps(
        {
            "instruction": "Respond to the public input using the response schema.",
            "kind": "act",
            "public_input": {"a": {"b": 3, "y": 2}, "z": 1},
            "response_schema": {"required": ["action"], "type": "object"},
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert "teacher-only" not in public_prompt
    assert "teacher-only" in teacher_prompt

    batch = build_aligned_batch(example, tokenizer, response_ids=(41, 42), max_length=512)
    student_prompt_length = batch.student_response_positions[0].item()
    teacher_prompt_length = batch.teacher_response_positions[0].item()
    assert batch.student_input_ids[:student_prompt_length].tolist() == [
        ord(character) for character in public_prompt
    ]
    assert batch.teacher_input_ids[:teacher_prompt_length].tolist() == [
        ord(character) for character in teacher_prompt
    ]


def test_build_aligned_batch_truncates_only_the_left_side_of_each_prompt() -> None:
    tokenizer = ToyTokenizer()
    example = _example(public_input={"observation": "P" * 200})

    public_prompt_ids = [ord(character) for character in render_public_prompt(example)]
    teacher_prompt_ids = [ord(character) for character in render_teacher_prompt(example)]
    batch = build_aligned_batch(example, tokenizer, response_ids=(41, 42), max_length=16)

    assert batch.student_input_ids.tolist() == public_prompt_ids[-14:] + [41, 42]
    assert batch.teacher_input_ids.tolist() == teacher_prompt_ids[-14:] + [41, 42]
    assert batch.student_response_positions.tolist() == [14, 15]
    assert batch.teacher_response_positions.tolist() == [14, 15]


def test_build_aligned_batch_rejects_a_response_that_cannot_fit() -> None:
    with pytest.raises(ValueError, match="response_ids.*max_length"):
        build_aligned_batch(_example(), ToyTokenizer(), response_ids=(41, 42, 43), max_length=2)
