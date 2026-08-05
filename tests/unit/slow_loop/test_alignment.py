from __future__ import annotations

import json

import pytest
import torch

from tau3_evolver.slow_loop.alignment import (
    build_aligned_batch,
    render_public_prompt,
    render_teacher_prompt,
)
from tau3_evolver.slow_loop.examples import OPDExample


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

    assert batch.student_input_ids.ndim == 2
    assert batch.teacher_input_ids.ndim == 2
    assert batch.student_input_ids.shape[0] == 1
    assert batch.teacher_input_ids.shape[0] == 1
    assert batch.student_attention_mask.shape == batch.student_input_ids.shape
    assert batch.teacher_attention_mask.shape == batch.teacher_input_ids.shape
    assert batch.student_response_positions.shape == batch.teacher_response_positions.shape
    assert batch.student_input_ids[0, batch.student_response_positions + 1].tolist() == [41, 42]
    assert batch.teacher_input_ids[0, batch.teacher_response_positions + 1].tolist() == [41, 42]
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
    assert teacher_prompt == public_prompt + "\n" + json.dumps(
        {"privileged_hindsight": example.privileged_hindsight},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    batch = build_aligned_batch(example, tokenizer, response_ids=(41, 42), max_length=512)
    student_prompt_length = batch.student_response_positions[0].item() + 1
    teacher_prompt_length = batch.teacher_response_positions[0].item() + 1
    assert batch.student_input_ids[0, :student_prompt_length].tolist() == [
        ord(character) for character in public_prompt
    ]
    assert batch.teacher_input_ids[0, :teacher_prompt_length].tolist() == [
        ord(character) for character in teacher_prompt
    ]


def test_build_aligned_batch_truncates_only_the_left_side_of_each_prompt() -> None:
    tokenizer = ToyTokenizer()
    example = _example(public_input={"observation": "P" * 200})

    public_prompt_ids = [ord(character) for character in render_public_prompt(example)]
    teacher_prompt_ids = [ord(character) for character in render_teacher_prompt(example)]
    batch = build_aligned_batch(example, tokenizer, response_ids=(41, 42), max_length=16)

    assert batch.student_input_ids.tolist() == [public_prompt_ids[-14:] + [41, 42]]
    assert batch.teacher_input_ids.tolist() == [teacher_prompt_ids[-14:] + [41, 42]]
    assert batch.student_response_positions.tolist() == [13, 14]
    assert batch.teacher_response_positions.tolist() == [13, 14]


def test_response_positions_select_the_causal_logits_for_the_generated_response() -> None:
    response_ids = (41, 42, 43)
    batch = build_aligned_batch(_example(), ToyTokenizer(), response_ids=response_ids, max_length=512)
    student_prompt_length = batch.student_response_positions[0].item() + 1
    teacher_prompt_length = batch.teacher_response_positions[0].item() + 1

    assert batch.student_response_positions[0].item() == student_prompt_length - 1
    assert batch.student_response_positions[-1].item() == student_prompt_length + len(response_ids) - 2
    assert batch.teacher_response_positions[0].item() == teacher_prompt_length - 1
    assert batch.teacher_response_positions[-1].item() == teacher_prompt_length + len(response_ids) - 2

    vocab_size = 64
    student_logits = torch.zeros((1, batch.student_input_ids.shape[1], vocab_size))
    teacher_logits = torch.zeros((1, batch.teacher_input_ids.shape[1], vocab_size))
    for position, response_id in zip(batch.student_response_positions, response_ids, strict=True):
        student_logits[0, position, response_id] = 1
    for position, response_id in zip(batch.teacher_response_positions, response_ids, strict=True):
        teacher_logits[0, position, response_id] = 1

    assert student_logits[0, batch.student_response_positions].argmax(dim=-1).tolist() == list(response_ids)
    assert teacher_logits[0, batch.teacher_response_positions].argmax(dim=-1).tolist() == list(response_ids)


@pytest.mark.parametrize(
    ("response_ids", "max_length", "match"),
    (
        ((), 32, "response_ids must not be empty"),
        ((41, 42), 2, "max_length must leave room for at least one prompt token"),
        ((41, 42, 43), 2, "response_ids.*max_length"),
    ),
)
def test_build_aligned_batch_rejects_responses_without_a_valid_prompt_budget(
    response_ids: tuple[int, ...], max_length: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_aligned_batch(
            _example(), ToyTokenizer(), response_ids=response_ids, max_length=max_length
        )
