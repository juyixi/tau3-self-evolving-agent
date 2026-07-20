from __future__ import annotations

import pytest
import torch

from tau3_retail_evolver.slow_loop.loss import token_forward_kl


def test_token_forward_kl_matches_the_hand_computed_full_vocabulary_value() -> None:
    student_logits = torch.tensor(
        [[[7.0, -3.0, 2.0], [0.5, -1.0, 3.0], [-4.0, 5.0, 1.0], [2.0, 0.0, -2.0]]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor(
        [[[-2.0, 6.0, 1.0], [1.0, 0.0, -2.0], [3.0, -1.0, 0.5], [-1.0, 2.0, 4.0]]],
        requires_grad=True,
    )
    student_positions = torch.tensor([1, 3])
    teacher_positions = torch.tensor([0, 2])

    loss = token_forward_kl(
        student_logits, teacher_logits, student_positions, teacher_positions
    )

    student_log_probs = torch.log_softmax(student_logits[0, student_positions], dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits[0, teacher_positions], dim=-1)
    teacher_probs = teacher_log_probs.exp()
    expected = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(-1).mean()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert student_logits.grad is not None
    assert teacher_logits.grad is None


def test_token_forward_kl_ignores_prompt_logits_and_is_zero_for_identical_selected_logits() -> None:
    selected_logits = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 4.0, 0.5]])
    student_logits = torch.tensor(
        [[[100.0, -100.0, 50.0], *selected_logits.tolist(), [9.0, 8.0, 7.0]]]
    )
    teacher_logits = torch.tensor(
        [[[-50.0, 60.0, -70.0], [1.0, -2.0, 3.0], [4.0, 5.0, 6.0], [-9.0, 2.0, 8.0]]]
    )
    positions = torch.tensor([1, 2])
    teacher_logits[0, positions] = selected_logits

    loss = token_forward_kl(student_logits, teacher_logits, positions, positions)

    torch.testing.assert_close(loss, torch.zeros((), dtype=loss.dtype))


def test_token_forward_kl_uses_every_vocabulary_logit() -> None:
    student_logits = torch.tensor([[[0.0, 0.0, -10.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[0.0, 0.0, 10.0]]])
    positions = torch.tensor([0])

    loss = token_forward_kl(student_logits, teacher_logits, positions, positions)

    assert loss.item() > 10.0


@pytest.mark.parametrize(
    ("student_logits", "teacher_logits", "student_positions", "teacher_positions", "match"),
    (
        (
            torch.zeros((2, 3, 4)),
            torch.zeros((1, 3, 4)),
            torch.tensor([0]),
            torch.tensor([0]),
            "batch dimension",
        ),
        (
            torch.zeros((1, 3, 4)),
            torch.zeros((1, 3, 5)),
            torch.tensor([0]),
            torch.tensor([0]),
            "vocabulary dimension",
        ),
        (
            torch.zeros((1, 3, 4)),
            torch.zeros((1, 3, 4)),
            torch.tensor([0, 1]),
            torch.tensor([0]),
            "same number",
        ),
        (
            torch.zeros((1, 3, 4)),
            torch.full((1, 3, 4), float("inf")),
            torch.tensor([0]),
            torch.tensor([0]),
            "finite",
        ),
    ),
)
def test_token_forward_kl_rejects_invalid_logits_and_position_maps(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_positions: torch.Tensor,
    teacher_positions: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        token_forward_kl(student_logits, teacher_logits, student_positions, teacher_positions)
