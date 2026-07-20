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
    student_logits = torch.tensor([[[1.5, -0.5, -4.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[-1.0, 0.25, 3.0]]])
    positions = torch.tensor([0])

    loss = token_forward_kl(student_logits, teacher_logits, positions, positions)

    student_log_probs = torch.log_softmax(student_logits[0, 0], dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits[0, 0], dim=-1)
    expected_full_vocab = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum()
    expected_without_last_vocab = (
        teacher_log_probs[:2].exp()
        * (teacher_log_probs[:2] - student_log_probs[:2])
    ).sum()

    torch.testing.assert_close(loss, expected_full_vocab)
    assert not torch.isclose(loss, expected_without_last_vocab)


def test_bfloat16_near_equal_large_vocabulary_kl_matches_float32_reference() -> None:
    vocabulary_size = 65_536
    base = torch.linspace(-8.0, 8.0, vocabulary_size, dtype=torch.float32)
    teacher_logits = torch.stack((base, base.flip(0))).reshape(
        1, 2, vocabulary_size
    ).to(torch.bfloat16)
    student_logits = teacher_logits.clone()
    student_logits[..., ::257] += torch.tensor(0.03125, dtype=torch.bfloat16)
    student_logits.requires_grad_()
    positions = torch.tensor([0, 1], dtype=torch.long)

    loss = token_forward_kl(
        student_logits,
        teacher_logits,
        positions,
        positions,
    )

    selected_student = student_logits[0, positions].float()
    selected_teacher = teacher_logits[0, positions].float()
    student_log_probs = torch.log_softmax(selected_student, dim=-1)
    teacher_log_probs = torch.log_softmax(selected_teacher, dim=-1)
    reference = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1).mean()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-7
    torch.testing.assert_close(loss, reference, rtol=1e-5, atol=1e-7)

    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


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
