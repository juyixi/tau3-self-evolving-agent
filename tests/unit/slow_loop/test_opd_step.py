from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tau3_retail_evolver.slow_loop.alignment import AlignedOPDBatch
from tau3_retail_evolver.slow_loop.opd_step import shared_policy_opd_step


class RecordingCausalModel(nn.Module):
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        super().__init__()
        self.vocabulary_bias = nn.Parameter(torch.tensor([0.5, -0.5, 1.0]))
        self.fail_on_call = fail_on_call
        self.calls: list[dict[str, object]] = []

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.calls.append(
            {
                "model_id": id(self),
                "training": self.training,
                "grad_enabled": torch.is_grad_enabled(),
                "input_ids": input_ids.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "parameter_ptr": self.vocabulary_bias.data_ptr(),
            }
        )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("intentional forward failure")
        logits = input_ids.to(dtype=self.vocabulary_bias.dtype).unsqueeze(-1) + self.vocabulary_bias
        return SimpleNamespace(logits=logits)


def _batch() -> AlignedOPDBatch:
    return AlignedOPDBatch(
        student_input_ids=torch.tensor([[1, 2, 3, 4]]),
        student_attention_mask=torch.tensor([[1, 1, 1, 1]]),
        teacher_input_ids=torch.tensor([[8, 9, 3, 4]]),
        teacher_attention_mask=torch.tensor([[1, 1, 1, 1]]),
        student_response_positions=torch.tensor([1, 2]),
        teacher_response_positions=torch.tensor([1, 2]),
    )


def test_shared_policy_opd_step_runs_teacher_then_student_through_one_model() -> None:
    model = RecordingCausalModel().train()
    batch = _batch()

    result = shared_policy_opd_step(model, batch)

    assert len(model.calls) == 2
    teacher_call, student_call = model.calls
    assert teacher_call["training"] is False
    assert teacher_call["grad_enabled"] is False
    assert teacher_call["input_ids"].tolist() == batch.teacher_input_ids.tolist()
    assert teacher_call["attention_mask"].tolist() == batch.teacher_attention_mask.tolist()
    assert student_call["training"] is True
    assert student_call["grad_enabled"] is True
    assert student_call["input_ids"].tolist() == batch.student_input_ids.tolist()
    assert student_call["attention_mask"].tolist() == batch.student_attention_mask.tolist()
    assert teacher_call["model_id"] == student_call["model_id"] == id(model)
    assert teacher_call["parameter_ptr"] == student_call["parameter_ptr"] == model.vocabulary_bias.data_ptr()
    assert model.training is True
    assert result.loss.requires_grad
    assert result.metrics["forward_kl"].requires_grad is False
    assert result.metrics["forward_kl"].ndim == 0

    result.loss.backward()
    assert model.vocabulary_bias.grad is not None


def test_shared_policy_opd_step_restores_the_original_mode_when_a_forward_raises() -> None:
    model = RecordingCausalModel(fail_on_call=2).eval()

    with pytest.raises(RuntimeError, match="intentional forward failure"):
        shared_policy_opd_step(model, _batch())

    assert len(model.calls) == 2
    assert model.calls[0]["training"] is False
    assert model.calls[0]["grad_enabled"] is False
    assert model.calls[1]["training"] is False
    assert model.calls[1]["grad_enabled"] is True
    assert model.training is False
