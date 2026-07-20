from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def token_forward_kl(
    student_logits: Tensor,
    teacher_logits: Tensor,
    student_positions: Tensor,
    teacher_positions: Tensor,
) -> Tensor:
    """Compute mean full-vocabulary forward KL at aligned causal-logit positions."""
    _validate_logits(student_logits, "student_logits")
    _validate_logits(teacher_logits, "teacher_logits")
    if student_logits.shape[0] != teacher_logits.shape[0]:
        raise ValueError("student and teacher logits must have the same batch dimension")
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError("student and teacher logits must have the same vocabulary dimension")

    selected_student_logits = _select_response_logits(
        student_logits, student_positions, "student_positions"
    )
    selected_teacher_logits = _select_response_logits(
        teacher_logits, teacher_positions, "teacher_positions"
    ).detach()
    if selected_student_logits.shape[0] != selected_teacher_logits.shape[0]:
        raise ValueError("student and teacher positions must select the same number of response logits")
    if not torch.isfinite(selected_student_logits).all():
        raise ValueError("student response logits must be finite")
    if not torch.isfinite(selected_teacher_logits).all():
        raise ValueError("teacher response logits must be finite")

    student_log_probs = F.log_softmax(selected_student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(selected_teacher_logits, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
    if not torch.isfinite(loss):
        raise ValueError("forward KL must be finite")
    return loss


def _validate_logits(logits: Tensor, name: str) -> None:
    if not isinstance(logits, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if logits.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, sequence, vocabulary]")
    if logits.shape[0] != 1:
        raise ValueError(f"{name} must have batch dimension 1")
    if logits.shape[1] < 1 or logits.shape[2] < 1:
        raise ValueError(f"{name} sequence and vocabulary dimensions must be non-empty")


def _select_response_logits(logits: Tensor, positions: Tensor, name: str) -> Tensor:
    if not isinstance(positions, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if positions.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if positions.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if positions.dtype != torch.long:
        raise ValueError(f"{name} must contain torch.long causal-logit indices")
    if positions.device != logits.device:
        raise ValueError(f"{name} must be on the same device as its logits")
    if positions.min().item() < 0 or positions.max().item() >= logits.shape[1]:
        raise ValueError(f"{name} contains an out-of-bounds causal-logit index")
    return logits[0, positions]
