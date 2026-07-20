from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from tau3_retail_evolver.slow_loop.alignment import AlignedOPDBatch
from tau3_retail_evolver.slow_loop.loss import token_forward_kl


@dataclass(frozen=True, slots=True)
class OPDStepResult:
    """Loss for backward plus detached scalar metrics for logging."""

    loss: Tensor
    metrics: Mapping[str, Tensor]


def shared_policy_opd_step(model: Any, batch: AlignedOPDBatch) -> OPDStepResult:
    """Evaluate a shared policy as frozen teacher, then trainable student."""
    if not isinstance(batch, AlignedOPDBatch):
        raise TypeError("batch must be an AlignedOPDBatch")
    if not hasattr(model, "training") or not callable(getattr(model, "train", None)):
        raise TypeError("model must expose training state and train(mode)")

    original_training = model.training
    try:
        with torch.no_grad():
            model.eval()
            teacher_logits = _model_logits(
                model,
                input_ids=batch.teacher_input_ids,
                attention_mask=batch.teacher_attention_mask,
            ).detach()

        model.train(original_training)
        student_logits = _model_logits(
            model,
            input_ids=batch.student_input_ids,
            attention_mask=batch.student_attention_mask,
        )
        loss = token_forward_kl(
            student_logits,
            teacher_logits,
            batch.student_response_positions,
            batch.teacher_response_positions,
        )
        return OPDStepResult(loss=loss, metrics={"forward_kl": loss.detach()})
    finally:
        model.train(original_training)


def _model_logits(model: Any, *, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    try:
        logits = output.logits
    except AttributeError as error:
        raise TypeError("model output must expose logits") from error
    if not isinstance(logits, Tensor):
        raise TypeError("model output logits must be a torch.Tensor")
    return logits
