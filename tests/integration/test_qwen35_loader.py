from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_QWEN35_INTEGRATION") != "1",
    reason="set RUN_QWEN35_INTEGRATION=1 to load the real Qwen3.5 model",
)


def test_qwen35_shared_policy_is_zero_impact_and_reloads_its_adapter(tmp_path: Path) -> None:
    import torch

    from tau3_retail_evolver.config import LoraConfig, ModelConfig, TrainingConfig
    from tau3_retail_evolver.models.lora import save_adapter_checkpoint
    from tau3_retail_evolver.models.qwen35 import (
        load_qwen35_tokenizer,
        load_shared_qwen35_policy,
    )

    model_config = ModelConfig(base_model="Qwen/Qwen3.5-9B")
    lora_config = LoraConfig()
    training_config = TrainingConfig()
    tokenizer = load_qwen35_tokenizer(model_config.base_model)
    model = load_shared_qwen35_policy(model_config, lora_config, training_config)

    base_parameters = [
        parameter for name, parameter in model.named_parameters() if "lora_" not in name
    ]
    lora_b_parameters = [
        parameter for name, parameter in model.named_parameters() if "lora_B" in name
    ]
    assert base_parameters
    assert not any(parameter.requires_grad for parameter in base_parameters)
    assert lora_b_parameters
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in lora_b_parameters)

    device = next(model.parameters()).device
    inputs = tokenizer("Return a short retail policy response.", return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    assert torch.isfinite(output.logits).all()

    checkpoint = save_adapter_checkpoint(model, tmp_path / "adapter")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded = load_shared_qwen35_policy(
        model_config,
        lora_config,
        training_config,
        adapter_path=checkpoint,
    )
    assert len(reloaded.peft_config) == 1
    assert not any(
        parameter.requires_grad
        for name, parameter in reloaded.named_parameters()
        if "lora_" not in name
    )
