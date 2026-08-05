"""Policy implementations and their decision boundary."""

from tau3_evolver.models.lora import (
    assert_only_lora_trainable,
    build_lora_config,
    save_adapter_checkpoint,
    validate_stage6_adapter_contract,
    validate_stage6_lora_settings,
)
from tau3_evolver.models.policy import DecisionRequest, DecisionResponse, Policy

__all__ = [
    "DecisionRequest",
    "DecisionResponse",
    "Policy",
    "assert_only_lora_trainable",
    "build_lora_config",
    "load_qwen35_processor",
    "load_qwen35_tokenizer",
    "load_shared_qwen35_policy",
    "save_adapter_checkpoint",
    "validate_stage6_adapter_contract",
    "validate_stage6_lora_settings",
]


def __getattr__(name: str) -> object:
    if name not in {
        "load_qwen35_processor",
        "load_qwen35_tokenizer",
        "load_shared_qwen35_policy",
    }:
        raise AttributeError(name)
    from tau3_evolver.models import qwen35

    value = getattr(qwen35, name)
    globals()[name] = value
    return value
