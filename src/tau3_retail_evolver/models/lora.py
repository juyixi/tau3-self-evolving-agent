"""PEFT helpers for the single shared OPD policy adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tau3_retail_evolver.config import LoraConfig as ProjectLoraConfig
from tau3_retail_evolver.config import TrainingConfig


_ADAPTER_NAME = "shared_policy"
_STAGE_6_LORA_R = 32
_STAGE_6_LORA_ALPHA = 64
_STAGE_6_LORA_DROPOUT = 0.05
_STAGE_6_TARGET_MODULES = "all-linear"


def build_lora_config(
    lora_config: ProjectLoraConfig,
    training_config: TrainingConfig,
    *,
    init_lora_weights: bool | str = True,
    peft_module: Any | None = None,
) -> Any:
    """Build the standard PEFT configuration with a zero-output LoRA update."""
    if not isinstance(lora_config, ProjectLoraConfig):
        raise TypeError("lora_config must be a LoraConfig")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")
    if lora_config.use_peft is not True:
        raise ValueError("Stage 6 requires use_peft=True")
    if lora_config.lora_r != _STAGE_6_LORA_R:
        raise ValueError(f"Stage 6 requires lora_r={_STAGE_6_LORA_R}")
    if lora_config.lora_alpha != _STAGE_6_LORA_ALPHA:
        raise ValueError(f"Stage 6 requires lora_alpha={_STAGE_6_LORA_ALPHA}")
    if lora_config.lora_dropout != _STAGE_6_LORA_DROPOUT:
        raise ValueError(f"Stage 6 requires lora_dropout={_STAGE_6_LORA_DROPOUT}")
    if training_config.target_modules != _STAGE_6_TARGET_MODULES:
        raise ValueError(f"Stage 6 requires target_modules='{_STAGE_6_TARGET_MODULES}'")
    if init_lora_weights is not True:
        raise ValueError("zero-impact LoRA requires init_lora_weights=True")

    peft = peft_module or _require_peft()
    return peft.LoraConfig(
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        init_lora_weights=True,
        target_modules=training_config.target_modules,
        task_type=peft.TaskType.CAUSAL_LM,
    )


def attach_shared_lora_adapter(
    base_model: Any,
    lora_config: ProjectLoraConfig,
    training_config: TrainingConfig,
    *,
    adapter_path: str | Path | None = None,
    peft_module: Any | None = None,
) -> Any:
    """Create or reload exactly one trainable adapter over ``base_model``."""
    peft = peft_module or _require_peft()
    if adapter_path is None:
        model = peft.get_peft_model(
            base_model,
            build_lora_config(lora_config, training_config, peft_module=peft),
            adapter_name=_ADAPTER_NAME,
        )
    else:
        model = peft.PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=_ADAPTER_NAME,
            is_trainable=True,
        )

    _assert_one_adapter(model)
    _freeze_non_lora_parameters(model)
    assert_only_lora_trainable(model)
    return model


def assert_only_lora_trainable(model: Any) -> int:
    """Ensure that all trainable parameters belong to the LoRA adapter."""
    if not callable(getattr(model, "named_parameters", None)):
        raise TypeError("model must expose named_parameters()")

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise ValueError("shared policy has no trainable LoRA parameters")
    non_lora_names = [name for name in trainable_names if not _is_lora_parameter(name)]
    if non_lora_names:
        joined_names = ", ".join(non_lora_names)
        raise ValueError(f"non-LoRA parameters are trainable: {joined_names}")
    return len(trainable_names)


def save_adapter_checkpoint(
    model: Any,
    checkpoint_path: str | Path,
    *,
    peft_module: Any | None = None,
) -> Path:
    """Save just the adapter tensors, refusing any base-model state."""
    peft = peft_module or _require_peft()
    adapter_state = peft.get_peft_model_state_dict(model, adapter_name=_ADAPTER_NAME)
    if not adapter_state:
        raise ValueError("adapter checkpoint has no LoRA tensors")
    base_tensor_names = [name for name in adapter_state if not _is_lora_parameter(name)]
    if base_tensor_names:
        joined_names = ", ".join(base_tensor_names)
        raise ValueError(f"adapter checkpoint contains a base-model tensor: {joined_names}")

    destination = Path(checkpoint_path)
    model.save_pretrained(
        destination,
        state_dict=adapter_state,
        safe_serialization=True,
        selected_adapters=[_ADAPTER_NAME],
    )
    adapter_directory = destination / _ADAPTER_NAME
    if not (adapter_directory / "adapter_config.json").is_file():
        raise RuntimeError(f"PEFT did not write a reloadable adapter to {adapter_directory}")
    return adapter_directory


def _freeze_non_lora_parameters(model: Any) -> None:
    if not callable(getattr(model, "named_parameters", None)):
        raise TypeError("model must expose named_parameters()")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = _is_lora_parameter(name)


def _assert_one_adapter(model: Any) -> None:
    peft_config = getattr(model, "peft_config", None)
    try:
        adapter_count = len(peft_config)
    except TypeError as error:
        raise TypeError("PEFT model must expose adapter configuration") from error
    if adapter_count != 1:
        raise ValueError(f"shared policy must have exactly one adapter, got {adapter_count}")


def _is_lora_parameter(name: str) -> bool:
    return "lora_" in name


def _require_peft() -> Any:
    try:
        import peft
    except ImportError as error:
        raise RuntimeError(
            "PEFT adapter loading requires the optional training dependency peft"
        ) from error
    return SimpleNamespace(
        LoraConfig=peft.LoraConfig,
        PeftModel=peft.PeftModel,
        TaskType=peft.TaskType,
        get_peft_model=peft.get_peft_model,
        get_peft_model_state_dict=peft.get_peft_model_state_dict,
    )
