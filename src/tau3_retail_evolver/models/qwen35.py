"""Lazy Qwen3.5 training loader for the shared OPD policy."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tau3_retail_evolver.config import LoraConfig, ModelConfig, TrainingConfig
from tau3_retail_evolver.models.lora import attach_shared_lora_adapter


def load_qwen35_tokenizer(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    transformers_module: Any | None = None,
) -> Any:
    """Load the text-only Qwen tokenizer from a local path or Hugging Face ID."""
    transformers = transformers_module or _require_transformers()
    load_options = _pretrained_options(
        revision=revision,
        local_files_only=local_files_only,
    )
    return transformers.AutoTokenizer.from_pretrained(
        str(model_id_or_path), **load_options
    )


def load_qwen35_processor(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    transformers_module: Any | None = None,
) -> Any:
    """Compatibility name for the text-only Qwen tokenizer loader."""
    return load_qwen35_tokenizer(
        model_id_or_path,
        revision=revision,
        local_files_only=local_files_only,
        transformers_module=transformers_module,
    )


def load_shared_qwen35_policy(
    model_config: ModelConfig,
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    *,
    revision: str | None = None,
    adapter_path: str | Path | None = None,
    local_files_only: bool = False,
    transformers_module: Any | None = None,
    peft_module: Any | None = None,
    torch_module: Any | None = None,
) -> Any:
    """Load one causal base model with one trainable shared LoRA adapter."""
    if not isinstance(model_config, ModelConfig):
        raise TypeError("model_config must be a ModelConfig")
    if not isinstance(lora_config, LoraConfig):
        raise TypeError("lora_config must be a LoraConfig")
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")

    transformers = transformers_module or _require_transformers()
    torch = torch_module or _require_torch()
    load_options = _pretrained_options(
        revision=revision,
        local_files_only=local_files_only,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        torch_dtype=_resolve_dtype(training_config.dtype, torch),
        **load_options,
    )
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    return attach_shared_lora_adapter(
        model,
        lora_config,
        training_config,
        adapter_path=adapter_path,
        peft_module=peft_module,
    )


def _resolve_dtype(dtype_name: str, torch_module: Any) -> Any:
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None:
        raise ValueError(f"unsupported training dtype: {dtype_name}")
    return dtype


def _pretrained_options(
    *, revision: str | None, local_files_only: bool
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if revision is not None:
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must not be empty")
        options["revision"] = revision
    if local_files_only:
        options["local_files_only"] = True
    return options


def _require_transformers() -> Any:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.5 loading requires the optional training dependency transformers"
        ) from error
    return SimpleNamespace(
        AutoModelForCausalLM=AutoModelForCausalLM,
        AutoTokenizer=AutoTokenizer,
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Qwen3.5 loading requires the optional training dependency torch"
        ) from error
    return torch
