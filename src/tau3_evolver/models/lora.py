"""PEFT helpers for the single shared OPD policy adapter."""

from __future__ import annotations

from collections.abc import Collection, Mapping
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tau3_evolver.config import LoraConfig as ProjectLoraConfig
from tau3_evolver.config import TrainingConfig


_ADAPTER_NAME = "shared_policy"
_STAGE_6_LORA_R = 32
_STAGE_6_LORA_ALPHA = 64
_STAGE_6_LORA_DROPOUT = 0.05
_STAGE_6_TARGET_MODULES = "all-linear"
_STAGE_6_ADAPTER_CONTRACT = "stage6_adapter_contract.json"
_STAGE_6_TARGET_INSTANCES_ATTR = "_stage6_all_linear_target_instances"


def build_lora_config(
    lora_config: ProjectLoraConfig,
    training_config: TrainingConfig,
    *,
    init_lora_weights: bool | str = True,
    peft_module: Any | None = None,
) -> Any:
    """Build the standard PEFT configuration with a zero-output LoRA update."""
    validate_stage6_lora_settings(lora_config, training_config)
    if init_lora_weights is not True:
        raise ValueError("zero-impact LoRA requires init_lora_weights=True")

    peft = peft_module or _require_peft()
    return peft.LoraConfig(
        r=lora_config.lora_r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        init_lora_weights=True,
        bias="none",
        use_rslora=False,
        use_dora=False,
        modules_to_save=None,
        rank_pattern={},
        alpha_pattern={},
        exclude_modules=None,
        layers_to_transform=None,
        layers_pattern=None,
        trainable_token_indices=None,
        layer_replication=None,
        lora_bias=False,
        target_parameters=None,
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
    validate_stage6_lora_settings(lora_config, training_config)
    peft = peft_module or _require_peft()
    eligible_instances = _eligible_linear_module_names(base_model)
    expected_targets: Collection[str] | None = None
    if adapter_path is None:
        model = peft.get_peft_model(
            base_model,
            build_lora_config(lora_config, training_config, peft_module=peft),
            adapter_name=_ADAPTER_NAME,
        )
    else:
        contract = validate_stage6_adapter_contract(adapter_path)
        expected_targets = contract["resolved_target_modules"]
        if tuple(contract["eligible_linear_modules"]) != eligible_instances:
            raise ValueError(
                "Stage 6 adapter contract eligible linear modules do not match the base model"
            )
        model = peft.PeftModel.from_pretrained(
            base_model,
            adapter_path,
            adapter_name=_ADAPTER_NAME,
            is_trainable=True,
        )

    _assert_one_adapter(model)
    _validate_loaded_adapter_config(model, expected_target_modules=expected_targets)
    targeted_instances = _loaded_target_module_instances(model)
    if targeted_instances != eligible_instances:
        raise ValueError(
            "shared adapter targeted module instances do not match all eligible linear modules"
        )
    setattr(model, _STAGE_6_TARGET_INSTANCES_ATTR, eligible_instances)
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
    resolved_targets = _validate_loaded_adapter_config(model)
    eligible_instances = getattr(model, _STAGE_6_TARGET_INSTANCES_ATTR, None)
    if not isinstance(eligible_instances, tuple) or not eligible_instances:
        raise ValueError("shared policy is missing its all-linear target instance record")
    targeted_instances = _loaded_target_module_instances(model)
    if targeted_instances != eligible_instances:
        raise ValueError(
            "shared adapter targeted module instances do not match all eligible linear modules"
        )
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
        safe_serialization=True,
        selected_adapters=[_ADAPTER_NAME],
    )
    adapter_directory = destination / _ADAPTER_NAME
    if not (adapter_directory / "adapter_config.json").is_file():
        raise RuntimeError(f"PEFT did not write a reloadable adapter to {adapter_directory}")
    contract = {
        "schema_version": 2,
        "requested_target_modules": _STAGE_6_TARGET_MODULES,
        "resolved_target_modules": list(resolved_targets),
        "eligible_linear_modules": list(eligible_instances),
        "targeted_module_instances": list(targeted_instances),
        "options": _stage6_adapter_options(),
    }
    (adapter_directory / _STAGE_6_ADAPTER_CONTRACT).write_text(
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    validate_stage6_adapter_contract(adapter_directory)
    return adapter_directory


def validate_stage6_adapter_contract(adapter_path: str | Path) -> dict[str, Any]:
    """Read and validate the dependency-free Stage 6 adapter contract."""
    contract_path = Path(adapter_path) / _STAGE_6_ADAPTER_CONTRACT
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {_STAGE_6_ADAPTER_CONTRACT}: {contract_path}") from error
    if not isinstance(contract, dict):
        raise ValueError("Stage 6 adapter contract must be a JSON object")
    expected_keys = {
        "schema_version",
        "requested_target_modules",
        "resolved_target_modules",
        "eligible_linear_modules",
        "targeted_module_instances",
        "options",
    }
    if set(contract) != expected_keys:
        raise ValueError("Stage 6 adapter contract fields are not exact")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 2:
        raise ValueError("Stage 6 adapter contract schema_version must be 2")
    if contract["requested_target_modules"] != _STAGE_6_TARGET_MODULES:
        raise ValueError("Stage 6 adapter contract requires logical target all-linear")

    targets = contract["resolved_target_modules"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("Stage 6 adapter contract resolved_target_modules must be non-empty")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ValueError("Stage 6 adapter contract resolved_target_modules must contain names")
    if targets != sorted(targets):
        raise ValueError("Stage 6 adapter contract resolved_target_modules must be sorted")
    if len(set(targets)) != len(targets):
        raise ValueError("Stage 6 adapter contract resolved_target_modules must be unique")

    eligible_instances = _validate_contract_name_list(
        contract["eligible_linear_modules"], "eligible_linear_modules"
    )
    targeted_instances = _validate_contract_name_list(
        contract["targeted_module_instances"], "targeted_module_instances"
    )
    if targeted_instances != eligible_instances:
        raise ValueError(
            "Stage 6 adapter contract targeted module instances must equal all eligible linear modules"
        )

    expected_options = _stage6_adapter_options()
    options = contract["options"]
    if not _has_exact_typed_values(options, expected_options):
        raise ValueError("Stage 6 adapter contract options do not match the exact contract")
    return contract


def _freeze_non_lora_parameters(model: Any) -> None:
    if not callable(getattr(model, "named_parameters", None)):
        raise TypeError("model must expose named_parameters()")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = _is_lora_parameter(name)


def validate_stage6_lora_settings(
    lora_config: ProjectLoraConfig,
    training_config: TrainingConfig,
) -> None:
    """Validate Stage 6 LoRA invariants without loading training dependencies."""
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


def _validate_loaded_adapter_config(
    model: Any,
    *,
    expected_target_modules: Collection[str] | None = None,
) -> tuple[str, ...]:
    try:
        adapter_config = model.peft_config[_ADAPTER_NAME]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError(f"loaded adapter must define {_ADAPTER_NAME!r} configuration") from error

    _require_loaded_value(adapter_config, "r", _STAGE_6_LORA_R)
    _require_loaded_value(adapter_config, "lora_alpha", _STAGE_6_LORA_ALPHA)
    _require_loaded_value(adapter_config, "lora_dropout", _STAGE_6_LORA_DROPOUT)
    if getattr(adapter_config, "init_lora_weights", None) is not True:
        raise ValueError("loaded adapter requires init_lora_weights=True")
    if getattr(adapter_config, "bias", None) != "none":
        raise ValueError("loaded adapter requires bias=none")
    if getattr(adapter_config, "use_rslora", None) is not False:
        raise ValueError("loaded adapter requires use_rslora=False")
    if getattr(adapter_config, "use_dora", None) is not False:
        raise ValueError("loaded adapter requires use_dora=False")
    if getattr(adapter_config, "modules_to_save", None) is not None:
        raise ValueError("loaded adapter requires modules_to_save=None")
    if getattr(adapter_config, "rank_pattern", None) != {}:
        raise ValueError("loaded adapter requires an empty rank_pattern")
    if getattr(adapter_config, "alpha_pattern", None) != {}:
        raise ValueError("loaded adapter requires an empty alpha_pattern")
    for field in (
        "exclude_modules",
        "layers_to_transform",
        "layers_pattern",
        "trainable_token_indices",
        "layer_replication",
        "target_parameters",
    ):
        if getattr(adapter_config, field, None) is not None:
            raise ValueError(f"loaded adapter requires {field}=None")
    if getattr(adapter_config, "lora_bias", None) is not False:
        raise ValueError("loaded adapter requires lora_bias=False")
    task_type = getattr(adapter_config, "task_type", None)
    if getattr(task_type, "value", task_type) != "CAUSAL_LM":
        raise ValueError("loaded adapter requires task_type=CAUSAL_LM")
    resolved_targets = _validate_loaded_target_modules(
        getattr(adapter_config, "target_modules", None)
    )
    if expected_target_modules is not None and resolved_targets != tuple(expected_target_modules):
        raise ValueError("loaded adapter target_modules do not match the adapter contract")
    return resolved_targets


def _require_loaded_value(adapter_config: Any, name: str, expected: int | float) -> None:
    if getattr(adapter_config, name, None) != expected:
        raise ValueError(f"loaded adapter requires {name}={expected}")


def _validate_loaded_target_modules(target_modules: Any) -> tuple[str, ...]:
    if target_modules == _STAGE_6_TARGET_MODULES:
        raise ValueError("loaded adapter target_modules must contain resolved concrete suffixes")
    if isinstance(target_modules, str):
        raise ValueError(
            "loaded adapter target_modules must be 'all-linear' or non-empty concrete suffixes"
        )
    if not isinstance(target_modules, Collection) or not target_modules:
        raise ValueError(
            "loaded adapter target_modules must be 'all-linear' or non-empty concrete suffixes"
        )
    if any(not isinstance(target, str) or not target.strip() for target in target_modules):
        raise ValueError(
            "loaded adapter target_modules must be 'all-linear' or non-empty concrete suffixes"
        )
    return tuple(sorted(set(target_modules)))


def _eligible_linear_module_names(model: Any) -> tuple[str, ...]:
    if not callable(getattr(model, "named_modules", None)):
        raise TypeError("base model must expose named_modules()")
    try:
        from torch import nn
    except ImportError as error:
        raise RuntimeError("all-linear target validation requires torch") from error
    output_layer = None
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        output_layer = get_output_embeddings()
    names = tuple(
        sorted(
            name
            for name, module in model.named_modules()
            if name and isinstance(module, nn.Linear) and module is not output_layer
        )
    )
    if not names:
        raise ValueError("base model has no eligible linear modules for all-linear LoRA")
    return names


def _loaded_target_module_instances(model: Any) -> tuple[str, ...]:
    for owner in (getattr(model, "base_model", None), model):
        value = getattr(owner, "targeted_module_names", None)
        if value is not None:
            return _validate_name_collection(value, "loaded targeted_module_names")
    raise ValueError("loaded adapter does not expose targeted_module_names")


def _validate_contract_name_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"Stage 6 adapter contract {field} must be a list")
    names = _validate_name_collection(value, f"Stage 6 adapter contract {field}")
    if value != list(names):
        raise ValueError(f"Stage 6 adapter contract {field} must be sorted and unique")
    return names


def _validate_name_collection(value: Any, label: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Collection)
        or not value
        or any(not isinstance(name, str) or not name.strip() for name in value)
    ):
        raise ValueError(f"{label} must contain module names")
    return tuple(sorted(set(value)))


def _stage6_adapter_options() -> dict[str, Any]:
    return {
        "alpha_pattern": {},
        "bias": "none",
        "exclude_modules": None,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": _STAGE_6_LORA_ALPHA,
        "lora_bias": False,
        "lora_dropout": _STAGE_6_LORA_DROPOUT,
        "modules_to_save": None,
        "r": _STAGE_6_LORA_R,
        "rank_pattern": {},
        "task_type": "CAUSAL_LM",
        "target_parameters": None,
        "trainable_token_indices": None,
        "use_dora": False,
        "use_rslora": False,
    }


def _has_exact_typed_values(actual: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    return all(
        type(actual[name]) is type(value) and actual[name] == value
        for name, value in expected.items()
    )


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
