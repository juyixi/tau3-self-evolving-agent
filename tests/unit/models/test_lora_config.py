from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from tau3_retail_evolver.config import LoraConfig, ModelConfig, TrainingConfig


def _lora_module() -> Any:
    spec = importlib.util.find_spec("tau3_retail_evolver.models.lora")
    assert spec is not None, "the LoRA adapter lifecycle module must exist"
    return importlib.import_module("tau3_retail_evolver.models.lora")


def _qwen35_module() -> Any:
    spec = importlib.util.find_spec("tau3_retail_evolver.models.qwen35")
    assert spec is not None, "the Qwen3.5 loader module must exist"
    return importlib.import_module("tau3_retail_evolver.models.qwen35")


class RecordingLoraConfig:
    calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.config = SimpleNamespace(use_cache=True)
        self.gradient_checkpointing_calls = 0

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_calls += 1


class FakeAdapterModel(nn.Module):
    def __init__(self, base_model: FakeBaseModel, adapter_config: RecordingLoraConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.lora_A = nn.Parameter(torch.ones((2, 2)))
        self.lora_B = nn.Parameter(torch.zeros((2, 2)))
        self.config = base_model.config
        self.peft_config = {"shared_policy": adapter_config}
        self.save_calls: list[dict[str, Any]] = []

    def save_pretrained(self, directory: str | Path, **kwargs: Any) -> None:
        self.save_calls.append({"directory": Path(directory), **kwargs})
        selected_adapters = kwargs["selected_adapters"]
        if selected_adapters != ["shared_policy"]:
            raise AssertionError("PEFT saves a non-default adapter by selected adapter name")
        adapter_directory = Path(directory) / selected_adapters[0]
        adapter_directory.mkdir(parents=True, exist_ok=True)
        (adapter_directory / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_directory / "adapter-model.bin").write_bytes(b"adapter only")


class FakeAutoModelForCausalLM:
    calls: list[dict[str, Any]] = []
    model: FakeBaseModel | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> FakeBaseModel:
        cls.calls.append({"model_id": model_id, **kwargs})
        cls.model = FakeBaseModel()
        return cls.model


class FakeAutoProcessor:
    calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> object:
        cls.calls.append({"model_id": model_id, **kwargs})
        return {"processor_for": model_id}


class FakePeftModule:
    LoraConfig = RecordingLoraConfig
    TaskType = SimpleNamespace(CAUSAL_LM="CAUSAL_LM")
    get_peft_calls: list[dict[str, Any]] = []
    load_calls: list[dict[str, Any]] = []
    state_dict_calls: list[dict[str, Any]] = []
    adapter_state: dict[str, torch.Tensor] = {}
    loaded_adapter_config: dict[str, Any] = {}

    @classmethod
    def get_peft_model(
        cls,
        model: FakeBaseModel,
        config: RecordingLoraConfig,
        *,
        adapter_name: str,
    ) -> FakeAdapterModel:
        cls.get_peft_calls.append(
            {"model": model, "config": config, "adapter_name": adapter_name}
        )
        return FakeAdapterModel(model, config)

    class PeftModel:
        @staticmethod
        def from_pretrained(
            model: FakeBaseModel,
            adapter_path: str | Path,
            *,
            adapter_name: str,
            is_trainable: bool,
        ) -> FakeAdapterModel:
            FakePeftModule.load_calls.append(
                {
                    "model": model,
                    "adapter_path": Path(adapter_path),
                    "adapter_name": adapter_name,
                    "is_trainable": is_trainable,
                }
            )
            return FakeAdapterModel(
                model,
                RecordingLoraConfig(**FakePeftModule.loaded_adapter_config),
            )

    @classmethod
    def get_peft_model_state_dict(
        cls,
        model: FakeAdapterModel,
        *,
        adapter_name: str,
    ) -> dict[str, torch.Tensor]:
        if adapter_name != "shared_policy":
            raise AssertionError("adapter state must be requested by its persistent name")
        cls.state_dict_calls.append({"model": model, "adapter_name": adapter_name})
        return cls.adapter_state or {
            "base_model.model.lora_A.shared_policy.weight": model.lora_A,
            "base_model.model.lora_B.shared_policy.weight": model.lora_B,
        }


@pytest.fixture(autouse=True)
def reset_fakes() -> None:
    RecordingLoraConfig.calls = []
    FakeAutoModelForCausalLM.calls = []
    FakeAutoModelForCausalLM.model = None
    FakeAutoProcessor.calls = []
    FakePeftModule.get_peft_calls = []
    FakePeftModule.load_calls = []
    FakePeftModule.state_dict_calls = []
    FakePeftModule.adapter_state = {}
    FakePeftModule.loaded_adapter_config = _loaded_adapter_config()


def _transformers_module() -> Any:
    return SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoProcessor=FakeAutoProcessor,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(base_model="Qwen/Qwen3.5-9B")


def _loaded_adapter_config(**overrides: Any) -> dict[str, Any]:
    return {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "init_lora_weights": True,
        "task_type": "CAUSAL_LM",
        "target_modules": "all-linear",
        **overrides,
    }


def test_build_lora_config_uses_the_project_zero_impact_settings() -> None:
    lora = _lora_module()

    config = lora.build_lora_config(
        LoraConfig(), TrainingConfig(), peft_module=FakePeftModule
    )

    assert isinstance(config, RecordingLoraConfig)
    assert config.kwargs == {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "init_lora_weights": True,
        "target_modules": "all-linear",
        "task_type": "CAUSAL_LM",
    }


def test_build_and_attach_paths_reuse_the_public_stage6_settings_validator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lora = _lora_module()
    calls: list[tuple[LoraConfig, TrainingConfig]] = []
    monkeypatch.setattr(
        lora,
        "validate_stage6_lora_settings",
        lambda lora_config, training_config: calls.append(
            (lora_config, training_config)
        ),
    )
    lora_config = LoraConfig()
    training_config = TrainingConfig()

    lora.build_lora_config(
        lora_config,
        training_config,
        peft_module=FakePeftModule,
    )
    lora.attach_shared_lora_adapter(
        FakeBaseModel(),
        lora_config,
        training_config,
        adapter_path=tmp_path / "adapter",
        peft_module=FakePeftModule,
    )

    assert calls == [
        (lora_config, training_config),
        (lora_config, training_config),
    ]


@pytest.mark.parametrize(
    ("lora_config", "training_config", "match"),
    (
        (LoraConfig(lora_r=16), TrainingConfig(), "lora_r=32"),
        (LoraConfig(lora_alpha=32), TrainingConfig(), "lora_alpha=64"),
        (LoraConfig(lora_dropout=0.1), TrainingConfig(), "lora_dropout=0.05"),
        (LoraConfig(), TrainingConfig(target_modules="q_proj"), "target_modules"),
    ),
)
def test_public_stage6_settings_validator_is_pure_and_rejects_deviations(
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    match: str,
) -> None:
    lora = _lora_module()

    with pytest.raises(ValueError, match=match):
        lora.validate_stage6_lora_settings(lora_config, training_config)


@pytest.mark.parametrize(
    ("lora_config", "training_config", "match"),
    (
        (
            LoraConfig.model_construct(
                use_peft=False,
                lora_r=32,
                lora_alpha=64,
                lora_dropout=0.05,
            ),
            TrainingConfig(),
            "use_peft",
        ),
        (LoraConfig(lora_r=16), TrainingConfig(), "lora_r=32"),
        (LoraConfig(lora_alpha=32), TrainingConfig(), "lora_alpha=64"),
        (LoraConfig(lora_dropout=0.1), TrainingConfig(), "lora_dropout=0.05"),
        (LoraConfig(), TrainingConfig(target_modules="q_proj"), "target_modules='all-linear'"),
    ),
)
def test_build_lora_config_rejects_values_outside_the_mandatory_stage_6_contract(
    lora_config: LoraConfig,
    training_config: TrainingConfig,
    match: str,
) -> None:
    lora = _lora_module()

    with pytest.raises(ValueError, match=match):
        lora.build_lora_config(lora_config, training_config, peft_module=FakePeftModule)

    assert RecordingLoraConfig.calls == []


@pytest.mark.parametrize("initializer", (False, "gaussian", "pissa"))
def test_build_lora_config_rejects_initializers_that_do_not_preserve_base_output(
    initializer: bool | str,
) -> None:
    lora = _lora_module()

    with pytest.raises(ValueError, match="zero-impact"):
        lora.build_lora_config(
            LoraConfig(),
            TrainingConfig(),
            init_lora_weights=initializer,
            peft_module=FakePeftModule,
        )

    assert RecordingLoraConfig.calls == []


def test_load_shared_policy_creates_one_zero_impact_adapter_and_freezes_base_parameters() -> None:
    qwen35 = _qwen35_module()

    model = qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        revision="model-commit-a",
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    assert FakeAutoModelForCausalLM.calls == [
        {
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "model-commit-a",
            "torch_dtype": torch.bfloat16,
        }
    ]
    assert FakeAutoModelForCausalLM.model is not None
    assert FakeAutoModelForCausalLM.model.gradient_checkpointing_calls == 1
    assert model.config.use_cache is False
    assert len(FakePeftModule.get_peft_calls) == 1
    assert FakePeftModule.get_peft_calls[0]["adapter_name"] == "shared_policy"
    assert FakePeftModule.load_calls == []
    assert model is not FakeAutoModelForCausalLM.model
    assert model.base_model.weight.requires_grad is False
    assert model.lora_A.requires_grad is True
    assert model.lora_B.requires_grad is True
    assert _lora_module().assert_only_lora_trainable(model) == 2


def test_load_shared_policy_reuses_one_trainable_existing_adapter(tmp_path: Path) -> None:
    qwen35 = _qwen35_module()
    adapter_path = tmp_path / "existing-adapter"

    model = qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        adapter_path=adapter_path,
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    assert FakePeftModule.get_peft_calls == []
    assert FakePeftModule.load_calls == [
        {
            "model": FakeAutoModelForCausalLM.model,
            "adapter_path": adapter_path,
            "adapter_name": "shared_policy",
            "is_trainable": True,
        }
    ]
    assert len(model.peft_config) == 1
    assert _lora_module().assert_only_lora_trainable(model) == 2


def test_attach_shared_adapter_rejects_project_settings_before_loading_an_adapter(
    tmp_path: Path,
) -> None:
    lora = _lora_module()

    with pytest.raises(ValueError, match="lora_r=32"):
        lora.attach_shared_lora_adapter(
            FakeBaseModel(),
            LoraConfig(lora_r=16),
            TrainingConfig(),
            adapter_path=tmp_path / "adapter",
            peft_module=FakePeftModule,
        )

    assert FakePeftModule.get_peft_calls == []
    assert FakePeftModule.load_calls == []


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"r": 16}, "r=32"),
        ({"lora_alpha": 32}, "lora_alpha=64"),
        ({"lora_dropout": 0.1}, "lora_dropout=0.05"),
        ({"init_lora_weights": False}, "init_lora_weights=True"),
        ({"init_lora_weights": "gaussian"}, "init_lora_weights=True"),
        ({"task_type": "SEQ_2_SEQ_LM"}, "task_type=CAUSAL_LM"),
        ({"target_modules": ""}, "target_modules"),
        ({"target_modules": "q_proj"}, "target_modules"),
    ),
)
def test_load_shared_policy_rejects_an_incompatible_loaded_adapter_config(
    tmp_path: Path,
    overrides: dict[str, Any],
    match: str,
) -> None:
    qwen35 = _qwen35_module()
    FakePeftModule.loaded_adapter_config = _loaded_adapter_config(**overrides)

    with pytest.raises(ValueError, match=match):
        qwen35.load_shared_qwen35_policy(
            _model_config(),
            LoraConfig(),
            TrainingConfig(),
            adapter_path=tmp_path / "adapter",
            transformers_module=_transformers_module(),
            peft_module=FakePeftModule,
        )


def test_load_shared_policy_accepts_expanded_loaded_target_modules(tmp_path: Path) -> None:
    qwen35 = _qwen35_module()
    FakePeftModule.loaded_adapter_config = _loaded_adapter_config(
        target_modules={"q_proj", "k_proj", "v_proj"}
    )

    model = qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        adapter_path=tmp_path / "adapter",
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    assert model.peft_config["shared_policy"].target_modules == {
        "q_proj",
        "k_proj",
        "v_proj",
    }


def test_load_qwen35_processor_accepts_a_model_id_or_local_path_without_loading_a_model(
    tmp_path: Path,
) -> None:
    qwen35 = _qwen35_module()

    processor = qwen35.load_qwen35_processor(
        tmp_path,
        revision="model-commit-a",
        transformers_module=_transformers_module(),
    )

    assert processor == {"processor_for": str(tmp_path)}
    assert FakeAutoProcessor.calls == [
        {"model_id": str(tmp_path), "revision": "model-commit-a"}
    ]
    assert FakeAutoModelForCausalLM.calls == []


def test_qwen_loaders_can_require_cached_files_without_downloading(tmp_path: Path) -> None:
    qwen35 = _qwen35_module()

    qwen35.load_qwen35_processor(
        tmp_path,
        revision="model-commit-a",
        local_files_only=True,
        transformers_module=_transformers_module(),
    )
    qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        revision="model-commit-a",
        local_files_only=True,
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    assert FakeAutoProcessor.calls == [
        {
            "model_id": str(tmp_path),
            "revision": "model-commit-a",
            "local_files_only": True,
        }
    ]
    assert FakeAutoModelForCausalLM.calls == [
        {
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "model-commit-a",
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
        }
    ]


def test_save_adapter_checkpoint_writes_only_peft_adapter_tensors(tmp_path: Path) -> None:
    qwen35 = _qwen35_module()
    model = qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    checkpoint = _lora_module().save_adapter_checkpoint(
        model, tmp_path / "adapter", peft_module=FakePeftModule
    )

    assert checkpoint == tmp_path / "adapter" / "shared_policy"
    assert model.save_calls == [
        {
            "directory": tmp_path / "adapter",
            "state_dict": {
                "base_model.model.lora_A.shared_policy.weight": model.lora_A,
                "base_model.model.lora_B.shared_policy.weight": model.lora_B,
            },
            "safe_serialization": True,
            "selected_adapters": ["shared_policy"],
        }
    ]
    assert FakePeftModule.state_dict_calls == [
        {"model": model, "adapter_name": "shared_policy"}
    ]
    assert (checkpoint / "adapter_config.json").is_file()
    assert (checkpoint / "adapter-model.bin").is_file()


def test_save_adapter_checkpoint_returns_the_exact_path_used_for_adapter_reload(
    tmp_path: Path,
) -> None:
    qwen35 = _qwen35_module()
    lora = _lora_module()
    model = qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    checkpoint = lora.save_adapter_checkpoint(
        model, tmp_path / "adapter", peft_module=FakePeftModule
    )
    qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        adapter_path=checkpoint,
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )

    assert checkpoint == tmp_path / "adapter" / "shared_policy"
    assert (checkpoint / "adapter_config.json").is_file()
    assert FakePeftModule.load_calls[-1]["adapter_path"] == checkpoint


def test_save_adapter_checkpoint_rejects_a_base_model_tensor(tmp_path: Path) -> None:
    qwen35 = _qwen35_module()
    model = qwen35.load_shared_qwen35_policy(
        _model_config(),
        LoraConfig(),
        TrainingConfig(),
        transformers_module=_transformers_module(),
        peft_module=FakePeftModule,
    )
    FakePeftModule.adapter_state = {"base_model.model.weight": model.base_model.weight}

    with pytest.raises(ValueError, match="base-model"):
        _lora_module().save_adapter_checkpoint(model, tmp_path / "adapter", peft_module=FakePeftModule)

    assert model.save_calls == []
