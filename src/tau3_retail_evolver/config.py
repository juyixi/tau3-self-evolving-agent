from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tau2Config(_ConfigModel):
    repo_path: Path
    domain: Literal["retail"]
    train_split: Literal["train"]
    eval_split: Literal["test"]
    user_llm: str
    user_llm_args: dict[str, Any] = Field(default_factory=dict)
    solo_mode: bool = True


class ModelConfig(_ConfigModel):
    base_model: Literal["Qwen/Qwen3.5-9B"]


class LoraConfig(_ConfigModel):
    use_peft: Literal[True] = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05


class RolloutConfig(_ConfigModel):
    temperature: float = 1.0
    top_p: float = 0.95
    max_episode_steps: int = 40


class MemoryConfig(_ConfigModel):
    enabled: bool = True
    max_entries: int = 1000


class TrainingConfig(_ConfigModel):
    seed: int = 42


class ProjectConfig(_ConfigModel):
    tau2: Tau2Config
    model: ModelConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def share_the_configured_model(self) -> ProjectConfig:
        if self.tau2.user_llm != self.model.base_model:
            raise ValueError("tau2.user_llm must match model.base_model")
        return self


def load_config(path: Path, overrides: Sequence[str] = ()) -> ProjectConfig:
    """Load a YAML config and apply dotted YAML-scalar overrides."""
    with path.open(encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")

    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid override: {override!r}")
        _set_override(data, key.split("."), yaml.safe_load(raw_value))

    return ProjectConfig.model_validate(data)


def _set_override(data: MutableMapping[str, Any], keys: list[str], value: Any) -> None:
    target = data
    for key in keys[:-1]:
        nested = target.get(key)
        if not isinstance(nested, MutableMapping):
            raise ValueError(f"unknown override path: {'.'.join(keys)}")
        target = nested
    if keys[-1] not in target:
        raise ValueError(f"unknown override path: {'.'.join(keys)}")
    target[keys[-1]] = value
