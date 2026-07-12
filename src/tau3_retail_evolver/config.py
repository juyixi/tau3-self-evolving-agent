from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from tau3_retail_evolver.credential_policy import is_credential_key


_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_AGENT_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


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
    agent_id: str = "retail"
    tiers: tuple[Literal["trajectory", "tip", "skill", "tool"], ...] = (
        "trajectory",
        "tip",
        "skill",
        "tool",
    )
    retrieve_top_k: int = 50
    teacher_memory_cap: int = 20
    score_threshold: float = 0.01
    maintenance_period: int = 30
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cuda"
    embedding_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    embedding_max_length: int = Field(default=2048, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)
    embedding_cache: bool = True

    @field_validator("agent_id")
    @classmethod
    def agent_id_must_be_a_safe_slug(cls, value: str) -> str:
        if not _AGENT_ID.fullmatch(value) or value in {".", ".."}:
            raise ValueError("agent_id must contain only ASCII letters, digits, '-' or '_'")
        return value


class TrainingConfig(_ConfigModel):
    seed: int = 42


class NLAssertionsConfig(_ConfigModel):
    model: str = "openrouter/openai/gpt-4.1"
    model_args: dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.0})
    api_key_env: str = "OPENROUTER_API_KEY"

    @field_validator("model")
    @classmethod
    def model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model must not be empty")
        return value

    @field_validator("api_key_env")
    @classmethod
    def api_key_env_must_be_a_valid_environment_variable_name(cls, value: str) -> str:
        if not _ENVIRONMENT_VARIABLE_NAME.fullmatch(value):
            raise ValueError("api_key_env must be a valid environment variable name")
        return value

    @field_validator("model_args", mode="before")
    @classmethod
    def model_args_must_not_contain_credentials(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("model_args must be a mapping")
        if _contains_credential_key(value):
            raise ValueError("model_args must not contain credential-bearing keys")
        return value


class EvaluationConfig(_ConfigModel):
    nl_assertions: NLAssertionsConfig = Field(default_factory=NLAssertionsConfig)


class ProjectConfig(_ConfigModel):
    tau2: Tau2Config
    model: ModelConfig
    lora: LoraConfig = Field(default_factory=LoraConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


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


def _contains_credential_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if is_credential_key(key):
                return True
            if _contains_credential_key(nested_value):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_credential_key(item) for item in value)
    return False
