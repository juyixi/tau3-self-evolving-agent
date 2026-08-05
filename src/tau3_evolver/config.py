from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from tau3_evolver.credential_policy import is_credential_key


_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Tau2Config(_ConfigModel):
    repo_path: Path
    user_llm: str
    user_llm_args: dict[str, Any] = Field(default_factory=dict)
    solo_mode: bool = True


class ModelConfig(_ConfigModel):
    base_model: Literal["Qwen/Qwen3.5-9B"]
    serving_base_url: str = "http://127.0.0.1:8000/v1"
    served_model_name: str = "Qwen/Qwen3.5-9B"
    api_key_env: str = "QWEN_API_KEY"
    max_tokens: int = Field(default=8192, ge=1)
    request_timeout_s: float = Field(default=600.0, gt=0)
    generation_settings: dict[str, Any] = Field(
        default_factory=lambda: {
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 20,
            "presence_penalty": 1.5,
            "parallel_tool_calls": False,
        }
    )

    @field_validator("api_key_env")
    @classmethod
    def api_key_env_must_be_valid(cls, value: str) -> str:
        if not _ENVIRONMENT_VARIABLE_NAME.fullmatch(value):
            raise ValueError("api_key_env must be a valid environment variable name")
        return value


class LoraConfig(_ConfigModel):
    use_peft: Literal[True] = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05


class RolloutConfig(_ConfigModel):
    temperature: float = 1.0
    top_p: float = 0.95
    max_episode_steps: int = 40


class ExecutionConfig(_ConfigModel):
    max_concurrency: int = Field(default=3, ge=1)
    seed: int = 42


class MemoryConfig(_ConfigModel):
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
    maintenance_tip_capacity: int = Field(default=200, ge=1)
    maintenance_similarity_threshold: float = Field(default=0.92, ge=-1.0, le=1.0)
    maintenance_priority_pair_limit: int = Field(default=24, ge=0, le=50)
    max_new_tips_per_episode: int = Field(default=2, ge=0)
    max_new_skills_per_episode: int = Field(default=1, ge=0)
    max_new_tools_per_episode: int = Field(default=1, ge=0)
    max_new_trajectories_per_episode: int = Field(default=1, ge=0)
    retrieval_mmr_lambda_tip: float = Field(default=0.65, ge=0.0, le=1.0)
    retrieval_mmr_lambda_skill: float = Field(default=0.80, ge=0.0, le=1.0)
    retrieval_mmr_lambda_tool: float = Field(default=0.85, ge=0.0, le=1.0)
    retrieval_mmr_lambda_trajectory: float = Field(default=0.75, ge=0.0, le=1.0)
    retrieval_global_mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    retrieval_quota_tip: int = Field(default=18, ge=0)
    retrieval_quota_skill: int = Field(default=18, ge=0)
    retrieval_quota_tool: int = Field(default=6, ge=0)
    retrieval_quota_trajectory: int = Field(default=4, ge=0)
    selection_max_total: int = Field(default=20, ge=1)
    selection_max_tip: int = Field(default=7, ge=0)
    selection_max_skill: int = Field(default=8, ge=0)
    selection_max_tool: int = Field(default=3, ge=0)
    selection_max_trajectory: int = Field(default=2, ge=0)
    embedding_provider: Literal["local"] = "local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cuda"
    embedding_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    embedding_max_length: int = Field(default=2048, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)
    embedding_cache: bool = True

class TrainingConfig(_ConfigModel):
    seed: int = 42
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    target_modules: str | tuple[str, ...] = "all-linear"
    max_sequence_length: int = Field(default=8192, ge=128)
    gradient_checkpointing: StrictBool = True
    learning_rate: float = Field(default=1e-5, gt=0)
    per_device_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_train_epochs: int = Field(default=3, ge=1)
    generation_max_new_tokens: int = Field(default=512, ge=1)
    loss_type: Literal["forward_kl"] = "forward_kl"


class SlowLoopConfig(_ConfigModel):
    tier_priors: dict[Literal["trajectory", "tip", "skill", "tool"], float] = Field(
        default_factory=lambda: {
            "trajectory": 0.9,
            "tip": 0.8,
            "skill": 1.0,
            "tool": 1.2,
        }
    )
    redundancy_threshold: float = Field(default=0.90, ge=-1.0, le=1.0)
    max_redundancy_pairs: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def tier_priors_must_have_exact_keys(self) -> "SlowLoopConfig":
        expected = {"trajectory", "tip", "skill", "tool"}
        if set(self.tier_priors) != expected:
            raise ValueError("tier_priors must define exactly trajectory, tip, skill, and tool")
        return self


class NLAssertionsConfig(_ConfigModel):
    model: str = "deepseek/deepseek-v4-pro"
    model_args: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
        }
    )
    api_key_env: str = "DEEPSEEK_API_KEY"

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
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    slow_loop: SlowLoopConfig = Field(default_factory=SlowLoopConfig)
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
