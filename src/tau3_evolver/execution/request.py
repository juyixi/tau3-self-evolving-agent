from __future__ import annotations

from enum import StrEnum
from pathlib import Path
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau3_evolver.execution.capabilities import ExecutionCapabilities


_SAFE_SLUG = re.compile(r"^[a-z0-9_-]+$")


class BenchmarkName(StrEnum):
    RETAIL = "retail"
    AIRLINE = "airline"


class ExecutionMode(StrEnum):
    TRAIN = "train"
    TEST = "test"


class ExecutionRequest(BaseModel):
    """Typed boundary between the CLI and online execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkName
    mode: ExecutionMode
    debug: bool = False
    memory_enabled: bool
    memory_source: str | None = None
    memory_snapshot: Path | None = None
    checkpoint: Path | None = None
    config_path: Path = Path("configs/default.yaml")
    overrides: tuple[str, ...] = ()
    run_id: str
    output_root: Path = Path("runs")

    @model_validator(mode="after")
    def validate_execution_combination(self) -> "ExecutionRequest":
        self._validate_slug(self.run_id, field="run_id")
        if self.memory_source is not None:
            self._validate_slug(self.memory_source, field="memory_source")
        if not self.memory_enabled:
            if self.memory_source is not None:
                raise ValueError("memory_source requires memory to be enabled")
            if self.memory_snapshot is not None:
                raise ValueError("memory_snapshot requires memory to be enabled")
        if (
            self.mode is ExecutionMode.TEST
            and self.memory_enabled
            and self.memory_snapshot is None
        ):
            raise ValueError("test mode with memory requires a frozen memory_snapshot")
        if (
            self.mode is ExecutionMode.TRAIN
            and self.memory_enabled
            and self.memory_source is not None
            and self.memory_source
            != self.destination_memory_namespace(self.benchmark.value)
            and self.memory_snapshot is None
        ):
            raise ValueError(
                "cross-domain training requires an explicit frozen memory_snapshot"
            )
        return self

    @property
    def capabilities(self) -> ExecutionCapabilities:
        is_train = self.mode is ExecutionMode.TRAIN
        return ExecutionCapabilities(
            can_read_memory=self.memory_enabled,
            can_write_memory=is_train and self.memory_enabled,
            can_run_maintenance=is_train and self.memory_enabled,
            can_use_train_split=is_train,
            can_use_test_split=not is_train,
            source_memory_read_only=(
                not is_train
                or (
                    self.memory_source is not None
                    and self.memory_source != self.benchmark.value
                )
            ),
        )

    def resolved_memory_source(self, default_namespace: str) -> str | None:
        if not self.memory_enabled:
            return None
        return self.memory_source or self.destination_memory_namespace(
            default_namespace
        )

    def destination_memory_namespace(self, default_namespace: str) -> str:
        self._validate_slug(default_namespace, field="default_memory_namespace")
        return f"{default_namespace}-debug" if self.debug else default_namespace

    def is_cross_domain_memory(self, default_namespace: str) -> bool:
        source = self.resolved_memory_source(default_namespace)
        return (
            source is not None
            and source != self.destination_memory_namespace(default_namespace)
        )

    @staticmethod
    def _validate_slug(value: str, *, field: str) -> None:
        if not _SAFE_SLUG.fullmatch(value) or value in {".", ".."}:
            raise ValueError(
                f"{field} must contain only lowercase ASCII letters, digits, '-' or '_'"
            )
