from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionCapabilities:
    """Resolved permissions for one online execution request."""

    can_read_memory: bool
    can_write_memory: bool
    can_run_maintenance: bool
    can_use_train_split: bool
    can_use_test_split: bool
    source_memory_read_only: bool

    @property
    def split(self) -> str:
        if self.can_use_train_split:
            return "train"
        return "test"
