from __future__ import annotations

from typing import Any, Protocol


class EventWriter(Protocol):
    def append(self, event: dict[str, Any]) -> None: ...


class LifecycleContext(Protocol):
    event_writer: EventWriter

    def event(
        self,
        event_type: str,
        task_id: str,
        **payload: Any,
    ) -> dict[str, Any]: ...


__all__ = ["EventWriter", "LifecycleContext"]
