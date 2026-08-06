from pathlib import Path

import pytest
from pydantic import ValidationError

from tau3_evolver.execution import ExecutionRequest


def _request(**overrides: object) -> ExecutionRequest:
    values: dict[str, object] = {
        "benchmark": "retail",
        "mode": "train",
        "memory_enabled": True,
        "run_id": "run-1",
    }
    values.update(overrides)
    return ExecutionRequest.model_validate(values)


def test_train_memory_enables_read_write_and_maintenance() -> None:
    request = _request()

    assert request.capabilities.can_read_memory
    assert request.capabilities.can_write_memory
    assert request.capabilities.can_run_maintenance
    assert request.capabilities.can_use_train_split
    assert not request.capabilities.can_use_test_split


def test_test_memory_requires_frozen_snapshot_and_is_read_only() -> None:
    request = _request(
        benchmark="airline",
        mode="test",
        memory_source="retail",
        memory_snapshot=Path("snapshots/s1"),
    )

    assert request.resolved_memory_source("airline") == "retail"
    assert request.is_cross_domain_memory("airline")
    assert request.capabilities.can_read_memory
    assert not request.capabilities.can_write_memory
    assert not request.capabilities.can_run_maintenance
    assert request.capabilities.source_memory_read_only


def test_test_memory_without_snapshot_is_rejected() -> None:
    with pytest.raises(ValidationError, match="memory_snapshot"):
        _request(mode="test")


@pytest.mark.parametrize("field", ("memory_source", "memory_snapshot"))
def test_no_memory_rejects_memory_inputs(field: str) -> None:
    value = "retail" if field == "memory_source" else Path("snapshot")
    with pytest.raises(ValidationError, match=field):
        _request(memory_enabled=False, **{field: value})


def test_default_memory_source_comes_from_benchmark_definition() -> None:
    request = _request(benchmark="airline", memory_source=None)

    assert request.resolved_memory_source("airline") == "airline"
    assert not request.is_cross_domain_memory("airline")


def test_cross_domain_training_is_resolved_after_benchmark_preparation() -> None:
    request = _request(benchmark="airline", memory_source="retail")

    assert request.memory_snapshot is None
    assert request.memory_source == "retail"


def test_debug_train_uses_isolated_default_memory_namespace() -> None:
    request = _request(debug=True)

    assert request.resolved_memory_source("retail") == "retail-debug"
    assert request.destination_memory_namespace("retail") == "retail-debug"
    assert not request.is_cross_domain_memory("retail")
    assert request.capabilities.can_write_memory


def test_debug_test_run_is_read_only() -> None:
    request = _request(mode="test", memory_enabled=False, debug=True)

    assert request.debug
    assert not request.capabilities.can_write_memory


def test_debug_train_defers_source_namespace_resolution() -> None:
    request = _request(debug=True, memory_source="retail")

    assert request.resolved_memory_source("retail") == "retail"
    assert request.destination_memory_namespace("retail") == "retail-debug"
