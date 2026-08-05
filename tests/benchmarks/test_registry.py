import pytest

from tau3_evolver.benchmarks.registry import BenchmarkRegistry, benchmark_registry


def test_default_registry_exposes_retail_and_airline() -> None:
    assert benchmark_registry.names() == ("airline", "retail")
    assert benchmark_registry.resolve("retail").default_memory_namespace == "retail"
    assert benchmark_registry.resolve("airline").default_memory_namespace == "airline"


def test_registry_rejects_unknown_or_duplicate_benchmarks() -> None:
    retail = benchmark_registry.resolve("retail")
    registry = BenchmarkRegistry((retail,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(retail)
    with pytest.raises(ValueError, match="unknown benchmark"):
        registry.resolve("missing")
