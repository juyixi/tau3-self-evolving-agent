import pytest

from tau3_evolver.benchmarks.registry import BenchmarkRegistry, benchmark_registry
from tau3_evolver.benchmarks.tau2.definitions import TAU2_BENCHMARK_DEFINITIONS


def test_default_registry_is_derived_from_static_definitions() -> None:
    expected = tuple(sorted(item.name for item in TAU2_BENCHMARK_DEFINITIONS))

    assert benchmark_registry.names() == expected
    for definition in TAU2_BENCHMARK_DEFINITIONS:
        assert benchmark_registry.resolve(definition.name) is definition


def test_registry_rejects_unknown_or_duplicate_benchmarks() -> None:
    definition = TAU2_BENCHMARK_DEFINITIONS[0]
    registry = BenchmarkRegistry((definition,))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)
    with pytest.raises(ValueError, match="unknown benchmark"):
        registry.resolve("missing")
