from __future__ import annotations

from collections.abc import Iterable

from tau3_evolver.benchmarks.types import BenchmarkDefinition


class BenchmarkRegistry:
    def __init__(self, definitions: Iterable[BenchmarkDefinition] = ()) -> None:
        self._definitions: dict[str, BenchmarkDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: BenchmarkDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"benchmark {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def resolve(self, name: str) -> BenchmarkDefinition:
        try:
            return self._definitions[name]
        except KeyError as error:
            available = ", ".join(self.names())
            raise ValueError(
                f"unknown benchmark {name!r}; available benchmarks: {available}"
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))


def _default_registry() -> BenchmarkRegistry:
    from tau3_evolver.benchmarks.tau2.definitions import AIRLINE, RETAIL

    return BenchmarkRegistry((RETAIL, AIRLINE))


benchmark_registry = _default_registry()
