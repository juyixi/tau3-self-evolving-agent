from __future__ import annotations

import ast
from pathlib import Path

import pytest


_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "tau3_evolver"


@pytest.mark.parametrize(
    ("package", "forbidden"),
    (
        (
            "memory",
            (
                "tau3_evolver.artifacts",
                "tau3_evolver.benchmarks",
                "tau3_evolver.config",
                "tau3_evolver.execution",
                "tau3_evolver.fast_loop",
                "tau3_evolver.models",
            ),
        ),
        (
            "artifacts",
            (
                "tau3_evolver.benchmarks",
                "tau3_evolver.config",
                "tau3_evolver.execution",
                "tau3_evolver.fast_loop",
                "tau3_evolver.memory",
                "tau3_evolver.models",
            ),
        ),
        (
            "persistence",
            (
                "tau3_evolver.artifacts",
                "tau3_evolver.benchmarks",
                "tau3_evolver.config",
                "tau3_evolver.execution",
                "tau3_evolver.fast_loop",
                "tau3_evolver.memory",
                "tau3_evolver.models",
                "tau3_evolver.security",
            ),
        ),
    ),
)
def test_lower_layers_do_not_import_application_or_sibling_domains(
    package: str,
    forbidden: tuple[str, ...],
) -> None:
    violations: list[str] = []
    for path in sorted((_PACKAGE_ROOT / package).rglob("*.py")):
        for imported in _imports(path):
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in forbidden
            ):
                violations.append(
                    f"{path.relative_to(_PACKAGE_ROOT)} imports {imported}"
                )

    assert not violations, "\n".join(violations)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return tuple(modules)
