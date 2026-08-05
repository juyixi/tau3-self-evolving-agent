from __future__ import annotations

from ast import literal_eval
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
import os
from pathlib import Path
import re

from tau3_evolver.config import ProjectConfig
from tau3_evolver.memory.paths import project_root


_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class EnvironmentLoadResult:
    path: Path
    loaded_names: tuple[str, ...]


class OnlineCredentialError(ValueError):
    """Raised before online execution when a required credential is unavailable."""


def load_project_environment(
    *,
    root: Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> EnvironmentLoadResult:
    """Load the canonical project .env without overriding process variables."""
    target = ((root or project_root()).resolve() / ".env").resolve()
    environment = os.environ if environ is None else environ
    if not target.is_file():
        return EnvironmentLoadResult(path=target, loaded_names=())

    values = _read_env_file(target)
    loaded: list[str] = []
    for name, value in values.items():
        if name in environment:
            continue
        environment[name] = value
        loaded.append(name)
    return EnvironmentLoadResult(path=target, loaded_names=tuple(loaded))


def preflight_online_credentials(
    config: ProjectConfig,
    *,
    environ: Mapping[str, str] | None = None,
    env_path: Path | None = None,
) -> None:
    """Fail before Benchmark preparation when online evaluator credentials are absent."""
    environment = os.environ if environ is None else environ
    requirements = (
        (config.tau2.user_api_key_env, "Tau2 user simulator"),
        (
            config.evaluation.nl_assertions.api_key_env,
            "Tau2 NL assertion evaluator",
        ),
    )
    consumers_by_name: dict[str, list[str]] = {}
    for name, consumer in requirements:
        consumers_by_name.setdefault(name, []).append(consumer)

    missing = {
        name: consumers
        for name, consumers in consumers_by_name.items()
        if not isinstance(environment.get(name), str)
        or not environment[name].strip()
    }
    if not missing:
        return

    details = "; ".join(
        f"{name} ({', '.join(consumers)})"
        for name, consumers in sorted(missing.items())
    )
    location = (env_path or (project_root() / ".env")).resolve()
    raise OnlineCredentialError(
        f"Missing or blank online credential environment variable(s): {details}. "
        f"Set them in the process environment or {location}"
    )


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        name, separator, raw_value = stripped.partition("=")
        name = name.strip()
        if not separator or not _ENVIRONMENT_VARIABLE_NAME.fullmatch(name):
            raise ValueError(f"invalid .env assignment at {path}:{line_number}")
        if name in values:
            raise ValueError(
                f"duplicate .env assignment for {name} at {path}:{line_number}"
            )
        values[name] = _parse_env_value(raw_value, path=path, line=line_number)
    return values


def _parse_env_value(raw_value: str, *, path: Path, line: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        try:
            parsed = literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"invalid quoted .env value at {path}:{line}") from error
        if not isinstance(parsed, str):
            raise ValueError(f"non-text .env value at {path}:{line}")
        return parsed
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


__all__ = [
    "EnvironmentLoadResult",
    "OnlineCredentialError",
    "load_project_environment",
    "preflight_online_credentials",
]
