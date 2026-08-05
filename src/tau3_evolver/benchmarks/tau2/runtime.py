from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class Tau2RuntimeBinding:
    source_root: Path
    package_version: str | None
    git_commit: str | None
    registry: Any
    run_domain: Callable[[Any], Any]
    text_run_config_type: type[Any]
    task_type: type[Any]
    half_duplex_agent_type: type[Any]
    assistant_message_type: type[Any]
    tool_call_type: type[Any]
    tool_message_type: type[Any]
    multi_tool_message_type: type[Any]


class Tau2Runtime:
    """Bind the first successfully imported Tau2 package for this process."""

    _binding: Tau2RuntimeBinding | None = None
    _lock = threading.Lock()

    @classmethod
    def bind(cls, repo_path: Path) -> Tau2RuntimeBinding:
        with cls._lock:
            if cls._binding is None:
                cls._binding = cls._import_first_runtime(repo_path)
            return cls._binding

    @classmethod
    def current(cls) -> Tau2RuntimeBinding | None:
        return cls._binding

    @classmethod
    def _import_first_runtime(cls, repo_path: Path) -> Tau2RuntimeBinding:
        inserted_source: Path | None = None
        existing_tau2_modules = {
            name for name in sys.modules if name == "tau2" or name.startswith("tau2.")
        }
        try:
            if "tau2" not in sys.modules:
                inserted_source = _source_root(repo_path)
                sys.path.insert(0, str(inserted_source))
                importlib.invalidate_caches()

            tau2_module = importlib.import_module("tau2")
            source_root = _module_source_root(tau2_module)
            registry_module = importlib.import_module("tau2.registry")
            runner_module = importlib.import_module("tau2.runner.batch")
            simulation_module = importlib.import_module("tau2.data_model.simulation")
            tasks_module = importlib.import_module("tau2.data_model.tasks")
            agent_module = importlib.import_module("tau2.agent.base_agent")
            message_module = importlib.import_module("tau2.data_model.message")
            return Tau2RuntimeBinding(
                source_root=source_root,
                package_version=_package_version(source_root),
                git_commit=_git_commit(source_root),
                registry=registry_module.registry,
                run_domain=runner_module.run_domain,
                text_run_config_type=simulation_module.TextRunConfig,
                task_type=tasks_module.Task,
                half_duplex_agent_type=agent_module.HalfDuplexAgent,
                assistant_message_type=message_module.AssistantMessage,
                tool_call_type=message_module.ToolCall,
                tool_message_type=message_module.ToolMessage,
                multi_tool_message_type=message_module.MultiToolMessage,
            )
        except Exception as error:
            if inserted_source is not None:
                try:
                    sys.path.remove(str(inserted_source))
                except ValueError:
                    pass
            for name in list(sys.modules):
                if (
                    name not in existing_tau2_modules
                    and (name == "tau2" or name.startswith("tau2."))
                ):
                    sys.modules.pop(name, None)
            raise RuntimeError(
                f"unable to import Tau2 runtime from the first available source: {error}"
            ) from error


def _source_root(repo_path: Path) -> Path:
    repo = Path(repo_path).expanduser().resolve()
    for candidate in (repo / "src", repo):
        if (candidate / "tau2" / "__init__.py").is_file():
            return candidate.resolve()
    raise RuntimeError(f"Tau2 package is unavailable under {repo}")


def _module_source_root(module: Any) -> Path:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        raise RuntimeError("imported Tau2 package has no filesystem origin")
    package_dir = Path(module_file).resolve().parent
    return package_dir.parent


def _package_version(source_root: Path) -> str | None:
    repo = source_root.parent if source_root.name == "src" else source_root
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        try:
            import tomllib

            with pyproject.open("rb") as source:
                value = tomllib.load(source).get("project", {}).get("version")
            if isinstance(value, str):
                return value
        except (OSError, tomllib.TOMLDecodeError):
            pass
    for distribution in ("tau2-bench", "tau2"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _git_commit(source_root: Path) -> str | None:
    repo = source_root.parent if source_root.name == "src" else source_root
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None
