from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeFingerprint:
    repo_path: Path
    git_commit: str
    package_version: str
    retail_tasks_path: Path
    retail_split_path: Path
    gym_available: bool
    domain: str = "retail"

    @property
    def tasks_path(self) -> Path:
        return self.retail_tasks_path

    @property
    def split_path(self) -> Path:
        return self.retail_split_path


@dataclass(frozen=True)
class Tau2RunDomainRuntime:
    run_domain: Callable[..., Any]
    text_run_config: type
    registry: Any
    half_duplex_agent: type
    assistant_message: type
    tool_call: type
    user_message: type
    tool_message: type
    multi_tool_message: type


class Tau2Runtime:
    """Probe a configured Tau2 checkout without import-time path mutation."""

    @staticmethod
    def inspect(repo_path: Path, *, domain: str = "retail") -> RuntimeFingerprint:
        fingerprint = Tau2Runtime.inspect_metadata(repo_path, domain=domain)
        Tau2Runtime.probe_gym(fingerprint.repo_path)
        return RuntimeFingerprint(
            repo_path=fingerprint.repo_path,
            git_commit=fingerprint.git_commit,
            package_version=fingerprint.package_version,
            retail_tasks_path=fingerprint.retail_tasks_path,
            retail_split_path=fingerprint.retail_split_path,
            gym_available=True,
            domain=fingerprint.domain,
        )

    @staticmethod
    def inspect_metadata(
        repo_path: Path,
        *,
        domain: str = "retail",
    ) -> RuntimeFingerprint:
        """Inspect checkout metadata and domain paths without importing Tau2."""
        if domain not in {"retail", "airline"}:
            raise ValueError(f"unsupported Tau2 domain: {domain!r}")
        resolved_repo_path = Path(repo_path).expanduser().resolve()
        if not resolved_repo_path.is_dir():
            raise RuntimeError(
                f"Tau2 checkout is missing at {resolved_repo_path}. "
                "Clone the tau2-bench repository to this path."
            )

        git_commit = _git_commit(resolved_repo_path)
        domain_root = resolved_repo_path / "data" / "tau2" / "domains" / domain
        retail_tasks_path = domain_root / "tasks.json"
        retail_split_path = domain_root / "split_tasks.json"
        for data_path in (retail_tasks_path, retail_split_path):
            if not data_path.is_file():
                raise RuntimeError(
                    f"Tau2 checkout at {resolved_repo_path} is missing {data_path}. "
                    f"Restore the {domain} data files from the tau2-bench checkout."
                )

        package_version = _package_version(resolved_repo_path)
        return RuntimeFingerprint(
            repo_path=resolved_repo_path,
            git_commit=git_commit,
            package_version=package_version,
            retail_tasks_path=retail_tasks_path,
            retail_split_path=retail_split_path,
            gym_available=False,
            domain=domain,
        )

    @staticmethod
    def probe_gym(repo_path: Path) -> None:
        """Probe the exact AgentGymEnv module without retaining imported state."""
        _require_tau2_gym(Path(repo_path).expanduser().resolve())

    @staticmethod
    def load_verified_gym_factory(repo_path: Path) -> Callable[..., Any]:
        """Load AgentGymEnv while binding Tau2 imports to the configured checkout."""
        return _import_agent_gym_env(Path(repo_path).expanduser().resolve(), bind=True)

    @staticmethod
    def load_verified_run_domain(repo_path: Path) -> Tau2RunDomainRuntime:
        """Bind and expose Tau2's official batch runner from the configured checkout."""
        return _import_run_domain_runtime(Path(repo_path).expanduser().resolve())

    @staticmethod
    def require_pinned_commit(fingerprint: RuntimeFingerprint) -> None:
        pin_path = fingerprint.repo_path.parent / f"{fingerprint.repo_path.name}.commit"
        try:
            pinned_commit = pin_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Tau2 pin is missing at {pin_path}; restore the committed checkout pin."
            ) from error
        except OSError as error:
            raise RuntimeError(f"Tau2 pin at {pin_path} is unreadable: {error}") from error
        if re.fullmatch(r"[0-9a-fA-F]{40}", pinned_commit) is None:
            raise RuntimeError(
                f"Tau2 pin at {pin_path} is malformed; expected one full 40-character Git commit."
            )
        if pinned_commit.casefold() != fingerprint.git_commit.casefold():
            raise RuntimeError(
                f"Tau2 pin mismatch at {pin_path}: expected {pinned_commit}, "
                f"checkout HEAD is {fingerprint.git_commit}. Check out the pinned commit."
            )


def _git_commit(repo_path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        raise RuntimeError(
            f"Unable to run Git for Tau2 checkout at {repo_path}. "
            "Install Git or fix PATH and the execution environment."
        ) from error
    if result.returncode != 0:
        raise RuntimeError(
            f"Tau2 path {repo_path} is not a usable Git checkout. "
            "Clone tau2-bench with its .git directory intact."
        )
    return result.stdout.strip()


def _package_version(repo_path: Path) -> str:
    pyproject_path = repo_path / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as source:
            project = tomllib.load(source)
        version = project["project"]["version"]
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"Tau2 checkout at {repo_path} has no readable package version in "
            f"{pyproject_path}. Restore the checkout metadata."
        ) from error
    if not isinstance(version, str):
        raise RuntimeError(
            f"Tau2 checkout at {repo_path} has an invalid package version in "
            f"{pyproject_path}. Restore the checkout metadata."
        )
    return version


def _require_tau2_gym(repo_path: Path) -> None:
    _import_agent_gym_env(repo_path, bind=False)


def _source_root(repo_path: Path) -> Path:
    source_root = next(
        (candidate for candidate in (repo_path / "src", repo_path) if (candidate / "tau2").is_dir()),
        None,
    )
    if source_root is None:
        raise RuntimeError(
            f"Tau2 checkout at {repo_path} does not contain the tau2 package. "
            "Restore the checkout or install its package dependencies."
        )
    return source_root.resolve()


def _import_agent_gym_env(repo_path: Path, *, bind: bool) -> Callable[..., Any]:
    source_root = _source_root(repo_path)

    original_sys_path = list(sys.path)
    existing_modules = {
        name: module for name, module in sys.modules.items() if name == "tau2" or name.startswith("tau2.")
    }
    succeeded = False
    try:
        for name in list(existing_modules):
            sys.modules.pop(name, None)
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != source_root]
        sys.path.insert(0, str(source_root))
        importlib.invalidate_caches()
        module = importlib.import_module("tau2.gym.gym_agent")
        module_file_value = getattr(module, "__file__", None)
        actual_origin = (
            Path(module_file_value).resolve()
            if isinstance(module_file_value, (str, bytes))
            else None
        )
        if actual_origin is None or not actual_origin.is_relative_to(source_root):
            actual_description = str(actual_origin) if actual_origin is not None else repr(module_file_value)
            raise RuntimeError(
                "Tau2 AgentGymEnv origin mismatch: expected module under "
                f"{source_root}, resolved {actual_description}. "
                "Remove the conflicting Tau2 installation or reinstall the configured checkout."
            )
        factory = getattr(module, "AgentGymEnv", None)
        if not callable(factory):
            raise RuntimeError(
                f"Tau2 module {actual_origin} must expose callable AgentGymEnv; "
                f"resolved {factory!r}. Restore the configured checkout."
            )
        succeeded = True
        return factory
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Unable to import tau2.gym.gym_agent from Tau2 checkout at {repo_path} "
            f"(expected source root {source_root}). "
            "Install the checkout's Python dependencies and retry."
        ) from error
    finally:
        if not bind or not succeeded:
            sys.path[:] = original_sys_path
            for name in [name for name in sys.modules if name == "tau2" or name.startswith("tau2.")]:
                sys.modules.pop(name, None)
            sys.modules.update(existing_modules)


def _import_run_domain_runtime(repo_path: Path) -> Tau2RunDomainRuntime:
    source_root = _source_root(repo_path)
    original_sys_path = list(sys.path)
    existing_modules = {
        name: module for name, module in sys.modules.items() if name == "tau2" or name.startswith("tau2.")
    }
    succeeded = False
    try:
        for name in list(existing_modules):
            sys.modules.pop(name, None)
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != source_root]
        sys.path.insert(0, str(source_root))
        importlib.invalidate_caches()

        run_module = importlib.import_module("tau2.run")
        simulation_module = importlib.import_module("tau2.data_model.simulation")
        registry_module = importlib.import_module("tau2.registry")
        agent_module = importlib.import_module("tau2.agent.base_agent")
        message_module = importlib.import_module("tau2.data_model.message")
        for module in (
            run_module,
            simulation_module,
            registry_module,
            agent_module,
            message_module,
        ):
            _require_module_under_source_root(module, source_root)

        run_domain = getattr(run_module, "run_domain", None)
        text_run_config = getattr(simulation_module, "TextRunConfig", None)
        registry = getattr(registry_module, "registry", None)
        half_duplex_agent = getattr(agent_module, "HalfDuplexAgent", None)
        required_message_types = {
            name: getattr(message_module, name, None)
            for name in (
                "AssistantMessage",
                "ToolCall",
                "UserMessage",
                "ToolMessage",
                "MultiToolMessage",
            )
        }
        if not callable(run_domain):
            raise RuntimeError("Tau2 runtime must expose callable tau2.run.run_domain")
        if not isinstance(text_run_config, type):
            raise RuntimeError("Tau2 runtime must expose TextRunConfig")
        if registry is None or not callable(
            getattr(registry, "register_agent_factory", None)
        ):
            raise RuntimeError("Tau2 runtime registry must support agent factories")
        if not isinstance(half_duplex_agent, type):
            raise RuntimeError("Tau2 runtime must expose HalfDuplexAgent")
        missing_types = [
            name for name, value in required_message_types.items() if not isinstance(value, type)
        ]
        if missing_types:
            raise RuntimeError(
                "Tau2 runtime is missing message types: " + ", ".join(missing_types)
            )
        succeeded = True
        return Tau2RunDomainRuntime(
            run_domain=run_domain,
            text_run_config=text_run_config,
            registry=registry,
            half_duplex_agent=half_duplex_agent,
            assistant_message=required_message_types["AssistantMessage"],
            tool_call=required_message_types["ToolCall"],
            user_message=required_message_types["UserMessage"],
            tool_message=required_message_types["ToolMessage"],
            multi_tool_message=required_message_types["MultiToolMessage"],
        )
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"Unable to import tau2.run.run_domain from Tau2 checkout at {repo_path} "
            f"(expected source root {source_root}). "
            "Install the checkout's Python dependencies and retry."
        ) from error
    finally:
        if not succeeded:
            sys.path[:] = original_sys_path
            for name in [name for name in sys.modules if name == "tau2" or name.startswith("tau2.")]:
                sys.modules.pop(name, None)
            sys.modules.update(existing_modules)


def _require_module_under_source_root(module: Any, source_root: Path) -> None:
    module_file_value = getattr(module, "__file__", None)
    actual_origin = (
        Path(module_file_value).resolve()
        if isinstance(module_file_value, (str, bytes))
        else None
    )
    if actual_origin is None or not actual_origin.is_relative_to(source_root):
        actual_description = str(actual_origin) if actual_origin is not None else repr(
            module_file_value
        )
        raise RuntimeError(
            "Tau2 module origin mismatch: expected module under "
            f"{source_root}, resolved {actual_description}. "
            "Remove the conflicting Tau2 installation or reinstall the configured checkout."
        )
