from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlsplit

from tau3_retail_evolver.credential_policy import is_credential_key
from tau3_retail_evolver.io.jsonl import _fsync_directory


_REDACTED = "[REDACTED]"


def create_manifest(
    path: Path,
    *,
    run_id: str,
    iteration: int,
    model_revision: str,
    parent_checkpoint: str | None,
    tau2_commit: str,
    split: str,
    split_hash: str,
    task_ids: Sequence[str],
    seed: int,
    user_simulator_config: Mapping[str, Any],
    environment_options: Mapping[str, Any],
    rollout_options: Mapping[str, Any],
    model_serving_contract: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    command: Sequence[str],
    adapter_revision: str | None = None,
    memory_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Atomically create one immutable run manifest."""
    _require_nonblank("model revision", model_revision)
    _require_optional_nonblank("parent checkpoint", parent_checkpoint)
    _require_optional_nonblank("adapter revision", adapter_revision)
    _require_optional_nonblank("memory snapshot id", memory_snapshot_id)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": iteration,
        "model_revision": model_revision,
        "parent_checkpoint": parent_checkpoint,
        "adapter_revision": adapter_revision,
        "memory_snapshot_id": memory_snapshot_id,
        "tau2_commit": tau2_commit,
        "split": split,
        "split_hash": split_hash,
        "task_ids": list(task_ids),
        "seed": seed,
        "user_simulator_config": sanitize_artifact_data(user_simulator_config),
        "environment_options": sanitize_artifact_data(environment_options),
        "rollout_options": sanitize_artifact_data(rollout_options),
        "model_serving_contract": sanitize_artifact_data(model_serving_contract),
        "evaluation_config": sanitize_artifact_data(evaluation_config),
        "command": _sanitize_command(command),
    }
    try:
        serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError("manifest must be JSON serializable") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as destination:
            temporary_path = Path(destination.name)
            destination.write(f"{serialized}\n")
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite existing manifest: {path}") from error
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def _require_nonblank(field: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must not be blank")


def _require_optional_nonblank(field: str, value: str | None) -> None:
    if value is not None:
        _require_nonblank(field, value)


def sanitize_artifact_data(value: Any) -> Any:
    """Recursively preserve public metadata while removing credential values."""
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if is_credential_key(key) else sanitize_artifact_data(nested)
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_artifact_data(item) for item in value]
    if isinstance(value, str) and _is_credential_bearing_url(value):
        return _REDACTED
    return value


def _sanitize_command(command: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for argument in command:
        text = str(argument)
        if redact_next:
            sanitized.append(_REDACTED)
            redact_next = False
        elif text.startswith("--"):
            key, separator, _ = text[2:].partition("=")
            if is_credential_key(key.replace("-", "_")):
                if separator:
                    sanitized.append(f"--{key}={_REDACTED}")
                else:
                    sanitized.append(text)
                    redact_next = True
            else:
                sanitized.append(sanitize_artifact_data(text))
        else:
            sanitized.append(sanitize_artifact_data(text))
    return sanitized


def _is_credential_bearing_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(
        parsed.username or parsed.password or parsed.query or parsed.fragment
    )
