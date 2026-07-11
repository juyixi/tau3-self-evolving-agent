from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


_REDACTED = "[REDACTED]"
_CREDENTIAL_TERMINAL_WORDS = {
    "token",
    "authorization",
    "secret",
    "secrets",
    "password",
    "passwords",
    "credential",
    "credentials",
}
_CREDENTIAL_KEY_SUFFIXES = (("api", "key"), ("private", "key"), ("access", "key"))
_COMPACT_CREDENTIAL_KEYS = {
    "apikey",
    "apitoken",
    "accesstoken",
    "authtoken",
    "clientsecret",
    "privatekey",
    "accesskey",
}


def create_manifest(
    path: Path,
    *,
    run_id: str,
    model_revision: str,
    tau2_commit: str,
    split: str,
    split_hash: str,
    task_ids: Sequence[str],
    seed: int,
    user_simulator_config: Mapping[str, Any],
    environment_options: Mapping[str, Any],
    rollout_options: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    """Atomically create one immutable, no-memory run manifest."""
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model_revision": model_revision,
        "adapter_revision": None,
        "memory_snapshot_id": None,
        "tau2_commit": tau2_commit,
        "split": split,
        "split_hash": split_hash,
        "task_ids": list(task_ids),
        "seed": seed,
        "user_simulator_config": sanitize_artifact_data(user_simulator_config),
        "environment_options": sanitize_artifact_data(environment_options),
        "rollout_options": sanitize_artifact_data(rollout_options),
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
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return manifest


def sanitize_artifact_data(value: Any) -> Any:
    """Recursively preserve public metadata while removing credential values."""
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_credential_key(key) else sanitize_artifact_data(nested)
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_artifact_data(item) for item in value]
    return value


def _sanitize_command(command: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for argument in command:
        text = str(argument)
        if redact_next:
            sanitized.append(_REDACTED)
            redact_next = False
        elif text.startswith("--") and _is_credential_key(text[2:].replace("-", "_")):
            if "=" in text:
                sanitized.append(f"{text.split('=', 1)[0]}={_REDACTED}")
            else:
                sanitized.append(text)
                redact_next = True
        else:
            sanitized.append(text)
    return sanitized


def _is_credential_key(key: Any) -> bool:
    key_text = str(key)
    compact = "".join(re.findall(r"[a-z0-9]+", key_text.casefold()))
    if compact in _COMPACT_CREDENTIAL_KEYS:
        return True
    separated = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", key_text)
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", separated)
    words = tuple(re.findall(r"[a-z0-9]+", separated.casefold()))
    return bool(words) and (
        words[-1] in _CREDENTIAL_TERMINAL_WORDS
        or any(words[-len(suffix) :] == suffix for suffix in _CREDENTIAL_KEY_SUFFIXES)
    )
