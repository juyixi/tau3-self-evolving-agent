from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any

from tau3_retail_evolver.memory.json_store import write_bytes_atomic


STATE_SCHEMA_VERSION = 1
STATE_FILE = "iteration_state.json"


class IterationState(StrEnum):
    CREATED = "created"
    ROLLOUT_COMPLETE = "rollout_complete"
    ATTRIBUTION_COMPLETE = "attribution_complete"
    DATASET_COMPLETE = "dataset_complete"
    TRAINING_COMPLETE = "training_complete"
    PROMOTED = "promoted"


_NEXT_STATE = {
    IterationState.CREATED: IterationState.ROLLOUT_COMPLETE,
    IterationState.ROLLOUT_COMPLETE: IterationState.ATTRIBUTION_COMPLETE,
    IterationState.ATTRIBUTION_COMPLETE: IterationState.DATASET_COMPLETE,
    IterationState.DATASET_COMPLETE: IterationState.TRAINING_COMPLETE,
    IterationState.TRAINING_COMPLETE: IterationState.PROMOTED,
}


class IterationStateStore:
    def __init__(self, iteration_dir: Path) -> None:
        self.iteration_dir = Path(iteration_dir).resolve()
        self.path = self.iteration_dir / STATE_FILE

    def create(self, identity: Mapping[str, Any]) -> dict[str, Any]:
        if self.path.exists():
            raise FileExistsError(f"iteration state already exists: {self.path}")
        record = {
            "schema_version": STATE_SCHEMA_VERSION,
            "state": IterationState.CREATED.value,
            "identity": _json_copy(identity),
            "stages": {},
        }
        self._write(record)
        return record

    def load_verified(self) -> dict[str, Any]:
        record = load_iteration_record(self.iteration_dir)
        _validate_record(record)
        for stage_name, stage in record["stages"].items():
            artifacts = stage.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError(f"iteration stage artifacts are invalid: {stage_name}")
            for artifact_name, artifact in artifacts.items():
                if not isinstance(artifact, Mapping):
                    raise ValueError(f"iteration artifact metadata is invalid: {artifact_name}")
                path = _resolve_artifact_path(
                    self.iteration_dir,
                    artifact.get("path"),
                )
                expected = artifact.get("sha256")
                actual = hash_artifact(path)
                if not isinstance(expected, str) or expected != actual:
                    raise ValueError(
                        f"artifact hash mismatch for {stage_name}.{artifact_name}: {path}"
                    )
        return record

    def advance(
        self,
        record: Mapping[str, Any],
        *,
        target: IterationState,
        stage_name: str,
        artifacts: Mapping[str, Path],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        current = IterationState(record["state"])
        if _NEXT_STATE.get(current) is not target:
            raise ValueError(f"invalid iteration state transition: {current} -> {target}")
        if stage_name in record["stages"]:
            raise ValueError(f"iteration stage already recorded: {stage_name}")
        artifact_records = {
            name: {
                "path": _relative_artifact_path(self.iteration_dir, path),
                "sha256": hash_artifact(path),
            }
            for name, path in sorted(artifacts.items())
        }
        updated = _json_copy(record)
        updated["state"] = target.value
        updated["stages"][stage_name] = {
            "artifacts": artifact_records,
            "metadata": _json_copy(metadata),
        }
        self._write(updated)
        return updated

    def _write(self, record: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        write_bytes_atomic(self.path, payload)


def load_iteration_record(iteration_dir: Path) -> dict[str, Any]:
    path = Path(iteration_dir).resolve() / STATE_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read iteration state: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"iteration state must be an object: {path}")
    return value


def hash_artifact(path: Path) -> str:
    resolved = Path(path).resolve()
    if resolved.is_file():
        return hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not resolved.is_dir():
        raise ValueError(f"iteration artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    for child in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if child.is_symlink():
            raise ValueError(f"iteration artifacts must not contain symlinks: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def _validate_record(record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("iteration state schema mismatch")
    try:
        state = IterationState(record.get("state"))
    except ValueError as error:
        raise ValueError("iteration state value is invalid") from error
    if not isinstance(record.get("identity"), Mapping):
        raise ValueError("iteration identity is missing")
    stages = record.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("iteration stages are missing")
    expected_count = list(IterationState).index(state)
    if len(stages) != expected_count:
        raise ValueError("iteration state does not match completed stage count")


def _relative_artifact_path(iteration_dir: Path, path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(iteration_dir).as_posix()
    except ValueError as error:
        raise ValueError(f"iteration artifact must stay inside iteration directory: {resolved}") from error


def _resolve_artifact_path(iteration_dir: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("iteration artifact path is invalid")
    resolved = (iteration_dir / value).resolve()
    try:
        resolved.relative_to(iteration_dir)
    except ValueError as error:
        raise ValueError("iteration artifact path escapes iteration directory") from error
    return resolved


def _json_copy(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )
    )
