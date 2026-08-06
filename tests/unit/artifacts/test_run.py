from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_evolver.artifacts.run import episode_artifact_metadata, write_run_record
from tau3_evolver.persistence.jsonl import JsonlWriter
from tau3_evolver.security.redaction import redact_public_data


def test_writes_immutable_run_record_with_sanitized_config(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes.jsonl"
    JsonlWriter(episodes).append({"schema_version": 1, "task_id": "1"})
    path = tmp_path / "run.json"

    record = write_run_record(
        path,
        {
            "run_id": "airline-test-001",
            "config": {"model": {"api_key": "secret", "api_key_env": "MODEL_KEY"}},
            "artifacts": {"episodes": episode_artifact_metadata(episodes)},
        },
    )

    assert record["schema_version"] == 1
    assert record["config"]["model"]["api_key"] == "[REDACTED]"
    assert record["config"]["model"]["api_key_env"] == "MODEL_KEY"
    assert json.loads(path.read_text(encoding="utf-8")) == record
    with pytest.raises(FileExistsError, match="overwrite"):
        write_run_record(path, record)


def test_episode_metadata_binds_path_bytes_rows_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    writer = JsonlWriter(path)
    writer.append({"task_id": "1"})
    writer.append({"task_id": "2"})

    metadata = episode_artifact_metadata(path)

    assert metadata["path"] == "episodes.jsonl"
    assert metadata["bytes"] == len(path.read_bytes())
    assert metadata["rows"] == 2
    assert len(metadata["sha256"]) == 64


def test_sanitizes_credential_bearing_urls() -> None:
    assert redact_public_data(
        {"endpoint": "https://user:secret@example.test/v1?token=x"}
    ) == {"endpoint": "[REDACTED]"}
