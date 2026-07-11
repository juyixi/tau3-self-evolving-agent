from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau3_retail_evolver.runs.manifest import create_manifest


def test_creates_a_no_memory_manifest_atomically_with_sanitized_runtime_data(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "baseline-001" / "manifest.json"

    manifest = create_manifest(
        path,
        run_id="baseline-001",
        model_revision="Qwen/Qwen3.5-9B@revision-a",
        tau2_commit="a" * 40,
        split="train",
        split_hash="b" * 64,
        task_ids=("task-1", "task-2"),
        seed=17,
        user_simulator_config={
            "user_llm": "simulator",
            "user_llm_args": {"api_key": "not-for-artifacts", "temperature": 0.0},
        },
        environment_options={"all_messages_as_observation": True},
        rollout_options={"temperature": 1.0, "top_p": 0.95, "max_episode_steps": 40},
        command=("python", "-m", "scripts.run_baseline", "--api-key", "secret"),
    )

    assert manifest == json.loads(path.read_text(encoding="utf-8"))
    assert manifest == {
        "adapter_revision": None,
        "command": ["python", "-m", "scripts.run_baseline", "--api-key", "[REDACTED]"],
        "environment_options": {"all_messages_as_observation": True},
        "memory_snapshot_id": None,
        "model_revision": "Qwen/Qwen3.5-9B@revision-a",
        "rollout_options": {"max_episode_steps": 40, "temperature": 1.0, "top_p": 0.95},
        "run_id": "baseline-001",
        "schema_version": 1,
        "seed": 17,
        "split": "train",
        "split_hash": "b" * 64,
        "task_ids": ["task-1", "task-2"],
        "tau2_commit": "a" * 40,
        "user_simulator_config": {
            "user_llm": "simulator",
            "user_llm_args": {"api_key": "[REDACTED]", "temperature": 0.0},
        },
    }


def test_refuses_to_overwrite_an_existing_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"existing":true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_manifest(
            path,
            run_id="baseline-001",
            model_revision="revision-a",
            tau2_commit="a" * 40,
            split="train",
            split_hash="b" * 64,
            task_ids=("task-1",),
            seed=17,
            user_simulator_config={},
            environment_options={},
            rollout_options={},
            command=("python",),
        )

    assert path.read_text(encoding="utf-8") == '{"existing":true}\n'
