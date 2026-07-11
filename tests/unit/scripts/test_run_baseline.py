from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tau3_retail_evolver.fast_loop.baseline_runner import RolloutSummary
from tau3_retail_evolver.models.policy import DecisionResponse
from scripts import run_baseline
from tests.support.policy import ScriptedPolicy


def test_rejects_non_train_before_loading_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_baseline,
        "require_learning_split",
        lambda split: (_ for _ in ()).throw(ValueError("train split required")),
    )
    monkeypatch.setattr(
        run_baseline,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("config must not load")),
    )

    with pytest.raises(ValueError, match="train split required"):
        run_baseline.main(
            [
                "--split",
                "test",
                "--task-id",
                "task-1",
                "--run-id",
                "baseline-001",
                "--output-root",
                str(tmp_path),
                "--qwen-base-url",
                "http://qwen.invalid/v1",
                "--model-revision",
                "revision-a",
            ]
        )


def test_creates_verified_no_memory_baseline_artifacts_without_api_key_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = SimpleNamespace(
        tau2=SimpleNamespace(
            repo_path=tmp_path / "external" / "tau2-bench",
            domain="retail",
            solo_mode=False,
            user_llm="simulator-model",
            user_llm_args={"api_key": "simulator-secret", "temperature": 0.0},
        ),
        model=SimpleNamespace(base_model="Qwen/Qwen3.5-9B"),
        rollout=SimpleNamespace(temperature=1.0, top_p=0.95, max_episode_steps=40),
        training=SimpleNamespace(seed=17),
    )
    runtime = SimpleNamespace(
        repo_path=config.tau2.repo_path,
        git_commit="a" * 40,
        retail_tasks_path=tmp_path / "tasks.json",
        retail_split_path=tmp_path / "split_tasks.json",
    )
    catalog = SimpleNamespace(
        split_sha256="b" * 64,
        task_ids=lambda split: ("task-1", "task-2"),
        require_official_compatibility=lambda: None,
    )
    client_args: dict[str, Any] = {}
    captured: dict[str, Any] = {}

    class FakeEnvironment:
        def __init__(self, task_id: str, config: Any, gym_factory: Any) -> None:
            captured.setdefault("environments", []).append((task_id, gym_factory))
            self.user_simulator_config = {
                "solo_mode": False,
                "user_llm": "resolved-simulator-model",
                "user_llm_args": {"api_key": "resolved-secret", "temperature": 0.2},
            }

        def close(self) -> None:
            captured["probe_close_calls"] = captured.get("probe_close_calls", 0) + 1

    def construct_client(**kwargs: Any) -> object:
        client_args.update(kwargs)
        return object()

    def fake_run(
        tasks: tuple[str, ...], env_factory: Any, policy: Any, context: Any
    ) -> RolloutSummary:
        captured["tasks"] = tasks
        captured["policy"] = policy
        captured["context"] = context
        environment = env_factory("task-1")
        environment.close()
        return RolloutSummary(episodes=())

    monkeypatch.setenv("QWEN_API_KEY", "top-secret-key")
    monkeypatch.setattr(run_baseline, "load_config", lambda path: config)
    monkeypatch.setattr(
        run_baseline,
        "Tau2Runtime",
        SimpleNamespace(
            inspect_metadata=lambda repo_path: runtime,
            require_pinned_commit=lambda fingerprint: None,
            load_verified_gym_factory=lambda repo_path: "verified-gym-factory",
        ),
    )
    monkeypatch.setattr(
        run_baseline,
        "RetailTaskCatalog",
        SimpleNamespace(from_files=lambda tasks_path, split_path: catalog),
    )
    monkeypatch.setattr(run_baseline, "Tau2RetailEnv", FakeEnvironment)
    monkeypatch.setattr(run_baseline, "OpenAICompatibleHttpClient", construct_client)
    monkeypatch.setattr(
        run_baseline,
        "OpenAICompatibleQwenPolicy",
        lambda *, client: ScriptedPolicy(
            [
                DecisionResponse(
                    raw_output="raw",
                    parsed_action="stop",
                    sampling_params={"temperature": 1.0, "top_p": 0.95},
                    latency_s=0.0,
                )
            ]
        ),
    )
    monkeypatch.setattr(run_baseline, "run_baseline", fake_run)

    returncode = run_baseline.main(
        [
            "--split",
            "train",
            "--task-id",
            "task-1",
            "--task-id",
            "task-2",
            "--run-id",
            "baseline-001",
            "--output-root",
            str(tmp_path / "runs"),
            "--qwen-base-url",
            "http://qwen.invalid/v1",
            "--model-revision",
            "revision-a",
        ]
    )

    manifest_path = tmp_path / "runs" / "baseline-001" / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert returncode == 0
    assert captured["tasks"] == ("task-1", "task-2")
    assert captured["context"].adapter_revision is None
    assert captured["context"].memory_snapshot_id is None
    assert captured["context"].seed == 17
    assert captured["environments"] == [
        ("task-1", "verified-gym-factory"),
        ("task-1", "verified-gym-factory"),
    ]
    assert captured["probe_close_calls"] == 2
    assert client_args == {
        "base_url": "http://qwen.invalid/v1",
        "model": "Qwen/Qwen3.5-9B",
        "api_key": "top-secret-key",
        "max_tokens": 8192,
        "generation_settings": {
            "chat_template_kwargs": {"enable_thinking": True},
            "top_k": 20,
            "presence_penalty": 1.5,
            "parallel_tool_calls": False,
        },
    }
    assert manifest["task_ids"] == ["task-1", "task-2"]
    assert manifest["iteration"] == 0
    assert manifest["parent_checkpoint"] is None
    assert manifest["user_simulator_config"] == {
        "solo_mode": False,
        "user_llm": "resolved-simulator-model",
        "user_llm_args": {"api_key": "[REDACTED]", "temperature": 0.2},
    }
    assert manifest["model_serving_contract"] == {
        "language_model_only": True,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "enable_thinking": True,
        "max_tokens": 8192,
        "top_k": 20,
        "presence_penalty": 1.5,
        "parallel_tool_calls": False,
    }
    assert manifest["rollout_options"] == {
        "max_episode_steps": 40,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert "top-secret-key" not in manifest_text
    assert "simulator-secret" not in manifest_text
    assert "top-secret-key" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "url",
    (
        "ftp://qwen.invalid/v1",
        "https://secret@qwen.invalid/v1",
        "https://qwen.invalid/v1?token=secret",
        "https://qwen.invalid/v1#secret",
    ),
)
def test_rejects_credential_or_non_http_qwen_urls_without_echoing_them(url: str) -> None:
    with pytest.raises(ValueError) as error:
        run_baseline._validate_qwen_base_url(url)

    assert "secret" not in str(error.value)
