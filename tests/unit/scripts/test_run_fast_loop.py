from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_fast_loop
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.runner import EpisodeResult, LifecycleResponse
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.repository import MemoryRepository


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tau2=SimpleNamespace(
            repo_path=tmp_path / "external" / "tau2-bench",
            domain="retail",
        ),
        model=SimpleNamespace(base_model="Qwen/Qwen3.5-9B"),
        rollout=SimpleNamespace(temperature=1.0, top_p=0.95, max_episode_steps=40),
        memory=SimpleNamespace(
            agent_id="retail",
            retrieve_top_k=50,
            maintenance_period=30,
        ),
        training=SimpleNamespace(seed=17),
        evaluation=SimpleNamespace(
            nl_assertions=SimpleNamespace(
                model="openrouter/openai/gpt-4.1",
                model_args={"temperature": 0.0},
                api_key_env="OPENROUTER_API_KEY",
            )
        ),
    )


def _episode(task_id: str, reward: float = 1.0) -> EpisodeResult:
    return EpisodeResult(
        task_id=task_id,
        final_reward=reward,
        steps=1,
        terminal_evaluation={"reward": reward, "task": task_id},
        simulation_result={"status": "done"},
        selected_memory_ids=(),
        written_memory_ids=(),
        truncated=False,
    )


def _install_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    episode_runner: Any,
) -> tuple[list[str], dict[str, Any]]:
    config = _config(tmp_path)
    runtime = SimpleNamespace(
        repo_path=config.tau2.repo_path,
        git_commit="a" * 40,
        retail_tasks_path=tmp_path / "tasks.json",
        retail_split_path=tmp_path / "split_tasks.json",
    )
    catalog = SimpleNamespace(
        split_sha256="b" * 64,
        task_ids=lambda split: ("task-1", "task-2"),
        require_official_compatibility=lambda: ordering.append("catalog-official"),
    )
    ordering: list[str] = []
    captured: dict[str, Any] = {}

    class RecordingRepository(MemoryRepository):
        def snapshot(self):  # type: ignore[no-untyped-def]
            ordering.append("snapshot")
            return super().snapshot()

    class FakeEnvironment:
        def __init__(self, task_id: str, config_value: Any, gym_factory: Any) -> None:
            ordering.append(f"environment:{task_id}")
            captured.setdefault("environments", []).append((task_id, gym_factory))
            self.user_simulator_config = {
                "solo_mode": False,
                "user_llm": "resolved-simulator",
                "user_llm_args": {"api_key": "simulator-secret", "temperature": 0.0},
            }

        def close(self) -> None:
            ordering.append("environment-close")

    def inspect_metadata(repo_path: Path) -> Any:
        ordering.append("runtime-inspect")
        return runtime

    def require_pinned_commit(fingerprint: Any) -> None:
        ordering.append("runtime-pin")

    def load_verified_gym_factory(repo_path: Path) -> str:
        ordering.append("gym")
        return "verified-gym-factory"

    def open_memory(config_value: Any, *, root: Path | None = None) -> MemoryRepository:
        ordering.append("memory-open")
        assert root == tmp_path / "isolated-project"
        repository = RecordingRepository(
            root / "history" / "agents" / config_value.agent_id / "memory"
        )
        captured["repository"] = repository
        return repository

    class FakeEmbeddingProvider:
        model_revision = "embedding@revision"
        dimension = 2

        def embed(self, text: str) -> tuple[float, float]:
            return (1.0, 0.0)

        def embed_batch(self, texts: list[str]) -> list[tuple[float, float]]:
            return [(1.0, 0.0) for _ in texts]

    def build_provider(memory_config: Any, memory_root: Path) -> FakeEmbeddingProvider:
        ordering.append("embedding")
        assert memory_root == captured["repository"].root
        return FakeEmbeddingProvider()

    def construct_client(**kwargs: Any) -> object:
        ordering.append("client")
        captured["client_kwargs"] = kwargs
        return object()

    def construct_policy(**kwargs: Any) -> object:
        ordering.append("policy")
        captured["policy_kwargs"] = kwargs
        return object()

    def bind_assertions(value: Any) -> dict[str, Any]:
        ordering.append("assertions")
        return {
            "model": value.model,
            "model_args": value.model_args,
            "api_key_env": value.api_key_env,
        }

    real_manifest = run_fast_loop.create_manifest

    def create_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        ordering.append("manifest")
        return real_manifest(*args, **kwargs)

    real_maintenance = run_fast_loop.run_due_maintenance

    def maintenance(**kwargs: Any):  # type: ignore[no-untyped-def]
        ordering.append(f"maintenance:{kwargs['completed_train_tasks']}")
        captured.setdefault("maintenance_contexts", []).append(kwargs["context"])
        return real_maintenance(**kwargs)

    monkeypatch.setattr(
        run_fast_loop,
        "load_config",
        lambda path: ordering.append("config") or config,
    )
    monkeypatch.setattr(
        run_fast_loop,
        "Tau2Runtime",
        SimpleNamespace(
            inspect_metadata=inspect_metadata,
            require_pinned_commit=require_pinned_commit,
            load_verified_gym_factory=load_verified_gym_factory,
        ),
    )
    monkeypatch.setattr(
        run_fast_loop,
        "RetailTaskCatalog",
        SimpleNamespace(
            from_files=lambda tasks_path, split_path: ordering.append("catalog") or catalog
        ),
    )
    monkeypatch.setattr(run_fast_loop, "Tau2RetailEnv", FakeEnvironment)
    monkeypatch.setattr(run_fast_loop, "open_training_memory", open_memory)
    monkeypatch.setattr(run_fast_loop, "build_embedding_provider", build_provider)
    monkeypatch.setattr(run_fast_loop, "OpenAICompatibleHttpClient", construct_client)
    monkeypatch.setattr(run_fast_loop, "OpenAICompatibleFastLoopPolicy", construct_policy)
    monkeypatch.setattr(run_fast_loop, "bind_tau2_nl_assertions", bind_assertions)
    monkeypatch.setattr(run_fast_loop, "create_manifest", create_manifest)
    monkeypatch.setattr(run_fast_loop, "run_fast_loop_episode", episode_runner)
    monkeypatch.setattr(run_fast_loop, "run_due_maintenance", maintenance)
    return ordering, captured


def test_rejects_non_train_before_loading_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_fast_loop,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("config must not load")),
    )

    with pytest.raises(ValueError, match="train"):
        run_fast_loop.main(
            [
                "--split",
                "test",
                "--task-id",
                "task-1",
                "--run-id",
                "learn-001",
                "--output-root",
                str(tmp_path),
                "--model-revision",
                "revision-a",
            ]
        )


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
        (("--run-id", "../escape"), "run ID"),
        (("--iteration", "-1"), "iteration"),
        (("--completed-train-tasks-before", "-1"), "completed"),
        (("--model-revision", "  "), "model revision"),
        (("--adapter-revision", "\t"), "adapter revision"),
    ),
)
def test_rejects_invalid_arguments_before_configuration_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: tuple[str, str],
    message: str,
) -> None:
    monkeypatch.setattr(
        run_fast_loop,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("config must not load")),
    )
    arguments = [
        "--split",
        "train",
        "--task-id",
        "task-1",
        "--run-id",
        "learn-001",
        "--output-root",
        str(tmp_path),
        "--model-revision",
        "revision-a",
    ]
    option = extra_args[0]
    if option in arguments:
        arguments[arguments.index(option) + 1] = extra_args[1]
    else:
        arguments.extend(extra_args)

    with pytest.raises(ValueError, match=message):
        run_fast_loop.main(arguments)


def test_requires_unique_official_train_task_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        run_fast_loop._require_explicit_train_tasks(("task-1", "task-1"), ("task-1",))
    with pytest.raises(ValueError, match="official train split"):
        run_fast_loop._require_explicit_train_tasks(("test-task",), ("task-1",))


def test_requires_qwen_base_url_after_catalog_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    runtime = SimpleNamespace(
        repo_path=config.tau2.repo_path,
        git_commit="a" * 40,
        retail_tasks_path=tmp_path / "tasks.json",
        retail_split_path=tmp_path / "split.json",
    )
    catalog = SimpleNamespace(
        split_sha256="b" * 64,
        task_ids=lambda split: ("task-1",),
        require_official_compatibility=lambda: None,
    )
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.setattr(run_fast_loop, "load_config", lambda path: config)
    monkeypatch.setattr(
        run_fast_loop,
        "Tau2Runtime",
        SimpleNamespace(
            inspect_metadata=lambda path: runtime,
            require_pinned_commit=lambda value: None,
        ),
    )
    monkeypatch.setattr(
        run_fast_loop,
        "RetailTaskCatalog",
        SimpleNamespace(from_files=lambda tasks, split: catalog),
    )

    with pytest.raises(ValueError, match="QWEN_BASE_URL"):
        run_fast_loop.main(
            [
                "--split",
                "train",
                "--task-id",
                "task-1",
                "--run-id",
                "learn-001",
                "--output-root",
                str(tmp_path / "runs"),
                "--model-revision",
                "revision-a",
            ]
        )


def test_refuses_existing_run_before_configuration_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "runs" / "learn-001").mkdir(parents=True)
    monkeypatch.setattr(
        run_fast_loop,
        "load_config",
        lambda path: (_ for _ in ()).throw(AssertionError("config must not load")),
    )

    with pytest.raises(FileExistsError, match="existing run"):
        run_fast_loop.main(
            [
                "--split",
                "train",
                "--task-id",
                "task-1",
                "--run-id",
                "learn-001",
                "--output-root",
                str(tmp_path / "runs"),
                "--model-revision",
                "revision-a",
            ]
        )


def test_creates_learning_artifacts_in_dependency_order_without_credential_leakage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_tasks: list[str] = []
    episode_contexts: list[RunContext] = []

    def episode_runner(**kwargs: Any) -> EpisodeResult:
        task_id = kwargs["task_id"]
        ordering.append(f"episode:{task_id}")
        captured_tasks.append(task_id)
        assert kwargs["task_instruction"] == (
            "Resolve the retail request shown in the current conversation."
        )
        assert task_id not in kwargs["task_instruction"]
        context = kwargs["context"]
        episode_contexts.append(context)
        repository = kwargs["repository"]
        captured["context"] = context
        repository.add(
            tier="tip",
            content=f"learned public workflow {task_id}",
            source_task_ids=(task_id,),
            created_round=context.iteration,
        )
        context.event_writer.append(
            context.event("EpisodeFinished", task_id, final_reward=1.0)
        )
        kwargs["environment"].close()
        return _episode(task_id)

    ordering, captured = _install_main_dependencies(
        monkeypatch, tmp_path, episode_runner=episode_runner
    )
    monkeypatch.setenv("QWEN_API_KEY", "qwen-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")

    returncode = run_fast_loop.main(
        [
            "--split",
            "train",
            "--task-id",
            "task-1",
            "--task-id",
            "task-2",
            "--run-id",
            "learn-001",
            "--output-root",
            str(tmp_path / "runs"),
            "--project-root",
            str(tmp_path / "isolated-project"),
            "--iteration",
            "4",
            "--completed-train-tasks-before",
            "25",
            "--qwen-base-url",
            "http://qwen.invalid/v1",
            "--model-revision",
            "model-revision-a",
            "--adapter-revision",
            "adapter-revision-b",
        ]
    )

    run_path = tmp_path / "runs" / "learn-001"
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    summary_text = (run_path / "fast_loop_summary.json").read_text(encoding="utf-8")
    summary = json.loads(summary_text)
    stdout = capsys.readouterr().out

    assert returncode == 0
    assert captured_tasks == ["task-1", "task-2"]
    assert all(context.mode is RunMode.LEARN for context in episode_contexts)
    assert all(context.adapter_revision == "adapter-revision-b" for context in episode_contexts)
    assert episode_contexts[0].memory_snapshot_id == manifest["memory_snapshot_id"]
    assert episode_contexts[1].memory_snapshot_id != episode_contexts[0].memory_snapshot_id
    assert all(
        context.task_group_for(task_id) == "retail"
        for context, task_id in zip(episode_contexts, captured_tasks, strict=True)
    )
    maintenance_contexts = captured["maintenance_contexts"]
    assert maintenance_contexts[0].memory_snapshot_id == episode_contexts[1].memory_snapshot_id
    assert maintenance_contexts[1].memory_snapshot_id == summary["output_memory_snapshot_id"]
    assert manifest["adapter_revision"] == "adapter-revision-b"
    assert manifest["parent_checkpoint"] is None
    assert manifest["task_ids"] == ["task-1", "task-2"]
    assert summary == {
        "completed_train_tasks_after": 27,
        "completed_train_tasks_before": 25,
        "episode_count": 2,
        "input_memory_snapshot_id": manifest["memory_snapshot_id"],
        "maintenance_rounds_executed": [],
        "output_memory_snapshot_id": summary["output_memory_snapshot_id"],
        "run_id": "learn-001",
        "successful_task_ids": ["task-1", "task-2"],
        "total_terminal_reward": 2.0,
    }
    assert summary["output_memory_snapshot_id"] != summary["input_memory_snapshot_id"]
    assert json.loads(stdout) == summary
    assert captured["client_kwargs"] == {
        "base_url": "http://qwen.invalid/v1",
        "model": "Qwen/Qwen3.5-9B",
        "api_key": "qwen-secret",
        "max_tokens": 8192,
        "generation_settings": run_fast_loop.QWEN_GENERATION_SETTINGS,
    }
    assert captured["policy_kwargs"]["temperature"] == 1.0
    assert captured["policy_kwargs"]["top_p"] == 0.95
    assert ordering == [
        "config",
        "runtime-inspect",
        "runtime-pin",
        "catalog",
        "catalog-official",
        "gym",
        "assertions",
        "environment:task-1",
        "environment-close",
        "memory-open",
        "embedding",
        "snapshot",
        "client",
        "policy",
        "manifest",
        "snapshot",
        "environment:task-1",
        "episode:task-1",
        "environment-close",
        "snapshot",
        "maintenance:26",
        "snapshot",
        "environment:task-2",
        "episode:task-2",
        "environment-close",
        "snapshot",
        "maintenance:27",
        "snapshot",
    ]
    rollout_events = [
        json.loads(line)
        for line in (run_path / "rollouts" / "events.jsonl").read_text("utf-8").splitlines()
    ]
    assert {event["task_group"] for event in rollout_events} == {"retail"}
    all_artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    for secret in ("qwen-secret", "openrouter-secret", "simulator-secret"):
        assert secret not in all_artifacts
        assert secret not in stdout


def test_episode_failure_preserves_events_without_success_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing_episode(**kwargs: Any) -> EpisodeResult:
        context = kwargs["context"]
        context.event_writer.append(
            context.event("EpisodeFailed", kwargs["task_id"], error={"type": "RuntimeError"})
        )
        raise RuntimeError("episode failed")

    _install_main_dependencies(monkeypatch, tmp_path, episode_runner=failing_episode)

    with pytest.raises(RuntimeError, match="episode failed"):
        run_fast_loop.main(
            [
                "--split",
                "train",
                "--task-id",
                "task-1",
                "--run-id",
                "learn-failed",
                "--output-root",
                str(tmp_path / "runs"),
                "--project-root",
                str(tmp_path / "isolated-project"),
                "--qwen-base-url",
                "http://qwen.invalid/v1",
                "--model-revision",
                "revision-a",
            ]
        )

    run_path = tmp_path / "runs" / "learn-failed"
    assert (run_path / "manifest.json").is_file()
    assert (run_path / "rollouts" / "events.jsonl").is_file()
    assert not (run_path / "fast_loop_summary.json").exists()


def test_thirty_successful_tasks_execute_exactly_maintenance_round_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    context = RunContext(
        run_id="thirty-task-smoke",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=repository.snapshot().memory_snapshot_id,
        seed=17,
        event_writer=JsonlWriter(tmp_path / "events.jsonl"),
        mode=RunMode.LEARN,
    )

    class EmptyMaintenancePolicy:
        calls = 0

        def generate(self, prompt: Any) -> LifecycleResponse:
            self.calls += 1
            return LifecycleResponse(
                raw_output='{"commands":[]}',
                sampling_params={"temperature": 0.0, "top_p": 1.0},
                latency_s=0.0,
            )

        def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
            raise AssertionError("valid empty maintenance decision must not need repair")

    policy = EmptyMaintenancePolicy()

    def successful_episode(**kwargs: Any) -> EpisodeResult:
        return _episode(kwargs["task_id"], reward=0.0)

    monkeypatch.setattr(run_fast_loop, "run_fast_loop_episode", successful_episode)
    results, executed_rounds = run_fast_loop._run_requested_tasks(
        task_ids=tuple(f"task-{index}" for index in range(1, 31)),
        env_factory=lambda task_id: object(),
        policy=policy,
        repository=repository,
        retriever=object(),
        fast_loop_config=object(),
        context=context,
        completed_train_tasks_before=0,
        maintenance_period=30,
    )

    state = json.loads((repository.root / "maintenance_state.json").read_text("utf-8"))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text("utf-8").splitlines()]
    starts = [event for event in events if event["event_type"] == "MaintenanceStarted"]
    assert len(results) == 30
    assert executed_rounds == (1,)
    assert policy.calls == 1
    assert state["completed_rounds"] == [1]
    assert len(starts) == 1
    assert starts[0]["completed_train_tasks"] == 30
    assert starts[0]["task_id"] == "maintenance-round-1"
