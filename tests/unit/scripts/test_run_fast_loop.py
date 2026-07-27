from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_fast_loop
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.envs.base import ResetResult, StepResult
from tau3_retail_evolver.fast_loop.runner import (
    EpisodeResult,
    FastLoopConfig,
    LifecycleResponse,
)
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever
from tau3_retail_evolver.memory.types import MemoryStatus


def _config(tmp_path: Path, *, memory_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        tau2=SimpleNamespace(
            repo_path=tmp_path / "external" / "tau2-bench",
            domain="retail",
        ),
        model=SimpleNamespace(base_model="Qwen/Qwen3.5-9B"),
        rollout=SimpleNamespace(temperature=1.0, top_p=0.95, max_episode_steps=40),
        memory=SimpleNamespace(
            enabled=memory_enabled,
            agent_id="retail",
            retrieve_top_k=50,
            maintenance_period=30,
            max_new_tips_per_episode=2,
            max_new_skills_per_episode=1,
            max_new_tools_per_episode=1,
            max_new_trajectories_per_episode=1,
            maintenance_tip_capacity=200,
            maintenance_similarity_threshold=0.92,
            maintenance_priority_pair_limit=24,
            retrieval_mmr_lambda_tip=0.65,
            retrieval_mmr_lambda_skill=0.80,
            retrieval_mmr_lambda_tool=0.85,
            retrieval_mmr_lambda_trajectory=0.75,
            retrieval_global_mmr_lambda=0.75,
            retrieval_quota_tip=18,
            retrieval_quota_skill=18,
            retrieval_quota_tool=6,
            retrieval_quota_trajectory=4,
            selection_max_total=20,
            selection_max_tip=7,
            selection_max_skill=8,
            selection_max_tool=3,
            selection_max_trajectory=2,
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


def _run_context(tmp_path: Path, *, events: list[dict[str, Any]] | None = None) -> RunContext:
    return RunContext(
        run_id="memory-switch-test",
        iteration=0,
        split="train",
        model_revision="revision-a",
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=17,
        event_writer=events if events is not None else JsonlWriter(tmp_path / "events.jsonl"),
        mode=RunMode.LEARN,
        default_task_group="retail",
    )


class _UnusedEmbeddingProvider:
    model_revision = "unused-embedding@1"
    dimension = 2

    def embed(self, text: str) -> tuple[float, float]:
        raise AssertionError("retriever must not be used")

    def embed_batch(self, texts: list[str]) -> list[tuple[float, float]]:
        raise AssertionError("retriever must not be used")


@pytest.mark.parametrize(
    ("memory_enabled", "use_repository", "use_retriever"),
    (
        (True, False, True),
        (True, True, False),
        (False, False, True),
        (False, True, False),
    ),
)
def test_run_requested_tasks_rejects_mismatched_memory_dependencies_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    memory_enabled: bool,
    use_repository: bool,
    use_retriever: bool,
) -> None:
    snapshot_calls: list[None] = []
    environment_calls: list[str] = []
    maintenance_calls: list[dict[str, Any]] = []

    class SnapshotRecordingRepository(MemoryRepository):
        def snapshot(self):  # type: ignore[no-untyped-def]
            snapshot_calls.append(None)
            return super().snapshot()

    repository = SnapshotRecordingRepository(tmp_path / "memory") if use_repository else None
    retriever = Retriever(_UnusedEmbeddingProvider()) if use_retriever else None
    monkeypatch.setattr(
        run_fast_loop,
        "run_due_maintenance",
        lambda **kwargs: maintenance_calls.append(kwargs),
    )

    with pytest.raises(ValueError):
        run_fast_loop._run_requested_tasks(
            task_ids=("task-1",),
            env_factory=lambda task_id: environment_calls.append(task_id),
            policy=object(),
            repository=repository,
            retriever=retriever,
            fast_loop_config=FastLoopConfig(memory_enabled=memory_enabled),
            context=_run_context(tmp_path),
            completed_train_tasks_before=0,
            maintenance_period=30,
        )

    assert snapshot_calls == []
    assert environment_calls == []
    assert maintenance_calls == []


@pytest.mark.parametrize(
    ("fast_loop_config_factory", "repository_factory", "retriever_factory", "error"),
    (
        (
            lambda _tmp_path: object(),
            lambda _tmp_path: None,
            lambda: None,
            "FastLoopConfig",
        ),
        (
            lambda _tmp_path: FastLoopConfig(),
            lambda tmp_path: ReadOnlyMemoryRepository(
                MemoryRepository(tmp_path / "memory").snapshot().path
            ),
            lambda: Retriever(_UnusedEmbeddingProvider()),
            "mutable MemoryRepository",
        ),
        (
            lambda _tmp_path: FastLoopConfig(),
            lambda _tmp_path: SimpleNamespace(
                snapshot=lambda: SimpleNamespace(memory_snapshot_id="fake-snapshot")
            ),
            lambda: Retriever(_UnusedEmbeddingProvider()),
            "mutable MemoryRepository",
        ),
        (
            lambda _tmp_path: FastLoopConfig(),
            lambda tmp_path: MemoryRepository(tmp_path / "memory"),
            lambda: object(),
            "Retriever",
        ),
    ),
    ids=("invalid-config", "read-only-repository", "fake-repository", "fake-retriever"),
)
def test_run_requested_tasks_validates_runtime_contract_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fast_loop_config_factory: Any,
    repository_factory: Any,
    retriever_factory: Any,
    error: str,
) -> None:
    snapshot_calls: list[None] = []
    environment_calls: list[str] = []
    maintenance_calls: list[dict[str, Any]] = []
    repository = repository_factory(tmp_path)
    if hasattr(repository, "snapshot"):
        original_snapshot = repository.snapshot

        def snapshot():  # type: ignore[no-untyped-def]
            snapshot_calls.append(None)
            return original_snapshot()

        monkeypatch.setattr(repository, "snapshot", snapshot)
    monkeypatch.setattr(
        run_fast_loop,
        "run_due_maintenance",
        lambda **kwargs: maintenance_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match=error):
        run_fast_loop._run_requested_tasks(
            task_ids=("task-1",),
            env_factory=lambda task_id: environment_calls.append(task_id),
            policy=object(),
            repository=repository,
            retriever=retriever_factory(),
            fast_loop_config=fast_loop_config_factory(tmp_path),
            context=_run_context(tmp_path),
            completed_train_tasks_before=0,
            maintenance_period=30,
        )

    assert snapshot_calls == []
    assert environment_calls == []
    assert maintenance_calls == []


def test_run_requested_tasks_disabled_uses_real_episode_without_memory_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[dict[str, Any]] = []
    environments: list[Any] = []

    class FakeEnvironment:
        def reset(self, *, seed: int) -> ResetResult:
            return ResetResult(
                observation="Customer asks for a refund",
                info={"policy": {"text": "Verify identity"}, "tools": []},
            )

        def step(self, action: str) -> StepResult:
            return StepResult(
                observation="Refund complete",
                reward=1.0,
                done=True,
                terminated=True,
                truncated=False,
                info={
                    "parse_error": None,
                    "reward_info": '{"score":1.0}',
                    "simulation_run": '{"status":"complete"}',
                },
            )

        def close(self) -> None:
            return None

    class ActionPolicy:
        def __init__(self) -> None:
            self.prompts: list[Any] = []

        def generate(self, prompt: Any) -> LifecycleResponse:
            self.prompts.append(prompt)
            return LifecycleResponse(
                raw_output='{"action":"finish"}',
                sampling_params={"temperature": 0.0, "top_p": 1.0},
                latency_s=0.0,
            )

        def repair(self, prompt: Any, raw_output: str, error: str) -> LifecycleResponse:
            raise AssertionError("valid action must not need repair")

    policy = ActionPolicy()
    monkeypatch.setattr(
        run_fast_loop,
        "run_due_maintenance",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("maintenance must not run")),
    )
    results, maintenance_rounds, failures = run_fast_loop._run_requested_tasks(
        task_ids=("task-1",),
        env_factory=lambda task_id: environments.append(FakeEnvironment()) or environments[-1],
        policy=policy,
        repository=None,
        retriever=None,
        fast_loop_config=FastLoopConfig(memory_enabled=False),
        context=_run_context(tmp_path, events=events),
        completed_train_tasks_before=0,
        maintenance_period=30,
    )

    memory_event_types = {
        "MemoryCandidatesRetrieved",
        "MemorySelected",
        "MemoryWriteProposed",
        "MemoryWriteCommitted",
        "MemoryWriteFailed",
    }
    assert environments
    assert maintenance_rounds == ()
    assert failures == ()
    assert results[0].selected_memory_ids == ()
    assert results[0].written_memory_ids == ()
    assert [prompt.kind for prompt in policy.prompts] == ["action"]
    assert "memories" not in policy.prompts[0].payload
    assert not memory_event_types.intersection(event["event_type"] for event in events)


def _install_main_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    episode_runner: Any,
    memory_enabled: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    config = _config(tmp_path, memory_enabled=memory_enabled)
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
    task_group_signatures = {
        "task-1": f"retail-actions-v1:{'1' * 64}",
        "task-2": f"retail-actions-v1:{'2' * 64}",
    }
    task_groups = SimpleNamespace(
        signature_for=lambda task_id: task_group_signatures[task_id]
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
    monkeypatch.setattr(
        run_fast_loop,
        "RetailTaskGroups",
        SimpleNamespace(
            from_file=lambda tasks_path, *, task_ids: ordering.append("task-groups")
            or task_groups
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
                "--completed-train-tasks-before",
                "0",
            ]
        )


def test_requires_explicit_completed_train_tasks_before() -> None:
    with pytest.raises(SystemExit) as error:
        run_fast_loop.parse_args(
            [
                "--split",
                "train",
                "--task-id",
                "task-1",
                "--run-id",
                "learn-001",
                "--model-revision",
                "revision-a",
            ]
        )

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
        (("--run-id", "../escape"), "run ID"),
        (("--iteration", "-1"), "iteration"),
        (("--completed-train-tasks-before", "-1"), "completed"),
        (("--seed", "-1"), "seed"),
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
        "--completed-train-tasks-before",
        "0",
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


def test_all_train_tasks_uses_seeded_reproducible_shuffle() -> None:
    official = tuple(str(index) for index in range(74))

    first = run_fast_loop._resolve_train_tasks(
        None,
        all_train_tasks=True,
        official_train_task_ids=official,
        seed=42,
    )
    repeated = run_fast_loop._resolve_train_tasks(
        None,
        all_train_tasks=True,
        official_train_task_ids=official,
        seed=42,
    )
    changed = run_fast_loop._resolve_train_tasks(
        None,
        all_train_tasks=True,
        official_train_task_ids=official,
        seed=43,
    )

    assert first == repeated
    assert first != changed
    assert set(first) == set(official)
    assert len(first) == 74


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
                "--completed-train-tasks-before",
                "0",
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
                "--completed-train-tasks-before",
                "0",
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
    assert [
        context.task_group_for(task_id)
        for context, task_id in zip(episode_contexts, captured_tasks, strict=True)
    ] == [f"retail-actions-v1:{'1' * 64}", f"retail-actions-v1:{'2' * 64}"]
    assert all(
        context.default_task_group == "retail-v2:maintenance"
        for context in episode_contexts
    )
    maintenance_contexts = captured["maintenance_contexts"]
    assert maintenance_contexts[0].memory_snapshot_id == episode_contexts[1].memory_snapshot_id
    assert maintenance_contexts[1].memory_snapshot_id == summary["output_memory_snapshot_id"]
    assert manifest["adapter_revision"] == "adapter-revision-b"
    assert manifest["parent_checkpoint"] is None
    assert manifest["task_ids"] == ["task-1", "task-2"]
    assert manifest["rollout_options"]["memory_enabled"] is True
    assert manifest["rollout_options"]["memory_agent_id"] == "retail"
    assert summary == {
        "attempted_task_count": 2,
        "completed_train_tasks_after": 27,
        "completed_train_tasks_before": 25,
        "episode_count": 2,
        "failed_maintenance_rounds": [],
        "failed_task_count": 0,
        "failed_task_ids": [],
        "failures": [],
        "input_memory_snapshot_id": manifest["memory_snapshot_id"],
        "maintenance_rounds_executed": [],
        "memory_enabled": True,
        "output_memory_snapshot_id": summary["output_memory_snapshot_id"],
        "input_memory_counts": {
            "trajectory": 0,
            "tip": 0,
            "skill": 0,
            "tool": 0,
        },
        "output_memory_counts": {
            "trajectory": 0,
            "tip": 2,
            "skill": 0,
            "tool": 0,
        },
        "memory_item_count": 2,
        "memory_selection_count": 0,
        "unique_reused_memory_count": 0,
        "memory_reuse_coverage": 0.0,
        "token_usage_episode_count": 0,
        "total_agent_prompt_tokens": 0,
        "total_agent_completion_tokens": 0,
        "total_agent_tokens": 0,
        "mean_agent_tokens": None,
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
        "max_tokens": 2048,
        "request_timeout_s": 600.0,
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
        "task-groups",
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
    assert {event["schema_version"] for event in rollout_events} == {2}
    assert {event["task_group"] for event in rollout_events} == {
        f"retail-actions-v1:{'1' * 64}",
        f"retail-actions-v1:{'2' * 64}",
    }
    all_artifacts = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.rglob("*")
        if path.is_file()
    )
    for secret in ("qwen-secret", "openrouter-secret", "simulator-secret"):
        assert secret not in all_artifacts
        assert secret not in stdout


def test_main_validates_enabled_memory_dependencies_before_input_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    episode_calls: list[str] = []

    def episode_runner(**kwargs: Any) -> EpisodeResult:
        episode_calls.append(kwargs["task_id"])
        return _episode(kwargs["task_id"])

    ordering, captured = _install_main_dependencies(
        monkeypatch,
        tmp_path,
        episode_runner=episode_runner,
    )
    mutable = MemoryRepository(tmp_path / "seed-memory")
    read_only = ReadOnlyMemoryRepository(mutable.snapshot().path)
    snapshot_calls: list[None] = []
    monkeypatch.setattr(
        read_only,
        "snapshot",
        lambda: snapshot_calls.append(None)
        or SimpleNamespace(memory_snapshot_id="invalid-input-snapshot"),
    )
    monkeypatch.setattr(
        run_fast_loop,
        "open_training_memory",
        lambda memory_config, *, root=None: read_only,
    )
    monkeypatch.setattr(
        run_fast_loop,
        "build_embedding_provider",
        lambda memory_config, memory_root: _UnusedEmbeddingProvider(),
    )

    with pytest.raises(ValueError, match="mutable MemoryRepository"):
        run_fast_loop.main(
            [
                "--split",
                "train",
                "--task-id",
                "task-1",
                "--run-id",
                "read-only-memory",
                "--output-root",
                str(tmp_path / "runs"),
                "--project-root",
                str(tmp_path / "isolated-project"),
                "--completed-train-tasks-before",
                "0",
                "--qwen-base-url",
                "http://qwen.invalid/v1",
                "--model-revision",
                "model-revision-a",
            ]
        )

    assert snapshot_calls == []
    assert len(captured["environments"]) == 1
    assert ordering.count("environment:task-1") == 1
    assert episode_calls == []


def test_disables_memory_dependencies_and_records_no_memory_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    episode_contexts: list[RunContext] = []

    def episode_runner(**kwargs: Any) -> EpisodeResult:
        assert kwargs["repository"] is None
        assert kwargs["retriever"] is None
        assert kwargs["config"].memory_enabled is False
        episode_contexts.append(kwargs["context"])
        kwargs["environment"].close()
        return _episode(kwargs["task_id"])

    _install_main_dependencies(
        monkeypatch,
        tmp_path,
        episode_runner=episode_runner,
        memory_enabled=False,
    )

    def memory_dependency_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("memory dependency must not be called when disabled")

    monkeypatch.setattr(run_fast_loop, "open_training_memory", memory_dependency_called)
    monkeypatch.setattr(run_fast_loop, "build_embedding_provider", memory_dependency_called)
    monkeypatch.setattr(run_fast_loop, "run_due_maintenance", memory_dependency_called)

    returncode = run_fast_loop.main(
        [
            "--split",
            "train",
            "--task-id",
            "task-1",
            "--run-id",
            "learn-no-memory",
            "--output-root",
            str(tmp_path / "runs"),
            "--project-root",
            str(tmp_path / "isolated-project"),
            "--qwen-base-url",
            "http://qwen.invalid/v1",
            "--model-revision",
            "revision-a",
            "--completed-train-tasks-before",
            "0",
        ]
    )

    run_path = tmp_path / "runs" / "learn-no-memory"
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_path / "fast_loop_summary.json").read_text(encoding="utf-8"))

    assert returncode == 0
    assert all(context.memory_snapshot_id is None for context in episode_contexts)
    assert manifest["rollout_options"]["memory_enabled"] is False
    assert manifest["rollout_options"]["memory_agent_id"] is None
    assert manifest["memory_snapshot_id"] is None
    assert summary["memory_enabled"] is False
    assert summary["input_memory_snapshot_id"] is None
    assert summary["output_memory_snapshot_id"] is None
    assert summary["maintenance_rounds_executed"] == []
    assert not (tmp_path / "isolated-project" / "history").exists()
    assert json.loads(capsys.readouterr().out) == summary


def test_episode_failure_is_recorded_and_next_task_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def failing_episode(**kwargs: Any) -> EpisodeResult:
        calls.append(kwargs["task_id"])
        if kwargs["task_id"] == "task-2":
            return _episode("task-2")
        context = kwargs["context"]
        context.event_writer.append(
            context.event("EpisodeFailed", kwargs["task_id"], error={"type": "RuntimeError"})
        )
        raise RuntimeError("episode failed")

    _install_main_dependencies(monkeypatch, tmp_path, episode_runner=failing_episode)

    returncode = run_fast_loop.main(
        [
            "--split",
            "train",
            "--task-id",
            "task-1",
            "--task-id",
            "task-2",
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
            "--completed-train-tasks-before",
            "0",
        ]
    )

    run_path = tmp_path / "runs" / "learn-failed"
    summary = json.loads(
        (run_path / "fast_loop_summary.json").read_text(encoding="utf-8")
    )
    events = [
        json.loads(line)
        for line in (run_path / "rollouts" / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert returncode == 0
    assert calls == ["task-1", "task-2"]
    assert (run_path / "manifest.json").is_file()
    assert (run_path / "rollouts" / "events.jsonl").is_file()
    assert summary["attempted_task_count"] == 2
    assert summary["episode_count"] == 1
    assert summary["successful_task_ids"] == ["task-2"]
    assert summary["failed_task_ids"] == ["task-1"]
    assert summary["completed_train_tasks_after"] == 2
    assert [event["event_type"] for event in events] == ["TaskFailed"]
    assert events[0]["task_id"] == "task-1"
    assert events[0]["error"]["types"] == ["RuntimeError"]
    assert events[0]["error"]["messages"] == ["episode failed"]


def test_episode_failure_deactivates_new_memory_before_continuing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events: list[dict[str, Any]] = []

    def episode(**kwargs: Any) -> EpisodeResult:
        if kwargs["task_id"] == "task-1":
            repository.add(
                tier="tip",
                content="Must not survive a failed task",
                source_task_ids=("task-1",),
                created_round=0,
            )
            raise RuntimeError("write failed after commit")
        return _episode(kwargs["task_id"])

    monkeypatch.setattr(run_fast_loop, "run_fast_loop_episode", episode)
    monkeypatch.setattr(
        run_fast_loop,
        "run_due_maintenance",
        lambda **kwargs: SimpleNamespace(executed=False),
    )

    results, _, failures = run_fast_loop._run_requested_tasks(
        task_ids=("task-1", "task-2"),
        env_factory=lambda task_id: object(),
        policy=object(),
        repository=repository,
        retriever=Retriever(_UnusedEmbeddingProvider()),
        fast_loop_config=FastLoopConfig(memory_enabled=True),
        context=_run_context(tmp_path, events=events),
        completed_train_tasks_before=0,
        maintenance_period=30,
    )

    assert [result.task_id for result in results] == ["task-2"]
    assert [failure["task_id"] for failure in failures] == ["task-1"]
    assert repository.list() == []
    discarded = repository.list(status=None)
    assert len(discarded) == 1
    assert discarded[0].status is MemoryStatus.RETIRED
    assert events[0]["event_type"] == "TaskFailed"
    assert events[0]["discarded_memory_ids"] == [discarded[0].id]


def test_maintenance_failure_is_recorded_and_next_task_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = MemoryRepository(tmp_path / "memory")
    events: list[dict[str, Any]] = []
    maintenance_calls = 0

    def maintenance(**kwargs: Any) -> SimpleNamespace:
        nonlocal maintenance_calls
        maintenance_calls += 1
        if maintenance_calls == 1:
            raise RuntimeError("maintenance failed")
        return SimpleNamespace(executed=False)

    monkeypatch.setattr(
        run_fast_loop,
        "run_fast_loop_episode",
        lambda **kwargs: _episode(kwargs["task_id"]),
    )
    monkeypatch.setattr(run_fast_loop, "run_due_maintenance", maintenance)

    results, rounds, failures = run_fast_loop._run_requested_tasks(
        task_ids=("task-1", "task-2"),
        env_factory=lambda task_id: object(),
        policy=object(),
        repository=repository,
        retriever=Retriever(_UnusedEmbeddingProvider()),
        fast_loop_config=FastLoopConfig(memory_enabled=True),
        context=_run_context(tmp_path, events=events),
        completed_train_tasks_before=0,
        maintenance_period=1,
    )

    assert [result.task_id for result in results] == ["task-1", "task-2"]
    assert rounds == ()
    assert len(failures) == 1
    assert failures[0]["stage"] == "maintenance"
    assert len(events) == 1
    assert events[0]["event_type"] == "MaintenanceTaskFailed"
    assert events[0]["task_id"] == "maintenance-round-1"


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
        default_task_group="retail",
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
    results, executed_rounds, failures = run_fast_loop._run_requested_tasks(
        task_ids=tuple(f"task-{index}" for index in range(1, 31)),
        env_factory=lambda task_id: object(),
        policy=policy,
        repository=repository,
        retriever=Retriever(_UnusedEmbeddingProvider()),
        fast_loop_config=FastLoopConfig(),
        context=context,
        completed_train_tasks_before=0,
        maintenance_period=30,
    )

    state = json.loads((repository.root / "maintenance_state.json").read_text("utf-8"))
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text("utf-8").splitlines()]
    starts = [event for event in events if event["event_type"] == "MaintenanceStarted"]
    assert len(results) == 30
    assert executed_rounds == (1,)
    assert failures == ()
    assert policy.calls == 1
    assert state["completed_rounds"] == [1]
    assert len(starts) == 1
    assert starts[0]["completed_train_tasks"] == 30
    assert starts[0]["task_id"] == "maintenance-round-1"
    assert starts[0]["task_group"] == "retail"
