from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tau3_evolver.execution.batch as batch_module
import tau3_evolver.benchmarks.tau2.executor as tau2_executor_module
from tau3_evolver.fast_loop.contracts import EpisodeResult, PendingEpisode
from tau3_evolver.fast_loop.settings import FastLoopConfig
from tau3_evolver.benchmarks.tau2.executor import Tau2BenchmarkExecutor
from tau3_evolver.benchmarks.types import PreparedBenchmark, RuntimeOrigin
from tau3_evolver.config import load_config
from tau3_evolver.execution import ExecutionRequest
from tau3_evolver.execution.memory_state import load_memory_state
from tau3_evolver.fast_loop.maintenance import MaintenanceResult
from tau3_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_evolver.memory.repository import MemoryRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _BaseAgent:
    def __init__(self, tools: list[Any], domain_policy: str) -> None:
        self.tools = tools
        self.domain_policy = domain_policy


class _Message:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _TextConfig:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


class _Registry:
    def __init__(self) -> None:
        self._agent_factories: dict[str, Any] = {}

    def register_agent_factory(self, factory: Any, name: str) -> None:
        self._agent_factories[name] = factory


def test_run_domain_receives_registered_tau3_factory_and_official_task_set(
    monkeypatch,
) -> None:
    registry = _Registry()
    created_agents: list[Any] = []
    received_config: list[Any] = []
    received_evaluator_config: list[Any] = []
    tools = [
        SimpleNamespace(
            openai_schema={
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
    ]

    def run_domain(config: Any) -> Any:
        received_config.append(config)
        factory = registry._agent_factories[config.agent]
        for task_id in config.task_ids:
            agent = factory(
                tools=tools,
                domain_policy="Domain policy",
                task=SimpleNamespace(id=task_id),
            )
            created_agents.append(agent)
        return SimpleNamespace(
            simulations=[
                SimpleNamespace(task_id=task_id, seed=index, reward_info=object())
                for index, task_id in enumerate(config.task_ids, start=1)
            ]
        )

    runtime = SimpleNamespace(
        half_duplex_agent_type=_BaseAgent,
        assistant_message_type=_Message,
        tool_call_type=_Message,
        tool_message_type=_Message,
        multi_tool_message_type=type("Multi", (), {}),
    )
    prepared = PreparedBenchmark(
        name="airline",
        task_ids=("a", "b"),
        split_name="test",
        split_hash="split-hash",
        runtime_origin=RuntimeOrigin(Path("tau2"), "1", "commit"),
        default_memory_namespace="airline",
        task_group="airline",
        executor=Tau2BenchmarkExecutor(
            benchmark_name="airline",
            split_name="test",
            runtime=SimpleNamespace(
                **runtime.__dict__,
                registry=registry,
                run_domain=run_domain,
                text_run_config_type=_TextConfig,
            ),
            evaluator_binding=received_evaluator_config.append,
        ),
    )
    request = ExecutionRequest(
        benchmark="airline",
        mode="test",
        memory_enabled=False,
        run_id="run-1",
    )
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    def finalize_simulation(**values: Any) -> PendingEpisode:
        task_id = str(values["simulation"].task_id)
        return PendingEpisode(
            result=EpisodeResult(
                task_id=task_id,
                final_reward=1.0,
                steps=1,
                terminal_evaluation={},
                selected_memory_ids=(),
                written_memory_ids=(),
                truncated=False,
            ),
            proposals=(),
        )

    monkeypatch.setattr(
        tau2_executor_module,
        "finalize_simulation",
        finalize_simulation,
    )
    result = batch_module.run_batch(
        prepared=prepared,
        request=request,
        project_config=config,
        policy=SimpleNamespace(),
        repository=None,
        destination_repository=None,
        retriever=None,
        fast_loop_config=FastLoopConfig(memory_enabled=False),
        input_memory_snapshot_id=None,
        memory_generation=0,
        episode_writer=None,
    )

    assert result.successful
    assert received_config[0].domain == "airline"
    assert received_config[0].task_split_name == "test"
    assert received_config[0].task_ids == ["a", "b"]
    assert received_config[0].max_concurrency == 2
    assert received_evaluator_config == [config.evaluation.nl_assertions]
    assert len(created_agents) == 2
    assert created_agents[0] is not created_agents[1]
    assert created_agents[0]._public_tools is not created_agents[1]._public_tools
    assert created_agents[0].get_init_state() is not created_agents[1].get_init_state()
    assert registry._agent_factories == {}


def test_successful_train_batch_runs_due_maintenance_before_output_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    registry = _Registry()

    def run_domain(config: Any) -> Any:
        factory = registry._agent_factories[config.agent]
        for task_id in config.task_ids:
            factory(tools=[], domain_policy="Policy", task=SimpleNamespace(id=task_id))
        return SimpleNamespace(
            simulations=[
                SimpleNamespace(task_id=task_id, seed=index, reward_info=object())
                for index, task_id in enumerate(config.task_ids, start=1)
            ]
        )

    runtime = SimpleNamespace(
        half_duplex_agent_type=_BaseAgent,
        assistant_message_type=_Message,
        tool_call_type=_Message,
        tool_message_type=_Message,
        multi_tool_message_type=type("Multi", (), {}),
    )
    prepared = PreparedBenchmark(
        name="retail",
        task_ids=("a", "b"),
        split_name="train",
        split_hash="split-hash",
        runtime_origin=RuntimeOrigin(Path("tau2"), "1", "commit"),
        default_memory_namespace="retail",
        task_group="retail",
        executor=Tau2BenchmarkExecutor(
            benchmark_name="retail",
            split_name="train",
            runtime=SimpleNamespace(
                **runtime.__dict__,
                registry=registry,
                run_domain=run_domain,
                text_run_config_type=_TextConfig,
            ),
        ),
    )
    request = ExecutionRequest(
        benchmark="retail",
        mode="train",
        memory_enabled=True,
        run_id="run-maintenance",
    )
    config = load_config(
        PROJECT_ROOT / "configs" / "default.yaml",
        overrides=("memory.maintenance_period=2",),
    )
    destination = MemoryRepository(tmp_path / "memory")
    input_snapshot = destination.snapshot()
    source = ReadOnlyMemoryRepository(input_snapshot.path)

    def finalize_simulation(**values: Any) -> PendingEpisode:
        task_id = str(values["simulation"].task_id)
        return PendingEpisode(
            result=EpisodeResult(
                task_id=task_id,
                final_reward=1.0,
                steps=1,
                terminal_evaluation={},
                selected_memory_ids=(),
                written_memory_ids=(),
                truncated=False,
            ),
            proposals=(),
        )

    received: list[dict[str, Any]] = []

    def run_due_maintenance(**values: Any) -> MaintenanceResult:
        received.append(values)
        context = values["context"]
        common = {"maintenance_round": 1}
        context.event_writer.append(
            context.event(
                "MaintenanceStarted",
                "maintenance-round-1",
                **common,
                completed_train_tasks=2,
                period=2,
                diagnostics={
                    tier: {"items": []}
                    for tier in ("trajectory", "tip", "skill", "tool")
                },
            )
        )
        context.event_writer.append(
            context.event(
                "MaintenanceProposed",
                "maintenance-round-1",
                **common,
                commands=[],
            )
        )
        context.event_writer.append(
            context.event(
                "MaintenanceCommitted",
                "maintenance-round-1",
                **common,
                looked_up_ids=[],
                created_ids=[],
                updated_ids=[],
            )
        )
        return MaintenanceResult(due=True, executed=True, maintenance_round=1)

    monkeypatch.setattr(
        tau2_executor_module,
        "finalize_simulation",
        finalize_simulation,
    )
    monkeypatch.setattr(batch_module, "run_due_maintenance", run_due_maintenance)
    monkeypatch.setattr(
        batch_module,
        "due_maintenance_rounds",
        lambda **values: (values["completed_train_tasks"] // values["period"],),
    )
    result = batch_module.run_batch(
        prepared=prepared,
        request=request,
        project_config=config,
        policy=SimpleNamespace(),
        repository=source,
        destination_repository=destination,
        retriever=SimpleNamespace(),
        fast_loop_config=FastLoopConfig(memory_enabled=True),
        input_memory_snapshot_id=input_snapshot.memory_snapshot_id,
        memory_generation=1,
        episode_writer=None,
    )

    assert result.successful
    assert received[0]["completed_train_tasks"] == 2
    assert received[0]["period"] == 2
    assert received[0]["maintenance_round"] == 1
    assert result.maintenance is not None
    assert result.maintenance.completed_train_tasks_before == 0
    assert result.maintenance.completed_train_tasks_after == 2
    assert result.maintenance.records[0]["maintenance_round"] == 1
    assert result.output_memory_snapshot_id == destination.snapshot().memory_snapshot_id

    def fail_maintenance(**_values: Any) -> MaintenanceResult:
        raise RuntimeError("maintenance failed")

    monkeypatch.setattr(batch_module, "run_due_maintenance", fail_maintenance)
    second_input = destination.snapshot()
    failed = batch_module.run_batch(
        prepared=prepared,
        request=request.model_copy(update={"run_id": "run-maintenance-failed"}),
        project_config=config,
        policy=SimpleNamespace(),
        repository=ReadOnlyMemoryRepository(second_input.path),
        destination_repository=destination,
        retriever=SimpleNamespace(),
        fast_loop_config=FastLoopConfig(memory_enabled=True),
        input_memory_snapshot_id=second_input.memory_snapshot_id,
        memory_generation=2,
        episode_writer=None,
    )

    assert not failed.successful
    assert failed.failures == ()
    assert failed.maintenance is not None
    assert failed.maintenance.failures[0].maintenance_round == 2
    assert failed.maintenance.failures[0].error_type == "RuntimeError"
    assert load_memory_state(destination.root).completed_tasks == 4
