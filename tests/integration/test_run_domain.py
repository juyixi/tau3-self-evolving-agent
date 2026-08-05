from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tau3_evolver.execution.batch as batch_module
from tau3_evolver.agent.lifecycle import PendingEpisode
from tau3_evolver.benchmarks.types import PreparedBenchmark, RuntimeOrigin
from tau3_evolver.config import load_config
from tau3_evolver.execution import ExecutionRequest
from tau3_evolver.agent.policy import EpisodeResult, FastLoopConfig


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
    tmp_path: Path,
) -> None:
    registry = _Registry()
    created_agents: list[Any] = []
    received_config: list[Any] = []
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
        task_type=SimpleNamespace,
        task_catalog=(SimpleNamespace(id="a"), SimpleNamespace(id="b")),
        task_ids=("a", "b"),
        split_name="test",
        split_hash="split-hash",
        environment_factory=lambda: None,
        runtime=runtime,
        run_domain=run_domain,
        text_run_config_type=_TextConfig,
        registry=registry,
        runtime_origin=RuntimeOrigin(Path("tau2"), "1", "commit"),
        default_memory_namespace="airline",
        task_group="airline",
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

    monkeypatch.setattr(batch_module, "finalize_simulation", finalize_simulation)
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
    assert len(created_agents) == 2
    assert created_agents[0] is not created_agents[1]
    assert created_agents[0]._public_tools is not created_agents[1]._public_tools
    assert created_agents[0].get_init_state() is not created_agents[1].get_init_state()
    assert registry._agent_factories == {}
