from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import threading
from typing import Any
import uuid

from tau3_evolver.benchmarks.tau2.agent import create_tau3_agent_factory
from tau3_evolver.benchmarks.tau2.episodes import finalize_simulation
from tau3_evolver.benchmarks.executor import (
    BenchmarkEpisode,
    BenchmarkExecutionRequest,
    BenchmarkExecutionResult,
    BenchmarkTaskFailure,
)
from tau3_evolver.benchmarks.tau2.runtime import Tau2RuntimeBinding


_REGISTRY_LOCK = threading.RLock()


@dataclass(slots=True)
class _TraceBuffer:
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class Tau2BenchmarkExecutor:
    """Own all Tau2 registration, run_domain, and simulation adapter details."""

    benchmark_name: str
    split_name: str
    runtime: Tau2RuntimeBinding
    evaluator_binding: Callable[..., Any] | None = None

    def execute(
        self,
        request: BenchmarkExecutionRequest,
    ) -> BenchmarkExecutionResult:
        agent = request.agent
        factory = create_tau3_agent_factory(
            runtime=self.runtime,
            benchmark=self.benchmark_name,
            policy=agent.policy,
            repository=agent.repository,
            retriever=agent.retriever,
            config=agent.config,
            memory_source_namespace=agent.memory_source_namespace,
            cross_domain_memory=agent.cross_domain_memory,
        )
        tau2_results = self._run_domain(request, factory)
        simulations = {
            str(item.task_id): item for item in tau2_results.simulations
        }
        missing = set(request.task_ids) - set(simulations)
        if missing:
            raise RuntimeError(
                f"Tau2 run_domain omitted task results: {sorted(missing)}"
            )

        episodes: list[BenchmarkEpisode] = []
        failures: list[BenchmarkTaskFailure] = []
        for task_id in request.task_ids:
            simulation = simulations[task_id]
            seed = int(
                simulation.seed
                if getattr(simulation, "seed", None) is not None
                else request.project_config.execution.seed
            )
            if _is_infrastructure_failure(simulation):
                failures.append(
                    BenchmarkTaskFailure(
                        task_id=task_id,
                        stage="run_domain",
                        error_type=_simulation_error_type(simulation),
                        seed=seed,
                    )
                )
                continue

            buffer = _TraceBuffer()
            context = request.context_factory(seed, buffer)
            try:
                episode = finalize_simulation(
                    runtime=self.runtime,
                    simulation=simulation,
                    policy=agent.policy,
                    config=agent.config,
                    context=context,
                    propose_experience=agent.propose_experience,
                )
            except Exception as error:
                failures.append(
                    BenchmarkTaskFailure(
                        task_id=task_id,
                        stage="finalize",
                        error_type=type(error).__name__,
                        seed=seed,
                    )
                )
                continue
            episodes.append(
                BenchmarkEpisode(
                    episode=episode,
                    events=tuple(buffer.events),
                    seed=seed,
                )
            )
        return BenchmarkExecutionResult(
            episodes=tuple(episodes),
            failures=tuple(failures),
        )

    def _run_domain(
        self,
        request: BenchmarkExecutionRequest,
        factory: Callable[..., Any],
    ) -> Any:
        config = request.project_config
        agent_name = f"tau3_agent_{uuid.uuid4().hex}"
        with _REGISTRY_LOCK:
            if self.evaluator_binding is not None:
                self.evaluator_binding(config.evaluation.nl_assertions)
            self.runtime.registry.register_agent_factory(factory, agent_name)
            try:
                with tempfile.TemporaryDirectory(
                    prefix="tau3-evolver-tau2-"
                ) as working:
                    return self.runtime.run_domain(
                        self.runtime.text_run_config_type(
                            domain=self.benchmark_name,
                            task_set_name=self.benchmark_name,
                            task_split_name=self.split_name,
                            task_ids=list(request.task_ids),
                            agent=agent_name,
                            llm_agent=config.model.base_model,
                            llm_args_agent={},
                            user="user_simulator",
                            llm_user=config.tau2.user_llm,
                            llm_args_user=dict(config.tau2.user_llm_args),
                            num_trials=1,
                            max_steps=max(4, config.rollout.max_episode_steps * 4),
                            max_errors=10,
                            save_to=str((Path(working) / "tau2").resolve()),
                            max_concurrency=min(
                                config.execution.max_concurrency,
                                len(request.task_ids),
                            ),
                            seed=config.execution.seed,
                            log_level="WARNING",
                            max_retries=2,
                            retry_delay=1.0,
                            auto_resume=False,
                            hallucination_retries=0,
                            enforce_communication_protocol=True,
                        )
                    )
            finally:
                _unregister_agent_factory(self.runtime.registry, agent_name)


def _unregister_agent_factory(registry: Any, agent_name: str) -> None:
    factories = getattr(registry, "_agent_factories", None)
    if not isinstance(factories, dict):
        raise RuntimeError("Tau2 Registry does not expose its Agent factory store")
    factories.pop(agent_name, None)


def _is_infrastructure_failure(simulation: Any) -> bool:
    reason = getattr(getattr(simulation, "termination_reason", None), "value", None)
    reason = str(reason or getattr(simulation, "termination_reason", ""))
    return reason == "infrastructure_error" or simulation.reward_info is None


def _simulation_error_type(simulation: Any) -> str:
    info = getattr(simulation, "info", None)
    if isinstance(info, Mapping) and isinstance(info.get("error_type"), str):
        return info["error_type"]
    return "InfrastructureError"


__all__ = ["Tau2BenchmarkExecutor"]
