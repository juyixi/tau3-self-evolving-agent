from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from tau3_retail_evolver.eval.guard import (
    EvaluationGuard,
    EvaluationMemory,
    EvaluationProtocol,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.maintenance import run_evaluation_maintenance
from tau3_retail_evolver.fast_loop.runner import (
    EpisodeResult,
    FastLoopConfig,
    FastLoopEnvironment,
    FastLoopPolicy,
    _run_lifecycle_episode,
)
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever


EnvironmentFactory = Callable[[str], FastLoopEnvironment]
RetrieverFactory = Callable[[EvaluationMemory], Retriever]


@dataclass(frozen=True, slots=True)
class TrialEpisode:
    trial_index: int
    seed: int
    result: EpisodeResult


@dataclass(frozen=True, slots=True)
class EvaluationRunResult:
    episodes: tuple[TrialEpisode, ...]
    maintenance_rounds_by_trial: tuple[tuple[int, ...], ...]
    output_memory_snapshot_ids: tuple[str | None, ...]


def run_evaluation_episode(
    *,
    task_id: str,
    task_instruction: str,
    environment: FastLoopEnvironment,
    policy: FastLoopPolicy,
    memory: EvaluationMemory,
    retriever: Retriever | None,
    config: FastLoopConfig,
    context: RunContext,
    guard: EvaluationGuard,
    trial_index: int,
) -> EpisodeResult:
    """Run one guarded held-out episode without exposing learning APIs."""
    guard.validate_episode(context, memory, trial_index=trial_index)
    _validate_dependencies(
        guard=guard,
        memory=memory,
        retriever=retriever,
        config=config,
    )
    return _run_lifecycle_episode(
        task_id=task_id,
        task_instruction=task_instruction,
        environment=environment,
        policy=policy,
        repository=memory.repository,
        retriever=retriever,
        config=config,
        context=context,
        write_memory=guard.capabilities.memory_write,
        memory_disabled_reason="protocol",
    )


def run_evaluation_trials(
    *,
    task_ids: Sequence[str],
    seeds: Sequence[int],
    env_factory: EnvironmentFactory,
    policy: FastLoopPolicy,
    guard: EvaluationGuard,
    retriever_factory: RetrieverFactory | None,
    config: FastLoopConfig,
    context: RunContext,
    maintenance_period: int,
    task_instruction: str = "Resolve the retail request shown in the current conversation.",
) -> EvaluationRunResult:
    """Run independent seeded trials in official task order."""
    tasks = tuple(task_ids)
    trial_seeds = tuple(seeds)
    _validate_run_inputs(
        tasks=tasks,
        seeds=trial_seeds,
        context=context,
        guard=guard,
        maintenance_period=maintenance_period,
        retriever_factory=retriever_factory,
        config=config,
    )

    episodes: list[TrialEpisode] = []
    maintenance_by_trial: list[tuple[int, ...]] = []
    output_snapshots: list[str | None] = []
    for trial_index, seed in enumerate(trial_seeds):
        memory = guard.open_memory(trial_index=trial_index)
        retriever = (
            retriever_factory(memory)
            if memory.repository is not None and retriever_factory is not None
            else None
        )
        executed_rounds: list[int] = []
        for task_index, task_id in enumerate(tasks, start=1):
            snapshot_id = _current_snapshot_id(memory)
            episode_context = replace(
                context,
                seed=seed,
                trial_index=trial_index,
                memory_snapshot_id=snapshot_id,
            )
            result = run_evaluation_episode(
                task_id=task_id,
                task_instruction=task_instruction,
                environment=env_factory(task_id),
                policy=policy,
                memory=memory,
                retriever=retriever,
                config=config,
                context=episode_context,
                guard=guard,
                trial_index=trial_index,
            )
            episodes.append(
                TrialEpisode(
                    trial_index=trial_index,
                    seed=seed,
                    result=result,
                )
            )
            if guard.protocol is EvaluationProtocol.TEST_STREAMING:
                repository = memory.repository
                assert isinstance(repository, MemoryRepository)
                maintenance_context = replace(
                    episode_context,
                    memory_snapshot_id=repository.snapshot().memory_snapshot_id,
                )
                maintenance = run_evaluation_maintenance(
                    completed_tasks=task_index,
                    period=maintenance_period,
                    repository=repository,
                    policy=policy,
                    context=maintenance_context,
                )
                if maintenance.executed:
                    executed_rounds.append(maintenance.maintenance_round)

        maintenance_by_trial.append(tuple(executed_rounds))
        output_snapshots.append(_current_snapshot_id(memory))

    return EvaluationRunResult(
        episodes=tuple(episodes),
        maintenance_rounds_by_trial=tuple(maintenance_by_trial),
        output_memory_snapshot_ids=tuple(output_snapshots),
    )


def _validate_dependencies(
    *,
    guard: EvaluationGuard,
    memory: EvaluationMemory,
    retriever: Retriever | None,
    config: FastLoopConfig,
) -> None:
    if not isinstance(config, FastLoopConfig):
        raise ValueError("evaluation config must be a FastLoopConfig")
    if guard.protocol is EvaluationProtocol.NO_MEMORY:
        if config.memory_enabled:
            raise ValueError("no_memory requires memory_enabled=false")
        if memory.repository is not None or retriever is not None:
            raise ValueError("no_memory requires no repository or Retriever")
        return
    if not config.memory_enabled:
        raise ValueError(f"{guard.protocol.value} requires memory_enabled=true")
    if not isinstance(retriever, Retriever):
        raise ValueError(f"{guard.protocol.value} requires a Retriever")
    if guard.protocol is EvaluationProtocol.TEST_STATIC:
        if not isinstance(memory.repository, ReadOnlyMemoryRepository):
            raise ValueError("test_static requires a read-only repository")
    elif not isinstance(memory.repository, MemoryRepository):
        raise ValueError("test_streaming requires a mutable repository")


def _validate_run_inputs(
    *,
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    context: RunContext,
    guard: EvaluationGuard,
    maintenance_period: int,
    retriever_factory: RetrieverFactory | None,
    config: FastLoopConfig,
) -> None:
    if not tasks or any(not isinstance(task_id, str) or not task_id for task_id in tasks):
        raise ValueError("evaluation task IDs must be non-empty strings")
    if len(tasks) != len(set(tasks)):
        raise ValueError("evaluation task IDs must be unique")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be unique")
    if type(maintenance_period) is not int or maintenance_period <= 0:
        raise ValueError("maintenance period must be positive")
    if context.mode is not RunMode.EVALUATE:
        raise ValueError("evaluation run requires EVALUATE mode")
    if context.run_id != guard.run_id or context.split != guard.split:
        raise ValueError("evaluation context does not match guard")
    if not isinstance(config, FastLoopConfig):
        raise ValueError("evaluation config must be a FastLoopConfig")
    expected_memory_enabled = guard.protocol is not EvaluationProtocol.NO_MEMORY
    if config.memory_enabled is not expected_memory_enabled:
        expectation = "true" if expected_memory_enabled else "false"
        raise ValueError(
            f"{guard.protocol.value} requires memory_enabled={expectation}"
        )
    if (
        guard.protocol is EvaluationProtocol.NO_MEMORY
        and retriever_factory is not None
    ):
        raise ValueError("no_memory requires no Retriever factory")
    if (
        guard.protocol is not EvaluationProtocol.NO_MEMORY
        and retriever_factory is None
    ):
        raise ValueError(f"{guard.protocol.value} requires a Retriever factory")


def _current_snapshot_id(memory: EvaluationMemory) -> str | None:
    repository = memory.repository
    if repository is None:
        return None
    if isinstance(repository, ReadOnlyMemoryRepository):
        return repository.memory_snapshot_id
    return repository.snapshot().memory_snapshot_id
