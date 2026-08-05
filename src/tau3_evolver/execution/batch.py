from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import tempfile
import threading
from typing import Any
import uuid

from tau3_evolver.agent.factory import create_tau3_agent_factory
from tau3_evolver.agent.lifecycle import PendingEpisode, finalize_simulation
from tau3_evolver.artifacts.episodes import (
    build_completed_episode,
    build_failed_episode,
)
from tau3_evolver.benchmarks.types import PreparedBenchmark
from tau3_evolver.config import ProjectConfig
from tau3_evolver.execution.events import BufferedEventWriter, EventWriter, ExecutionContext
from tau3_evolver.execution.request import ExecutionRequest
from tau3_evolver.execution.results import BatchFailure, BatchResult
from tau3_evolver.agent.policy import FastLoopConfig, FastLoopPolicy
from tau3_evolver.memory.batches import commit_batch_state
from tau3_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.retrieval import Retriever


_REGISTRY_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class EpisodeCommit:
    written_ids: tuple[str, ...]
    replayed_ids: tuple[str, ...]


def run_batch(
    *,
    prepared: PreparedBenchmark,
    request: ExecutionRequest,
    project_config: ProjectConfig,
    policy: FastLoopPolicy,
    repository: MemoryRepository | ReadOnlyMemoryRepository | None,
    destination_repository: MemoryRepository | None,
    retriever: Retriever | None,
    fast_loop_config: FastLoopConfig,
    input_memory_snapshot_id: str | None,
    memory_generation: int,
    episode_writer: EventWriter | None,
) -> BatchResult:
    source_namespace = request.resolved_memory_source(
        prepared.default_memory_namespace
    )
    cross_domain = request.is_cross_domain_memory(
        prepared.default_memory_namespace
    )
    factory = create_tau3_agent_factory(
        runtime=_runtime_view(prepared),
        benchmark=prepared.name,
        policy=policy,
        repository=repository,
        retriever=retriever,
        config=fast_loop_config,
        memory_source_namespace=source_namespace,
        cross_domain_memory=cross_domain,
    )
    agent_name = f"tau3_agent_{uuid.uuid4().hex}"
    with _REGISTRY_LOCK:
        if prepared.evaluator_binding is not None:
            prepared.evaluator_binding(project_config.evaluation.nl_assertions)
        prepared.registry.register_agent_factory(factory, agent_name)
        try:
            with tempfile.TemporaryDirectory(prefix="tau3-evolver-tau2-") as working:
                tau2_results = prepared.run_domain(
                    prepared.text_run_config_type(
                        domain=prepared.name,
                        task_set_name=prepared.name,
                        task_split_name=prepared.split_name,
                        task_ids=list(prepared.task_ids),
                        agent=agent_name,
                        llm_agent=project_config.model.base_model,
                        llm_args_agent={},
                        user="user_simulator",
                        llm_user=project_config.tau2.user_llm,
                        llm_args_user=dict(project_config.tau2.user_llm_args),
                        num_trials=1,
                        max_steps=max(4, project_config.rollout.max_episode_steps * 4),
                        max_errors=10,
                        save_to=str((Path(working) / "tau2").resolve()),
                        max_concurrency=min(
                            project_config.execution.max_concurrency,
                            len(prepared.task_ids),
                        ),
                        seed=project_config.execution.seed,
                        log_level="WARNING",
                        max_retries=2,
                        retry_delay=1.0,
                        auto_resume=False,
                        hallucination_retries=0,
                        enforce_communication_protocol=True,
                    )
                )
        finally:
            prepared.registry._agent_factories.pop(agent_name, None)

    simulations = {str(item.task_id): item for item in tau2_results.simulations}
    missing = set(prepared.task_ids) - set(simulations)
    if missing:
        raise RuntimeError(f"Tau2 run_domain omitted task results: {sorted(missing)}")

    pending: list[tuple[PendingEpisode, BufferedEventWriter]] = []
    failures: list[BatchFailure] = []
    failure_seeds: dict[str, int] = {}
    for task_id in prepared.task_ids:
        simulation = simulations[task_id]
        seed = int(
            simulation.seed
            if getattr(simulation, "seed", None) is not None
            else project_config.execution.seed
        )
        if _is_infrastructure_failure(simulation):
            failures.append(
                BatchFailure(
                    task_id=task_id,
                    stage="run_domain",
                    error_type=_simulation_error_type(simulation),
                )
            )
            failure_seeds[task_id] = seed
            continue
        buffer = BufferedEventWriter()
        context = _context(
            prepared=prepared,
            request=request,
            source_namespace=source_namespace,
            input_memory_snapshot_id=input_memory_snapshot_id,
            cross_domain=cross_domain,
            memory_generation=memory_generation,
            seed=seed,
            writer=buffer,
            project_config=project_config,
        )
        try:
            episode = finalize_simulation(
                runtime=_runtime_view(prepared),
                simulation=simulation,
                policy=policy,
                config=fast_loop_config,
                context=context,
                propose_experience=request.capabilities.can_write_memory,
            )
        except Exception as error:
            failures.append(
                BatchFailure(
                    task_id=task_id,
                    stage="finalize",
                    error_type=type(error).__name__,
                )
            )
            failure_seeds[task_id] = seed
            continue
        pending.append((episode, buffer))

    output_snapshot_id: str | None = None
    if failures and destination_repository is not None:
        for episode, buffer in pending:
            buffer.append(
                _context(
                    prepared=prepared,
                    request=request,
                    source_namespace=source_namespace,
                    input_memory_snapshot_id=input_memory_snapshot_id,
                    cross_domain=cross_domain,
                    memory_generation=memory_generation,
                    seed=project_config.execution.seed,
                    writer=buffer,
                    project_config=project_config,
                ).event(
                    "MemoryBatchDiscarded",
                    episode.result.task_id,
                    reason="batch_failed",
                    proposal_count=len(episode.proposals),
                )
            )
    elif destination_repository is not None:
        commits = commit_pending_experience(
            destination_repository,
            [episode for episode, _ in pending],
        )
        for index, (episode, buffer) in enumerate(pending):
            commit = commits[index]
            buffer.append(
                _context(
                    prepared=prepared,
                    request=request,
                    source_namespace=source_namespace,
                    input_memory_snapshot_id=input_memory_snapshot_id,
                    cross_domain=cross_domain,
                    memory_generation=memory_generation,
                    seed=project_config.execution.seed,
                    writer=buffer,
                    project_config=project_config,
                ).event(
                    "MemoryWriteCommitted",
                    episode.result.task_id,
                    written_memory_ids=list(commit.written_ids),
                    replayed_memory_ids=list(commit.replayed_ids),
                )
            )
            pending[index] = (
                replace(
                    episode,
                    result=replace(
                        episode.result, written_memory_ids=commit.written_ids
                    ),
                ),
                buffer,
            )
        snapshot = destination_repository.snapshot()
        output_snapshot_id = snapshot.memory_snapshot_id
        commit_batch_state(
            destination_repository.root,
            expected_generation=memory_generation,
            completed_tasks=len(pending),
            snapshot_id=output_snapshot_id,
        )

    if episode_writer is not None:
        completed = {
            episode.result.task_id: (episode, buffer) for episode, buffer in pending
        }
        failed = {failure.task_id: failure for failure in failures}
        for task_id in prepared.task_ids:
            if task_id in completed:
                episode, buffer = completed[task_id]
                episode_writer.append(
                    build_completed_episode(episode.result, buffer.events)
                )
            else:
                episode_writer.append(
                    build_failed_episode(
                        failed[task_id],
                        task_group=prepared.task_group,
                        seed=failure_seeds[task_id],
                    )
                )
    return BatchResult(
        episodes=tuple(episode.result for episode, _ in pending),
        failures=tuple(failures),
        input_memory_snapshot_id=input_memory_snapshot_id,
        output_memory_snapshot_id=output_snapshot_id,
    )


def commit_pending_experience(
    repository: MemoryRepository,
    episodes: Sequence[PendingEpisode],
) -> tuple[EpisodeCommit, ...]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        for proposal in episode.proposals:
            groups.setdefault(proposal["memory_id"], []).append(proposal)

    existed_before: set[str] = set()
    materialized: dict[str, Mapping[str, Any]] = {}
    for memory_id, group in groups.items():
        first = group[0]
        if any(not _same_memory(first, candidate) for candidate in group[1:]):
            raise ValueError(f"conflicting Memory proposals in one batch: {memory_id}")
        existing = repository.get(memory_id)
        if existing is not None:
            if not _same_existing_memory(existing, first):
                raise ValueError(f"Memory proposal conflicts with repository: {memory_id}")
            existed_before.add(memory_id)
        materialized[memory_id] = _coalesce_provenance(group)

    for memory_id, proposal in materialized.items():
        if memory_id not in existed_before:
            repository.add(**proposal["add_kwargs"])

    first_episode_for_id: dict[str, int] = {}
    commits: list[EpisodeCommit] = []
    for index, episode in enumerate(episodes):
        written_ids = tuple(proposal["memory_id"] for proposal in episode.proposals)
        replayed_ids: list[str] = []
        for memory_id in written_ids:
            owner = first_episode_for_id.setdefault(memory_id, index)
            if memory_id in existed_before or owner != index:
                replayed_ids.append(memory_id)
        commits.append(EpisodeCommit(written_ids, tuple(replayed_ids)))
    return tuple(commits)


def _same_memory(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_kwargs = left["add_kwargs"]
    right_kwargs = right["add_kwargs"]
    fields = ("tier", "tier_schema_version", "payload", "content", "retrieval_text")
    return left["memory_id"] == right["memory_id"] and all(
        left_kwargs.get(field) == right_kwargs.get(field) for field in fields
    )


def _same_existing_memory(existing: Any, proposal: Mapping[str, Any]) -> bool:
    kwargs = proposal["add_kwargs"]
    return (
        existing.id == proposal["memory_id"]
        and existing.tier == kwargs["tier"]
        and existing.tier_schema_version == kwargs.get("tier_schema_version", 1)
        and existing.payload == kwargs.get("payload", {})
        and existing.content == kwargs["content"]
    )


def _coalesce_provenance(
    proposals: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if len(proposals) == 1:
        return proposals[0]
    first = proposals[0]
    kwargs = dict(first["add_kwargs"])
    source_task_ids = tuple(
        dict.fromkeys(
            task_id
            for proposal in proposals
            for task_id in proposal["add_kwargs"].get("source_task_ids", ())
        )
    )
    kwargs["source_task_ids"] = source_task_ids
    kwargs["metadata"] = {
        "batch_contributions": [
            {
                "source_task_ids": list(
                    proposal["add_kwargs"].get("source_task_ids", ())
                ),
                "metadata": proposal["add_kwargs"].get("metadata", {}),
            }
            for proposal in proposals
        ]
    }
    return {**first, "add_kwargs": kwargs}


def _context(
    *,
    prepared: PreparedBenchmark,
    request: ExecutionRequest,
    source_namespace: str | None,
    input_memory_snapshot_id: str | None,
    cross_domain: bool,
    memory_generation: int,
    seed: int,
    writer: EventWriter,
    project_config: ProjectConfig,
) -> ExecutionContext:
    checkpoint = str(request.checkpoint) if request.checkpoint is not None else None
    return ExecutionContext(
        run_id=request.run_id,
        benchmark=prepared.name,
        mode=request.mode.value,
        split=prepared.split_name,
        model_revision=project_config.model.served_model_name,
        checkpoint=checkpoint,
        memory_source_namespace=source_namespace,
        memory_snapshot_id=input_memory_snapshot_id,
        cross_domain_memory=cross_domain,
        memory_generation=memory_generation,
        seed=seed,
        event_writer=writer,
        default_task_group=prepared.task_group,
    )


def _runtime_view(prepared: PreparedBenchmark) -> Any:
    runtime = prepared.runtime
    if runtime is None:
        raise RuntimeError("PreparedBenchmark has no bound Tau2 runtime")
    return runtime


def _is_infrastructure_failure(simulation: Any) -> bool:
    reason = getattr(getattr(simulation, "termination_reason", None), "value", None)
    reason = str(reason or getattr(simulation, "termination_reason", ""))
    return reason == "infrastructure_error" or simulation.reward_info is None


def _simulation_error_type(simulation: Any) -> str:
    info = getattr(simulation, "info", None)
    if isinstance(info, Mapping) and isinstance(info.get("error_type"), str):
        return info["error_type"]
    return "InfrastructureError"
