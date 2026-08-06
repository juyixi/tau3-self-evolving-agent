from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from tau3_evolver.fast_loop.contracts import FastLoopPolicy, PendingEpisode
from tau3_evolver.fast_loop.settings import FastLoopConfig
from tau3_evolver.artifacts.maintenance import build_completed_maintenance
from tau3_evolver.artifacts.contracts import (
    CompletedEpisodeProjection,
    FailedEpisodeProjection,
)
from tau3_evolver.artifacts.episodes import (
    build_completed_episode,
    build_failed_episode,
)
from tau3_evolver.benchmarks.executor import (
    BenchmarkAgentSpec,
    BenchmarkExecutionRequest,
)
from tau3_evolver.benchmarks.types import PreparedBenchmark
from tau3_evolver.config import ProjectConfig
from tau3_evolver.execution.events import BufferedEventWriter, EventWriter, ExecutionContext
from tau3_evolver.execution.request import ExecutionRequest
from tau3_evolver.execution.results import (
    BatchFailure,
    BatchResult,
    MaintenanceBatchResult,
    MaintenanceFailure,
)
from tau3_evolver.execution.memory_state import commit_memory_state, load_memory_state
from tau3_evolver.fast_loop.maintenance import (
    due_maintenance_rounds,
    run_due_maintenance,
)
from tau3_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_evolver.memory.repository import MemoryRepository
from tau3_evolver.memory.retrieval import Retriever


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
    execution = prepared.executor.execute(
        BenchmarkExecutionRequest(
            task_ids=prepared.task_ids,
            project_config=project_config,
            agent=BenchmarkAgentSpec(
                policy=policy,
                repository=repository,
                retriever=retriever,
                config=fast_loop_config,
                memory_source_namespace=source_namespace,
                cross_domain_memory=cross_domain,
                propose_experience=request.capabilities.can_write_memory,
            ),
            context_factory=lambda seed, writer: _context(
                prepared=prepared,
                request=request,
                source_namespace=source_namespace,
                input_memory_snapshot_id=input_memory_snapshot_id,
                cross_domain=cross_domain,
                memory_generation=memory_generation,
                seed=seed,
                writer=writer,
                project_config=project_config,
            ),
        )
    )
    pending: list[tuple[PendingEpisode, BufferedEventWriter]] = [
        (
            item.episode,
            BufferedEventWriter(events=list(item.events)),
        )
        for item in execution.episodes
    ]
    failures = [
        BatchFailure(
            task_id=failure.task_id,
            stage=failure.stage,
            error_type=failure.error_type,
        )
        for failure in execution.failures
    ]
    failure_seeds = {
        failure.task_id: failure.seed for failure in execution.failures
    }

    output_snapshot_id: str | None = None
    maintenance_batch: MaintenanceBatchResult | None = None
    if failures and destination_repository is not None:
        batch_state = load_memory_state(destination_repository.root)
        maintenance_batch = MaintenanceBatchResult(
            period=project_config.memory.maintenance_period,
            completed_train_tasks_before=batch_state.completed_tasks,
            completed_train_tasks_after=batch_state.completed_tasks,
        )
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
        batch_state = load_memory_state(destination_repository.root)
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
        completed_train_tasks = batch_state.completed_tasks + len(pending)
        maintenance_records: list[Mapping[str, Any]] = []
        maintenance_failures: list[MaintenanceFailure] = []
        if request.capabilities.can_run_maintenance:
            due_rounds = due_maintenance_rounds(
                completed_train_tasks=completed_train_tasks,
                period=project_config.memory.maintenance_period,
                repository=destination_repository,
            )
            for maintenance_round in due_rounds:
                maintenance_buffer = BufferedEventWriter()
                maintenance_snapshot = destination_repository.snapshot()
                maintenance_context = _context(
                    prepared=prepared,
                    request=request,
                    source_namespace=source_namespace,
                    input_memory_snapshot_id=maintenance_snapshot.memory_snapshot_id,
                    cross_domain=cross_domain,
                    memory_generation=memory_generation,
                    seed=project_config.execution.seed,
                    writer=maintenance_buffer,
                    project_config=project_config,
                )
                try:
                    maintenance = run_due_maintenance(
                        completed_train_tasks=completed_train_tasks,
                        period=project_config.memory.maintenance_period,
                        repository=destination_repository,
                        policy=policy,
                        context=maintenance_context,
                        tip_capacity=fast_loop_config.maintenance_tip_capacity,
                        similarity_threshold=(
                            fast_loop_config.maintenance_similarity_threshold
                        ),
                        priority_pair_limit=(
                            fast_loop_config.maintenance_priority_pair_limit
                        ),
                        maintenance_round=maintenance_round,
                    )
                    if maintenance.executed:
                        maintenance_records.append(
                            build_completed_maintenance(maintenance_buffer.events)
                        )
                except Exception as error:
                    maintenance_failures.append(
                        MaintenanceFailure(
                            maintenance_round=maintenance_round,
                            trigger_task_index=completed_train_tasks,
                            error_type=type(error).__name__,
                        )
                    )
                    break

        snapshot = destination_repository.snapshot()
        output_snapshot_id = snapshot.memory_snapshot_id
        commit_memory_state(
            destination_repository.root,
            expected_generation=memory_generation,
            completed_tasks=len(pending),
            snapshot_id=output_snapshot_id,
        )
        maintenance_batch = MaintenanceBatchResult(
            period=project_config.memory.maintenance_period,
            completed_train_tasks_before=batch_state.completed_tasks,
            completed_train_tasks_after=completed_train_tasks,
            records=tuple(maintenance_records),
            failures=tuple(maintenance_failures),
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
                    build_completed_episode(
                        CompletedEpisodeProjection(
                            task_id=episode.result.task_id,
                            final_reward=episode.result.final_reward,
                            steps=episode.result.steps,
                            terminal_evaluation=episode.result.terminal_evaluation,
                            truncated=episode.result.truncated,
                            project_truncated=episode.result.project_truncated,
                            parse_error_count=episode.result.parse_error_count,
                            response_parse_error_count=(
                                episode.result.response_parse_error_count
                            ),
                            response_count=episode.result.response_count,
                            agent_prompt_tokens=episode.result.agent_prompt_tokens,
                            agent_completion_tokens=(
                                episode.result.agent_completion_tokens
                            ),
                        ),
                        buffer.events,
                    )
                )
            else:
                failure = failed[task_id]
                episode_writer.append(
                    build_failed_episode(
                        FailedEpisodeProjection(
                            task_id=failure.task_id,
                            stage=failure.stage,
                            error_type=failure.error_type,
                        ),
                        task_group=prepared.task_group,
                        seed=failure_seeds[task_id],
                    )
                )
    return BatchResult(
        episodes=tuple(episode.result for episode, _ in pending),
        failures=tuple(failures),
        input_memory_snapshot_id=input_memory_snapshot_id,
        output_memory_snapshot_id=output_snapshot_id,
        maintenance=maintenance_batch,
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
