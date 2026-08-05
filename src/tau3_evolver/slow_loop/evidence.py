from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau3_evolver.agent.decisions import MaintenanceDecision
from tau3_evolver.agent.prompts import MAX_DIAGNOSTIC_CONTENT_CHARS
from tau3_evolver.artifacts.maintenance import maintenance_record_sha256
from tau3_evolver.memory.operations import DeleteCommand, LookupCommand, MergeCommand
from tau3_evolver.memory.paths import training_memory_root
from tau3_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_evolver.memory.tier_contracts import (
    TIER_SCHEMA_VERSION,
    ToolPayload,
    TrajectoryPayload,
    render_tier_payload,
    validate_stored_tier_payload,
    validate_tool_payload_against_tools,
)
from tau3_evolver.memory.types import (
    MemoryStatus,
    MemoryTier,
    canonical_content,
    stable_memory_id,
)
from tau3_evolver.slow_loop.source_runs import SourceRun, SourceRunSet
from tau3_evolver.slow_loop.task_grouping import canonicalize_task_group


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MemoryCandidateEvidence(_EvidenceModel):
    memory_id: str
    memory_version: int = Field(ge=1)
    tier: MemoryTier
    rank: int = Field(ge=1)
    similarity: float
    content: str
    content_sha256: str


class TrajectoryStepEvidence(_EvidenceModel):
    turn: int = Field(ge=0)
    observation: Any
    action: str
    next_observation: Any
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    public_info: dict[str, Any]


class WriteProposalEvidence(_EvidenceModel):
    memory_id: str
    generation_mode: Literal["llm", "rule"] = "llm"
    tier: MemoryTier
    tier_schema_version: Literal[2] = 2
    payload: dict[str, Any]
    content: str
    retrieval_text: str
    metadata: dict[str, Any]
    source_task_ids: tuple[str, ...]
    created_round: int = Field(ge=0)

    @model_validator(mode="after")
    def tier_payload_must_match_content(self) -> WriteProposalEvidence:
        stored = validate_stored_tier_payload(self.tier, self.payload)
        if canonical_content(render_tier_payload(self.tier, stored)) != canonical_content(
            self.content
        ):
            raise ValueError("write proposal content does not match its tier payload")
        return self


class PublicMemoryEvidence(_EvidenceModel):
    id: str
    tier: MemoryTier
    content: str
    version: int = Field(ge=1)
    status: MemoryStatus


class MaintenanceMemoryEvidence(_EvidenceModel):
    id: str
    tier: MemoryTier
    content: str
    version: int = Field(ge=1)
    status: MemoryStatus
    usage_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    last_used: str | None
    embedding: tuple[float, ...] | None
    embedding_model_revision: str | None


class EpisodeEvidence(_EvidenceModel):
    evidence_schema_version: Literal[1] = 1
    episode_id: str
    run_id: str
    source_episode_row: int = Field(ge=1)
    source_episode_sha256: str
    memory_generation: int = Field(ge=0)
    task_id: str
    task_group: str
    model_revision: str
    adapter_revision: str | None
    runtime_revision: str
    split_hash: str
    memory_namespace: str
    memory_snapshot_id: str
    seed: int = Field(ge=0)
    policy: Any
    tools: tuple[dict[str, Any], ...]
    initial_observation: Any
    query_hash: str
    retriever_revision: str
    candidates: tuple[MemoryCandidateEvidence, ...]
    selected_memory_ids: tuple[str, ...]
    trajectory: tuple[TrajectoryStepEvidence, ...]
    terminal_evaluation: dict[str, Any]
    final_reward: float
    terminated: bool
    truncated: bool
    write_proposals: tuple[WriteProposalEvidence, ...]
    proposed_memory_ids: tuple[str, ...]
    committed_new_memory_ids: tuple[str, ...]
    replayed_memory_ids: tuple[str, ...]


class MaintenanceEvidence(_EvidenceModel):
    evidence_schema_version: Literal[1] = 1
    maintenance_id: str
    run_id: str
    source_record_index: int = Field(ge=1)
    source_record_sha256: str
    memory_generation: int = Field(ge=0)
    maintenance_round: int = Field(ge=1)
    trigger_task_index: int = Field(ge=1)
    period: int = Field(ge=1)
    memory_snapshot_id: str
    prior_episode_ids: tuple[str, ...]
    public_repository: tuple[PublicMemoryEvidence, ...]
    repository_state: tuple[MaintenanceMemoryEvidence, ...]
    commands: tuple[dict[str, Any], ...]
    looked_up_ids: tuple[str, ...]
    created_ids: tuple[str, ...]
    updated_ids: tuple[str, ...]


class EvidenceLedger(_EvidenceModel):
    evidence_schema_version: Literal[1] = 1
    memory_generation: int = Field(ge=0)
    model_revision: str
    adapter_revision: str | None
    runtime_revision: str
    split_hash: str
    memory_namespace: str
    source_run_ids: tuple[str, ...]
    episodes: tuple[EpisodeEvidence, ...]
    maintenance: tuple[MaintenanceEvidence, ...]


def build_evidence(
    source_runs: SourceRunSet,
    *,
    memory_root: Path,
) -> EvidenceLedger:
    if not isinstance(source_runs, SourceRunSet):
        raise TypeError("source_runs must be a SourceRunSet")
    memory_root = memory_root.resolve()
    if memory_root.name != "memory" or memory_root.parent.name != source_runs.memory_namespace:
        raise ValueError("evidence memory namespace does not match source runs")

    snapshots: dict[tuple[Path, str], ReadOnlyMemoryRepository] = {}
    episodes: list[EpisodeEvidence] = []
    maintenance: list[MaintenanceEvidence] = []
    for source_run in source_runs.runs:
        local_task_ids: list[str] = []
        for row_number, row in enumerate(source_run.episodes, start=1):
            episode = _build_episode_record(
                source_run,
                row_number,
                row,
                memory_root=memory_root,
                project_root=source_runs.project_root,
                snapshots=snapshots,
            )
            episodes.append(episode)
            local_task_ids.append(episode.task_id)
        if len(local_task_ids) != source_run.summary["episode_count"]:
            raise ValueError(f"evidence episode count mismatch for run {source_run.run_id}")
        if tuple(local_task_ids) != source_run.task_ids:
            raise ValueError(f"evidence task order mismatch for run {source_run.run_id}")
        local_maintenance_rounds: list[int] = []
        for record_index, record in enumerate(
            source_run.maintenance_records,
            start=1,
        ):
            item = _build_maintenance_record(
                source_run,
                record_index,
                record,
                prior_episode_ids=tuple(item.episode_id for item in episodes),
                memory_root=memory_root,
                snapshots=snapshots,
            )
            maintenance.append(item)
            local_maintenance_rounds.append(item.maintenance_round)
        if tuple(local_maintenance_rounds) != tuple(
            source_run.summary["maintenance_rounds_executed"]
        ):
            raise ValueError(
                f"evidence maintenance rounds mismatch for run {source_run.run_id}"
            )

    return EvidenceLedger(
        memory_generation=source_runs.memory_generation,
        model_revision=source_runs.model_revision,
        adapter_revision=source_runs.adapter_revision,
        runtime_revision=source_runs.runtime_revision,
        split_hash=source_runs.split_hash,
        memory_namespace=source_runs.memory_namespace,
        source_run_ids=tuple(run.run_id for run in source_runs.runs),
        episodes=tuple(episodes),
        maintenance=tuple(maintenance),
    )


def _build_maintenance_record(
    source_run: SourceRun,
    record_index: int,
    record: Mapping[str, Any],
    *,
    prior_episode_ids: tuple[str, ...],
    memory_root: Path,
    snapshots: dict[tuple[Path, str], ReadOnlyMemoryRepository],
) -> MaintenanceEvidence:
    record_hash = _nonblank(record, "record_sha256", "maintenance record")
    if not _SHA256.fullmatch(record_hash) or record_hash != maintenance_record_sha256(
        record
    ):
        raise ValueError("maintenance record hash mismatch")
    maintenance_round = record.get("maintenance_round")
    trigger_task_index = record.get("trigger_task_index")
    period = record.get("period")
    expected_trigger = source_run.summary["completed_train_tasks_after"]
    if type(maintenance_round) is not int or maintenance_round <= 0:
        raise ValueError("maintenance round must be positive")
    if trigger_task_index != expected_trigger:
        raise ValueError("maintenance trigger task index mismatch")
    if type(period) is not int or period <= 0:
        raise ValueError("maintenance period is invalid")
    if maintenance_round > trigger_task_index // period:
        raise ValueError("maintenance round does not match trigger task index")

    snapshot_id = _nonblank(record, "memory_snapshot_id", "maintenance record")
    snapshot = _snapshot(memory_root, snapshot_id, snapshots)
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "trajectory",
        "tip",
        "skill",
        "tool",
    }:
        raise ValueError("maintenance diagnostics must contain exactly four tiers")
    public_repository: list[PublicMemoryEvidence] = []
    repository_state: list[MaintenanceMemoryEvidence] = []
    for tier in MemoryTier:
        tier_payload = diagnostics[tier.value]
        if not isinstance(tier_payload, Mapping):
            raise ValueError("maintenance tier diagnostics must be an object")
        items = _list_of_mappings(tier_payload.get("items"), "maintenance items")
        for raw in items:
            public = PublicMemoryEvidence.model_validate(raw)
            if public.tier != tier:
                raise ValueError(f"maintenance public tier mismatch: {public.id}")
            memory_item = snapshot.get(public.id)
            if memory_item is None:
                raise ValueError(
                    f"maintenance memory missing from snapshot: {public.id}"
                )
            content_matches = public.content == memory_item.content or (
                0 < len(public.content) <= MAX_DIAGNOSTIC_CONTENT_CHARS
                and memory_item.content.startswith(public.content)
            )
            if (
                memory_item.tier != public.tier
                or not content_matches
                or memory_item.version != public.version
                or memory_item.status != public.status
            ):
                raise ValueError(f"maintenance public memory mismatch: {public.id}")
            public_repository.append(public)
            repository_state.append(
                MaintenanceMemoryEvidence(
                    id=memory_item.id,
                    tier=memory_item.tier,
                    content=memory_item.content,
                    version=memory_item.version,
                    status=memory_item.status,
                    usage_count=memory_item.usage_count,
                    success_count=memory_item.success_count,
                    last_used=memory_item.last_used,
                    embedding=memory_item.embedding,
                    embedding_model_revision=memory_item.embedding_model_revision,
                )
            )

    commands = _list_of_mappings(record.get("commands"), "maintenance commands")
    try:
        decision = MaintenanceDecision.model_validate({"commands": commands})
    except ValueError as error:
        raise ValueError("maintenance commands are invalid") from error
    expected_looked_up, expected_created, expected_updated = _maintenance_result_ids(
        snapshot,
        decision,
        maintenance_round=maintenance_round,
    )
    looked_up_ids = _string_tuple(record.get("looked_up_ids"), "looked_up_ids")
    created_ids = _string_tuple(record.get("created_ids"), "created_ids")
    updated_ids = _string_tuple(record.get("updated_ids"), "updated_ids")
    if (
        looked_up_ids != expected_looked_up
        or created_ids != expected_created
        or updated_ids != expected_updated
    ):
        raise ValueError("maintenance commit result does not match proposed commands")

    return MaintenanceEvidence(
        maintenance_id=(
            f"{source_run.run_id}:maintenance-round-{maintenance_round}"
        ),
        run_id=source_run.run_id,
        source_record_index=record_index,
        source_record_sha256=record_hash,
        memory_generation=source_run.run["memory"]["generation"],
        maintenance_round=maintenance_round,
        trigger_task_index=trigger_task_index,
        period=period,
        memory_snapshot_id=snapshot_id,
        prior_episode_ids=prior_episode_ids,
        public_repository=tuple(public_repository),
        repository_state=tuple(repository_state),
        commands=tuple(
            command.model_dump(mode="json") for command in decision.commands
        ),
        looked_up_ids=looked_up_ids,
        created_ids=created_ids,
        updated_ids=updated_ids,
    )


def _build_episode_record(
    source_run: SourceRun,
    row_number: int,
    row: Mapping[str, Any],
    *,
    memory_root: Path,
    project_root: Path,
    snapshots: dict[tuple[Path, str], ReadOnlyMemoryRepository],
) -> EpisodeEvidence:
    task_id = _nonblank(row, "task_id", "source episode")
    task = _json_mapping(row.get("task"), "task")
    memory = _json_mapping(row.get("memory"), "memory")
    retrieval = _json_mapping(memory.get("retrieval"), "retrieval")
    outcome = _json_mapping(row.get("outcome"), "outcome")
    source_memory = source_run.run["memory"]
    snapshot_id = _nonblank(source_memory, "input_snapshot_id", "source run Memory")
    input_memory_root = training_memory_root(
        source_memory["source_namespace"], root=project_root
    )
    input_snapshot = _snapshot(input_memory_root, snapshot_id, snapshots)

    raw_candidates = _list_of_mappings(retrieval.get("candidates"), "candidates")
    candidates = tuple(
        _candidate_evidence(raw, expected_rank=index, snapshot=input_snapshot)
        for index, raw in enumerate(raw_candidates, start=1)
    )
    candidate_ids = tuple(candidate.memory_id for candidate in candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"duplicate candidate memory ID for task {task_id}")
    selected_ids = _string_tuple(
        memory.get("selected_memory_ids"), "selected_memory_ids"
    )
    if not set(selected_ids) <= set(candidate_ids):
        raise ValueError(f"selected memory is not a candidate for task {task_id}")

    raw_trajectory = _list_of_mappings(row.get("trajectory"), "trajectory")
    trajectory = tuple(
        _trajectory_step(raw, expected_turn=index, task_id=task_id)
        for index, raw in enumerate(raw_trajectory)
    )
    if not trajectory:
        raise ValueError(f"trajectory is empty for task {task_id}")
    if trajectory[0].observation != task.get("initial_observation"):
        raise ValueError(f"trajectory initial observation mismatch for task {task_id}")
    if any(
        previous.next_observation != current.observation
        for previous, current in zip(trajectory, trajectory[1:], strict=False)
    ):
        raise ValueError(f"trajectory observation chain mismatch for task {task_id}")
    if outcome.get("steps") != len(trajectory):
        raise ValueError(f"episode step count mismatch for task {task_id}")
    final_reward = _finite_number(outcome.get("final_reward"), "final_reward")
    if not math.isclose(
        final_reward,
        trajectory[-1].reward,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"final reward does not match trajectory for task {task_id}")

    tools = _list_of_mappings(task.get("tools"), "tools")
    raw_writes = _list_of_mappings(memory.get("writes"), "Memory writes")
    write_proposals = tuple(
        _write_proposal(
            raw,
            task_id=task_id,
            run_id=source_run.run_id,
            task_group=str(row["task_group"]),
            final_reward=final_reward,
            trajectory=trajectory,
            tools=tools,
        )
        for raw in raw_writes
    )
    proposal_ids = tuple(item.memory_id for item in write_proposals)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError(f"duplicate write proposal for task {task_id}")
    dispositions = {
        str(raw["memory_id"]): raw.get("disposition") for raw in raw_writes
    }
    invalid = set(dispositions.values()) - {"created", "replayed"}
    if invalid:
        raise ValueError(
            f"uncommitted Memory proposal for task {task_id}: {sorted(invalid)}"
        )
    committed_new_ids = tuple(
        memory_id for memory_id in proposal_ids if dispositions[memory_id] == "created"
    )
    replayed_ids = tuple(
        memory_id for memory_id in proposal_ids if dispositions[memory_id] == "replayed"
    )

    output_snapshot = _snapshot(
        memory_root,
        source_memory["output_snapshot_id"],
        snapshots,
    )
    proposal_by_id = {item.memory_id: item for item in write_proposals}
    for memory_id in replayed_ids:
        if input_snapshot.get(memory_id) is None:
            raise ValueError(f"replayed memory missing from input snapshot: {memory_id}")
    for memory_id in committed_new_ids:
        if input_snapshot.get(memory_id) is not None:
            raise ValueError(f"new memory already exists in input snapshot: {memory_id}")
    for memory_id in proposal_ids:
        item = output_snapshot.get(memory_id)
        proposal = proposal_by_id[memory_id]
        if item is None:
            raise ValueError(f"committed memory missing from output snapshot: {memory_id}")
        if (
            item.tier != proposal.tier
            or item.tier_schema_version != proposal.tier_schema_version
            or item.payload != proposal.payload
            or item.content != proposal.content
        ):
            raise ValueError(f"committed memory content mismatch: {memory_id}")

    return EpisodeEvidence(
        episode_id=f"{source_run.run_id}:{task_id}",
        run_id=source_run.run_id,
        source_episode_row=row_number,
        source_episode_sha256=_row_hash(row),
        memory_generation=source_memory["generation"],
        task_id=task_id,
        task_group=canonicalize_task_group(str(row["task_group"])),
        model_revision=source_run.run["policy"]["model_revision"],
        adapter_revision=source_run.adapter_revision,
        runtime_revision=source_run.runtime_revision,
        split_hash=source_run.run["execution"]["split_hash"],
        memory_namespace=source_run.run["memory"].get(
            "destination_namespace", source_run.run["execution"]["benchmark"]
        ),
        memory_snapshot_id=snapshot_id,
        seed=row["seed"],
        policy=_json_copy(task.get("policy"), "policy"),
        tools=tuple(_json_mapping(tool, "tool") for tool in tools),
        initial_observation=_json_copy(
            task.get("initial_observation"), "initial_observation"
        ),
        query_hash=_sha256_value(retrieval, "query_hash", "retrieval"),
        retriever_revision=_nonblank(retrieval, "retriever_revision", "retrieval"),
        candidates=candidates,
        selected_memory_ids=selected_ids,
        trajectory=trajectory,
        # Evaluator assertions and rubrics remain in the immutable Source Run for
        # diagnosis, but never cross into a trainable Slow Loop artifact.
        terminal_evaluation={},
        final_reward=final_reward,
        terminated=trajectory[-1].terminated,
        truncated=_strict_bool(outcome.get("truncated"), "truncated"),
        write_proposals=write_proposals,
        proposed_memory_ids=proposal_ids,
        committed_new_memory_ids=committed_new_ids,
        replayed_memory_ids=replayed_ids,
    )


def _trajectory_step(
    raw: Mapping[str, Any],
    *,
    expected_turn: int,
    task_id: str,
) -> TrajectoryStepEvidence:
    if raw.get("turn") != expected_turn:
        raise ValueError(f"trajectory turn is invalid for task {task_id}")
    return TrajectoryStepEvidence(
        turn=expected_turn,
        observation=_json_copy(raw.get("observation"), "observation"),
        action=_nonblank(raw, "action", "trajectory step"),
        next_observation=_json_copy(raw.get("next_observation"), "next_observation"),
        reward=_finite_number(raw.get("reward"), "step reward"),
        done=_strict_bool(raw.get("done"), "done"),
        terminated=_strict_bool(raw.get("terminated"), "terminated"),
        truncated=_strict_bool(raw.get("truncated"), "truncated"),
        public_info=_json_mapping(raw.get("public_info"), "public_info"),
    )


def _candidate_evidence(
    raw: Mapping[str, Any],
    *,
    expected_rank: int,
    snapshot: ReadOnlyMemoryRepository,
) -> MemoryCandidateEvidence:
    memory_id = _nonblank(raw, "memory_id", "candidate")
    item = snapshot.get(memory_id)
    if item is None:
        raise ValueError(f"candidate missing from snapshot: {memory_id}")
    version = raw.get("memory_version")
    if type(version) is not int or version != item.version:
        raise ValueError(f"candidate version mismatch: {memory_id}")
    try:
        tier = MemoryTier(raw.get("tier"))
    except ValueError as error:
        raise ValueError(f"candidate tier is invalid: {memory_id}") from error
    if tier != item.tier:
        raise ValueError(f"candidate tier mismatch: {memory_id}")
    if raw.get("rank") != expected_rank:
        raise ValueError(f"candidate rank mismatch: {memory_id}")
    similarity = _finite_number(raw.get("similarity"), "candidate similarity")
    return MemoryCandidateEvidence(
        memory_id=memory_id,
        memory_version=version,
        tier=tier,
        rank=expected_rank,
        similarity=similarity,
        content=item.content,
        content_sha256=hashlib.sha256(item.content.encode("utf-8")).hexdigest(),
    )


def _write_proposal(
    raw: Mapping[str, Any],
    *,
    task_id: str,
    run_id: str,
    task_group: str,
    final_reward: float,
    trajectory: Sequence[TrajectoryStepEvidence],
    tools: Sequence[Mapping[str, Any]],
) -> WriteProposalEvidence:
    memory_id = _nonblank(raw, "memory_id", "write proposal")
    try:
        tier = MemoryTier(raw.get("tier"))
    except ValueError as error:
        raise ValueError(f"write proposal tier is invalid: {memory_id}") from error
    if raw.get("tier_schema_version") != TIER_SCHEMA_VERSION:
        raise ValueError(f"write proposal tier schema is invalid: {memory_id}")
    raw_payload = raw.get("payload")
    if not isinstance(raw_payload, Mapping):
        raise ValueError(f"write proposal payload is invalid: {memory_id}")
    stored_payload = validate_stored_tier_payload(tier, raw_payload)
    if isinstance(stored_payload, ToolPayload):
        validate_tool_payload_against_tools(stored_payload, tools)
    if isinstance(stored_payload, TrajectoryPayload):
        expected_episode_id = f"{run_id}:{task_id}"
        if stored_payload.source_episode_id != expected_episode_id:
            raise ValueError(f"trajectory proposal episode mismatch: {memory_id}")
        if stored_payload.task_group != task_group:
            raise ValueError(f"trajectory proposal task group mismatch: {memory_id}")
        if not math.isclose(
            stored_payload.final_reward,
            final_reward,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"trajectory proposal reward mismatch: {memory_id}")
        expected_steps = tuple(
            (index, step.action, step.reward, step.done)
            for index, step in enumerate(trajectory, start=1)
        )
        actual_steps = tuple(
            (step.order, step.action, step.reward, step.done)
            for step in stored_payload.steps
        )
        if actual_steps != expected_steps:
            raise ValueError(f"trajectory proposal steps mismatch: {memory_id}")
        if raw.get("generation_mode") == "rule":
            expected_runtime_steps = tuple(
                (
                    _runtime_step_text(step.observation),
                    _runtime_step_text(step.next_observation),
                    step.terminated,
                    step.truncated,
                )
                for step in trajectory
            )
            actual_runtime_steps = tuple(
                (
                    step.observation,
                    step.result,
                    step.terminated,
                    step.truncated,
                )
                for step in stored_payload.steps
            )
            if actual_runtime_steps != expected_runtime_steps:
                raise ValueError(
                    f"runtime trajectory proposal evidence mismatch: {memory_id}"
                )
    content = _nonblank(raw, "content", "write proposal")
    if canonical_content(content) != canonical_content(
        render_tier_payload(tier, stored_payload)
    ):
        raise ValueError(f"write proposal content/payload mismatch: {memory_id}")
    if memory_id != stable_memory_id(tier, content):
        raise ValueError(f"write proposal stable ID mismatch: {memory_id}")
    source_task_ids = _string_tuple(raw.get("source_task_ids"), "source_task_ids")
    if source_task_ids != (task_id,):
        raise ValueError(f"write proposal source task mismatch: {memory_id}")
    created_round = raw.get("memory_generation")
    if type(created_round) is not int or created_round < 0:
        raise ValueError(f"write proposal round is invalid: {memory_id}")
    return WriteProposalEvidence(
        memory_id=memory_id,
        generation_mode=raw.get("generation_mode", "llm"),
        tier=tier,
        payload=stored_payload.model_dump(mode="json"),
        content=content,
        retrieval_text=_nonblank(raw, "retrieval_text", "write proposal"),
        metadata=_json_mapping(raw.get("metadata"), "write metadata"),
        source_task_ids=source_task_ids,
        created_round=created_round,
    )


def _runtime_step_text(value: Any) -> str | None:
    """Reproduce the runtime payload's normalize, bound, then revalidate order."""
    normalized = str(value).strip()[:500].strip()
    return normalized or None


def _maintenance_result_ids(
    snapshot: ReadOnlyMemoryRepository,
    decision: MaintenanceDecision,
    *,
    maintenance_round: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    has_lookup = any(
        isinstance(command, LookupCommand) for command in decision.commands
    )
    has_write = any(
        not isinstance(command, LookupCommand) for command in decision.commands
    )
    if has_lookup and has_write:
        raise ValueError("maintenance lookup commands cannot be mixed with writes")
    state = {
        item.id: [item.tier, item.status]
        for item in snapshot.list(status=None)
    }
    looked_up: list[str] = []
    created: list[str] = []
    updated: list[str] = []
    for command in decision.commands:
        if isinstance(command, LookupCommand):
            for memory_id in command.memory_ids:
                tier_status = state.get(memory_id)
                if tier_status is None or tier_status[1] != MemoryStatus.ACTIVE:
                    raise ValueError(
                        f"maintenance lookup references inactive Memory: {memory_id}"
                    )
                looked_up.append(memory_id)
        elif isinstance(command, DeleteCommand):
            if command.updated_round != maintenance_round:
                raise ValueError("maintenance command round mismatch")
            for memory_id in command.memory_ids:
                tier_status = state.get(memory_id)
                if tier_status is None or tier_status[1] != MemoryStatus.ACTIVE:
                    raise ValueError(
                        f"maintenance delete references inactive Memory: {memory_id}"
                    )
                tier_status[1] = MemoryStatus.RETIRED
                _append_once(updated, memory_id)
        elif isinstance(command, MergeCommand):
            if command.updated_round != maintenance_round:
                raise ValueError("maintenance command round mismatch")
            sources: list[list[Any]] = []
            for memory_id in command.source_ids:
                tier_status = state.get(memory_id)
                if tier_status is None or tier_status[1] != MemoryStatus.ACTIVE:
                    raise ValueError(
                        f"maintenance merge references inactive Memory: {memory_id}"
                    )
                sources.append(tier_status)
            tiers = {source[0] for source in sources}
            if len(tiers) != 1:
                raise ValueError("maintenance merge crosses Memory tiers")
            tier = next(iter(tiers))
            target_id = stable_memory_id(tier, command.content)
            if target_id in state:
                raise ValueError(
                    f"maintenance merge target already exists: {target_id}"
                )
            state[target_id] = [tier, MemoryStatus.ACTIVE]
            created.append(target_id)
            for memory_id, source in zip(
                command.source_ids,
                sources,
                strict=True,
            ):
                source[1] = MemoryStatus.RETIRED
                _append_once(updated, memory_id)
        else:
            raise TypeError(
                f"unsupported maintenance command: {type(command).__name__}"
            )
    return tuple(looked_up), tuple(created), tuple(updated)


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _snapshot(
    memory_root: Path,
    snapshot_id: str,
    cache: dict[tuple[Path, str], ReadOnlyMemoryRepository],
) -> ReadOnlyMemoryRepository:
    key = (memory_root.resolve(), snapshot_id)
    if key not in cache:
        cache[key] = ReadOnlyMemoryRepository(memory_root / "snapshots" / snapshot_id)
    return cache[key]


def _row_hash(row: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        row,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _list_of_mappings(value: Any, label: str) -> list[Mapping[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not all(isinstance(item, Mapping) for item in value)
    ):
        raise ValueError(f"{label} must be a list of objects")
    return list(value)


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{label} must be a list of non-blank strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    return result


def _nonblank(value: Mapping[str, Any], key: str, label: str) -> str:
    resolved = value.get(key)
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError(f"{label} {key} must be a non-blank string")
    return resolved


def _sha256_value(value: Mapping[str, Any], key: str, label: str) -> str:
    resolved = _nonblank(value, key, label)
    if not _SHA256.fullmatch(resolved):
        raise ValueError(f"{label} {key} must be a lowercase SHA256")
    return resolved


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean")
    return value


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    copied = _json_copy(value, label)
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be a JSON object")
    return copied


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                default=_json_default,
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-safe") from error


def _json_default(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
