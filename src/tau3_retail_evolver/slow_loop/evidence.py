from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau3_retail_evolver.fast_loop.decisions import MaintenanceDecision
from tau3_retail_evolver.fast_loop.prompts import MAX_DIAGNOSTIC_CONTENT_CHARS
from tau3_retail_evolver.io.jsonl import iter_jsonl_objects
from tau3_retail_evolver.memory.read_only import ReadOnlyMemoryRepository
from tau3_retail_evolver.memory.operations import DeleteCommand, LookupCommand, MergeCommand
from tau3_retail_evolver.memory.tier_contracts import (
    TIER_SCHEMA_VERSION,
    ToolPayload,
    TrajectoryPayload,
    render_tier_payload,
    validate_stored_tier_payload,
    validate_tool_payload_against_tools,
)
from tau3_retail_evolver.memory.types import (
    MemoryStatus,
    MemoryTier,
    canonical_content,
    stable_memory_id,
)
from tau3_retail_evolver.slow_loop.source_runs import SourceRun, SourceRunSet
from tau3_retail_evolver.slow_loop.task_grouping import canonicalize_retail_task_group


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
    source_event_start: int = Field(ge=1)
    source_event_end: int = Field(ge=1)
    source_event_sha256: str
    iteration: int = Field(ge=0)
    task_id: str
    task_group: str
    model_revision: str
    adapter_revision: str | None
    tau2_commit: str
    split_hash: str
    memory_agent_id: str
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
    simulation_result: dict[str, Any]
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
    source_event_start: int = Field(ge=1)
    source_event_end: int = Field(ge=1)
    source_event_sha256: str
    iteration: int = Field(ge=0)
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
    iteration: int = Field(ge=0)
    model_revision: str
    adapter_revision: str | None
    tau2_commit: str
    split_hash: str
    memory_agent_id: str
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
    if memory_root.name != "memory" or memory_root.parent.name != source_runs.memory_agent_id:
        raise ValueError("evidence memory namespace does not match source runs")

    snapshots: dict[str, ReadOnlyMemoryRepository] = {}
    episodes: list[EpisodeEvidence] = []
    maintenance: list[MaintenanceEvidence] = []
    for source_run in source_runs.runs:
        rows = _load_source_events(source_run)
        for _, event in rows:
            _require_event_provenance(source_run, event)
        cursor = 0
        local_episode_count = 0
        local_task_ids: list[str] = []
        local_maintenance_rounds: list[int] = []
        while cursor < len(rows):
            row_number, event = rows[cursor]
            event_type = event.get("event_type")
            if event_type == "EpisodeStarted":
                block, cursor = _take_episode_block(rows, cursor)
                episode = _build_episode(
                    source_run,
                    block,
                    memory_root=memory_root,
                    snapshots=snapshots,
                )
                episodes.append(episode)
                local_episode_count += 1
                local_task_ids.append(episode.task_id)
                continue
            if event_type == "TaskFailed":
                local_task_ids.append(event["task_id"])
                cursor += 1
                continue
            if event_type == "MaintenanceStarted":
                block, cursor = _take_maintenance_block(rows, cursor)
                expected_task_index = (
                    source_run.summary["completed_train_tasks_before"]
                    + len(local_task_ids)
                )
                item = _build_maintenance(
                        source_run,
                        block,
                        prior_episode_ids=tuple(item.episode_id for item in episodes),
                        expected_task_index=expected_task_index,
                        memory_root=memory_root,
                        snapshots=snapshots,
                    )
                maintenance.append(item)
                local_maintenance_rounds.append(item.maintenance_round)
                continue
            if event_type == "MaintenanceTaskFailed":
                cursor += 1
                continue
            raise ValueError(
                f"unexpected source event {event_type!r} at {source_run.events_path}:{row_number}"
            )
        if local_episode_count != source_run.summary["episode_count"]:
            raise ValueError(f"evidence episode count mismatch for run {source_run.run_id}")
        if tuple(local_task_ids) != tuple(source_run.manifest["task_ids"]):
            raise ValueError(f"evidence task order mismatch for run {source_run.run_id}")
        if tuple(local_maintenance_rounds) != tuple(
            source_run.summary["maintenance_rounds_executed"]
        ):
            raise ValueError(
                f"evidence maintenance rounds mismatch for run {source_run.run_id}"
            )

    return EvidenceLedger(
        iteration=source_runs.iteration,
        model_revision=source_runs.model_revision,
        adapter_revision=source_runs.adapter_revision,
        tau2_commit=source_runs.tau2_commit,
        split_hash=source_runs.split_hash,
        memory_agent_id=source_runs.memory_agent_id,
        source_run_ids=tuple(run.run_id for run in source_runs.runs),
        episodes=tuple(episodes),
        maintenance=tuple(maintenance),
    )


def _load_source_events(source_run: SourceRun) -> list[tuple[int, dict[str, Any]]]:
    actual_hash = hashlib.sha256(source_run.events_path.read_bytes()).hexdigest()
    if actual_hash != source_run.events_sha256:
        raise ValueError(f"source event hash changed: {source_run.events_path}")
    return list(enumerate(iter_jsonl_objects(source_run.events_path), start=1))


def _take_episode_block(
    rows: list[tuple[int, dict[str, Any]]],
    start: int,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    task_id = rows[start][1].get("task_id")
    if start + 1 >= len(rows):
        raise ValueError(f"incomplete episode lifecycle for task {task_id}")
    next_type = rows[start + 1][1].get("event_type")
    if next_type == "MemorySelected":
        raise ValueError(f"MemorySelected before candidates for task {task_id}")
    if next_type != "MemoryCandidatesRetrieved":
        raise ValueError(f"incomplete retrieval lifecycle for task {task_id}")
    if start + 2 >= len(rows) or rows[start + 2][1].get("event_type") != "MemorySelected":
        raise ValueError(f"incomplete selection lifecycle for task {task_id}")

    cursor = start + 3
    turn = 0
    while cursor < len(rows) and rows[cursor][1].get("event_type") == "DecisionMade":
        if cursor + 1 >= len(rows) or rows[cursor + 1][1].get("event_type") != "EnvironmentStepped":
            raise ValueError(f"incomplete action lifecycle for task {task_id}")
        if rows[cursor][1].get("turn") != turn or rows[cursor + 1][1].get("turn") != turn:
            raise ValueError(f"non-monotonic trajectory turn for task {task_id}")
        cursor += 2
        turn += 1
    if turn == 0 or cursor >= len(rows) or rows[cursor][1].get("event_type") != "EpisodeFinished":
        raise ValueError(f"incomplete episode lifecycle for task {task_id}")
    cursor += 1
    if cursor < len(rows) and rows[cursor][1].get("event_type") == "EpisodeFinished":
        raise ValueError(f"duplicate EpisodeFinished for task {task_id}")
    if cursor >= len(rows) or rows[cursor][1].get("event_type") != "MemoryWriteProposed":
        raise ValueError(f"incomplete write lifecycle for task {task_id}")
    cursor += 1
    if cursor >= len(rows) or rows[cursor][1].get("event_type") != "MemoryWriteCommitted":
        raise ValueError(f"incomplete write lifecycle for task {task_id}")
    cursor += 1
    block = rows[start:cursor]
    _require_block_task(block, task_id)
    return block, cursor


def _take_maintenance_block(
    rows: list[tuple[int, dict[str, Any]]],
    start: int,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    if start + 2 >= len(rows):
        raise ValueError("incomplete maintenance lifecycle")
    block = rows[start : start + 3]
    if tuple(event.get("event_type") for _, event in block) != (
        "MaintenanceStarted",
        "MaintenanceProposed",
        "MaintenanceCommitted",
    ):
        raise ValueError("incomplete maintenance lifecycle")
    task_id = block[0][1].get("task_id")
    _require_block_task(block, task_id)
    return block, start + 3


def _require_block_task(
    block: Sequence[tuple[int, dict[str, Any]]],
    task_id: Any,
) -> None:
    if not isinstance(task_id, str) or any(event.get("task_id") != task_id for _, event in block):
        raise ValueError("source event block crosses task provenance")
    snapshots = {event.get("memory_snapshot_id") for _, event in block}
    if len(snapshots) != 1:
        raise ValueError(f"source event block crosses snapshot provenance for task {task_id}")


def _build_episode(
    source_run: SourceRun,
    block: list[tuple[int, dict[str, Any]]],
    *,
    memory_root: Path,
    snapshots: dict[str, ReadOnlyMemoryRepository],
) -> EpisodeEvidence:
    started = block[0][1]
    retrieved = block[1][1]
    selected_event = block[2][1]
    finished = block[-3][1]
    proposed = block[-2][1]
    committed = block[-1][1]
    task_id = started["task_id"]
    snapshot_id = started["memory_snapshot_id"]
    snapshot = _snapshot(memory_root, snapshot_id, snapshots)

    raw_candidates = _list_of_mappings(retrieved.get("candidates"), "candidates")
    candidates = tuple(
        _candidate_evidence(raw, expected_rank=index, snapshot=snapshot)
        for index, raw in enumerate(raw_candidates, start=1)
    )
    candidate_ids = tuple(candidate.memory_id for candidate in candidates)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"duplicate candidate memory ID for task {task_id}")
    selected_ids = _string_tuple(
        selected_event.get("selected_memory_ids"), "selected_memory_ids"
    )
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError(f"duplicate selected memory ID for task {task_id}")
    if not set(selected_ids) <= set(candidate_ids):
        raise ValueError(f"selected memory is not a candidate for task {task_id}")
    raw_selected = _list_of_mappings(selected_event.get("selected"), "selected candidates")
    selected_detail_ids = tuple(
        _nonblank(raw, "memory_id", "selected candidate") for raw in raw_selected
    )
    if (
        len(selected_detail_ids) != len(set(selected_detail_ids))
        or set(selected_detail_ids) != set(selected_ids)
    ):
        raise ValueError(f"selected candidate details mismatch for task {task_id}")
    raw_by_id = {raw["memory_id"]: raw for raw in raw_candidates}
    if any(raw != raw_by_id.get(raw["memory_id"]) for raw in raw_selected):
        raise ValueError(f"selected candidate details mismatch for task {task_id}")

    trajectory = _trajectory(block[3:-3], task_id=task_id)
    if trajectory[0].observation != started.get("observation"):
        raise ValueError(f"trajectory initial observation mismatch for task {task_id}")
    if any(
        previous.next_observation != current.observation
        for previous, current in zip(trajectory, trajectory[1:], strict=False)
    ):
        raise ValueError(f"trajectory observation chain mismatch for task {task_id}")
    if finished.get("steps") != len(trajectory):
        raise ValueError(f"EpisodeFinished step count mismatch for task {task_id}")
    final_reward = _finite_number(finished.get("final_reward"), "final_reward")
    if not math.isclose(final_reward, trajectory[-1].reward, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"final reward does not match trajectory for task {task_id}")

    tools = started.get("tools")
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        raise ValueError(f"EpisodeStarted tools are invalid for task {task_id}")
    raw_proposals = _list_of_mappings(proposed.get("proposals"), "write proposals")
    write_proposals = tuple(
        _write_proposal(
            raw,
            task_id=task_id,
            run_id=source_run.run_id,
            task_group=started["task_group"],
            final_reward=final_reward,
            trajectory=trajectory,
            tools=tools,
        )
        for raw in raw_proposals
    )
    proposal_ids = tuple(item.memory_id for item in write_proposals)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ValueError(f"duplicate write proposal for task {task_id}")
    written_ids = _string_tuple(committed.get("written_memory_ids"), "written_memory_ids")
    replayed_ids = _string_tuple(committed.get("replayed_memory_ids"), "replayed_memory_ids")
    if written_ids != proposal_ids:
        raise ValueError(f"write proposal/commit mismatch for task {task_id}")
    if not set(replayed_ids) <= set(written_ids):
        raise ValueError(f"replayed memory is not committed for task {task_id}")
    committed_new_ids = tuple(memory_id for memory_id in written_ids if memory_id not in replayed_ids)
    output_snapshot = _snapshot(
        memory_root,
        source_run.summary["output_memory_snapshot_id"],
        snapshots,
    )
    proposal_by_id = {item.memory_id: item for item in write_proposals}
    for memory_id in replayed_ids:
        if snapshot.get(memory_id) is None:
            raise ValueError(f"replayed memory missing from input snapshot: {memory_id}")
    for memory_id in committed_new_ids:
        if snapshot.get(memory_id) is not None:
            raise ValueError(f"new memory already exists in input snapshot: {memory_id}")
    for memory_id in written_ids:
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

    _json_mapping(finished.get("terminal_evaluation"), "terminal_evaluation")
    _json_mapping(finished.get("simulation_result"), "simulation_result")
    truncated = _strict_bool(finished.get("truncated"), "truncated")
    return EpisodeEvidence(
        episode_id=f"{source_run.run_id}:{task_id}",
        run_id=source_run.run_id,
        source_event_start=block[0][0],
        source_event_end=block[-1][0],
        source_event_sha256=_block_hash(block),
        iteration=source_run.manifest["iteration"],
        task_id=task_id,
        task_group=canonicalize_retail_task_group(started["task_group"]),
        model_revision=source_run.manifest["model_revision"],
        adapter_revision=source_run.manifest["adapter_revision"],
        tau2_commit=source_run.manifest["tau2_commit"],
        split_hash=source_run.manifest["split_hash"],
        memory_agent_id=source_run.manifest["rollout_options"]["memory_agent_id"],
        memory_snapshot_id=snapshot_id,
        seed=source_run.manifest["seed"],
        policy=_json_copy(started.get("policy"), "policy"),
        tools=tuple(_json_mapping(tool, "tool") for tool in tools),
        initial_observation=_json_copy(started.get("observation"), "observation"),
        query_hash=_sha256_value(retrieved, "query_hash", "retrieval event"),
        retriever_revision=_nonblank(
            retrieved, "retriever_revision", "retrieval event"
        ),
        candidates=candidates,
        selected_memory_ids=selected_ids,
        trajectory=trajectory,
        terminal_evaluation={"reward": final_reward},
        simulation_result={},
        final_reward=final_reward,
        terminated=trajectory[-1].terminated,
        truncated=truncated,
        write_proposals=write_proposals,
        proposed_memory_ids=proposal_ids,
        committed_new_memory_ids=committed_new_ids,
        replayed_memory_ids=replayed_ids,
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


def _trajectory(
    rows: Sequence[tuple[int, dict[str, Any]]],
    *,
    task_id: str,
) -> tuple[TrajectoryStepEvidence, ...]:
    if not rows or len(rows) % 2:
        raise ValueError(f"trajectory is incomplete for task {task_id}")
    steps: list[TrajectoryStepEvidence] = []
    for offset in range(0, len(rows), 2):
        decision = rows[offset][1]
        stepped = rows[offset + 1][1]
        turn = offset // 2
        if decision.get("event_type") != "DecisionMade" or stepped.get(
            "event_type"
        ) != "EnvironmentStepped":
            raise ValueError(f"trajectory event order is invalid for task {task_id}")
        if decision.get("turn") != turn or stepped.get("turn") != turn:
            raise ValueError(f"trajectory turn is invalid for task {task_id}")
        action = _nonblank(decision, "parsed_action", "decision event")
        if stepped.get("action") != action:
            raise ValueError(f"trajectory action mismatch for task {task_id}")
        steps.append(
            TrajectoryStepEvidence(
                turn=turn,
                observation=_json_copy(decision.get("observation"), "observation"),
                action=action,
                next_observation=_json_copy(
                    stepped.get("observation"), "next_observation"
                ),
                reward=_finite_number(stepped.get("reward"), "step reward"),
                done=_strict_bool(stepped.get("done"), "done"),
                terminated=_strict_bool(stepped.get("terminated"), "terminated"),
                truncated=_strict_bool(stepped.get("truncated"), "truncated"),
                public_info=_json_mapping(stepped.get("public_info"), "public_info"),
            )
        )
    return tuple(steps)


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
                    str(step.observation).strip()[:500],
                    str(step.next_observation).strip()[:500] or None,
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
    created_round = raw.get("created_round")
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


def _build_maintenance(
    source_run: SourceRun,
    block: list[tuple[int, dict[str, Any]]],
    *,
    prior_episode_ids: tuple[str, ...],
    expected_task_index: int,
    memory_root: Path,
    snapshots: dict[str, ReadOnlyMemoryRepository],
) -> MaintenanceEvidence:
    started = block[0][1]
    proposed = block[1][1]
    committed = block[2][1]
    round_values = [event.get("maintenance_round") for _, event in block]
    if not all(type(value) is int for value in round_values) or len(set(round_values)) != 1:
        raise ValueError("maintenance round provenance mismatch")
    maintenance_round = round_values[0]
    if maintenance_round <= 0:
        raise ValueError("maintenance round must be positive")
    trigger_task_index = started.get("completed_train_tasks")
    period = started.get("period")
    if trigger_task_index != expected_task_index:
        raise ValueError("maintenance trigger task index mismatch")
    if type(period) is not int or period <= 0:
        raise ValueError("maintenance period is invalid")
    if trigger_task_index % period or trigger_task_index // period != maintenance_round:
        raise ValueError("maintenance round does not match trigger task index")

    snapshot_id = started["memory_snapshot_id"]
    snapshot = _snapshot(memory_root, snapshot_id, snapshots)
    diagnostics = started.get("diagnostics")
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
            item = snapshot.get(public.id)
            if item is None:
                raise ValueError(f"maintenance memory missing from snapshot: {public.id}")
            content_matches = public.content == item.content or (
                0 < len(public.content) <= MAX_DIAGNOSTIC_CONTENT_CHARS
                and item.content.startswith(public.content)
            )
            if (
                item.tier != public.tier
                or not content_matches
                or item.version != public.version
                or item.status != public.status
            ):
                raise ValueError(f"maintenance public memory mismatch: {public.id}")
            public_repository.append(public)
            repository_state.append(
                MaintenanceMemoryEvidence(
                    id=item.id,
                    tier=item.tier,
                    content=item.content,
                    version=item.version,
                    status=item.status,
                    usage_count=item.usage_count,
                    success_count=item.success_count,
                    last_used=item.last_used,
                    embedding=item.embedding,
                    embedding_model_revision=item.embedding_model_revision,
                )
            )

    commands = _list_of_mappings(proposed.get("commands"), "maintenance commands")
    try:
        decision = MaintenanceDecision.model_validate({"commands": commands})
    except ValueError as error:
        raise ValueError("maintenance commands are invalid") from error
    expected_looked_up, expected_created, expected_updated = _maintenance_result_ids(
        snapshot,
        decision,
        maintenance_round=maintenance_round,
    )
    looked_up_ids = _string_tuple(committed.get("looked_up_ids"), "looked_up_ids")
    created_ids = _string_tuple(committed.get("created_ids"), "created_ids")
    updated_ids = _string_tuple(committed.get("updated_ids"), "updated_ids")
    if (
        looked_up_ids != expected_looked_up
        or created_ids != expected_created
        or updated_ids != expected_updated
    ):
        raise ValueError("maintenance commit result does not match proposed commands")
    completed_rounds = _positive_int_tuple(
        committed.get("completed_rounds"), "completed_rounds"
    )
    if maintenance_round not in completed_rounds:
        raise ValueError("maintenance commit does not include its completed round")
    return MaintenanceEvidence(
        maintenance_id=f"{source_run.run_id}:maintenance-round-{maintenance_round}",
        run_id=source_run.run_id,
        source_event_start=block[0][0],
        source_event_end=block[-1][0],
        source_event_sha256=_block_hash(block),
        iteration=source_run.manifest["iteration"],
        maintenance_round=maintenance_round,
        trigger_task_index=trigger_task_index,
        period=period,
        memory_snapshot_id=snapshot_id,
        prior_episode_ids=prior_episode_ids,
        public_repository=tuple(public_repository),
        repository_state=tuple(repository_state),
        commands=tuple(command.model_dump(mode="json") for command in decision.commands),
        looked_up_ids=looked_up_ids,
        created_ids=created_ids,
        updated_ids=updated_ids,
    )


def _maintenance_result_ids(
    snapshot: ReadOnlyMemoryRepository,
    decision: MaintenanceDecision,
    *,
    maintenance_round: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    has_lookup = any(isinstance(command, LookupCommand) for command in decision.commands)
    has_write = any(not isinstance(command, LookupCommand) for command in decision.commands)
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
                    raise ValueError(f"maintenance lookup references inactive Memory: {memory_id}")
                looked_up.append(memory_id)
        elif isinstance(command, DeleteCommand):
            if command.updated_round != maintenance_round:
                raise ValueError("maintenance command round mismatch")
            for memory_id in command.memory_ids:
                tier_status = state.get(memory_id)
                if tier_status is None or tier_status[1] != MemoryStatus.ACTIVE:
                    raise ValueError(f"maintenance delete references inactive Memory: {memory_id}")
                tier_status[1] = MemoryStatus.RETIRED
                _append_once(updated, memory_id)
        elif isinstance(command, MergeCommand):
            if command.updated_round != maintenance_round:
                raise ValueError("maintenance command round mismatch")
            sources = []
            for memory_id in command.source_ids:
                tier_status = state.get(memory_id)
                if tier_status is None or tier_status[1] != MemoryStatus.ACTIVE:
                    raise ValueError(f"maintenance merge references inactive Memory: {memory_id}")
                sources.append(tier_status)
            tiers = {source[0] for source in sources}
            if len(tiers) != 1:
                raise ValueError("maintenance merge crosses Memory tiers")
            tier = next(iter(tiers))
            target_id = stable_memory_id(tier, command.content)
            if target_id in state:
                raise ValueError(f"maintenance merge target already exists: {target_id}")
            state[target_id] = [tier, MemoryStatus.ACTIVE]
            created.append(target_id)
            for memory_id, source in zip(command.source_ids, sources, strict=True):
                source[1] = MemoryStatus.RETIRED
                _append_once(updated, memory_id)
        else:
            raise TypeError(f"unsupported maintenance command: {type(command).__name__}")
    return tuple(looked_up), tuple(created), tuple(updated)


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _snapshot(
    memory_root: Path,
    snapshot_id: str,
    cache: dict[str, ReadOnlyMemoryRepository],
) -> ReadOnlyMemoryRepository:
    if snapshot_id not in cache:
        cache[snapshot_id] = ReadOnlyMemoryRepository(
            memory_root / "snapshots" / snapshot_id
        )
    return cache[snapshot_id]


def _require_event_provenance(source_run: SourceRun, event: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": 2,
        "run_id": source_run.run_id,
        "iteration": source_run.manifest["iteration"],
        "split": "train",
        "mode": "learn",
        "model_revision": source_run.manifest["model_revision"],
        "adapter_revision": source_run.manifest["adapter_revision"],
        "seed": source_run.manifest["seed"],
    }
    if any(event.get(key) != value for key, value in expected.items()):
        raise ValueError(f"source event provenance mismatch for run {source_run.run_id}")


def _block_hash(block: Sequence[tuple[int, dict[str, Any]]]) -> str:
    canonical = json.dumps(
        [event for _, event in block],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _list_of_mappings(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} must be a list of objects")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a list of non-blank strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    return result


def _positive_int_tuple(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        type(item) is int and item > 0 for item in value
    ):
        raise ValueError(f"{label} must be a list of positive integers")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
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
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be JSON-safe") from error
