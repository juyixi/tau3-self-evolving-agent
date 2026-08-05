from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tau3_evolver.agent.policy import EpisodeResult
from tau3_evolver.execution.results import BatchFailure
from tau3_evolver.artifacts.sanitize import sanitize_artifact_data


EPISODE_SCHEMA_VERSION = 1
_RUN_CONTEXT_FIELDS = frozenset(
    {
        "schema_version",
        "event_type",
        "run_id",
        "benchmark",
        "mode",
        "split",
        "task_id",
        "task_group",
        "model_revision",
        "checkpoint",
        "memory_source_namespace",
        "memory_snapshot_id",
        "cross_domain_memory",
        "memory_generation",
        "seed",
    }
)


def build_completed_episode(
    result: EpisodeResult,
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collapse one internal lifecycle into the canonical task-level artifact."""
    if not events:
        raise ValueError(f"episode has no lifecycle evidence: {result.task_id}")
    task_ids = {event.get("task_id") for event in events}
    if task_ids != {result.task_id}:
        raise ValueError(f"episode lifecycle crosses task identity: {result.task_id}")

    started = _one(events, "EpisodeStarted", required=True)
    finished = _one(events, "EpisodeFinished", required=True)
    retrieved = _one(events, "MemoryCandidatesRetrieved")
    selected = _one(events, "MemorySelected")
    disabled = _one(events, "MemoryDisabled")
    proposed = _one(events, "MemoryWriteProposed")
    committed = _one(events, "MemoryWriteCommitted")
    discarded = _one(events, "MemoryBatchDiscarded")

    if (retrieved is None) != (selected is None):
        raise ValueError(f"incomplete Memory selection lifecycle: {result.task_id}")
    if retrieved is None and disabled is None:
        raise ValueError(f"episode has no Memory mode evidence: {result.task_id}")
    if retrieved is not None and disabled is not None:
        raise ValueError(f"episode has conflicting Memory mode evidence: {result.task_id}")

    trajectory = _trajectory(events, task_id=result.task_id)
    if len(trajectory) != result.steps:
        raise ValueError(f"episode step count mismatch: {result.task_id}")
    if float(finished.get("final_reward")) != float(result.final_reward):
        raise ValueError(f"episode reward mismatch: {result.task_id}")

    memory = (
        _enabled_memory(retrieved, selected, proposed, committed, discarded)
        if retrieved is not None and selected is not None
        else {
            "enabled": False,
            "reason": str(disabled.get("reason", "disabled")),
            "writes": [],
        }
    )
    record = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "task_id": result.task_id,
        "task_group": str(started["task_group"]),
        "seed": int(started["seed"]),
        "status": "completed",
        "task": {
            "initial_observation": started.get("observation"),
            "policy": started.get("policy"),
            "tools": started.get("tools", []),
        },
        "trajectory": trajectory,
        "outcome": {
            "final_reward": result.final_reward,
            "steps": result.steps,
            "terminal_evaluation": dict(result.terminal_evaluation),
            "truncated": result.truncated,
            "project_truncated": result.project_truncated,
            "parse_error_count": result.parse_error_count,
            "response_parse_error_count": result.response_parse_error_count,
            "response_count": result.response_count,
            "agent_prompt_tokens": result.agent_prompt_tokens,
            "agent_completion_tokens": result.agent_completion_tokens,
        },
        "memory": memory,
    }
    return sanitize_artifact_data(record)


def build_failed_episode(
    failure: BatchFailure,
    *,
    task_group: str,
    seed: int,
) -> dict[str, Any]:
    return sanitize_artifact_data(
        {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "task_id": failure.task_id,
            "task_group": task_group,
            "seed": seed,
            "status": "failed",
            "failure": {
                "stage": failure.stage,
                "error_type": failure.error_type,
            },
        }
    )


def _enabled_memory(
    retrieved: Mapping[str, Any],
    selected: Mapping[str, Any],
    proposed: Mapping[str, Any] | None,
    committed: Mapping[str, Any] | None,
    discarded: Mapping[str, Any] | None,
) -> dict[str, Any]:
    retrieval = _payload(retrieved)
    selection = _payload(selected)
    selection.pop("selected", None)
    selected_ids = selection.pop("selected_memory_ids", [])

    proposals = list(proposed.get("proposals", [])) if proposed is not None else []
    written = (
        {str(value) for value in committed.get("written_memory_ids", [])}
        if committed is not None
        else set()
    )
    replayed = (
        {str(value) for value in committed.get("replayed_memory_ids", [])}
        if committed is not None
        else set()
    )
    proposal_ids = {str(proposal.get("memory_id")) for proposal in proposals}
    if written and written != proposal_ids:
        raise ValueError("Memory proposal and commit identities differ")
    writes = []
    for proposal in proposals:
        memory_id = str(proposal["memory_id"])
        disposition = (
            "discarded"
            if discarded is not None
            else "replayed"
            if memory_id in replayed
            else "created"
            if memory_id in written
            else "proposed"
        )
        writes.append({**dict(proposal), "disposition": disposition})

    write_audit = _payload(proposed) if proposed is not None else {}
    write_audit.pop("proposals", None)
    return {
        "enabled": True,
        "retrieval": retrieval,
        "selected_memory_ids": list(selected_ids),
        "selection": selection,
        "writes": writes,
        "write_audit": write_audit,
    }


def _trajectory(
    events: Sequence[Mapping[str, Any]],
    *,
    task_id: str,
) -> list[dict[str, Any]]:
    decisions = sorted(
        (event for event in events if event.get("event_type") == "DecisionMade"),
        key=lambda event: int(event["turn"]),
    )
    steps = sorted(
        (event for event in events if event.get("event_type") == "EnvironmentStepped"),
        key=lambda event: int(event["turn"]),
    )
    if len(decisions) != len(steps):
        raise ValueError(f"incomplete trajectory: {task_id}")
    result: list[dict[str, Any]] = []
    for turn, (decision, stepped) in enumerate(zip(decisions, steps, strict=True)):
        if decision.get("turn") != turn or stepped.get("turn") != turn:
            raise ValueError(f"non-monotonic trajectory: {task_id}")
        action = decision.get("parsed_action")
        if action != stepped.get("action"):
            raise ValueError(f"trajectory action mismatch: {task_id}")
        result.append(
            {
                "turn": turn,
                "observation": decision.get("observation"),
                "action": action,
                "next_observation": stepped.get("observation"),
                "reward": stepped.get("reward"),
                "done": stepped.get("done"),
                "terminated": stepped.get("terminated"),
                "truncated": stepped.get("truncated"),
                "public_info": stepped.get("public_info", {}),
                "decision": {
                    "sampling_params": decision.get("sampling_params", {}),
                    "latency_s": decision.get("latency_s", 0.0),
                    "repair_used": decision.get("repair_used", False),
                    "prompt_tokens": decision.get("prompt_tokens"),
                    "completion_tokens": decision.get("completion_tokens"),
                },
            }
        )
    return result


def _one(
    events: Sequence[Mapping[str, Any]],
    event_type: str,
    *,
    required: bool = False,
) -> Mapping[str, Any] | None:
    matches = [event for event in events if event.get("event_type") == event_type]
    if len(matches) > 1:
        raise ValueError(f"duplicate {event_type} event")
    if required and not matches:
        raise ValueError(f"missing {event_type} event")
    return matches[0] if matches else None


def _payload(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key not in _RUN_CONTEXT_FIELDS}


__all__ = [
    "EPISODE_SCHEMA_VERSION",
    "build_completed_episode",
    "build_failed_episode",
]
