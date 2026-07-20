from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from tau3_retail_evolver.fast_loop.decisions import maintenance_command_schemas
from tau3_retail_evolver.slow_loop.attribution import MemoryScore
from tau3_retail_evolver.slow_loop.evidence import (
    EpisodeEvidence,
    EvidenceLedger,
    MaintenanceEvidence,
    MaintenanceMemoryEvidence,
    MemoryCandidateEvidence,
)
from tau3_retail_evolver.slow_loop.leakage import (
    audit_artifact_payload,
    audit_public_input,
)


ExampleKind = Literal["sel", "act", "write", "maint"]

ONLINE_SAMPLING_CONTRACT = {
    "mode": "online",
    "student_completion_source": "stage6_current_policy_generation",
    "current_model_and_lora_revision_required": True,
    "teacher_uses_same_completion_token_prefix": True,
    "teacher_forward_stop_gradient": True,
    "fast_loop_history_role": "behavior_provenance_only",
}


class OPDExample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    example_id: str
    kind: ExampleKind
    public_input: dict[str, Any]
    privileged_hindsight: dict[str, Any]
    response_schema: dict[str, Any]
    sampling_contract: dict[str, Any]
    provenance: dict[str, Any]


def build_selection_examples(
    ledger: EvidenceLedger,
    scores: Sequence[MemoryScore],
    *,
    score_threshold: float = 0.01,
) -> tuple[OPDExample, ...]:
    score_by_id = _score_index(scores)
    _validate_score_provenance(ledger, scores)
    threshold = _threshold(score_threshold)
    examples: list[OPDExample] = []
    for episode in ledger.episodes:
        if not episode.candidates:
            continue
        candidate_scores = [_score_for(score_by_id, item.memory_id) for item in episode.candidates]
        if not any(
            score.value is not None and abs(score.value) >= threshold
            for score in candidate_scores
        ):
            continue
        public_input = {
            "policy": _json_copy(episode.policy),
            "tools": [_json_copy(tool) for tool in episode.tools],
            "observation": _json_copy(episode.initial_observation),
            "candidates": [_candidate_public(candidate) for candidate in episode.candidates],
        }
        privileged = {
            "candidate_scores": [
                {
                    "memory_id": score.memory_id,
                    "tier": score.tier.value,
                    "value": score.value,
                    "attribution": score.attribution,
                    "gamma": score.confidence,
                    "confidence": score.confidence,
                    "status": score.status,
                    "qualified_for_supervision": score.qualified_for_supervision,
                }
                for score in candidate_scores
            ]
        }
        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "memory_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [candidate.memory_id for candidate in episode.candidates],
                    },
                    "uniqueItems": True,
                }
            },
            "required": ["memory_ids"],
        }
        examples.append(
            _make_example(
                "sel",
                public_input,
                privileged,
                response_schema,
                _episode_provenance(episode),
            )
        )
    return tuple(examples)


def build_action_examples(
    ledger: EvidenceLedger,
    scores: Sequence[MemoryScore],
    *,
    score_threshold: float = 0.01,
    teacher_memory_cap: int = 20,
) -> tuple[OPDExample, ...]:
    score_by_id = _score_index(scores)
    _validate_score_provenance(ledger, scores)
    threshold = _threshold(score_threshold)
    cap = _positive_int(teacher_memory_cap, "teacher_memory_cap")
    examples: list[OPDExample] = []
    for episode in ledger.episodes:
        candidate_by_id = {
            candidate.memory_id: candidate for candidate in episode.candidates
        }
        valuable: list[tuple[MemoryScore, MemoryCandidateEvidence]] = []
        for memory_id in episode.selected_memory_ids:
            candidate = candidate_by_id.get(memory_id)
            if candidate is None:
                raise ValueError(
                    f"selected memory is absent from candidates: {episode.episode_id}/{memory_id}"
                )
            score = _score_for(score_by_id, memory_id)
            if score.value is not None and score.value >= threshold:
                valuable.append((score, candidate))
        valuable.sort(
            key=lambda pair: (-float(pair[0].value), pair[1].rank, pair[0].memory_id)
        )
        valuable = valuable[:cap]
        if not valuable:
            continue
        successful = _successful_trajectory(episode, ledger.episodes)
        if successful is None:
            continue
        privileged = {
            "valuable_selected_memories": [
                {
                    "memory_id": score.memory_id,
                    "tier": score.tier.value,
                    "content": candidate.content,
                    "rank": candidate.rank,
                    "value": score.value,
                    "confidence": score.confidence,
                    "status": score.status,
                }
                for score, candidate in valuable
            ],
            "successful_trajectory": {
                "episode_id": successful.episode_id,
                "task_group": successful.task_group,
                "final_reward": successful.final_reward,
                "trajectory": [
                    step.model_dump(mode="json") for step in successful.trajectory
                ],
            },
        }
        for turn, step in enumerate(episode.trajectory):
            history = [
                previous.model_dump(mode="json")
                for previous in episode.trajectory[:turn]
            ]
            public_input = {
                "policy": _json_copy(episode.policy),
                "tools": [_json_copy(tool) for tool in episode.tools],
                "history": history,
                "observation": _json_copy(step.observation),
            }
            provenance = {
                **_episode_provenance(episode),
                "turn": turn,
                "successful_trajectory_episode_id": successful.episode_id,
            }
            examples.append(
                _make_example(
                    "act",
                    public_input,
                    privileged,
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"action": {"type": "string", "minLength": 1}},
                        "required": ["action"],
                    },
                    provenance,
                )
            )
    return tuple(examples)


def build_writing_examples(
    ledger: EvidenceLedger,
    scores: Sequence[MemoryScore],
    *,
    score_threshold: float = 0.01,
) -> tuple[OPDExample, ...]:
    score_by_id = _score_index(scores)
    _validate_score_provenance(ledger, scores)
    threshold = _threshold(score_threshold)
    examples: list[OPDExample] = []
    for episode in ledger.episodes:
        if not episode.committed_new_memory_ids:
            continue
        proposal_by_id = {
            proposal.memory_id: proposal for proposal in episode.write_proposals
        }
        rows: list[dict[str, Any]] = []
        for memory_id in episode.committed_new_memory_ids:
            proposal = proposal_by_id.get(memory_id)
            if proposal is None:
                raise ValueError(
                    f"committed new memory has no proposal: {episode.episode_id}/{memory_id}"
                )
            score = _score_for(score_by_id, memory_id)
            if score.creator_episode_id != episode.episode_id:
                raise ValueError(f"new memory creator provenance mismatch: {memory_id}")
            if episode.episode_id in score.source_episode_ids:
                raise ValueError(f"creator episode leaked into future attribution: {memory_id}")
            rows.append(
                {
                    "memory_id": memory_id,
                    "tier": proposal.tier.value,
                    "content": proposal.content,
                    "retrieval_text": proposal.retrieval_text,
                    "creator_episode_id": score.creator_episode_id,
                    "source_episode_ids": list(score.source_episode_ids),
                    "value": score.value,
                    "attribution": score.attribution,
                    "confidence": score.confidence,
                    "status": score.status,
                    "groups": [group.model_dump(mode="json") for group in score.groups],
                }
            )
        if not any(
            row["value"] is not None and abs(row["value"]) >= threshold for row in rows
        ):
            continue
        selected_candidates = {
            candidate.memory_id: candidate for candidate in episode.candidates
        }
        public_input = {
            "policy": _json_copy(episode.policy),
            "tools": [_json_copy(tool) for tool in episode.tools],
            "initial_observation": _json_copy(episode.initial_observation),
            "trajectory": [step.model_dump(mode="json") for step in episode.trajectory],
            "final_reward": episode.final_reward,
            "selected_memories": [
                _candidate_public(selected_candidates[memory_id])
                for memory_id in episode.selected_memory_ids
            ],
        }
        examples.append(
            _make_example(
                "write",
                public_input,
                {"written_memory_scores": rows},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "memories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "tier": {
                                        "type": "string",
                                        "enum": ["trajectory", "tip", "skill", "tool"],
                                    },
                                    "content": {"type": "string", "minLength": 1},
                                    "retrieval_text": {"type": "string", "minLength": 1},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["tier", "content"],
                            },
                        }
                    },
                    "required": ["memories"],
                },
                _episode_provenance(episode),
            )
        )
    return tuple(examples)


def build_maintenance_examples(
    ledger: EvidenceLedger,
    scores: Sequence[MemoryScore],
    *,
    teacher_memory_cap: int = 20,
    redundancy_threshold: float = 0.90,
    max_redundancy_pairs: int = 50,
) -> tuple[OPDExample, ...]:
    score_by_id = _score_index(scores)
    _validate_score_provenance(ledger, scores)
    cap = _positive_int(teacher_memory_cap, "teacher_memory_cap")
    if (
        not isinstance(redundancy_threshold, (int, float))
        or isinstance(redundancy_threshold, bool)
        or not math.isfinite(redundancy_threshold)
        or not -1.0 <= redundancy_threshold <= 1.0
    ):
        raise ValueError("redundancy_threshold must be finite and between -1 and 1")
    if type(max_redundancy_pairs) is not int or max_redundancy_pairs < 0:
        raise ValueError("max_redundancy_pairs must be a non-negative integer")
    episode_by_id = {episode.episode_id: episode for episode in ledger.episodes}
    examples: list[OPDExample] = []
    for maintenance in ledger.maintenance:
        prior = []
        for episode_id in maintenance.prior_episode_ids:
            episode = episode_by_id.get(episode_id)
            if episode is None:
                raise ValueError(
                    f"maintenance references unknown prior episode: {episode_id}"
                )
            prior.append(episode)
        redundancy_pairs = _redundancy_pairs(
            maintenance.repository_state,
            threshold=float(redundancy_threshold),
            limit=max_redundancy_pairs,
        )
        diagnostics = _maintenance_diagnostics(
            maintenance,
            prior,
            score_by_id,
            redundancy_pairs,
            cap=cap,
        )
        public_input = {
            "repository": [
                item.model_dump(mode="json") for item in maintenance.public_repository
            ],
            "interaction_history": [
                {
                    "episode_id": episode.episode_id,
                    "trajectory": [
                        step.model_dump(mode="json") for step in episode.trajectory
                    ],
                    "final_reward": episode.final_reward,
                }
                for episode in prior
            ],
            "tools": list(maintenance_command_schemas()),
        }
        privileged = {
            "memory_diagnostics": diagnostics,
            "redundancy_pairs": redundancy_pairs,
        }
        response_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            command["schema"] for command in maintenance_command_schemas()
                        ]
                    },
                }
            },
            "required": ["commands"],
        }
        provenance = {
            "maintenance_id": maintenance.maintenance_id,
            "run_id": maintenance.run_id,
            "iteration": maintenance.iteration,
            "maintenance_round": maintenance.maintenance_round,
            "trigger_task_index": maintenance.trigger_task_index,
            "memory_snapshot_id": maintenance.memory_snapshot_id,
            "source_event_sha256": maintenance.source_event_sha256,
        }
        examples.append(
            _make_example(
                "maint", public_input, privileged, response_schema, provenance
            )
        )
    return tuple(examples)


def audit_example_boundaries(example: OPDExample) -> None:
    if not isinstance(example, OPDExample):
        raise TypeError("example must be an OPDExample")
    audit_public_input(example.kind, example.public_input)
    for payload in (
        example.public_input,
        example.privileged_hindsight,
        example.response_schema,
        example.sampling_contract,
        example.provenance,
    ):
        audit_artifact_payload(payload)
    if example.sampling_contract != ONLINE_SAMPLING_CONTRACT:
        raise ValueError("missing online sampling contract")
    if example.kind == "sel":
        public_ids = {
            row["memory_id"] for row in example.public_input.get("candidates", [])
        }
        score_rows = example.privileged_hindsight.get("candidate_scores", [])
        score_ids = {row.get("memory_id") for row in score_rows}
        if public_ids != score_ids or any("content" in row for row in score_rows):
            raise ValueError("selection public/privileged candidate boundary mismatch")
    if example.kind == "act":
        public_text = _canonical_json(example.public_input)
        for row in example.privileged_hindsight.get(
            "valuable_selected_memories", []
        ):
            for field in ("memory_id", "content"):
                value = row.get(field)
                if isinstance(value, str) and value and value in public_text:
                    raise ValueError("action public input contains memory")
    if example.kind == "write":
        for row in example.privileged_hindsight.get("written_memory_scores", []):
            if row.get("creator_episode_id") in row.get("source_episode_ids", []):
                raise ValueError("writing example uses creator episode as future evidence")
    if example.kind == "maint":
        repository_ids = {
            row.get("id") for row in example.public_input.get("repository", [])
        }
        diagnostic_ids = {
            row.get("memory_id")
            for row in example.privileged_hindsight.get("memory_diagnostics", [])
        }
        pair_ids = {
            memory_id
            for pair in example.privileged_hindsight.get("redundancy_pairs", [])
            for memory_id in (
                pair.get("left_memory_id"),
                pair.get("right_memory_id"),
            )
        }
        if not diagnostic_ids <= repository_ids or not pair_ids <= repository_ids:
            raise ValueError("maintenance privileged IDs do not join public repository")


def _make_example(
    kind: ExampleKind,
    public_input: dict[str, Any],
    privileged_hindsight: dict[str, Any],
    response_schema: dict[str, Any],
    provenance: dict[str, Any],
) -> OPDExample:
    identity = {
        "kind": kind,
        "public_input": public_input,
        "privileged_hindsight": privileged_hindsight,
        "response_schema": response_schema,
        "provenance": provenance,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()
    example = OPDExample(
        example_id=f"opd_{kind}_{digest[:24]}",
        kind=kind,
        public_input=_json_copy(public_input),
        privileged_hindsight=_json_copy(privileged_hindsight),
        response_schema=_json_copy(response_schema),
        sampling_contract=dict(ONLINE_SAMPLING_CONTRACT),
        provenance=_json_copy(provenance),
    )
    audit_example_boundaries(example)
    return example


def _episode_provenance(episode: EpisodeEvidence) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "run_id": episode.run_id,
        "iteration": episode.iteration,
        "task_id": episode.task_id,
        "task_group": episode.task_group,
        "model_revision": episode.model_revision,
        "adapter_revision": episode.adapter_revision,
        "memory_snapshot_id": episode.memory_snapshot_id,
        "source_event_sha256": episode.source_event_sha256,
    }


def _candidate_public(candidate: MemoryCandidateEvidence) -> dict[str, Any]:
    return {
        "memory_id": candidate.memory_id,
        "memory_version": candidate.memory_version,
        "tier": candidate.tier.value,
        "rank": candidate.rank,
        "similarity": candidate.similarity,
        "content": candidate.content,
    }


def _successful_trajectory(
    current: EpisodeEvidence,
    episodes: Sequence[EpisodeEvidence],
) -> EpisodeEvidence | None:
    if current.final_reward == 1.0:
        return current
    candidates = [
        episode
        for episode in episodes
        if episode.task_group == current.task_group and episode.final_reward == 1.0
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda episode: (
            -episode.final_reward,
            len(episode.trajectory),
            episode.episode_id,
        ),
    )


def _maintenance_diagnostics(
    maintenance: MaintenanceEvidence,
    prior: Sequence[EpisodeEvidence],
    score_by_id: Mapping[str, MemoryScore],
    redundancy_pairs: list[dict[str, Any]],
    *,
    cap: int,
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    state_by_id = {item.id: item for item in maintenance.repository_state}
    for memory_id in sorted(state_by_id):
        state = state_by_id[memory_id]
        score = _score_for(score_by_id, memory_id)
        successful_selected_count = sum(
            memory_id in episode.selected_memory_ids and episode.final_reward == 1.0
            for episode in prior
        )
        rows[memory_id] = {
            "memory_id": memory_id,
            "tier": state.tier.value,
            "value": score.value,
            "attribution": score.attribution,
            "gamma": score.confidence,
            "confidence": score.confidence,
            "status": score.status,
            "retrieved_count": score.retrieved_count,
            "selected_count": score.selected_count,
            "usage_count": state.usage_count,
            "success_count": state.success_count,
            "successful_selected_count": successful_selected_count,
            "last_used": state.last_used,
            "embedding_model_revision": state.embedding_model_revision,
            "embedding_dimension": len(state.embedding) if state.embedding is not None else None,
        }
    high_value = sorted(
        rows,
        key=lambda memory_id: (
            rows[memory_id]["value"] is None,
            -(rows[memory_id]["value"] or 0.0),
            memory_id,
        ),
    )
    low_value = sorted(
        rows,
        key=lambda memory_id: (
            rows[memory_id]["value"] is None,
            rows[memory_id]["value"] or 0.0,
            memory_id,
        ),
    )
    high_usage = sorted(
        rows,
        key=lambda memory_id: (-rows[memory_id]["usage_count"], memory_id),
    )
    redundancy_endpoints: list[str] = []
    for pair in redundancy_pairs:
        redundancy_endpoints.extend(
            (pair["left_memory_id"], pair["right_memory_id"])
        )
    buckets = (high_value, low_value, high_usage, redundancy_endpoints)
    selected: list[str] = []
    selected_set: set[str] = set()
    max_bucket_length = max((len(bucket) for bucket in buckets), default=0)
    for index in range(max_bucket_length):
        for bucket in buckets:
            if index >= len(bucket):
                continue
            memory_id = bucket[index]
            if memory_id in rows and memory_id not in selected_set:
                selected.append(memory_id)
                selected_set.add(memory_id)
                if len(selected) == cap:
                    return [rows[item] for item in sorted(selected)]
    for memory_id in sorted(rows):
        if memory_id not in selected_set:
            selected.append(memory_id)
            if len(selected) == cap:
                break
    return [rows[item] for item in sorted(selected)]


def _redundancy_pairs(
    states: Sequence[MaintenanceMemoryEvidence],
    *,
    threshold: float,
    limit: int,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    ordered = sorted(states, key=lambda item: item.id)
    for left_index, left in enumerate(ordered):
        if left.embedding is None or left.embedding_model_revision is None:
            continue
        for right in ordered[left_index + 1 :]:
            if (
                right.embedding is None
                or right.embedding_model_revision != left.embedding_model_revision
                or len(right.embedding) != len(left.embedding)
            ):
                continue
            similarity = _cosine(left.embedding, right.embedding)
            if similarity is None or similarity < threshold:
                continue
            pairs.append(
                {
                    "left_memory_id": left.id,
                    "right_memory_id": right.id,
                    "kappa": similarity,
                    "embedding_model_revision": left.embedding_model_revision,
                    "embedding_dimension": len(left.embedding),
                }
            )
    pairs.sort(
        key=lambda pair: (
            -pair["kappa"],
            pair["left_memory_id"],
            pair["right_memory_id"],
        )
    )
    return pairs[:limit]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or len(left) != len(right):
        return None
    if not all(math.isfinite(value) for value in (*left, *right)):
        return None
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, similarity))


def _score_index(scores: Sequence[MemoryScore]) -> dict[str, MemoryScore]:
    result = {score.memory_id: score for score in scores}
    if len(result) != len(scores):
        raise ValueError("Memory scores contain duplicate IDs")
    return result


def _validate_score_provenance(
    ledger: EvidenceLedger,
    scores: Sequence[MemoryScore],
) -> None:
    episode_ids = {episode.episode_id for episode in ledger.episodes}
    memory_ids = {
        candidate.memory_id
        for episode in ledger.episodes
        for candidate in episode.candidates
    }
    memory_ids.update(
        proposal.memory_id
        for episode in ledger.episodes
        for proposal in episode.write_proposals
    )
    memory_ids.update(
        item.id
        for maintenance in ledger.maintenance
        for item in maintenance.public_repository
    )
    for score in scores:
        if score.memory_id not in memory_ids:
            raise ValueError(f"Memory score references another build: {score.memory_id}")
        if not set(score.source_episode_ids) <= episode_ids:
            raise ValueError(f"Memory score references another build: {score.memory_id}")
        if score.creator_episode_id is not None and score.creator_episode_id not in episode_ids:
            raise ValueError(f"Memory creator references another build: {score.memory_id}")
        for group in score.groups:
            if not set(group.source_episode_ids) <= set(score.source_episode_ids):
                raise ValueError(
                    f"Memory group score has inconsistent source episodes: {score.memory_id}"
                )


def _score_for(scores: Mapping[str, MemoryScore], memory_id: str) -> MemoryScore:
    try:
        return scores[memory_id]
    except KeyError as error:
        raise ValueError(f"missing Memory score: {memory_id}") from error


def _threshold(value: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("score_threshold must be finite and non-negative")
    return float(value)


def _positive_int(value: int, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError) as error:
        raise ValueError("OPD example payload must be JSON-safe") from error
