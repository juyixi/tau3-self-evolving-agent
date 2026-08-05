from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from tau3_evolver.memory.types import MemoryTier
from tau3_evolver.slow_loop.evidence import EvidenceLedger


class _AttributionModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MemoryGroupScore(_AttributionModel):
    group: str
    retrieved_count: int = Field(ge=2)
    selected_count: int = Field(ge=1)
    not_selected_count: int = Field(ge=1)
    selected_reward_mean: float
    not_selected_reward_mean: float
    rho: float = Field(ge=0.0, le=1.0)
    delta: float
    contribution: float
    source_episode_ids: tuple[str, ...]


class MemoryScore(_AttributionModel):
    memory_id: str
    tier: MemoryTier
    observed_versions: tuple[int, ...]
    creator_episode_id: str | None
    source_episode_ids: tuple[str, ...]
    groups: tuple[MemoryGroupScore, ...]
    retrieved_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    not_selected_count: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    tier_prior: float
    attribution: float | None
    value: float | None
    status: Literal["scored", "insufficient_evidence"]
    qualified_for_supervision: bool


@dataclass(slots=True)
class _MemoryFacts:
    tier: MemoryTier
    versions: set[int] = field(default_factory=set)
    creator_index: int | None = None
    creator_episode_id: str | None = None


@dataclass(slots=True)
class _GroupObservations:
    selected: list[tuple[str, float]] = field(default_factory=list)
    not_selected: list[tuple[str, float]] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)


def compute_memory_scores(
    ledger: EvidenceLedger,
    *,
    tier_priors: Mapping[str, float],
    score_threshold: float,
) -> tuple[MemoryScore, ...]:
    if not isinstance(ledger, EvidenceLedger):
        raise TypeError("ledger must be an EvidenceLedger")
    priors = _validate_priors(tier_priors)
    if (
        not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not math.isfinite(score_threshold)
    ):
        raise ValueError("score_threshold must be finite")

    facts = _collect_memory_facts(ledger)
    episode_ids = [episode.episode_id for episode in ledger.episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("evidence ledger contains duplicate episode IDs")

    scores = [
        _score_memory(
            memory_id,
            facts[memory_id],
            ledger=ledger,
            tier_prior=priors[facts[memory_id].tier],
            score_threshold=float(score_threshold),
        )
        for memory_id in sorted(facts)
    ]
    return tuple(scores)


def _collect_memory_facts(ledger: EvidenceLedger) -> dict[str, _MemoryFacts]:
    facts: dict[str, _MemoryFacts] = {}
    for episode_index, episode in enumerate(ledger.episodes):
        candidate_ids = [candidate.memory_id for candidate in episode.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"duplicate candidate in episode {episode.episode_id}")
        if len(episode.selected_memory_ids) != len(set(episode.selected_memory_ids)):
            raise ValueError(f"duplicate selection in episode {episode.episode_id}")
        if not set(episode.selected_memory_ids) <= set(candidate_ids):
            raise ValueError(f"selection is not a candidate in episode {episode.episode_id}")
        if not math.isfinite(episode.final_reward):
            raise ValueError(f"episode reward must be finite: {episode.episode_id}")
        for candidate in episode.candidates:
            memory = _facts_for(facts, candidate.memory_id, candidate.tier)
            memory.versions.add(candidate.memory_version)
        proposal_by_id = {proposal.memory_id: proposal for proposal in episode.write_proposals}
        if len(proposal_by_id) != len(episode.write_proposals):
            raise ValueError(f"duplicate write proposal in episode {episode.episode_id}")
        for proposal in episode.write_proposals:
            _facts_for(facts, proposal.memory_id, proposal.tier)
        for memory_id in episode.committed_new_memory_ids:
            proposal = proposal_by_id.get(memory_id)
            if proposal is None:
                raise ValueError(
                    f"committed new memory has no proposal in episode {episode.episode_id}"
                )
            memory = _facts_for(facts, memory_id, proposal.tier)
            if memory.creator_index is not None:
                raise ValueError(f"memory has multiple creator episodes: {memory_id}")
            memory.creator_index = episode_index
            memory.creator_episode_id = episode.episode_id
            memory.versions.add(1)
    for maintenance in ledger.maintenance:
        for item in maintenance.public_repository:
            memory = _facts_for(facts, item.id, item.tier)
            memory.versions.add(item.version)
    return facts


def _facts_for(
    facts: dict[str, _MemoryFacts],
    memory_id: str,
    tier: MemoryTier,
) -> _MemoryFacts:
    existing = facts.get(memory_id)
    if existing is None:
        existing = _MemoryFacts(tier=tier)
        facts[memory_id] = existing
    elif existing.tier != tier:
        raise ValueError(f"memory tier changed across evidence: {memory_id}")
    return existing


def _score_memory(
    memory_id: str,
    facts: _MemoryFacts,
    *,
    ledger: EvidenceLedger,
    tier_prior: float,
    score_threshold: float,
) -> MemoryScore:
    observations: dict[str, _GroupObservations] = {}
    source_episode_ids: list[str] = []
    selected_count = 0
    not_selected_count = 0
    for episode_index, episode in enumerate(ledger.episodes):
        if facts.creator_index is not None and episode_index <= facts.creator_index:
            continue
        candidate = next(
            (
                candidate
                for candidate in episode.candidates
                if candidate.memory_id == memory_id
            ),
            None,
        )
        if candidate is None:
            continue
        if candidate.tier != facts.tier:
            raise ValueError(f"memory tier changed across evidence: {memory_id}")
        source_episode_ids.append(episode.episode_id)
        group = observations.setdefault(episode.task_group, _GroupObservations())
        group.source_episode_ids.append(episode.episode_id)
        observation = (episode.episode_id, float(episode.final_reward))
        if memory_id in episode.selected_memory_ids:
            group.selected.append(observation)
            selected_count += 1
        else:
            group.not_selected.append(observation)
            not_selected_count += 1

    group_scores: list[MemoryGroupScore] = []
    for group_name in sorted(observations):
        group = observations[group_name]
        if not group.selected or not group.not_selected:
            continue
        selected_mean = math.fsum(reward for _, reward in group.selected) / len(
            group.selected
        )
        not_selected_mean = math.fsum(
            reward for _, reward in group.not_selected
        ) / len(group.not_selected)
        retrieved_count = len(group.selected) + len(group.not_selected)
        rho = len(group.selected) / retrieved_count
        delta = selected_mean - not_selected_mean
        contribution = rho * delta
        group_scores.append(
            MemoryGroupScore(
                group=group_name,
                retrieved_count=retrieved_count,
                selected_count=len(group.selected),
                not_selected_count=len(group.not_selected),
                selected_reward_mean=selected_mean,
                not_selected_reward_mean=not_selected_mean,
                rho=rho,
                delta=delta,
                contribution=contribution,
                source_episode_ids=tuple(group.source_episode_ids),
            )
        )

    confidence = 1.0 - 1.0 / math.sqrt(1.0 + selected_count)
    if group_scores:
        attribution = math.fsum(group.contribution for group in group_scores)
        value = tier_prior * confidence * attribution
        if not math.isfinite(attribution) or not math.isfinite(value):
            raise ValueError(f"non-finite attribution result for memory {memory_id}")
        status: Literal["scored", "insufficient_evidence"] = "scored"
    else:
        attribution = None
        value = None
        status = "insufficient_evidence"
    return MemoryScore(
        memory_id=memory_id,
        tier=facts.tier,
        observed_versions=tuple(sorted(facts.versions)),
        creator_episode_id=facts.creator_episode_id,
        source_episode_ids=tuple(source_episode_ids),
        groups=tuple(group_scores),
        retrieved_count=selected_count + not_selected_count,
        selected_count=selected_count,
        not_selected_count=not_selected_count,
        confidence=confidence,
        tier_prior=tier_prior,
        attribution=attribution,
        value=value,
        status=status,
        qualified_for_supervision=value is not None and value >= score_threshold,
    )


def _validate_priors(tier_priors: Mapping[str, float]) -> dict[MemoryTier, float]:
    expected = {tier.value for tier in MemoryTier}
    if set(tier_priors) != expected:
        raise ValueError("tier_priors must define exactly trajectory, tip, skill, and tool")
    resolved: dict[MemoryTier, float] = {}
    for tier in MemoryTier:
        value = tier_priors[tier.value]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"tier prior must be finite and non-negative: {tier.value}")
        resolved[tier] = float(value)
    return resolved
