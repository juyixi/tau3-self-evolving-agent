from __future__ import annotations

import hashlib
import json

import pytest

from tau3_evolver.slow_loop.attribution import MemoryGroupScore, MemoryScore
from tau3_evolver.slow_loop.evidence import (
    EpisodeEvidence,
    EvidenceLedger,
    MaintenanceEvidence,
    MaintenanceMemoryEvidence,
    MemoryCandidateEvidence,
    PublicMemoryEvidence,
    TrajectoryStepEvidence,
    WriteProposalEvidence,
)
from tau3_evolver.slow_loop.examples import (
    audit_example_boundaries,
    build_action_examples,
    build_maintenance_examples,
    build_selection_examples,
    build_writing_examples,
)
from tau3_evolver.memory.tier_contracts import (
    SkillPayload,
    SkillStep,
    TipPayload,
    materialize_rule_trajectory_memory,
    render_tier_payload,
)
from tau3_evolver.memory.types import MemoryTier, stable_memory_id


def _candidate(memory_id: str, tier: str, rank: int) -> MemoryCandidateEvidence:
    content = f"Public guidance for {memory_id}."
    return MemoryCandidateEvidence(
        memory_id=memory_id,
        memory_version=1,
        tier=tier,
        rank=rank,
        similarity=1.0 - rank / 10,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )


def _step(
    action: str,
    reward: float,
    *,
    turn: int = 0,
    done: bool = True,
) -> TrajectoryStepEvidence:
    return TrajectoryStepEvidence(
        turn=turn,
        observation=f"Observation {turn}.",
        action=action,
        next_observation=f"Observation {turn + 1}.",
        reward=reward,
        done=done,
        terminated=done,
        truncated=False,
        public_info={"status": "done" if done else "continue"},
    )


def _episode(
    episode_id: str,
    *,
    group: str = "returns",
    reward: float,
    candidates: tuple[MemoryCandidateEvidence, ...] = (),
    selected_ids: tuple[str, ...] = (),
    proposals: tuple[WriteProposalEvidence, ...] = (),
    committed_new_ids: tuple[str, ...] = (),
    replayed_ids: tuple[str, ...] = (),
    trajectory: tuple[TrajectoryStepEvidence, ...] | None = None,
) -> EpisodeEvidence:
    return EpisodeEvidence(
        episode_id=episode_id,
        run_id="run-a",
        source_episode_row=1,
        source_episode_sha256="e" * 64,
        memory_generation=3,
        task_id=f"task-{episode_id}",
        task_group=group,
        model_revision="model-a",
        adapter_revision="adapter-a",
        runtime_revision="c" * 40,
        split_hash="d" * 64,
        memory_namespace="retail",
        memory_snapshot_id="snapshot-a",
        seed=17,
        policy="Follow public retail policy.",
        tools=({"type": "function", "function": {"name": "lookup_order"}},),
        initial_observation="Customer asks for help.",
        query_hash="a" * 64,
        retriever_revision="embedding-a",
        candidates=candidates,
        selected_memory_ids=selected_ids,
        trajectory=trajectory or (_step("lookup_order(order_id='1')", reward),),
        terminal_evaluation={},
        final_reward=reward,
        terminated=True,
        truncated=False,
        write_proposals=proposals,
        proposed_memory_ids=tuple(proposal.memory_id for proposal in proposals),
        committed_new_memory_ids=committed_new_ids,
        replayed_memory_ids=replayed_ids,
    )


def _ledger(
    episodes: tuple[EpisodeEvidence, ...],
    maintenance: tuple[MaintenanceEvidence, ...] = (),
) -> EvidenceLedger:
    return EvidenceLedger(
        memory_generation=3,
        model_revision="model-a",
        adapter_revision="adapter-a",
        runtime_revision="c" * 40,
        split_hash="d" * 64,
        memory_namespace="retail",
        source_run_ids=("run-a",),
        episodes=episodes,
        maintenance=maintenance,
    )


def _score(
    memory_id: str,
    *,
    tier: str,
    value: float | None,
    creator: str | None = None,
    source_episode_ids: tuple[str, ...] = ("selected", "control"),
) -> MemoryScore:
    scored = value is not None
    groups = (
        MemoryGroupScore(
            group="returns",
            retrieved_count=2,
            selected_count=1,
            not_selected_count=1,
            selected_reward_mean=1.0,
            not_selected_reward_mean=0.0,
            rho=0.5,
            delta=1.0,
            contribution=0.5,
            source_episode_ids=source_episode_ids,
        ),
    ) if scored else ()
    return MemoryScore(
        memory_id=memory_id,
        tier=tier,
        observed_versions=(1,),
        creator_episode_id=creator,
        source_episode_ids=source_episode_ids,
        groups=groups,
        retrieved_count=len(source_episode_ids),
        selected_count=1 if source_episode_ids else 0,
        not_selected_count=max(0, len(source_episode_ids) - 1),
        confidence=0.5 if source_episode_ids else 0.0,
        tier_prior=0.8 if tier == "tip" else 1.0,
        attribution=value,
        value=value,
        status="scored" if scored else "insufficient_evidence",
        qualified_for_supervision=value is not None and value > 0.01,
    )


def _selection_action_fixture():
    mem_a = _candidate("mem-a", "tip", 1)
    mem_b = _candidate("mem-b", "tool", 2)
    failed = _episode(
        "failed",
        reward=0.0,
        candidates=(mem_a, mem_b),
        selected_ids=("mem-a",),
    )
    success = _episode(
        "success",
        reward=1.0,
        candidates=(mem_a, mem_b),
        selected_ids=("mem-a",),
        trajectory=(
            _step("Ask the customer to authenticate.", 0.0, turn=0, done=False),
            _step("lookup_order(order_id='1')", 1.0, turn=1),
        ),
    )
    scores = (
        _score(
            "mem-a", tier="tip", value=0.8, source_episode_ids=("failed", "success")
        ),
        _score(
            "mem-b", tier="tool", value=-0.2, source_episode_ids=("failed", "success")
        ),
    )
    return _ledger((failed, success)), scores


def test_selection_example_keeps_all_candidate_scores_and_online_contract() -> None:
    ledger, scores = _selection_action_fixture()

    example = build_selection_examples(
        ledger, scores, score_threshold=0.01
    )[0]

    assert example.kind == "sel"
    assert "value" not in json.dumps(example.public_input).lower()
    assert {row["memory_id"] for row in example.privileged_hindsight["candidate_scores"]} == {
        "mem-a",
        "mem-b",
    }
    assert all(
        "content" not in row
        for row in example.privileged_hindsight["candidate_scores"]
    )
    assert example.privileged_hindsight["candidate_scores"][0]["gamma"] == 0.5
    assert example.sampling_contract["mode"] == "online"
    assert example.provenance["sample_unit"] == "task"
    assert "student_output" not in type(example).model_fields
    audit_example_boundaries(example)


def test_action_example_is_one_task_level_sample_for_its_own_successful_trajectory() -> None:
    ledger, scores = _selection_action_fixture()

    examples = build_action_examples(
        ledger,
        scores,
        score_threshold=0.01,
        teacher_memory_cap=20,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.schema_version == 2
    public_text = json.dumps(example.public_input)
    assert "mem-a" not in public_text
    assert "Public guidance" not in public_text
    assert example.privileged_hindsight["valuable_selected_memories"][0][
        "memory_id"
    ] == "mem-a"
    assert example.privileged_hindsight["successful_trajectory"]["task_group"] == (
        example.provenance["task_group"]
    )
    assert example.privileged_hindsight["successful_trajectory"]["episode_id"] == "success"
    assert len(example.privileged_hindsight["successful_trajectory"]["trajectory"]) == 2
    assert example.provenance["episode_id"] == "success"
    assert example.provenance["sample_unit"] == "task"
    assert example.provenance["source_trajectory_step_count"] == 2
    assert "turn" not in example.provenance
    assert "solution" in example.response_schema["properties"]
    audit_example_boundaries(example)


def test_writing_example_uses_only_future_scored_committed_memories() -> None:
    proposal_payload = SkillPayload(
        goal="A new public workflow",
        steps=(
            SkillStep(order=1, instruction="Inspect the current state."),
            SkillStep(order=2, instruction="Apply the verified next action."),
        ),
        success_condition="The requested operation is complete.",
    )
    replay_payload = TipPayload(guidance="Existing guidance.")
    proposal = WriteProposalEvidence(
        memory_id="mem-new",
        tier="skill",
        payload=proposal_payload.model_dump(mode="json"),
        content=render_tier_payload(MemoryTier.SKILL, proposal_payload),
        retrieval_text="A new public workflow",
        metadata={},
        source_task_ids=("task-creator",),
        created_round=3,
    )
    replay = WriteProposalEvidence(
        memory_id="mem-replay",
        tier="tip",
        payload=replay_payload.model_dump(mode="json"),
        content=render_tier_payload(MemoryTier.TIP, replay_payload),
        retrieval_text="Existing guidance.",
        metadata={},
        source_task_ids=("task-creator",),
        created_round=3,
    )
    rule_trajectory = materialize_rule_trajectory_memory(
        task_instruction="Complete the customer request.",
        run_id="run-a",
        task_id="task-creator",
        task_group="returns",
        final_reward=1.0,
        outcome_class="success",
        trajectory=(
            {
                "observation": "Observation 0.",
                "action": "lookup_order(order_id='1')",
                "next_observation": "Observation 1.",
                "reward": 1.0,
                "done": True,
                "terminated": True,
                "truncated": False,
            },
        ),
    )
    rule_id = stable_memory_id(MemoryTier.TRAJECTORY, rule_trajectory.content)
    rule_proposal = WriteProposalEvidence(
        memory_id=rule_id,
        generation_mode="rule",
        tier="trajectory",
        payload=rule_trajectory.payload,
        content=rule_trajectory.content,
        retrieval_text=rule_trajectory.retrieval_text,
        metadata={"generation_mode": "rule"},
        source_task_ids=("task-creator",),
        created_round=3,
    )
    creator = _episode(
        "creator",
        reward=1.0,
        proposals=(rule_proposal, proposal, replay),
        committed_new_ids=(rule_id, "mem-new"),
        replayed_ids=("mem-replay",),
    )
    score = _score(
        "mem-new",
        tier="skill",
        value=0.6,
        creator="creator",
        source_episode_ids=("future-selected", "future-control"),
    )

    future_selected = _episode("future-selected", reward=1.0)
    future_control = _episode("future-control", reward=0.0)
    example = build_writing_examples(
        _ledger((creator, future_selected, future_control)),
        (score,),
        score_threshold=0.01,
    )[0]

    rows = example.privileged_hindsight["written_memory_scores"]
    assert [row["memory_id"] for row in rows] == ["mem-new"]
    assert rows[0]["creator_episode_id"] == "creator"
    assert rows[0]["tier_schema_version"] == 2
    assert rows[0]["payload"] == proposal_payload.model_dump(mode="json")
    assert "creator" not in rows[0]["source_episode_ids"]
    assert "mem-replay" not in json.dumps(example.privileged_hindsight)
    assert rule_id not in json.dumps(example.privileged_hindsight)
    assert example.provenance["sample_unit"] == "task"
    audit_example_boundaries(example)


def _maintenance_fixture():
    mem_a = PublicMemoryEvidence(
        id="mem-a", tier="tip", content="First guidance.", version=1, status="active"
    )
    mem_b = PublicMemoryEvidence(
        id="mem-b", tier="tip", content="Second guidance.", version=1, status="active"
    )
    state_a = MaintenanceMemoryEvidence(
        **mem_a.model_dump(mode="python"),
        usage_count=10,
        success_count=8,
        last_used="2026-07-20T00:00:00Z",
        embedding=(1.0, 0.0),
        embedding_model_revision="embedding-a",
    )
    state_b = MaintenanceMemoryEvidence(
        **mem_b.model_dump(mode="python"),
        usage_count=2,
        success_count=1,
        last_used=None,
        embedding=(0.999, 0.001),
        embedding_model_revision="embedding-a",
    )
    episode = _episode("prior", reward=1.0)
    control = _episode("prior-control", reward=0.0)
    maintenance = MaintenanceEvidence(
        maintenance_id="run-a:maintenance-round-1",
        run_id="run-a",
        source_event_start=9,
        source_event_end=11,
        source_event_sha256="f" * 64,
        memory_generation=3,
        maintenance_round=1,
        trigger_task_index=30,
        period=30,
        memory_snapshot_id="snapshot-a",
        prior_episode_ids=("prior", "prior-control"),
        public_repository=(mem_a, mem_b),
        repository_state=(state_a, state_b),
        commands=(),
        looked_up_ids=(),
        created_ids=(),
        updated_ids=(),
    )
    scores = (
        _score(
            "mem-a",
            tier="tip",
            value=0.7,
            source_episode_ids=("prior", "prior-control"),
        ),
        _score(
            "mem-b",
            tier="tip",
            value=-0.3,
            source_episode_ids=("prior", "prior-control"),
        ),
    )
    return _ledger((episode, control), (maintenance,)), scores


def test_maintenance_example_keeps_usage_and_redundancy_privileged() -> None:
    ledger, scores = _maintenance_fixture()

    example = build_maintenance_examples(
        ledger,
        scores,
        teacher_memory_cap=2,
        redundancy_threshold=0.90,
        max_redundancy_pairs=50,
    )[0]

    assert "usage" not in json.dumps(example.public_input).lower()
    assert "task_group" not in json.dumps(example.public_input)
    assert "usage" in json.dumps(example.privileged_hindsight).lower()
    assert example.privileged_hindsight["memory_diagnostics"][0]["gamma"] == 0.5
    assert len(example.privileged_hindsight["memory_diagnostics"]) <= 2
    assert example.privileged_hindsight["redundancy_pairs"][0]["left_memory_id"] == "mem-a"
    assert example.privileged_hindsight["redundancy_pairs"][0]["right_memory_id"] == "mem-b"
    assert example.provenance["sample_unit"] == "maintenance_round"
    audit_example_boundaries(example)


def test_example_builder_rejects_score_for_memory_outside_ledger() -> None:
    ledger, scores = _selection_action_fixture()
    external = _score(
        "mem-external",
        tier="tip",
        value=0.4,
        source_episode_ids=("failed", "success"),
    )

    with pytest.raises(ValueError, match="another build"):
        build_selection_examples(ledger, (*scores, external))
