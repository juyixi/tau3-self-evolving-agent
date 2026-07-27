from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.tier_contracts import (
    SkillPayload,
    SkillStep,
    render_tier_payload,
)
from tau3_retail_evolver.memory.types import stable_memory_id, MemoryTier
from tau3_retail_evolver.fast_loop.prompts import MAX_DIAGNOSTIC_CONTENT_CHARS
from tau3_retail_evolver.slow_loop.evidence import build_evidence
from tau3_retail_evolver.slow_loop.task_grouping import RETAIL_TASK_GROUP
from tau3_retail_evolver.slow_loop.source_runs import SourceRunSet, load_source_runs


GROUP = f"retail-actions-v1:{'a' * 64}"


def _common(
    run_id: str,
    task_id: str,
    snapshot_id: str,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 3,
        "split": "train",
        "mode": "learn",
        "task_id": task_id,
        "task_group": GROUP,
        "model_revision": "model-a",
        "adapter_revision": "adapter-a",
        "memory_snapshot_id": snapshot_id,
        "seed": seed,
    }


def _episode_events(
    *,
    run_id: str,
    task_id: str,
    snapshot_id: str,
    candidates: list[dict[str, Any]],
    selected_ids: list[str],
    proposals: list[dict[str, Any]],
    written_ids: list[str],
    replayed_ids: list[str],
) -> list[dict[str, Any]]:
    common = _common(run_id, task_id, snapshot_id)
    action = "lookup_order(order_id='1')"
    return [
        {
            **common,
            "event_type": "EpisodeStarted",
            "observation": "How can I help?",
            "policy": "Follow retail policy.",
            "tools": [{"type": "function", "function": {"name": "lookup_order"}}],
        },
        {
            **common,
            "event_type": "MemoryCandidatesRetrieved",
            "query_hash": "a" * 64,
            "retriever_revision": "embedding-a",
            "candidates": candidates,
        },
        {
            **common,
            "event_type": "MemorySelected",
            "selected_memory_ids": selected_ids,
            "selected": [
                candidate for candidate in candidates if candidate["memory_id"] in selected_ids
            ],
        },
        {
            **common,
            "event_type": "DecisionMade",
            "turn": 0,
            "observation": "How can I help?",
            "parsed_action": action,
            "sampling_params": {"temperature": 1.0},
            "latency_s": 0.1,
            "repair_used": False,
        },
        {
            **common,
            "event_type": "EnvironmentStepped",
            "turn": 0,
            "action": action,
            "observation": "Order found.",
            "reward": 1.0,
            "done": True,
            "terminated": True,
            "truncated": False,
            "public_info": {"status": "done"},
        },
        {
            **common,
            "event_type": "EpisodeFinished",
            "steps": 1,
            "final_reward": 1.0,
            "terminal_evaluation": {
                "reward": 1.0,
                "nl_assertions": [
                    {
                        "nl_assertion": "Private evaluator rubric.",
                        "justification": "Private evaluator reasoning.",
                        "met": True,
                    }
                ],
                "action_checks": [{"arguments": {"private": "golden"}}],
            },
            "simulation_result": {
                "status": "done",
                "messages": [{"raw_data": {"reasoning_content": "private"}}],
            },
            "truncated": False,
            "project_truncated": False,
        },
        {
            **common,
            "event_type": "MemoryWriteProposed",
            "proposals": proposals,
            "repair_used": False,
        },
        {
            **common,
            "event_type": "MemoryWriteCommitted",
            "written_memory_ids": written_ids,
            "replayed_memory_ids": replayed_ids,
        },
    ]


def _write_run(
    tmp_path: Path,
    *,
    run_id: str,
    task_ids: list[str],
    input_snapshot: str,
    output_snapshot: str,
    events: list[dict[str, Any]],
    total_reward: float,
) -> Path:
    run_path = tmp_path / "runs" / run_id
    (run_path / "rollouts").mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 3,
        "model_revision": "model-a",
        "adapter_revision": "adapter-a",
        "memory_snapshot_id": input_snapshot,
        "tau2_commit": "c" * 40,
        "split": "train",
        "split_hash": "d" * 64,
        "task_ids": task_ids,
        "seed": 17,
        "environment_options": {"domain": "retail"},
        "rollout_options": {"memory_enabled": True, "memory_agent_id": "retail"},
    }
    summary = {
        "run_id": run_id,
        "episode_count": len(task_ids),
        "completed_train_tasks_before": 0,
        "completed_train_tasks_after": len(task_ids),
        "input_memory_snapshot_id": input_snapshot,
        "output_memory_snapshot_id": output_snapshot,
        "memory_enabled": True,
        "successful_task_ids": task_ids,
        "maintenance_rounds_executed": [1] if len(task_ids) == 30 else [],
        "total_terminal_reward": total_reward,
    }
    (run_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_path / "fast_loop_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (run_path / "rollouts" / "events.jsonl").write_text(
        "".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8"
    )
    return run_path


def _catalog(*task_ids: str) -> SimpleNamespace:
    return SimpleNamespace(task_ids=lambda split: task_ids, split_sha256="d" * 64)


def _schema2_source_with_committed_write(
    tmp_path: Path,
) -> tuple[SourceRunSet, Path, dict[str, str]]:
    memory_root = tmp_path / "history" / "agents" / "retail" / "memory"
    repository = MemoryRepository(memory_root)
    tip = repository.add(
        tier="tip",
        content="Confirm the order number before changing it.",
        source_task_ids=("seed-tip",),
        created_round=0,
    )
    tool = repository.add(
        tier="tool",
        content="Use lookup_order before a mutation.",
        source_task_ids=("seed-tool",),
        created_round=0,
    )
    input_snapshot = repository.snapshot()
    skill_payload = SkillPayload(
        goal="Verify the retrieved order before the final action",
        steps=(
            SkillStep(order=1, instruction="Look up the order."),
            SkillStep(order=2, instruction="Verify the returned order state."),
        ),
        success_condition="The verified order is ready for the final action.",
    )
    skill_content = render_tier_payload(MemoryTier.SKILL, skill_payload)
    skill_id = stable_memory_id(MemoryTier.SKILL, skill_content)
    candidates = [
        {
            "memory_id": tip.id,
            "memory_version": tip.version,
            "tier": tip.tier.value,
            "rank": 1,
            "similarity": 0.9,
        },
        {
            "memory_id": tool.id,
            "memory_version": tool.version,
            "tier": tool.tier.value,
            "rank": 2,
            "similarity": 0.7,
        },
    ]
    proposal = {
        "memory_id": skill_id,
        "tier": "skill",
        "tier_schema_version": 2,
        "payload": skill_payload.model_dump(mode="json"),
        "content": skill_content,
        "retrieval_text": "Verify the retrieved order before the final action.",
        "metadata": {"source_run_id": "run-a"},
        "source_task_ids": ["1"],
        "created_round": 3,
    }
    events = _episode_events(
        run_id="run-a",
        task_id="1",
        snapshot_id=input_snapshot.memory_snapshot_id,
        candidates=candidates,
        selected_ids=[tip.id],
        proposals=[proposal],
        written_ids=[skill_id],
        replayed_ids=[],
    )
    repository.add(
        tier="skill",
        tier_schema_version=2,
        payload=skill_payload.model_dump(mode="json"),
        content=skill_content,
        source_task_ids=("1",),
        created_round=3,
        metadata={"source_run_id": "run-a"},
    )
    output_snapshot = repository.snapshot()
    run_path = _write_run(
        tmp_path,
        run_id="run-a",
        task_ids=["1"],
        input_snapshot=input_snapshot.memory_snapshot_id,
        output_snapshot=output_snapshot.memory_snapshot_id,
        events=events,
        total_reward=1.0,
    )
    source = load_source_runs(
        [run_path], catalog=_catalog("1"), memory_root=memory_root
    )
    return source, memory_root, {"tip": tip.id, "tool": tool.id, "skill": skill_id}


def test_build_evidence_reconstructs_episode_from_frozen_snapshot(
    tmp_path: Path,
) -> None:
    source, memory_root, ids = _schema2_source_with_committed_write(tmp_path)

    ledger = build_evidence(source, memory_root=memory_root)

    episode = ledger.episodes[0]
    assert episode.task_id == "1"
    assert episode.task_group == RETAIL_TASK_GROUP
    assert [candidate.memory_id for candidate in episode.candidates] == [
        ids["tip"],
        ids["tool"],
    ]
    assert episode.candidates[0].content == "Confirm the order number before changing it."
    assert episode.selected_memory_ids == (ids["tip"],)
    assert episode.trajectory[0].action == "lookup_order(order_id='1')"
    assert episode.final_reward == 1.0
    assert episode.terminal_evaluation == {"reward": 1.0}
    assert episode.simulation_result == {}
    assert episode.terminated is True
    assert episode.truncated is False
    assert episode.committed_new_memory_ids == (ids["skill"],)
    assert episode.replayed_memory_ids == ()
    assert episode.source_event_start == 1
    assert episode.source_event_end == 8


def test_build_evidence_skips_recorded_failed_task(
    tmp_path: Path,
) -> None:
    source, memory_root, _ = _schema2_source_with_committed_write(tmp_path)
    run_path = source.runs[0].path
    manifest_path = run_path / "manifest.json"
    summary_path = run_path / "fast_loop_summary.json"
    events_path = run_path / "rollouts" / "events.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
    ]
    failed_event = {
        **_common("run-a", "0", manifest["memory_snapshot_id"]),
        "event_type": "TaskFailed",
        "error": {"types": ["RuntimeError"], "fingerprint": "a" * 16},
    }
    manifest["task_ids"] = ["0", "1"]
    summary.update(
        attempted_task_count=2,
        failed_task_count=1,
        failed_task_ids=["0"],
        completed_train_tasks_after=2,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    events_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in [failed_event, *events]),
        encoding="utf-8",
    )
    loaded = load_source_runs(
        [run_path],
        catalog=_catalog("0", "1"),
        memory_root=memory_root,
    )

    ledger = build_evidence(loaded, memory_root=memory_root)

    assert [episode.task_id for episode in ledger.episodes] == ["1"]


def test_build_evidence_accepts_selected_details_in_retrieval_order(
    tmp_path: Path,
) -> None:
    def select_in_teacher_order(
        events: list[dict[str, Any]], ids: dict[str, str]
    ) -> None:
        events[2]["selected_memory_ids"] = [ids["tool"], ids["tip"]]
        events[2]["selected"] = list(events[1]["candidates"])

    source, memory_root = _mutated_source(tmp_path, select_in_teacher_order)

    ledger = build_evidence(source, memory_root=memory_root)

    assert ledger.episodes[0].selected_memory_ids == (
        stable_memory_id(MemoryTier.TOOL, "Use lookup_order before a mutation."),
        stable_memory_id(
            MemoryTier.TIP, "Confirm the order number before changing it."
        ),
    )


def _mutated_source(
    tmp_path: Path,
    mutation: Callable[[list[dict[str, Any]], dict[str, str]], None],
) -> tuple[SourceRunSet, Path]:
    source, memory_root, ids = _schema2_source_with_committed_write(tmp_path)
    events_path = source.runs[0].events_path
    events = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]
    mutation(events, ids)
    events_path.write_text(
        "".join(f"{json.dumps(event)}\n" for event in events), encoding="utf-8"
    )
    run = replace(
        source.runs[0],
        events_sha256=hashlib.sha256(events_path.read_bytes()).hexdigest(),
    )
    return replace(source, runs=(run,)), memory_root


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("selection_before_retrieval", "MemorySelected before candidates"),
        ("selected_not_candidate", "selected memory is not a candidate"),
        ("duplicate_finish", "duplicate EpisodeFinished"),
        ("proposal_without_commit", "incomplete write lifecycle"),
        ("candidate_missing_from_snapshot", "candidate missing from snapshot"),
        ("candidate_version_mismatch", "candidate version mismatch"),
        ("selected_detail_mismatch", "selected candidate details mismatch"),
        ("cross_run_provenance", "event provenance"),
    ],
)
def test_build_evidence_rejects_invalid_lifecycle(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    def mutate(events: list[dict[str, Any]], ids: dict[str, str]) -> None:
        if mutation == "selection_before_retrieval":
            events[1], events[2] = events[2], events[1]
        elif mutation == "selected_not_candidate":
            events[2]["selected_memory_ids"] = [ids["skill"]]
        elif mutation == "duplicate_finish":
            events.insert(6, dict(events[5]))
        elif mutation == "proposal_without_commit":
            events.pop()
        elif mutation == "candidate_missing_from_snapshot":
            events[1]["candidates"][1]["memory_id"] = ids["skill"]
        elif mutation == "candidate_version_mismatch":
            events[1]["candidates"][0]["memory_version"] = 2
        elif mutation == "selected_detail_mismatch":
            events[2]["selected"][0]["similarity"] = 0.1
        elif mutation == "cross_run_provenance":
            events[4]["run_id"] = "other-run"

    source, memory_root = _mutated_source(tmp_path, mutate)

    with pytest.raises(ValueError, match=message):
        build_evidence(source, memory_root=memory_root)


def _source_with_maintenance(
    tmp_path: Path,
    *,
    content: str = "Keep this public.",
    committed_created_ids: list[str] | None = None,
) -> tuple[SourceRunSet, Path, str]:
    memory_root = tmp_path / "history" / "agents" / "retail" / "memory"
    repository = MemoryRepository(memory_root)
    public_memory = repository.add(
        tier="tip",
        content=content,
        source_task_ids=("seed-maintenance",),
        created_round=0,
    )
    snapshot = repository.snapshot()
    events: list[dict[str, Any]] = []
    task_ids = [str(index) for index in range(1, 31)]
    for task_id in task_ids:
        events.extend(
            _episode_events(
                run_id="run-maint",
                task_id=task_id,
                snapshot_id=snapshot.memory_snapshot_id,
                candidates=[],
                selected_ids=[],
                proposals=[],
                written_ids=[],
                replayed_ids=[],
            )
        )
    maintenance_common = {
        **_common("run-maint", "maintenance-round-1", snapshot.memory_snapshot_id),
        "task_group": "retail-actions-v1:maintenance",
        "maintenance_round": 1,
    }
    public_item = {
        "id": public_memory.id,
        "tier": "tip",
        "content": content[:MAX_DIAGNOSTIC_CONTENT_CHARS],
        "version": 1,
        "status": "active",
    }
    events.extend(
        [
            {
                **maintenance_common,
                "event_type": "MaintenanceStarted",
                "completed_train_tasks": 30,
                "period": 30,
                "diagnostics": {
                    "trajectory": {"items": []},
                    "tip": {"items": [public_item]},
                    "skill": {"items": []},
                    "tool": {"items": []},
                },
            },
            {
                **maintenance_common,
                "event_type": "MaintenanceProposed",
                "commands": [],
            },
            {
                **maintenance_common,
                "event_type": "MaintenanceCommitted",
                "looked_up_ids": [],
                "created_ids": committed_created_ids or [],
                "updated_ids": [],
                "completed_rounds": [1],
            },
        ]
    )
    run_path = _write_run(
        tmp_path,
        run_id="run-maint",
        task_ids=task_ids,
        input_snapshot=snapshot.memory_snapshot_id,
        output_snapshot=snapshot.memory_snapshot_id,
        events=events,
        total_reward=30.0,
    )
    return (
        load_source_runs(
            [run_path], catalog=_catalog(*task_ids), memory_root=memory_root
        ),
        memory_root,
        public_memory.id,
    )


def test_maintenance_evidence_uses_public_repository_and_prior_history(
    tmp_path: Path,
) -> None:
    source, memory_root, public_memory_id = _source_with_maintenance(tmp_path)

    ledger = build_evidence(source, memory_root=memory_root)

    maintenance = ledger.maintenance[0]
    assert maintenance.maintenance_round == 1
    assert maintenance.trigger_task_index == 30
    assert maintenance.public_repository[0].model_dump(mode="json") == {
        "id": public_memory_id,
        "tier": "tip",
        "content": "Keep this public.",
        "version": 1,
        "status": "active",
    }
    assert maintenance.repository_state[0].id == public_memory_id
    assert maintenance.repository_state[0].usage_count == 0
    assert maintenance.repository_state[0].embedding is None
    assert maintenance.commands == ()
    assert len(maintenance.prior_episode_ids) == 30


def test_maintenance_evidence_accepts_canonical_public_content_truncation(
    tmp_path: Path,
) -> None:
    content = "x" * (MAX_DIAGNOSTIC_CONTENT_CHARS + 17)
    source, memory_root, _ = _source_with_maintenance(tmp_path, content=content)

    maintenance = build_evidence(source, memory_root=memory_root).maintenance[0]

    assert maintenance.public_repository[0].content == content[:MAX_DIAGNOSTIC_CONTENT_CHARS]
    assert maintenance.repository_state[0].content == content


def test_maintenance_evidence_rejects_commit_result_not_produced_by_commands(
    tmp_path: Path,
) -> None:
    source, memory_root, _ = _source_with_maintenance(
        tmp_path,
        committed_created_ids=["mem-tip-not-produced"],
    )

    with pytest.raises(ValueError, match="maintenance commit result"):
        build_evidence(source, memory_root=memory_root)
