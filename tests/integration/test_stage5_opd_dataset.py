from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tau3_retail_evolver.envs.runtime import RuntimeFingerprint
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.tier_contracts import (
    SkillPayload,
    SkillStep,
    render_tier_payload,
)
from tau3_retail_evolver.memory.types import MemoryTier
from tau3_retail_evolver.slow_loop import dataset as dataset_module
from tau3_retail_evolver.slow_loop.audit import audit_dataset
from tau3_retail_evolver.slow_loop.dataset import DatasetBuildRequest, build_opd_dataset
from tau3_retail_evolver.slow_loop.task_grouping import RetailTaskGroups


TAU2_COMMIT = "c" * 40
MODEL_REVISION = "model-a"
ADAPTER_REVISION = "adapter-a"


def test_repeated_task_passes_build_and_audit_four_opd_views(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_runs, config_path, project_root = _write_two_continuous_runs(tmp_path)
    retail_root = project_root / "external" / "tau2-bench" / "data" / "tau2" / "domains" / "retail"
    fingerprint = RuntimeFingerprint(
        repo_path=project_root / "external" / "tau2-bench",
        git_commit=TAU2_COMMIT,
        package_version="fixture",
        retail_tasks_path=retail_root / "tasks.json",
        retail_split_path=retail_root / "split_tasks.json",
        gym_available=False,
    )
    monkeypatch.setattr(
        dataset_module.Tau2Runtime,
        "inspect_metadata",
        staticmethod(lambda path: fingerprint),
    )
    monkeypatch.setattr(
        dataset_module.Tau2Runtime,
        "require_pinned_commit",
        staticmethod(lambda runtime: None),
    )
    monkeypatch.setattr(dataset_module, "_git_revision", lambda project: "b" * 40)

    first = build_opd_dataset(
        DatasetBuildRequest(
            source_run_paths=source_runs,
            dataset_build_id="opd-iter0-first",
            output_root=project_root / "published",
            config_path=config_path,
            project_root=project_root,
        )
    )
    second = build_opd_dataset(
        DatasetBuildRequest(
            source_run_paths=tuple(reversed(source_runs)),
            dataset_build_id="opd-iter0-second",
            output_root=project_root / "published",
            config_path=config_path,
            project_root=project_root,
        )
    )

    assert first.audit_report["passed"] is True
    assert second.audit_report["passed"] is True
    assert _content_hashes(first.dataset_dir) == _content_hashes(second.dataset_dir)
    assert all(
        _jsonl_count(first.dataset_dir / "datasets" / f"{kind}.jsonl") > 0
        for kind in ("sel", "act", "write", "maint")
    )
    manifest = first.manifest
    assert manifest["memory"]["snapshot_chain"][0] != manifest["memory"][
        "snapshot_chain"
    ][-1]
    assert manifest["counts"]["evidence_episodes"] == 30
    assert manifest["counts"]["evidence_maintenance"] == 1

    source_events = source_runs[0] / "rollouts" / "events.jsonl"
    source_events.write_text(
        source_events.read_text(encoding="utf-8").replace(
            "Customer needs retail help.", "Tampered source observation.", 1
        ),
        encoding="utf-8",
    )
    report = audit_dataset(first.dataset_dir)
    assert report.passed is False
    assert "source_evidence_mismatch" in {error.code for error in report.errors}


def _write_two_continuous_runs(
    tmp_path: Path,
) -> tuple[tuple[Path, Path], Path, Path]:
    project = tmp_path / "project"
    retail_root = project / "external" / "tau2-bench" / "data" / "tau2" / "domains" / "retail"
    retail_root.mkdir(parents=True)
    split_fixture = Path(__file__).parents[1] / "fixtures" / "tau2_retail" / "split_tasks.json"
    split_data = json.loads(split_fixture.read_text(encoding="utf-8"))
    (retail_root / "split_tasks.json").write_text(
        json.dumps(split_data), encoding="utf-8"
    )
    all_task_ids = tuple(split_data["base"])
    tasks = {
        "tasks": [
            {
                "id": task_id,
                "evaluation_criteria": {
                    "actions": [{"name": "return_delivered_order_items"}]
                },
            }
            for task_id in all_task_ids
        ]
    }
    tasks_path = retail_root / "tasks.json"
    tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
    catalog = RetailTaskCatalog.from_files(
        tasks_path,
        retail_root / "split_tasks.json",
    )
    catalog.require_official_compatibility()
    task_ids = tuple(catalog.task_ids("train")[:30])
    group = RetailTaskGroups.from_file(tasks_path, task_ids=task_ids).signature_for(
        task_ids[0]
    )

    memory_root = project / "history" / "agents" / "retail" / "memory"
    repository = MemoryRepository(memory_root)
    old = repository.add(
        tier="tip",
        content="Check the order before changing it.",
        source_task_ids=("seed",),
        created_round=0,
        embedding=(1.0, 0.0),
        embedding_model_revision="embedding-a",
    )
    snapshot_a = repository.snapshot().memory_snapshot_id
    new_payload = SkillPayload(
        goal="Verify eligibility before completing a return",
        steps=(
            SkillStep(order=1, instruction="Look up the order."),
            SkillStep(order=2, instruction="Verify item return eligibility."),
        ),
        success_condition="The eligible return is ready to complete.",
    )
    new_content = render_tier_payload(MemoryTier.SKILL, new_payload)
    new = repository.add(
        tier="skill",
        tier_schema_version=2,
        payload=new_payload.model_dump(mode="json"),
        content=new_content,
        retrieval_text="Verify eligibility before completing a return.",
        source_task_ids=(task_ids[0],),
        created_round=0,
        metadata={"source_run_id": "run-a"},
        embedding=(0.999, 0.001),
        embedding_model_revision="embedding-a",
    )
    snapshot_b = repository.snapshot().memory_snapshot_id

    old_candidate = _candidate(old, rank=1, similarity=0.95)
    new_candidate_rank_two = _candidate(new, rank=2, similarity=0.85)
    new_candidate = _candidate(new, rank=1, similarity=0.92)
    proposal = {
        "memory_id": new.id,
        "tier": "skill",
        "tier_schema_version": 2,
        "payload": new_payload.model_dump(mode="json"),
        "content": new.content,
        "retrieval_text": new.retrieval_text,
        "metadata": {"source_run_id": "run-a"},
        "source_task_ids": [task_ids[0]],
        "created_round": 0,
    }

    episode_specs: list[dict[str, Any]] = []
    for index, task_id in enumerate(task_ids):
        spec: dict[str, Any] = {
            "task_id": task_id,
            "snapshot_id": snapshot_b if index else snapshot_a,
            "candidates": [],
            "selected_ids": [],
            "reward": 1.0,
            "proposals": [],
            "written_ids": [],
        }
        if index == 0:
            spec.update(
                candidates=[old_candidate],
                selected_ids=[old.id],
                reward=1.0,
                proposals=[proposal],
                written_ids=[new.id],
            )
        elif index == 1:
            spec.update(
                candidates=[old_candidate, new_candidate_rank_two],
                selected_ids=[new.id],
                reward=0.0,
            )
        elif index == 2:
            spec.update(candidates=[new_candidate], selected_ids=[], reward=1.0)
        episode_specs.append(spec)

    run_a_specs = episode_specs[:15]
    run_b_specs = [
        {**spec, "task_id": task_ids[index]}
        for index, spec in enumerate(episode_specs[15:])
    ]
    run_a = _write_run(
        project,
        run_id="run-a",
        specs=run_a_specs,
        before=0,
        input_snapshot=snapshot_a,
        output_snapshot=snapshot_b,
        split_hash=catalog.split_sha256,
        group=group,
    )
    run_b = _write_run(
        project,
        run_id="run-b",
        specs=run_b_specs,
        before=15,
        input_snapshot=snapshot_b,
        output_snapshot=snapshot_b,
        split_hash=catalog.split_sha256,
        group=group,
        maintenance=_maintenance_events(
            repository,
            run_id="run-b",
            snapshot_id=snapshot_b,
        ),
    )

    config_path = project / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "tau2": {
                    "repo_path": "external/tau2-bench",
                    "domain": "retail",
                    "train_split": "train",
                    "eval_split": "test",
                    "user_llm": "fixture",
                    "user_llm_args": {},
                    "solo_mode": False,
                },
                "model": {"base_model": "Qwen/Qwen3.5-9B"},
                "memory": {
                    "enabled": True,
                    "agent_id": "retail",
                    "embedding_device": "cpu",
                    "embedding_dtype": "float32",
                },
            }
        ),
        encoding="utf-8",
    )
    return (run_a, run_b), config_path, project


def _candidate(item: Any, *, rank: int, similarity: float) -> dict[str, Any]:
    return {
        "memory_id": item.id,
        "memory_version": item.version,
        "tier": item.tier.value,
        "rank": rank,
        "similarity": similarity,
    }


def _write_run(
    project: Path,
    *,
    run_id: str,
    specs: list[dict[str, Any]],
    before: int,
    input_snapshot: str,
    output_snapshot: str,
    split_hash: str,
    group: str,
    maintenance: list[dict[str, Any]] | None = None,
) -> Path:
    run_path = project / "source-runs" / run_id
    events_path = run_path / "rollouts" / "events.jsonl"
    events_path.parent.mkdir(parents=True)
    task_ids = [spec["task_id"] for spec in specs]
    events: list[dict[str, Any]] = []
    for spec in specs:
        events.extend(_episode_events(run_id=run_id, group=group, **spec))
    events.extend(maintenance or [])
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 0,
        "model_revision": MODEL_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "memory_snapshot_id": input_snapshot,
        "tau2_commit": TAU2_COMMIT,
        "split": "train",
        "split_hash": split_hash,
        "task_ids": task_ids,
        "seed": 17,
        "environment_options": {"domain": "retail"},
        "rollout_options": {"memory_enabled": True, "memory_agent_id": "retail"},
    }
    summary = {
        "run_id": run_id,
        "episode_count": len(task_ids),
        "completed_train_tasks_before": before,
        "completed_train_tasks_after": before + len(task_ids),
        "input_memory_snapshot_id": input_snapshot,
        "output_memory_snapshot_id": output_snapshot,
        "memory_enabled": True,
        "successful_task_ids": task_ids,
        "maintenance_rounds_executed": [1] if maintenance else [],
        "total_terminal_reward": sum(spec["reward"] for spec in specs),
    }
    (run_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_path / "fast_loop_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    events_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return run_path


def _episode_events(
    *,
    run_id: str,
    task_id: str,
    snapshot_id: str,
    group: str,
    candidates: list[dict[str, Any]],
    selected_ids: list[str],
    reward: float,
    proposals: list[dict[str, Any]],
    written_ids: list[str],
) -> list[dict[str, Any]]:
    common = _common(run_id, task_id, snapshot_id, group)
    selected = [row for row in candidates if row["memory_id"] in selected_ids]
    action = "lookup_order(order_id='1')"
    return [
        {
            **common,
            "event_type": "EpisodeStarted",
            "observation": "Customer needs retail help.",
            "policy": "Follow the retail policy.",
            "tools": [{"type": "function", "function": {"name": "lookup_order"}}],
        },
        {
            **common,
            "event_type": "MemoryCandidatesRetrieved",
            "query_hash": hashlib.sha256(task_id.encode()).hexdigest(),
            "retriever_revision": "embedding-a",
            "candidates": candidates,
        },
        {
            **common,
            "event_type": "MemorySelected",
            "selected_memory_ids": selected_ids,
            "selected": selected,
        },
        {
            **common,
            "event_type": "DecisionMade",
            "turn": 0,
            "observation": "Customer needs retail help.",
            "parsed_action": action,
        },
        {
            **common,
            "event_type": "EnvironmentStepped",
            "turn": 0,
            "action": action,
            "observation": "Request complete.",
            "reward": reward,
            "done": True,
            "terminated": True,
            "truncated": False,
            "public_info": {},
        },
        {
            **common,
            "event_type": "EpisodeFinished",
            "steps": 1,
            "final_reward": reward,
            "terminal_evaluation": {},
            "simulation_result": {},
            "truncated": False,
        },
        {**common, "event_type": "MemoryWriteProposed", "proposals": proposals},
        {
            **common,
            "event_type": "MemoryWriteCommitted",
            "written_memory_ids": written_ids,
            "replayed_memory_ids": [],
        },
    ]


def _maintenance_events(
    repository: MemoryRepository,
    *,
    run_id: str,
    snapshot_id: str,
) -> list[dict[str, Any]]:
    common = _common(
        run_id,
        "maintenance-round-1",
        snapshot_id,
        "retail-actions-v1:maintenance",
    )
    common["maintenance_round"] = 1
    diagnostics = {
        tier: {"items": []} for tier in ("trajectory", "tip", "skill", "tool")
    }
    for item in repository.list(status=None):
        diagnostics[item.tier.value]["items"].append(
            {
                "id": item.id,
                "tier": item.tier.value,
                "content": item.content,
                "version": item.version,
                "status": item.status.value,
            }
        )
    return [
        {
            **common,
            "event_type": "MaintenanceStarted",
            "completed_train_tasks": 30,
            "period": 30,
            "diagnostics": diagnostics,
        },
        {**common, "event_type": "MaintenanceProposed", "commands": []},
        {
            **common,
            "event_type": "MaintenanceCommitted",
            "looked_up_ids": [],
            "created_ids": [],
            "updated_ids": [],
            "completed_rounds": [1],
        },
    ]


def _common(
    run_id: str,
    task_id: str,
    snapshot_id: str,
    group: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "iteration": 0,
        "split": "train",
        "mode": "learn",
        "task_id": task_id,
        "task_group": group,
        "model_revision": MODEL_REVISION,
        "adapter_revision": ADAPTER_REVISION,
        "memory_snapshot_id": snapshot_id,
        "seed": 17,
    }


def _content_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.jsonl"))
    }


def _jsonl_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
