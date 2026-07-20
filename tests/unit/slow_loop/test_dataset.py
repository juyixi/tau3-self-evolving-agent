from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tau3_retail_evolver.envs.task_catalog import OFFICIAL_SPLIT_SHA256
from tau3_retail_evolver.slow_loop import dataset as dataset_module
from tau3_retail_evolver.slow_loop.attribution import MemoryScore
from tau3_retail_evolver.slow_loop.audit import audit_dataset
from tau3_retail_evolver.slow_loop.audit import AuditError, AuditReport
from tau3_retail_evolver.slow_loop.dataset import (
    DatasetBuildRequest,
    build_opd_dataset,
)
from tau3_retail_evolver.slow_loop.evidence import (
    EpisodeEvidence,
    EvidenceLedger,
    MemoryCandidateEvidence,
    TrajectoryStepEvidence,
)
from tau3_retail_evolver.slow_loop.examples import ONLINE_SAMPLING_CONTRACT, OPDExample


def _episode() -> EpisodeEvidence:
    content = "Check the order before changing it."
    return EpisodeEvidence(
        episode_id="run-a:1",
        run_id="run-a",
        source_event_start=1,
        source_event_end=8,
        source_event_sha256="e" * 64,
        iteration=0,
        task_id="1",
        task_group="retail-actions-v1:" + "a" * 64,
        model_revision="model-a",
        adapter_revision="adapter-a",
        tau2_commit="c" * 40,
        split_hash=OFFICIAL_SPLIT_SHA256,
        memory_agent_id="retail",
        memory_snapshot_id="snapshot-a",
        seed=42,
        policy="Public policy.",
        tools=(),
        initial_observation="Help with order 1.",
        query_hash="q" * 64,
        retriever_revision="embedding-a",
        candidates=(
            MemoryCandidateEvidence(
                memory_id="mem-a",
                memory_version=1,
                tier="tip",
                rank=1,
                similarity=0.9,
                content=content,
                content_sha256=hashlib.sha256(content.encode()).hexdigest(),
            ),
        ),
        selected_memory_ids=("mem-a",),
        trajectory=(
            TrajectoryStepEvidence(
                turn=0,
                observation="Help with order 1.",
                action="lookup_order",
                next_observation="Done.",
                reward=1.0,
                done=True,
                terminated=True,
                truncated=False,
                public_info={},
            ),
        ),
        terminal_evaluation={},
        simulation_result={},
        final_reward=1.0,
        terminated=True,
        truncated=False,
        write_proposals=(),
        proposed_memory_ids=(),
        committed_new_memory_ids=(),
        replayed_memory_ids=(),
    )


def _example(kind: str) -> OPDExample:
    public = {"observation": "Public state."}
    privileged: dict[str, object] = {}
    if kind == "sel":
        public["candidates"] = [
            {"memory_id": "mem-a", "tier": "tip", "content": "Public tip."}
        ]
        privileged = {"candidate_scores": [{"memory_id": "mem-a", "value": None}]}
    elif kind == "write":
        privileged = {"written_memory_scores": []}
    elif kind == "maint":
        public = {"repository": [{"id": "mem-a", "content": "Public tip."}]}
        privileged = {
            "memory_diagnostics": [{"memory_id": "mem-a", "value": None}],
            "redundancy_pairs": [],
        }
    return OPDExample(
        example_id=f"opd_{kind}_fixture",
        kind=kind,
        public_input=public,
        privileged_hindsight=privileged,
        response_schema={"type": "object"},
        sampling_contract=ONLINE_SAMPLING_CONTRACT,
        provenance={"episode_id": "run-a:1", "task_id": "1"},
    )


def _materialized() -> dataset_module._MaterializedDataset:
    episode = _episode()
    ledger = EvidenceLedger(
        iteration=0,
        model_revision="model-a",
        adapter_revision="adapter-a",
        tau2_commit="c" * 40,
        split_hash=OFFICIAL_SPLIT_SHA256,
        memory_agent_id="retail",
        source_run_ids=("run-a",),
        episodes=(episode,),
        maintenance=(),
    )
    score = MemoryScore(
        memory_id="mem-a",
        tier="tip",
        observed_versions=(1,),
        creator_episode_id=None,
        source_episode_ids=(episode.episode_id,),
        groups=(),
        retrieved_count=1,
        selected_count=1,
        not_selected_count=0,
        confidence=1.0 - 1.0 / 2**0.5,
        tier_prior=0.8,
        attribution=None,
        value=None,
        status="insufficient_evidence",
        qualified_for_supervision=False,
    )
    return dataset_module._MaterializedDataset(
        ledger=ledger,
        scores=(score,),
        examples={kind: (_example(kind),) for kind in ("sel", "act", "write", "maint")},
        source_runs=(
            {
                "run_id": "run-a",
                "task_ids": ["1"],
                "manifest_sha256": "1" * 64,
                "events_sha256": "2" * 64,
                "summary_sha256": "3" * 64,
                "input_memory_snapshot_id": "snapshot-a",
                "output_memory_snapshot_id": "snapshot-b",
                "completed_train_tasks_before": 0,
                "completed_train_tasks_after": 1,
            },
        ),
        official_train_task_ids=tuple(str(index) for index in range(74)),
        snapshot_chain=("snapshot-a", "snapshot-b"),
        resolved_config={
            "tier_priors": {"trajectory": 0.9, "tip": 0.8, "skill": 1.0, "tool": 1.2},
            "score_threshold": 0.01,
            "teacher_memory_cap": 20,
            "redundancy_threshold": 0.9,
            "max_redundancy_pairs": 50,
        },
        build_code_revision="b" * 40,
    )


def _request(tmp_path: Path) -> DatasetBuildRequest:
    return DatasetBuildRequest(
        source_run_paths=(tmp_path / "source-run",),
        dataset_build_id="opd-iter0-001",
        output_root=tmp_path / "runs",
        config_path=tmp_path / "config.yaml",
        project_root=tmp_path,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(dataset_module, "_materialize", lambda request: _materialized())
    return build_opd_dataset(_request(tmp_path)).dataset_dir


def test_dataset_build_writes_canonical_files_and_manifest_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _built(tmp_path, monkeypatch)

    assert (root / "evidence" / "episodes.jsonl").is_file()
    assert (root / "attribution" / "memory_scores.jsonl").is_file()
    assert all(
        (root / "datasets" / f"{kind}.jsonl").is_file()
        for kind in ("sel", "act", "write", "maint")
    )
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for relative_path, artifact in manifest["artifacts"].items():
        assert _sha256(root / relative_path) == artifact["sha256"]
        assert len((root / relative_path).read_text(encoding="utf-8").splitlines()) == artifact["line_count"]
    assert audit_dataset(root).passed is True


def test_dataset_build_refuses_existing_build_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataset_module, "_materialize", lambda request: _materialized())
    request = _request(tmp_path)
    build_opd_dataset(request)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_opd_dataset(request)


def test_dataset_build_does_not_publish_when_internal_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataset_module, "_materialize", lambda request: _materialized())
    monkeypatch.setattr(
        dataset_module,
        "audit_dataset",
        lambda path: AuditReport(
            dataset_build_id="opd-iter0-001",
            passed=False,
            checked_artifacts=(),
            errors=(AuditError(code="forced_failure", message="failed"),),
        ),
    )

    with pytest.raises(ValueError, match="audit failed"):
        build_opd_dataset(_request(tmp_path))

    output_root = tmp_path / "runs"
    assert not (output_root / "opd-iter0-001").exists()
    assert not list(output_root.glob(".opd-iter0-001.tmp-*"))


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("duplicate_example", "duplicate_example_id"),
        ("public_value_leak", "public_privileged_leak"),
        ("artifact_hash_changed", "artifact_hash_mismatch"),
        ("missing_online_contract", "missing_online_sampling_contract"),
        ("test_task_id", "non_train_source"),
        ("manifest_count", "manifest_count_mismatch"),
        ("broken_snapshot_chain", "source_lineage_invalid"),
        ("tampered_score", "attribution_recompute_mismatch"),
        ("privileged_score", "example_score_mismatch"),
    ],
)
def test_auditor_fails_closed_on_mutated_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error_code: str,
) -> None:
    root = _built(tmp_path, monkeypatch)
    selection = root / "datasets" / "sel.jsonl"
    evidence = root / "evidence" / "episodes.jsonl"
    manifest_path = root / "dataset_manifest.json"
    if mutation == "duplicate_example":
        selection.write_bytes(selection.read_bytes() + selection.read_bytes())
    elif mutation == "public_value_leak":
        row = json.loads(selection.read_text(encoding="utf-8"))
        row["public_input"]["memory_value"] = 0.8
        selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif mutation == "artifact_hash_changed":
        selection.write_bytes(selection.read_bytes() + b"\n")
    elif mutation == "missing_online_contract":
        row = json.loads(selection.read_text(encoding="utf-8"))
        row["sampling_contract"] = {}
        selection.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif mutation == "test_task_id":
        row = json.loads(evidence.read_text(encoding="utf-8"))
        row["task_id"] = "test-only"
        evidence.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif mutation == "manifest_count":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["counts"]["sel"] = 999
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    elif mutation == "broken_snapshot_chain":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["memory"]["snapshot_chain"] = ["wrong", "snapshot-b"]
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    elif mutation == "tampered_score":
        score_path = root / "attribution" / "memory_scores.jsonl"
        row = json.loads(score_path.read_text(encoding="utf-8"))
        row["attribution"] = 1.0
        row["value"] = 0.8
        row["status"] = "scored"
        row["qualified_for_supervision"] = True
        score_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    elif mutation == "privileged_score":
        row = json.loads(selection.read_text(encoding="utf-8"))
        row["privileged_hindsight"]["candidate_scores"][0]["value"] = 9.0
        selection.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = audit_dataset(root)

    assert report.passed is False
    assert error_code in {error.code for error in report.errors}
