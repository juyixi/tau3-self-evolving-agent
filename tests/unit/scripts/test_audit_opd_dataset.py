from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_opd_dataset as audit_script
from tau3_retail_evolver.slow_loop.audit import AuditError, AuditReport


def test_audit_cli_returns_nonzero_when_report_fails(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    report = AuditReport(
        dataset_build_id="opd-iter0-001",
        passed=False,
        checked_artifacts=("datasets/sel.jsonl",),
        errors=(AuditError(code="artifact_hash_mismatch", message="changed"),),
    )
    monkeypatch.setattr(audit_script, "audit_dataset", lambda path, **kwargs: report)

    exit_code = audit_script.main(["--dataset-dir", str(tmp_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output == report.model_dump(mode="json")


def test_audit_cli_returns_zero_when_report_passes(monkeypatch, capsys, tmp_path: Path) -> None:
    report = AuditReport(
        dataset_build_id="opd-iter0-001",
        passed=True,
        checked_artifacts=(),
        errors=(),
    )
    monkeypatch.setattr(audit_script, "audit_dataset", lambda path, **kwargs: report)

    assert audit_script.main(["--dataset-dir", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_audit_cli_passes_explicit_project_root(monkeypatch, capsys, tmp_path: Path) -> None:
    report = AuditReport(
        dataset_build_id="opd-iter0-001",
        passed=True,
        checked_artifacts=(),
        errors=(),
    )
    received: dict[str, Path | None] = {}

    def fake_audit(path: Path, *, project_root: Path | None = None) -> AuditReport:
        received["dataset_dir"] = path
        received["project_root"] = project_root
        return report

    monkeypatch.setattr(audit_script, "audit_dataset", fake_audit)
    project_root = tmp_path / "project"
    dataset_dir = tmp_path / "dataset"

    assert audit_script.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--project-root",
            str(project_root),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    assert received == {
        "dataset_dir": dataset_dir,
        "project_root": project_root,
    }
