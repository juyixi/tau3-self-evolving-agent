from __future__ import annotations

import json
from pathlib import Path

from scripts import build_opd_dataset as build_script
from tau3_retail_evolver.slow_loop.dataset import DatasetBuildResult


def test_build_cli_requires_explicit_source_runs_and_prints_canonical_summary(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_build(request):
        captured["request"] = request
        return DatasetBuildResult(
            dataset_dir=tmp_path / "runs" / request.dataset_build_id / "slow_loop",
            manifest={"dataset_build_id": request.dataset_build_id, "counts": {"sel": 2}},
            audit_report={"passed": True},
        )

    monkeypatch.setattr(build_script, "build_opd_dataset", fake_build)

    exit_code = build_script.main(
        [
            "--config",
            "configs/default.yaml",
            "--source-run",
            "runs/a",
            "--source-run",
            "runs/b",
            "--dataset-build-id",
            "opd-iter0-001",
            "--output-root",
            str(tmp_path / "runs"),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary == {
        "audit_passed": True,
        "counts": {"sel": 2},
        "dataset_build_id": "opd-iter0-001",
        "dataset_dir": str(tmp_path / "runs" / "opd-iter0-001" / "slow_loop"),
    }
    assert captured["request"].source_run_paths == (Path("runs/a"), Path("runs/b"))
