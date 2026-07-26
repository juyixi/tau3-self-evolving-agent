from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import build_stage8_report
from tau3_retail_evolver.eval.experiment import (
    BASE_NO_MEMORY,
    BASE_WITH_MEMORY,
    OPD_NO_MEMORY,
    OPD_WITH_MEMORY,
)


def _arguments(tmp_path: Path) -> list[str]:
    return [
        "--experiment-id",
        "stage8-main",
        "--base-no-memory-report",
        "a.json",
        "--base-with-memory-report",
        "b.json",
        "--opd-with-memory-report",
        "c.json",
        "--opd-no-memory-report",
        "d.json",
        "--train-run",
        "train-1",
        "--train-run",
        "train-2",
        "--train-run",
        "train-3",
        "--dataset-dir",
        "dataset",
        "--training-dir",
        "training",
        "--memory-snapshot",
        "snapshot",
        "--output-dir",
        str(tmp_path / "report"),
    ]


def test_parser_defaults_to_three_train_passes(tmp_path: Path) -> None:
    args = build_stage8_report.parse_args(_arguments(tmp_path))

    assert args.train_passes == 3
    assert args.bootstrap_samples == 2_000
    assert len(args.train_runs) == 3


def test_main_writes_json_and_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, Any] = {}
    fake_report = {
        "evaluation": {
            "cells": {
                BASE_NO_MEMORY: {"pass_at_1": 0.25},
                BASE_WITH_MEMORY: {"pass_at_1": 0.5},
                OPD_WITH_MEMORY: {"pass_at_1": 0.75},
                OPD_NO_MEMORY: {"pass_at_1": 0.5},
            }
        }
    }

    monkeypatch.setattr(
        build_stage8_report,
        "load_labeled_evaluation_reports",
        lambda paths: captured.setdefault("report_paths", paths) or {},
    )

    def build(**kwargs: Any) -> dict[str, Any]:
        captured["build"] = kwargs
        return fake_report

    monkeypatch.setattr(
        build_stage8_report,
        "build_stage8_experiment_report",
        build,
    )

    def write_report(path: Path, report: Any) -> None:
        captured["report_path"] = path
        path.write_text("{}\n", encoding="utf-8")

    def write_dashboard(path: Path, report: Any) -> None:
        captured["dashboard_path"] = path
        path.write_text("<html></html>\n", encoding="utf-8")

    monkeypatch.setattr(
        build_stage8_report,
        "write_stage8_experiment_report",
        write_report,
    )
    monkeypatch.setattr(
        build_stage8_report,
        "write_stage8_dashboard",
        write_dashboard,
    )

    result = build_stage8_report.main(_arguments(tmp_path))
    output = json.loads(capsys.readouterr().out)

    assert result == 0
    assert set(captured["report_paths"]) == {
        BASE_NO_MEMORY,
        BASE_WITH_MEMORY,
        OPD_WITH_MEMORY,
        OPD_NO_MEMORY,
    }
    assert captured["build"]["expected_train_passes"] == 3
    assert captured["report_path"].name == "stage8_experiment_report.json"
    assert captured["dashboard_path"].name == "stage8_dashboard.html"
    assert output["pass_at_1"][OPD_WITH_MEMORY] == 0.75
