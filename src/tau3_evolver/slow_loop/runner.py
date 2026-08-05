from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from tau3_evolver.slow_loop.audit import audit_dataset
from tau3_evolver.slow_loop.dataset import DatasetBuildRequest, build_opd_dataset


def execute_slow_loop(action: str, argv: Sequence[str]) -> int:
    """Dispatch one manually initiated offline Slow Loop operation."""
    if action == "build":
        return _build(argv)
    if action == "audit":
        return _audit(argv)
    if action == "train":
        from tau3_evolver.slow_loop.training_suite import main

        return main(argv)
    raise ValueError(f"unknown slow-loop action: {action}")


def _build(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="tau3 slow-loop build")
    parser.add_argument(
        "--source-run", dest="source_runs", type=Path, action="append", required=True
    )
    parser.add_argument("--dataset-build-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(list(argv))
    result = build_opd_dataset(
        DatasetBuildRequest(
            source_run_paths=tuple(args.source_runs),
            dataset_build_id=args.dataset_build_id,
            output_root=args.output_root,
            config_path=args.config,
            project_root=args.project_root,
        )
    )
    _print_json(
        {
            "audit_passed": result.audit_report.get("passed") is True,
            "counts": result.manifest.get("counts", {}),
            "dataset_build_id": result.manifest.get("dataset_build_id"),
            "dataset_dir": str(result.dataset_dir),
        }
    )
    return 0


def _audit(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="tau3 slow-loop audit")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(list(argv))
    report = audit_dataset(args.dataset_dir, project_root=args.project_root)
    _print_json(report.model_dump(mode="json"))
    return 0 if report.passed else 1


def _print_json(value: object) -> None:
    sys.stdout.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
