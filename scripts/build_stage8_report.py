from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from tau3_retail_evolver.eval.experiment import (
    BASE_NO_MEMORY,
    BASE_WITH_MEMORY,
    DEFAULT_TRAIN_PASSES,
    OPD_NO_MEMORY,
    OPD_WITH_MEMORY,
    build_stage8_experiment_report,
    load_labeled_evaluation_reports,
    write_stage8_experiment_report,
)
from tau3_retail_evolver.eval.visualization import write_stage8_dashboard


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the controlled Stage 8 report and HTML charts."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--base-no-memory-report", type=Path, required=True)
    parser.add_argument("--base-with-memory-report", type=Path, required=True)
    parser.add_argument("--opd-with-memory-report", type=Path, required=True)
    parser.add_argument("--opd-no-memory-report", type=Path, required=True)
    parser.add_argument(
        "--train-run",
        dest="train_runs",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--memory-snapshot", type=Path, required=True)
    parser.add_argument("--train-passes", type=int, default=DEFAULT_TRAIN_PASSES)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite Stage 8 report directory: {output_dir}"
        )
    reports = load_labeled_evaluation_reports(
        {
            BASE_NO_MEMORY: args.base_no_memory_report,
            BASE_WITH_MEMORY: args.base_with_memory_report,
            OPD_WITH_MEMORY: args.opd_with_memory_report,
            OPD_NO_MEMORY: args.opd_no_memory_report,
        }
    )
    report = build_stage8_experiment_report(
        experiment_id=args.experiment_id,
        evaluation_reports=reports,
        train_run_dirs=args.train_runs,
        dataset_dir=args.dataset_dir,
        training_dir=args.training_dir,
        memory_snapshot_path=args.memory_snapshot,
        expected_train_passes=args.train_passes,
        bootstrap_samples=args.bootstrap_samples,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage8_experiment_report.json"
    dashboard_path = output_dir / "stage8_dashboard.html"
    write_stage8_experiment_report(report_path, report)
    write_stage8_dashboard(dashboard_path, report)
    sys.stdout.write(
        json.dumps(
            {
                "experiment_id": args.experiment_id,
                "report": str(report_path),
                "dashboard": str(dashboard_path),
                "pass_at_1": {
                    label: cell["pass_at_1"]
                    for label, cell in report["evaluation"]["cells"].items()
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
