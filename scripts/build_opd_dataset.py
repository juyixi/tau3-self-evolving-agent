from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from tau3_retail_evolver.slow_loop.dataset import (
    DatasetBuildRequest,
    build_opd_dataset,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Stage 5 retail OPD dataset."
    )
    parser.add_argument("--source-run", dest="source_runs", type=Path, action="append", required=True)
    parser.add_argument("--dataset-build-id", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--project-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_opd_dataset(
        DatasetBuildRequest(
            source_run_paths=tuple(args.source_runs),
            dataset_build_id=args.dataset_build_id,
            output_root=args.output_root,
            config_path=args.config,
            project_root=args.project_root,
        )
    )
    summary = {
        "audit_passed": result.audit_report.get("passed") is True,
        "counts": result.manifest.get("counts", {}),
        "dataset_build_id": result.manifest.get("dataset_build_id"),
        "dataset_dir": str(result.dataset_dir),
    }
    sys.stdout.write(_canonical_json(summary))
    return 0


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

