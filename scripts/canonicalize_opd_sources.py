from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.memory.paths import project_root as default_project_root
from tau3_retail_evolver.memory.paths import training_memory_root
from tau3_retail_evolver.slow_loop.canonicalize import (
    CanonicalizeRequest,
    canonicalize_opd_sources,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an audited, deduplicated OPSD source from fragmented train runs."
        )
    )
    parser.add_argument(
        "--source-run",
        dest="source_runs",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--maintenance-events",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--final-memory-snapshot", required=True)
    parser.add_argument(
        "--expected-seed",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--skip-evidence-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project = (args.project_root or default_project_root()).resolve()
    config_path = _resolve(args.config, project)
    config = load_config(config_path)
    runtime = Tau2Runtime.inspect_metadata(_resolve(config.tau2.repo_path, project))
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.retail_tasks_path,
        runtime.retail_split_path,
    )
    catalog.require_official_compatibility()
    expected_seeds = tuple(args.expected_seed or (42, 43, 44))
    result = canonicalize_opd_sources(
        CanonicalizeRequest(
            source_run_paths=tuple(_resolve(path, project) for path in args.source_runs),
            maintenance_event_paths=tuple(
                _resolve(path, project) for path in args.maintenance_events
            ),
            output_root=_resolve(args.output_root, project),
            build_id=args.build_id,
            final_memory_snapshot_id=args.final_memory_snapshot,
            maintenance_period=config.memory.maintenance_period,
            expected_seeds=expected_seeds,
            catalog=catalog,
            memory_root=training_memory_root(config.memory.agent_id, root=project),
            deep_validate=not args.skip_evidence_validation,
        )
    )
    summary = {
        "build_id": args.build_id,
        "canonical_root": str(result.root),
        "coverage": result.index["coverage"],
        "index_path": str(result.index_path),
        "source_runs": [str(path) for path in result.source_run_paths],
        "validation": result.index["validation"],
    }
    sys.stdout.write(
        json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    return 0


def _resolve(path: Path, project: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (project / value).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
