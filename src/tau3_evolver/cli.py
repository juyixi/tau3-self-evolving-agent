from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from tau3_evolver.execution.request import ExecutionRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tau3")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run one benchmark task set.")
    run.add_argument("--benchmark", choices=("retail", "airline"), required=True)
    run.add_argument("--mode", choices=("train", "test"), required=True)
    run.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Run a deterministic split subset sized to execution.max_concurrency."
        ),
    )
    memory = run.add_mutually_exclusive_group(required=True)
    memory.add_argument("--memory", dest="memory_enabled", action="store_true")
    memory.add_argument("--no-memory", dest="memory_enabled", action="store_false")
    run.add_argument("--memory-source")
    run.add_argument("--memory-snapshot", type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--config", dest="config_path", type=Path, default=Path("configs/default.yaml"))
    run.add_argument("--set", dest="overrides", action="append", default=[])
    run.add_argument("--run-id", required=True)
    run.add_argument("--output-root", type=Path, default=Path("runs"))

    slow_loop = commands.add_parser(
        "slow-loop", help="Run manually initiated offline data or training work."
    )
    slow_loop.add_argument(
        "action", choices=("build", "audit", "train"), help="Offline operation."
    )
    slow_loop.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def parse_execution_request(argv: Sequence[str]) -> ExecutionRequest:
    namespace = build_parser().parse_args(list(argv))
    if namespace.command != "run":
        raise ValueError("expected the run command")
    return ExecutionRequest.model_validate(
        {
            "benchmark": namespace.benchmark,
            "mode": namespace.mode,
            "debug": namespace.debug,
            "memory_enabled": namespace.memory_enabled,
            "memory_source": namespace.memory_source,
            "memory_snapshot": namespace.memory_snapshot,
            "checkpoint": namespace.checkpoint,
            "config_path": namespace.config_path,
            "overrides": tuple(namespace.overrides),
            "run_id": namespace.run_id,
            "output_root": namespace.output_root,
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    namespace = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if namespace.command == "run":
            request = ExecutionRequest.model_validate(
                {
                    "benchmark": namespace.benchmark,
                    "mode": namespace.mode,
                    "debug": namespace.debug,
                    "memory_enabled": namespace.memory_enabled,
                    "memory_source": namespace.memory_source,
                    "memory_snapshot": namespace.memory_snapshot,
                    "checkpoint": namespace.checkpoint,
                    "config_path": namespace.config_path,
                    "overrides": tuple(namespace.overrides),
                    "run_id": namespace.run_id,
                    "output_root": namespace.output_root,
                }
            )
            from tau3_evolver.execution.runner import execute

            result = execute(request)
            return 0 if result.successful else 1

        from tau3_evolver.slow_loop.runner import execute_slow_loop

        return execute_slow_loop(namespace.action, namespace.arguments)
    except (ValidationError, ValueError) as error:
        parser.error(str(error))
    return 2
