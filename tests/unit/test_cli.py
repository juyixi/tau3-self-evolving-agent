from pathlib import Path

import pytest

from tau3_evolver.cli import parse_execution_request


def test_run_cli_builds_typed_request_without_task_or_iteration_fields() -> None:
    request = parse_execution_request(
        [
            "run",
            "--benchmark",
            "airline",
            "--mode",
            "test",
            "--memory",
            "--memory-source",
            "retail",
            "--memory-snapshot",
            "snapshots/s1",
            "--checkpoint",
            "checkpoints/opd",
            "--run-id",
            "generalization-1",
        ]
    )

    assert request.benchmark.value == "airline"
    assert request.mode.value == "test"
    assert request.memory_source == "retail"
    assert request.memory_snapshot == Path("snapshots/s1")
    assert not hasattr(request, "task_id")
    assert not hasattr(request, "iteration")


@pytest.mark.parametrize("argument", ("--task-id", "--iteration", "--all-tasks"))
def test_run_cli_does_not_publish_legacy_selection_arguments(argument: str) -> None:
    with pytest.raises(SystemExit):
        parse_execution_request(
            [
                "run",
                "--benchmark",
                "retail",
                "--mode",
                "train",
                "--no-memory",
                "--run-id",
                "run-1",
                argument,
                "value",
            ]
        )
