from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any

from tau3_retail_evolver.config import ProjectConfig, load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.memory.factory import open_training_memory
from tau3_retail_evolver.pipeline.executor import CommandIterationExecutor
from tau3_retail_evolver.pipeline.iteration import IterationRequest, run_iteration
from tau3_retail_evolver.pipeline.sampling import select_train_tasks
from tau3_retail_evolver.pipeline.state import IterationState


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable train-only fast/slow OPD iteration."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/iterations"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--parent-iteration-dir", type=Path)
    parser.add_argument("--completed-train-tasks-before", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--task-id", dest="task_ids", action="append", default=[])
    parser.add_argument("--qwen-base-url")
    parser.add_argument(
        "--stop-after",
        choices=[
            IterationState.ROLLOUT_COMPLETE.value,
            IterationState.ATTRIBUTION_COMPLETE.value,
            IterationState.DATASET_COMPLETE.value,
            IterationState.TRAINING_COMPLETE.value,
        ],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    request = _build_request(args)
    result = run_iteration(
        request,
        CommandIterationExecutor(),
        stop_after=(IterationState(args.stop_after) if args.stop_after else None),
    )
    promotion = result.promotion_manifest
    _print_json(
        {
            "adapter_revision": (
                promotion["child"]["adapter_revision"]
                if promotion is not None
                else request.adapter_revision
            ),
            "iteration": request.iteration,
            "iteration_dir": str(result.iteration_dir),
            "iteration_id": request.iteration_id,
            "memory_snapshot_id": (
                promotion["memory"]["output_snapshot_id"]
                if promotion is not None
                else None
            ),
            "state": result.state.value,
        }
    )
    return 0


def _build_request(args: argparse.Namespace) -> IterationRequest:
    project_root = (args.project_root or Path.cwd()).resolve()
    config_path = _resolve(project_root, args.config)
    output_root = _resolve(project_root, args.output_root)
    iteration_dir = output_root / args.iteration_id
    persisted_input_snapshot = _persisted_input_memory_snapshot(iteration_dir)
    config = load_config(config_path)
    official_train_task_ids = _load_official_train_tasks(config, project_root)
    task_count = args.task_count or config.pipeline.iteration_task_count
    task_ids = select_train_tasks(
        official_train_task_ids,
        task_count=task_count,
        seed=config.training.seed,
        iteration=args.iteration,
        shuffle=config.pipeline.shuffle_train_tasks,
        explicit_task_ids=tuple(args.task_ids),
    )
    current_snapshot = open_training_memory(
        config.memory,
        root=project_root,
    ).snapshot()

    parent_iteration_dir: Path | None = None
    parent_checkpoint: Path | None = None
    adapter_revision = args.adapter_revision
    completed_before = args.completed_train_tasks_before
    input_snapshot = (
        persisted_input_snapshot or current_snapshot.memory_snapshot_id
    )
    if args.parent_iteration_dir is not None:
        parent_iteration_dir = _resolve(project_root, args.parent_iteration_dir)
        promotion = _read_json(parent_iteration_dir / "promotion_manifest.json")
        child = promotion.get("child")
        memory = promotion.get("memory")
        if not isinstance(child, Mapping) or not isinstance(memory, Mapping):
            raise ValueError("parent promotion lineage is incomplete")
        inherited_adapter = _nonblank(child.get("adapter_revision"), "parent adapter revision")
        if adapter_revision is not None and adapter_revision != inherited_adapter:
            raise ValueError("--adapter-revision does not match promoted parent")
        adapter_revision = inherited_adapter
        parent_checkpoint = Path(
            _nonblank(child.get("checkpoint"), "parent checkpoint")
        )
        inherited_completed = promotion.get("completed_train_tasks_after")
        if type(inherited_completed) is not int or inherited_completed < 0:
            raise ValueError("parent completed task count is invalid")
        if completed_before is not None and completed_before != inherited_completed:
            raise ValueError(
                "--completed-train-tasks-before does not match promoted parent"
            )
        completed_before = inherited_completed
        inherited_snapshot = _nonblank(
            memory.get("output_snapshot_id"),
            "parent memory snapshot",
        )
        if (
            persisted_input_snapshot is None
            and current_snapshot.memory_snapshot_id != inherited_snapshot
        ):
            raise ValueError("current Memory repository does not match promoted parent snapshot")
        if (
            persisted_input_snapshot is not None
            and persisted_input_snapshot != inherited_snapshot
        ):
            raise ValueError(
                "persisted iteration Memory snapshot does not match promoted parent"
            )
        input_snapshot = inherited_snapshot
    else:
        if args.iteration != 0:
            raise ValueError("nonzero iteration requires --parent-iteration-dir")
        if completed_before is None:
            raise ValueError(
                "initial iteration requires --completed-train-tasks-before"
            )

    adapter_revision = _nonblank(adapter_revision, "adapter revision")
    assert completed_before is not None
    return IterationRequest(
        iteration_id=args.iteration_id,
        iteration=args.iteration,
        iteration_dir=iteration_dir,
        project_root=project_root,
        config_path=config_path,
        model_revision=_nonblank(args.model_revision, "model revision"),
        adapter_revision=adapter_revision,
        parent_checkpoint=parent_checkpoint,
        parent_iteration_dir=parent_iteration_dir,
        input_memory_snapshot_id=input_snapshot,
        memory_snapshots_dir=current_snapshot.path.parent,
        task_ids=task_ids,
        official_train_task_ids=tuple(official_train_task_ids),
        completed_train_tasks_before=completed_before,
        qwen_base_url=args.qwen_base_url or os.environ.get("QWEN_BASE_URL"),
    )


def _load_official_train_tasks(
    config: ProjectConfig,
    project_root: Path,
) -> tuple[str, ...]:
    tau2_path = _resolve(project_root, config.tau2.repo_path)
    runtime = Tau2Runtime.inspect_metadata(tau2_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.retail_tasks_path,
        runtime.retail_split_path,
    )
    catalog.require_official_compatibility()
    return catalog.task_ids("train")


def _resolve(project_root: Path, path: Path) -> Path:
    candidate = Path(path).expanduser()
    return (
        (project_root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read parent promotion manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"parent promotion manifest must be an object: {path}")
    return value


def _persisted_input_memory_snapshot(iteration_dir: Path) -> str | None:
    state_path = iteration_dir / "iteration_state.json"
    if not state_path.exists():
        return None
    record = _read_json(state_path)
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError(f"iteration identity is missing: {state_path}")
    return _nonblank(
        identity.get("input_memory_snapshot_id"),
        "persisted input memory snapshot",
    )


def _nonblank(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _print_json(value: Mapping[str, Any]) -> None:
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


if __name__ == "__main__":
    raise SystemExit(main())
