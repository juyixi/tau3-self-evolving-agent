from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlsplit

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.split_guard import require_learning_split
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.envs.tau2_retail import Tau2RetailEnv
from tau3_retail_evolver.evaluation.tau2_nl_assertions import bind_tau2_nl_assertions
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.maintenance import run_due_maintenance
from tau3_retail_evolver.fast_loop.runner import (
    EpisodeResult,
    FastLoopConfig,
    run_fast_loop_episode,
)
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import build_embedding_provider
from tau3_retail_evolver.memory.factory import open_training_memory
from tau3_retail_evolver.memory.json_store import write_bytes_atomic
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever
from tau3_retail_evolver.models.openai_compatible import (
    OpenAICompatibleFastLoopPolicy,
    OpenAICompatibleHttpClient,
)
from tau3_retail_evolver.runs.manifest import create_manifest


MODEL_SERVING_CONTRACT = {
    "language_model_only": True,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "enable_thinking": True,
    "max_tokens": 8192,
    "top_k": 20,
    "presence_penalty": 1.5,
    "parallel_tool_calls": False,
}

QWEN_GENERATION_SETTINGS = {
    "chat_template_kwargs": {"enable_thinking": True},
    "top_k": 20,
    "presence_penalty": 1.5,
    "parallel_tool_calls": False,
}

TASK_INSTRUCTION = "Resolve the retail request shown in the current conversation."
_SAFE_SLUG = re.compile(r"^[a-z0-9_-]+$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical Qwen retail fast-loop learning.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-id", dest="task_ids", action="append", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--completed-train-tasks-before", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_learning_split(args.split)
    _validate_early_arguments(args)

    run_path = args.output_root / args.run_id
    if run_path.exists():
        raise FileExistsError(f"refusing to reuse existing run: {run_path}")

    config = load_config(args.config)
    runtime = Tau2Runtime.inspect_metadata(config.tau2.repo_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(runtime.retail_tasks_path, runtime.retail_split_path)
    catalog.require_official_compatibility()
    _require_explicit_train_tasks(args.task_ids, catalog.task_ids("train"))

    base_url = _validate_qwen_base_url(args.qwen_base_url or os.environ.get("QWEN_BASE_URL"))
    if not base_url:
        raise ValueError("QWEN_BASE_URL or --qwen-base-url is required")

    gym_factory = Tau2Runtime.load_verified_gym_factory(runtime.repo_path)
    evaluation_provenance = bind_tau2_nl_assertions(config.evaluation.nl_assertions)
    probe = Tau2RetailEnv(args.task_ids[0], config, gym_factory=gym_factory)
    try:
        user_simulator_config = probe.user_simulator_config
    finally:
        probe.close()

    repository = open_training_memory(config.memory, root=args.project_root)
    embedding_provider = build_embedding_provider(config.memory, repository.root)
    retriever = Retriever(embedding_provider)
    input_snapshot = repository.snapshot()

    client = OpenAICompatibleHttpClient(
        base_url=base_url,
        model=config.model.base_model,
        api_key=os.environ.get("QWEN_API_KEY") or "EMPTY",
        max_tokens=MODEL_SERVING_CONTRACT["max_tokens"],
        generation_settings=QWEN_GENERATION_SETTINGS,
    )
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
    )

    create_manifest(
        run_path / "manifest.json",
        run_id=args.run_id,
        iteration=args.iteration,
        model_revision=args.model_revision,
        parent_checkpoint=None,
        adapter_revision=args.adapter_revision,
        memory_snapshot_id=input_snapshot.memory_snapshot_id,
        tau2_commit=runtime.git_commit,
        split=args.split,
        split_hash=catalog.split_sha256,
        task_ids=tuple(args.task_ids),
        seed=config.training.seed,
        user_simulator_config=user_simulator_config,
        environment_options={
            "domain": config.tau2.domain,
            "all_messages_as_observation": True,
        },
        rollout_options={
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "max_episode_steps": config.rollout.max_episode_steps,
        },
        model_serving_contract=MODEL_SERVING_CONTRACT,
        evaluation_config={"nl_assertions": evaluation_provenance},
        command=_command_for_manifest(argv),
    )
    context = RunContext(
        run_id=args.run_id,
        iteration=args.iteration,
        split=args.split,
        model_revision=args.model_revision,
        adapter_revision=args.adapter_revision,
        memory_snapshot_id=input_snapshot.memory_snapshot_id,
        seed=config.training.seed,
        event_writer=JsonlWriter(run_path / "rollouts" / "events.jsonl"),
        mode=RunMode.LEARN,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
    )
    fast_loop_config = FastLoopConfig(
        retrieve_top_k=config.memory.retrieve_top_k,
        max_episode_steps=config.rollout.max_episode_steps,
    )
    results, maintenance_rounds = _run_requested_tasks(
        task_ids=tuple(args.task_ids),
        env_factory=lambda task_id: Tau2RetailEnv(
            task_id,
            config,
            gym_factory=gym_factory,
        ),
        policy=policy,
        repository=repository,
        retriever=retriever,
        fast_loop_config=fast_loop_config,
        context=context,
        completed_train_tasks_before=args.completed_train_tasks_before,
        maintenance_period=config.memory.maintenance_period,
    )
    output_snapshot = repository.snapshot()
    summary = {
        "run_id": args.run_id,
        "episode_count": len(results),
        "total_terminal_reward": sum(result.final_reward for result in results),
        "successful_task_ids": [result.task_id for result in results],
        "maintenance_rounds_executed": list(maintenance_rounds),
        "completed_train_tasks_before": args.completed_train_tasks_before,
        "completed_train_tasks_after": args.completed_train_tasks_before + len(results),
        "input_memory_snapshot_id": input_snapshot.memory_snapshot_id,
        "output_memory_snapshot_id": output_snapshot.memory_snapshot_id,
    }
    summary_bytes = _canonical_json_bytes(summary)
    write_bytes_atomic(run_path / "fast_loop_summary.json", summary_bytes)
    sys.stdout.write(summary_bytes.decode("utf-8"))
    return 0


def _run_requested_tasks(
    *,
    task_ids: Sequence[str],
    env_factory: Callable[[str], Any],
    policy: Any,
    repository: MemoryRepository,
    retriever: Any,
    fast_loop_config: Any,
    context: RunContext,
    completed_train_tasks_before: int,
    maintenance_period: int,
) -> tuple[tuple[EpisodeResult, ...], tuple[int, ...]]:
    results: list[EpisodeResult] = []
    maintenance_rounds: list[int] = []
    for index, task_id in enumerate(task_ids, start=1):
        result = run_fast_loop_episode(
            task_id=task_id,
            task_instruction=TASK_INSTRUCTION,
            environment=env_factory(task_id),
            policy=policy,
            repository=repository,
            retriever=retriever,
            config=fast_loop_config,
            context=context,
        )
        results.append(result)
        maintenance = run_due_maintenance(
            completed_train_tasks=completed_train_tasks_before + index,
            period=maintenance_period,
            repository=repository,
            policy=policy,
            context=context,
        )
        if maintenance.executed:
            maintenance_rounds.append(maintenance.maintenance_round)
    return tuple(results), tuple(maintenance_rounds)


def _validate_early_arguments(args: argparse.Namespace) -> None:
    if not _SAFE_SLUG.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        raise ValueError(
            "run ID must contain only lowercase ASCII letters, digits, '-' or '_'"
        )
    if args.iteration < 0:
        raise ValueError("iteration must be non-negative")
    if args.completed_train_tasks_before < 0:
        raise ValueError("completed train tasks before must be non-negative")
    if not args.model_revision.strip():
        raise ValueError("model revision must not be blank")
    if args.adapter_revision is not None and not args.adapter_revision.strip():
        raise ValueError("adapter revision must not be blank")


def _require_explicit_train_tasks(task_ids: Sequence[str], train_task_ids: Sequence[str]) -> None:
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("requested task IDs must be unique")
    train_set = set(train_task_ids)
    unknown = [task_id for task_id in task_ids if task_id not in train_set]
    if unknown:
        raise ValueError(
            "requested task IDs are not in the official train split: " + ", ".join(unknown)
        )


def _validate_qwen_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Qwen base URL must be an http(s) URL without credentials, query, or fragment")
    return value


def _command_for_manifest(argv: Sequence[str] | None) -> list[str]:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    return [sys.executable, "-m", "scripts.run_fast_loop", *arguments]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
