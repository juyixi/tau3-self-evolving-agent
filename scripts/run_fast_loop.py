from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any
from urllib.parse import urlsplit

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.credential_policy import is_credential_key
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
    validate_fast_loop_dependencies,
)
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import build_embedding_provider
from tau3_retail_evolver.memory.factory import open_training_memory
from tau3_retail_evolver.memory.json_store import write_bytes_atomic
from tau3_retail_evolver.memory.repository import MemoryRepository
from tau3_retail_evolver.memory.retrieval import Retriever
from tau3_retail_evolver.memory.types import MEMORY_TIERS, MemoryStatus
from tau3_retail_evolver.models.openai_compatible import (
    OpenAICompatibleFastLoopPolicy,
    OpenAICompatibleHttpClient,
)
from tau3_retail_evolver.runs.manifest import create_manifest
from tau3_retail_evolver.slow_loop.task_grouping import (
    MAINTENANCE_TASK_GROUP,
    RetailTaskGroups,
)


MODEL_SERVING_CONTRACT = {
    "language_model_only": True,
    "reasoning_parser": "qwen3",
    "tool_call_parser": "qwen3_coder",
    "enable_thinking": True,
    "max_tokens": 8192,
    "request_timeout_s": 600.0,
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


@dataclass(slots=True)
class _EventBuffer:
    events: list[dict[str, Any]] = field(default_factory=list)

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def flush_to(self, writer: Any) -> None:
        for event in self.events:
            writer.append(event)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run canonical Qwen retail fast-loop learning.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split", required=True)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--task-id", dest="task_ids", action="append")
    tasks.add_argument("--all-train-tasks", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--completed-train-tasks-before", type=int, required=True)
    parser.add_argument("--seed", type=int)
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
    run_seed = config.training.seed if args.seed is None else args.seed
    task_ids = _resolve_train_tasks(
        args.task_ids,
        all_train_tasks=args.all_train_tasks,
        official_train_task_ids=catalog.task_ids("train"),
        seed=run_seed,
    )

    base_url = _validate_qwen_base_url(args.qwen_base_url or os.environ.get("QWEN_BASE_URL"))
    if not base_url:
        raise ValueError("QWEN_BASE_URL or --qwen-base-url is required")

    task_groups = RetailTaskGroups.from_file(
        runtime.retail_tasks_path,
        task_ids=task_ids,
    )

    gym_factory = Tau2Runtime.load_verified_gym_factory(runtime.repo_path)
    evaluation_provenance = bind_tau2_nl_assertions(config.evaluation.nl_assertions)
    probe = Tau2RetailEnv(task_ids[0], config, gym_factory=gym_factory)
    try:
        user_simulator_config = probe.user_simulator_config
    finally:
        probe.close()

    fast_loop_config = FastLoopConfig(
        retrieve_top_k=config.memory.retrieve_top_k,
        max_episode_steps=config.rollout.max_episode_steps,
        memory_enabled=config.memory.enabled,
         max_new_tips_per_episode=config.memory.max_new_tips_per_episode,
        max_new_skills_per_episode=config.memory.max_new_skills_per_episode,
        max_new_tools_per_episode=config.memory.max_new_tools_per_episode,
        max_new_trajectories_per_episode=config.memory.max_new_trajectories_per_episode,
        maintenance_tip_capacity=config.memory.maintenance_tip_capacity,
        maintenance_similarity_threshold=config.memory.maintenance_similarity_threshold,
        maintenance_priority_pair_limit=config.memory.maintenance_priority_pair_limit,
        retrieval_mmr_lambda_tip=config.memory.retrieval_mmr_lambda_tip,
        retrieval_mmr_lambda_skill=config.memory.retrieval_mmr_lambda_skill,
        retrieval_mmr_lambda_tool=config.memory.retrieval_mmr_lambda_tool,
        retrieval_mmr_lambda_trajectory=config.memory.retrieval_mmr_lambda_trajectory,
        retrieval_global_mmr_lambda=config.memory.retrieval_global_mmr_lambda,
        retrieval_quota_tip=config.memory.retrieval_quota_tip,
        retrieval_quota_skill=config.memory.retrieval_quota_skill,
        retrieval_quota_tool=config.memory.retrieval_quota_tool,
        retrieval_quota_trajectory=config.memory.retrieval_quota_trajectory,
        selection_max_total=config.memory.selection_max_total,
        selection_max_tip=config.memory.selection_max_tip,
        selection_max_skill=config.memory.selection_max_skill,
        selection_max_tool=config.memory.selection_max_tool,
         selection_max_trajectory=config.memory.selection_max_trajectory,
    )
    if fast_loop_config.memory_enabled:
        repository = open_training_memory(config.memory, root=args.project_root)
        embedding_provider = build_embedding_provider(config.memory, repository.root)
        retriever = Retriever(embedding_provider)
    else:
        repository = None
        retriever = None
    validate_fast_loop_dependencies(
        config=fast_loop_config,
        repository=repository,
        retriever=retriever,
    )
    if fast_loop_config.memory_enabled:
        assert repository is not None
        input_memory_snapshot = repository.snapshot()
        input_memory_snapshot_id = input_memory_snapshot.memory_snapshot_id
        input_memory_counts = input_memory_snapshot.counts
    else:
        input_memory_snapshot_id = None
        input_memory_counts = {tier: 0 for tier in MEMORY_TIERS}

    client = OpenAICompatibleHttpClient(
        base_url=base_url,
        model=config.model.base_model,
        api_key=os.environ.get("QWEN_API_KEY") or "EMPTY",
        max_tokens=MODEL_SERVING_CONTRACT["max_tokens"],
        request_timeout_s=MODEL_SERVING_CONTRACT["request_timeout_s"],
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
        memory_snapshot_id=input_memory_snapshot_id,
        tau2_commit=runtime.git_commit,
        split=args.split,
        split_hash=catalog.split_sha256,
        task_ids=task_ids,
        seed=run_seed,
        user_simulator_config=user_simulator_config,
        environment_options={
            "domain": config.tau2.domain,
            "all_messages_as_observation": True,
        },
        rollout_options={
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "max_episode_steps": config.rollout.max_episode_steps,
            "memory_enabled": fast_loop_config.memory_enabled,
            "memory_agent_id": (
                config.memory.agent_id if fast_loop_config.memory_enabled else None
            ),
            "task_order_seed": run_seed,
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
        memory_snapshot_id=input_memory_snapshot_id,
        seed=run_seed,
        event_writer=JsonlWriter(run_path / "rollouts" / "events.jsonl"),
        mode=RunMode.LEARN,
        task_groups={
            task_id: task_groups.signature_for(task_id) for task_id in task_ids
        },
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        default_task_group=MAINTENANCE_TASK_GROUP,
    )
    results, maintenance_rounds, failures = _run_requested_tasks(
        task_ids=task_ids,
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
    if fast_loop_config.memory_enabled:
        assert repository is not None
        output_memory_snapshot = repository.snapshot()
        output_memory_snapshot_id = output_memory_snapshot.memory_snapshot_id
        output_memory_counts = output_memory_snapshot.counts
    else:
        output_memory_snapshot_id = None
        output_memory_counts = {tier: 0 for tier in MEMORY_TIERS}
    token_results = [
        result
        for result in results
        if result.agent_prompt_tokens is not None
        and result.agent_completion_tokens is not None
    ]
    selected_memory_ids = {
        memory_id
        for result in results
        for memory_id in result.selected_memory_ids
    }
    memory_item_count = sum(output_memory_counts.values())
    task_failures = [
        failure for failure in failures if failure["stage"] == "episode"
    ]
    maintenance_failures = [
        failure for failure in failures if failure["stage"] == "maintenance"
    ]
    summary = {
        "run_id": args.run_id,
        "attempted_task_count": len(task_ids),
        "episode_count": len(results),
        "total_terminal_reward": sum(result.final_reward for result in results),
        "successful_task_ids": [result.task_id for result in results],
        "failed_task_count": len(task_failures),
        "failed_task_ids": [failure["task_id"] for failure in task_failures],
        "failures": list(failures),
        "failed_maintenance_rounds": [
            failure["maintenance_round"] for failure in maintenance_failures
        ],
        "maintenance_rounds_executed": list(maintenance_rounds),
        "completed_train_tasks_before": args.completed_train_tasks_before,
        "completed_train_tasks_after": (
            args.completed_train_tasks_before + len(task_ids)
        ),
        "memory_enabled": fast_loop_config.memory_enabled,
        "input_memory_snapshot_id": input_memory_snapshot_id,
        "output_memory_snapshot_id": output_memory_snapshot_id,
        "input_memory_counts": input_memory_counts,
        "output_memory_counts": output_memory_counts,
        "memory_item_count": memory_item_count,
        "memory_selection_count": sum(
            len(result.selected_memory_ids) for result in results
        ),
        "unique_reused_memory_count": len(selected_memory_ids),
        "memory_reuse_coverage": (
            len(selected_memory_ids) / memory_item_count
            if memory_item_count
            else None
        ),
        "token_usage_episode_count": len(token_results),
        "total_agent_prompt_tokens": sum(
            result.agent_prompt_tokens or 0 for result in token_results
        ),
        "total_agent_completion_tokens": sum(
            result.agent_completion_tokens or 0 for result in token_results
        ),
        "total_agent_tokens": sum(
            (result.agent_prompt_tokens or 0)
            + (result.agent_completion_tokens or 0)
            for result in token_results
        ),
        "mean_agent_tokens": (
            sum(
                (result.agent_prompt_tokens or 0)
                + (result.agent_completion_tokens or 0)
                for result in token_results
            )
            / len(token_results)
            if token_results
            else None
        ),
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
    repository: MemoryRepository | None,
    retriever: Retriever | None,
    fast_loop_config: FastLoopConfig,
    context: RunContext,
    completed_train_tasks_before: int,
    maintenance_period: int,
) -> tuple[
    tuple[EpisodeResult, ...],
    tuple[int, ...],
    tuple[dict[str, Any], ...],
]:
    validate_fast_loop_dependencies(
        config=fast_loop_config,
        repository=repository,
        retriever=retriever,
    )
    results: list[EpisodeResult] = []
    maintenance_rounds: list[int] = []
    failures: list[dict[str, Any]] = []
    canonical_writer = context.event_writer
    for index, task_id in enumerate(task_ids, start=1):
        episode_buffer = _EventBuffer()
        episode_context = replace(context, event_writer=episode_buffer)
        memory_ids_before: set[str] | None = None
        try:
            if fast_loop_config.memory_enabled:
                assert repository is not None
                episode_context = replace(
                    episode_context,
                    memory_snapshot_id=repository.snapshot().memory_snapshot_id,
                )
                memory_ids_before = {
                    item.id for item in repository.list(status=None)
                }
            else:
                episode_context = replace(episode_context, memory_snapshot_id=None)
            result = run_fast_loop_episode(
                task_id=task_id,
                task_instruction=TASK_INSTRUCTION,
                environment=env_factory(task_id),
                policy=policy,
                repository=repository,
                retriever=retriever,
                config=fast_loop_config,
                context=episode_context,
            )
        except Exception as error:
            discarded_ids, rollback_error = _discard_new_memories(
                repository,
                memory_ids_before,
                updated_round=context.iteration,
            )
            failure = _failure_record(error, task_id=task_id, stage="episode")
            if rollback_error is not None:
                failure["rollback_error"] = rollback_error
            failures.append(failure)
            canonical_writer.append(
                episode_context.event(
                    "TaskFailed",
                    task_id,
                    error=failure["error"],
                    discarded_memory_ids=list(discarded_ids),
                    rollback_error=failure.get("rollback_error"),
                )
            )
        else:
            episode_buffer.flush_to(canonical_writer)
            results.append(result)

        if fast_loop_config.memory_enabled:
            assert repository is not None
            completed_tasks = completed_train_tasks_before + index
            maintenance_buffer = _EventBuffer()
            maintenance_context = replace(context, event_writer=maintenance_buffer)
            try:
                maintenance_context = replace(
                    maintenance_context,
                    memory_snapshot_id=repository.snapshot().memory_snapshot_id,
                )
                maintenance = run_due_maintenance(
                    completed_train_tasks=completed_tasks,
                    period=maintenance_period,
                    repository=repository,
                    policy=policy,
                    context=maintenance_context,
                )
            except Exception as error:
                maintenance_round = max(1, completed_tasks // maintenance_period)
                failure = _failure_record(
                    error,
                    task_id=f"maintenance-round-{maintenance_round}",
                    stage="maintenance",
                )
                failure["maintenance_round"] = maintenance_round
                failures.append(failure)
                canonical_writer.append(
                    maintenance_context.event(
                        "MaintenanceTaskFailed",
                        failure["task_id"],
                        error=failure["error"],
                        maintenance_round=maintenance_round,
                    )
                )
            else:
                maintenance_buffer.flush_to(canonical_writer)
                if maintenance.executed:
                    maintenance_rounds.append(maintenance.maintenance_round)
    return tuple(results), tuple(maintenance_rounds), tuple(failures)


def _failure_record(
    error: Exception,
    *,
    task_id: str,
    stage: str,
) -> dict[str, Any]:
    error_types: list[str] = []
    error_messages: list[str] = []
    current: BaseException | None = error
    while current is not None and len(error_types) < 8:
        error_types.append(type(current).__name__)
        error_messages.append(_redact_error_text(str(current)))
        current = current.__cause__ or current.__context__
    fingerprint_source = f"{stage}|{'|'.join(error_types)}|{error}"
    return {
        "task_id": task_id,
        "stage": stage,
        "error": {
            "types": error_types,
            "messages": error_messages,
            "fingerprint": hashlib.sha256(
                fingerprint_source.encode("utf-8", errors="replace")
            ).hexdigest()[:16],
        },
    }


def _redact_error_text(value: str) -> str:
    redacted = value
    for key, secret in os.environ.items():
        if is_credential_key(key) and secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted[:4000]


def _discard_new_memories(
    repository: MemoryRepository | None,
    memory_ids_before: set[str] | None,
    *,
    updated_round: int,
) -> tuple[tuple[str, ...], dict[str, Any] | None]:
    if repository is None or memory_ids_before is None:
        return (), None
    try:
        new_ids = tuple(
            sorted(
                item.id
                for item in repository.list(status=None)
                if item.id not in memory_ids_before
            )
        )
        for memory_id in new_ids:
            repository.update_status(
                memory_id,
                MemoryStatus.RETIRED,
                updated_round=updated_round,
            )
        return new_ids, None
    except Exception as error:
        return (), _failure_record(
            error,
            task_id="memory-rollback",
            stage="rollback",
        )["error"]


def _validate_early_arguments(args: argparse.Namespace) -> None:
    if not _SAFE_SLUG.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        raise ValueError(
            "run ID must contain only lowercase ASCII letters, digits, '-' or '_'"
        )
    if args.iteration < 0:
        raise ValueError("iteration must be non-negative")
    if args.completed_train_tasks_before < 0:
        raise ValueError("completed train tasks before must be non-negative")
    if args.seed is not None and args.seed < 0:
        raise ValueError("seed must be non-negative")
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


def _resolve_train_tasks(
    task_ids: Sequence[str] | None,
    *,
    all_train_tasks: bool,
    official_train_task_ids: Sequence[str],
    seed: int,
) -> tuple[str, ...]:
    official = tuple(official_train_task_ids)
    if all_train_tasks:
        ordered = list(official)
        random.Random(f"{seed}:tau3-retail-stage8-fast-loop").shuffle(ordered)
        return tuple(ordered)
    if task_ids is None:
        raise ValueError("explicit task IDs are required unless --all-train-tasks is set")
    _require_explicit_train_tasks(task_ids, official)
    return tuple(task_ids)


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
