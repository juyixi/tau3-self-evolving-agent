from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from collections.abc import Sequence
from urllib.parse import urlsplit

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.split_guard import require_learning_split
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.evaluation.tau2_nl_assertions import bind_tau2_nl_assertions
from tau3_retail_evolver.fast_loop.baseline_runner import (
    EpisodeSummary,
    RolloutSummary,
)
from tau3_retail_evolver.fast_loop.events import RunContext
from tau3_retail_evolver.fast_loop.runner import FastLoopConfig
from tau3_retail_evolver.fast_loop.tau2_run_domain import run_tau2_fast_loop_batch
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.models.openai_compatible import (
    OpenAICompatibleHttpClient,
    OpenAICompatibleFastLoopPolicy,
)
from tau3_retail_evolver.runs.manifest import create_manifest
from tau3_retail_evolver.slow_loop.task_grouping import (
    RetailTaskGroups,
    maintenance_task_group,
)


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a canonical no-memory Qwen Tau2 baseline."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-id", dest="task_ids", action="append", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", type=Path, default=Path("runs"))
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_learning_split(args.split)
    if args.split != "train":
        raise ValueError("baseline supports only the train split")
    if not args.model_revision.strip():
        raise ValueError("model revision is required")

    config = load_config(args.config)
    runtime = Tau2Runtime.inspect_metadata(
        config.tau2.repo_path,
        domain=config.tau2.domain,
    )
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.tasks_path,
        runtime.split_path,
        domain=config.tau2.domain,
    )
    catalog.require_official_compatibility()
    _require_explicit_train_tasks(args.task_ids, catalog.task_ids("train"))

    base_url = _validate_qwen_base_url(args.qwen_base_url or os.environ.get("QWEN_BASE_URL"))
    if not base_url:
        raise ValueError("QWEN_BASE_URL or --qwen-base-url is required")
    api_key = os.environ.get("QWEN_API_KEY") or "EMPTY"
    if config.tau2.solo_mode:
        raise ValueError("run_domain baseline execution requires tau2.solo_mode=false")
    run_domain_runtime = Tau2Runtime.load_verified_run_domain(runtime.repo_path)
    evaluation_provenance = bind_tau2_nl_assertions(config.evaluation.nl_assertions)
    user_simulator_config = {
        "solo_mode": False,
        "user_llm": config.tau2.user_llm,
        "user_llm_args": dict(config.tau2.user_llm_args),
    }
    client = OpenAICompatibleHttpClient(
        base_url=base_url,
        model=config.model.base_model,
        api_key=api_key,
        max_tokens=MODEL_SERVING_CONTRACT["max_tokens"],
        generation_settings=QWEN_GENERATION_SETTINGS,
    )
    policy = OpenAICompatibleFastLoopPolicy(
        client=client,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
    )
    task_groups = RetailTaskGroups.from_file(
        runtime.tasks_path,
        task_ids=args.task_ids,
        domain=config.tau2.domain,
    )

    run_path = args.output_root / args.run_id
    create_manifest(
        run_path / "manifest.json",
        run_id=args.run_id,
        iteration=args.iteration,
        model_revision=args.model_revision,
        parent_checkpoint=None,
        tau2_commit=runtime.git_commit,
        split=args.split,
        split_hash=catalog.split_sha256,
        task_ids=tuple(args.task_ids),
        seed=config.training.seed,
        user_simulator_config=user_simulator_config,
        environment_options={
            "domain": config.tau2.domain,
            "execution_backend": "tau2.run.run_domain",
            "max_concurrency": config.rollout.max_concurrency,
        },
        rollout_options={
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "max_episode_steps": config.rollout.max_episode_steps,
            "max_concurrency": config.rollout.max_concurrency,
            "memory_enabled": False,
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
        adapter_revision=None,
        memory_snapshot_id=None,
        seed=config.training.seed,
        event_writer=JsonlWriter(run_path / "rollouts" / "events.jsonl"),
        task_groups={
            task_id: task_groups.signature_for(task_id)
            for task_id in args.task_ids
        },
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        default_task_group=maintenance_task_group(config.tau2.domain),
    )
    batch = run_tau2_fast_loop_batch(
        runtime=run_domain_runtime,
        domain=config.tau2.domain,
        split=args.split,
        task_ids=tuple(args.task_ids),
        run_seed=config.training.seed,
        max_concurrency=config.rollout.max_concurrency,
        user_llm=config.tau2.user_llm,
        user_llm_args=config.tau2.user_llm_args,
        agent_model=config.model.base_model,
        policy=policy,
        repository=None,
        retriever=None,
        config=FastLoopConfig(
            retrieve_top_k=config.memory.retrieve_top_k,
            max_episode_steps=config.rollout.max_episode_steps,
            memory_enabled=False,
        ),
        context_factory=lambda _task_id, _tau2_seed: context,
        task_instruction=(
            f"Resolve the {config.tau2.domain} request shown in the current conversation."
        ),
        write_memory=False,
        memory_disabled_reason="baseline",
    )
    if batch.failures:
        failed = ", ".join(
            f"{failure.task_id} ({failure.stage}:{failure.error_type})"
            for failure in batch.failures
        )
        raise RuntimeError(f"Tau2 run_domain baseline failures: {failed}")
    summary = RolloutSummary(
        episodes=tuple(
            EpisodeSummary(
                task_id=episode.result.task_id,
                final_reward=episode.result.final_reward,
                steps=episode.result.steps,
                terminal_evaluation=episode.result.terminal_evaluation,
                simulation_result=episode.result.simulation_result,
            )
            for episode in batch.episodes
        )
    )
    _print_summary(args.run_id, summary)
    return 0


def _require_explicit_train_tasks(task_ids: Sequence[str], train_task_ids: Sequence[str]) -> None:
    train_set = set(train_task_ids)
    unknown = [task_id for task_id in task_ids if task_id not in train_set]
    if unknown:
        raise ValueError(f"requested task IDs are not in the official train split: {', '.join(unknown)}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("requested task IDs must be unique")


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
    return [sys.executable, "-m", "scripts.run_baseline", *arguments]


def _print_summary(run_id: str, summary: RolloutSummary) -> None:
    print(
        json.dumps(
            {
                "run_id": run_id,
                "episode_count": summary.episode_count,
                "total_reward": summary.total_reward,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
