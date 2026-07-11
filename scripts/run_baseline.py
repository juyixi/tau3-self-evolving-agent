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
from tau3_retail_evolver.envs.tau2_retail import Tau2RetailEnv
from tau3_retail_evolver.fast_loop.baseline_runner import RolloutSummary, run_baseline
from tau3_retail_evolver.fast_loop.events import RunContext
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.models.openai_compatible import (
    OpenAICompatibleHttpClient,
    OpenAICompatibleQwenPolicy,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a canonical no-memory Qwen retail baseline.")
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
    runtime = Tau2Runtime.inspect_metadata(config.tau2.repo_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(runtime.retail_tasks_path, runtime.retail_split_path)
    catalog.require_official_compatibility()
    _require_explicit_train_tasks(args.task_ids, catalog.task_ids("train"))

    base_url = _validate_qwen_base_url(args.qwen_base_url or os.environ.get("QWEN_BASE_URL"))
    if not base_url:
        raise ValueError("QWEN_BASE_URL or --qwen-base-url is required")
    api_key = os.environ.get("QWEN_API_KEY") or "EMPTY"
    gym_factory = Tau2Runtime.load_verified_gym_factory(runtime.repo_path)
    probe = Tau2RetailEnv(args.task_ids[0], config, gym_factory=gym_factory)
    try:
        user_simulator_config = probe.user_simulator_config
    finally:
        probe.close()
    client = OpenAICompatibleHttpClient(
        base_url=base_url,
        model=config.model.base_model,
        api_key=api_key,
        max_tokens=MODEL_SERVING_CONTRACT["max_tokens"],
        generation_settings=QWEN_GENERATION_SETTINGS,
    )
    policy = OpenAICompatibleQwenPolicy(client=client)

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
            "all_messages_as_observation": True,
        },
        rollout_options={
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "max_episode_steps": config.rollout.max_episode_steps,
        },
        model_serving_contract=MODEL_SERVING_CONTRACT,
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
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
    )
    summary = run_baseline(
        tuple(args.task_ids),
        lambda task_id: Tau2RetailEnv(task_id, config, gym_factory=gym_factory),
        policy,
        context,
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
