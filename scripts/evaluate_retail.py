from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.envs.tau2_retail import Tau2RetailEnv
from tau3_retail_evolver.eval.guard import EvaluationGuard, EvaluationProtocol
from tau3_retail_evolver.eval.metrics import (
    EvaluationProvenance,
    build_evaluation_report,
    write_evaluation_json,
)
from tau3_retail_evolver.eval.runner import run_evaluation_trials
from tau3_retail_evolver.evaluation.tau2_nl_assertions import (
    bind_tau2_nl_assertions,
)
from tau3_retail_evolver.fast_loop.events import RunContext, RunMode
from tau3_retail_evolver.fast_loop.runner import FastLoopConfig
from tau3_retail_evolver.io.jsonl import JsonlWriter
from tau3_retail_evolver.memory.embeddings import build_embedding_provider
from tau3_retail_evolver.memory.paths import project_root as default_project_root
from tau3_retail_evolver.memory.retrieval import Retriever
from tau3_retail_evolver.models.openai_compatible import (
    OpenAICompatibleFastLoopPolicy,
    OpenAICompatibleHttpClient,
)
from tau3_retail_evolver.runs.manifest import create_manifest
from tau3_retail_evolver.slow_loop.task_grouping import RetailTaskGroups


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

_SAFE_SLUG = re.compile(r"^[a-z0-9_-]+$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen/LoRA on held-out Tau2 Retail tasks."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--protocol",
        type=EvaluationProtocol,
        choices=tuple(EvaluationProtocol),
        required=True,
    )
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--official-base-reproduction", action="store_true")
    parser.add_argument("--task-id", dest="task_ids", action="append")
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-revision")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--memory-snapshot", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_early_arguments(args)
    split = "base" if args.official_base_reproduction else args.split
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else default_project_root()
    )
    run_path = args.output_root / args.run_id
    if run_path.exists():
        raise FileExistsError(f"refusing to reuse existing evaluation run: {run_path}")

    config = load_config(args.config)
    seeds = _resolve_seeds(
        args.seeds,
        num_trials=args.num_trials,
        base_seed=config.training.seed,
    )
    runtime = Tau2Runtime.inspect_metadata(config.tau2.repo_path)
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(
        runtime.retail_tasks_path,
        runtime.retail_split_path,
    )
    catalog.require_official_compatibility()
    task_ids = _resolve_task_ids(args.task_ids, catalog.task_ids(split))

    base_url = _validate_qwen_base_url(
        args.qwen_base_url or os.environ.get("QWEN_BASE_URL")
    )
    if not base_url:
        raise ValueError("QWEN_BASE_URL or --qwen-base-url is required")

    guard = EvaluationGuard(
        protocol=args.protocol,
        run_id=args.run_id,
        agent_id=config.memory.agent_id,
        project_root=project_root,
        split=split,
        official_base_reproduction=args.official_base_reproduction,
        memory_snapshot_path=args.memory_snapshot,
    )
    source_memory_snapshot_id = guard.source_memory_snapshot_id()
    task_groups = RetailTaskGroups.from_file(
        runtime.retail_tasks_path,
        task_ids=task_ids,
    )
    gym_factory = Tau2Runtime.load_verified_gym_factory(runtime.repo_path)
    nl_evaluator = bind_tau2_nl_assertions(config.evaluation.nl_assertions)
    probe = Tau2RetailEnv(task_ids[0], config, gym_factory=gym_factory)
    try:
        user_simulator_config = probe.user_simulator_config
    finally:
        probe.close()

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
    memory_enabled = args.protocol is not EvaluationProtocol.NO_MEMORY
    fast_loop_config = FastLoopConfig(
        retrieve_top_k=config.memory.retrieve_top_k,
        max_episode_steps=config.rollout.max_episode_steps,
        memory_enabled=memory_enabled,
    )
    if memory_enabled:
        retriever_factory = lambda memory: Retriever(
            build_embedding_provider(
                config.memory,
                _require_memory_root(memory.root),
            )
        )
    else:
        retriever_factory = None

    checkpoint = str(args.checkpoint) if args.checkpoint is not None else None
    create_manifest(
        run_path / "manifest.json",
        run_id=args.run_id,
        iteration=args.iteration,
        model_revision=args.model_revision,
        parent_checkpoint=checkpoint,
        adapter_revision=args.adapter_revision,
        memory_snapshot_id=source_memory_snapshot_id,
        tau2_commit=runtime.git_commit,
        split=split,
        split_hash=catalog.split_sha256,
        task_ids=task_ids,
        seed=seeds[0],
        user_simulator_config=user_simulator_config,
        environment_options={
            "domain": config.tau2.domain,
            "all_messages_as_observation": True,
            "official_base_reproduction": args.official_base_reproduction,
        },
        rollout_options={
            "protocol": args.protocol.value,
            "temperature": config.rollout.temperature,
            "top_p": config.rollout.top_p,
            "max_episode_steps": config.rollout.max_episode_steps,
            "memory_enabled": memory_enabled,
            "memory_write": guard.capabilities.memory_write,
            "memory_agent_id": (
                config.memory.agent_id if memory_enabled else None
            ),
            "num_trials": len(seeds),
            "seeds": list(seeds),
            "task_order": list(task_ids),
            "capabilities": guard.capabilities.as_dict(),
        },
        model_serving_contract=MODEL_SERVING_CONTRACT,
        evaluation_config={
            "nl_assertions": nl_evaluator,
            "protocol": args.protocol.value,
        },
        command=_command_for_manifest(argv),
    )
    context = RunContext(
        run_id=args.run_id,
        iteration=args.iteration,
        split=split,
        model_revision=args.model_revision,
        adapter_revision=args.adapter_revision,
        memory_snapshot_id=source_memory_snapshot_id,
        seed=seeds[0],
        event_writer=JsonlWriter(run_path / "rollouts" / "events.jsonl"),
        mode=RunMode.EVALUATE,
        task_groups={
            task_id: task_groups.signature_for(task_id)
            for task_id in task_ids
        },
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        default_task_group="retail-actions-v1:evaluation-maintenance",
    )
    run = run_evaluation_trials(
        task_ids=task_ids,
        seeds=seeds,
        env_factory=lambda task_id: Tau2RetailEnv(
            task_id,
            config,
            gym_factory=gym_factory,
        ),
        policy=policy,
        guard=guard,
        retriever_factory=retriever_factory,
        config=fast_loop_config,
        context=context,
        maintenance_period=config.memory.maintenance_period,
    )
    provenance = EvaluationProvenance(
        run_id=args.run_id,
        protocol=args.protocol,
        official_base_reproduction=args.official_base_reproduction,
        split=split,
        checkpoint=checkpoint,
        base_model=config.model.base_model,
        model_revision=args.model_revision,
        adapter_revision=args.adapter_revision,
        tau2_commit=runtime.git_commit,
        split_hash=catalog.split_sha256,
        task_ids=task_ids,
        seeds=seeds,
        user_simulator_config=user_simulator_config,
        nl_evaluator=nl_evaluator,
        memory_snapshot_id=source_memory_snapshot_id,
        max_episode_steps=config.rollout.max_episode_steps,
        model_serving_contract=MODEL_SERVING_CONTRACT,
        capabilities=guard.capabilities.as_dict(),
    )
    report = build_evaluation_report(provenance, run)
    report_path = run_path / "evaluation_report.json"
    write_evaluation_json(report_path, report)
    output = {
        "run_id": args.run_id,
        "report_path": str(report_path),
        **report["summary"],
    }
    sys.stdout.write(
        json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    return 0


def _validate_early_arguments(args: argparse.Namespace) -> None:
    if not _SAFE_SLUG.fullmatch(args.run_id) or args.run_id in {".", ".."}:
        raise ValueError(
            "run ID must contain only lowercase ASCII letters, digits, '-' or '_'"
        )
    if args.iteration < 0:
        raise ValueError("iteration must be non-negative")
    if args.num_trials < 1:
        raise ValueError("num-trials must be positive")
    if not args.model_revision.strip():
        raise ValueError("model revision must not be blank")
    if args.adapter_revision is not None and not args.adapter_revision.strip():
        raise ValueError("adapter revision must not be blank")
    if (args.adapter_revision is None) != (args.checkpoint is None):
        raise ValueError(
            "adapter-revision and checkpoint must be provided together"
        )
    if args.protocol is EvaluationProtocol.TEST_STATIC:
        if args.memory_snapshot is None:
            raise ValueError("test_static requires --memory-snapshot")
    elif args.memory_snapshot is not None:
        raise ValueError(
            f"{args.protocol.value} does not accept --memory-snapshot"
        )
    if (
        args.official_base_reproduction
        and args.protocol is not EvaluationProtocol.NO_MEMORY
    ):
        raise ValueError("official base reproduction supports only no_memory")


def _resolve_seeds(
    requested: Sequence[int] | None,
    *,
    num_trials: int,
    base_seed: int,
) -> tuple[int, ...]:
    if num_trials < 1:
        raise ValueError("num-trials must be positive")
    seeds = (
        tuple(requested)
        if requested is not None
        else tuple(base_seed + index for index in range(num_trials))
    )
    if len(seeds) != num_trials:
        raise ValueError("the number of --seed values must equal --num-trials")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("evaluation seeds must be non-negative integers")
    if len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be unique")
    return seeds


def _resolve_task_ids(
    requested: Sequence[str] | None,
    official_task_ids: Sequence[str],
) -> tuple[str, ...]:
    official = tuple(official_task_ids)
    task_ids = tuple(requested) if requested is not None else official
    if not task_ids:
        raise ValueError("evaluation requires at least one task")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("requested task IDs must be unique")
    unknown = [task_id for task_id in task_ids if task_id not in set(official)]
    if unknown:
        raise ValueError(
            "requested task IDs are not in the official evaluation split: "
            + ", ".join(unknown)
        )
    return task_ids


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
        raise ValueError(
            "Qwen base URL must be an http(s) URL without credentials, query, or fragment"
        )
    return value


def _require_memory_root(root: Path | None) -> Path:
    if root is None:
        raise ValueError("evaluation Memory root is missing")
    return root


def _command_for_manifest(argv: Sequence[str] | None) -> list[str]:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    return [sys.executable, "-m", "scripts.evaluate_retail", *arguments]


if __name__ == "__main__":
    raise SystemExit(main())
