from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.slow_loop.task_grouping import RetailTaskGroups


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a pinned Tau2 domain and its run_domain integration."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split", choices=("train", "test", "base"), required=True)
    parser.add_argument("--task-id", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    if args.task_id not in catalog.task_ids(args.split):
        raise ValueError(
            f"task {args.task_id!r} is not in the {args.split!r} "
            f"{config.tau2.domain} split"
        )
    groups = RetailTaskGroups.from_file(
        runtime.tasks_path,
        task_ids=(args.task_id,),
        domain=config.tau2.domain,
    )
    run_domain_runtime = Tau2Runtime.load_verified_run_domain(runtime.repo_path)
    probe = run_domain_runtime.text_run_config(
        domain=config.tau2.domain,
        task_split_name=args.split,
        task_ids=[args.task_id],
        agent="llm_agent",
        llm_agent=config.model.base_model,
        llm_args_agent={},
        user="user_simulator",
        llm_user=config.tau2.user_llm,
        llm_args_user=dict(config.tau2.user_llm_args),
        num_trials=1,
        max_steps=max(4, config.rollout.max_episode_steps * 4),
        max_errors=10,
        save_to=None,
        max_concurrency=1,
        seed=config.training.seed,
        log_level="WARNING",
        max_retries=0,
        retry_delay=1.0,
        auto_resume=False,
        hallucination_retries=0,
        enforce_communication_protocol=True,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "backend": "tau2.run.run_domain",
                "domain": config.tau2.domain,
                "tau2_commit": runtime.git_commit,
                "tau2_package_version": runtime.package_version,
                "split": args.split,
                "split_hash": catalog.split_sha256,
                "task_id": args.task_id,
                "task_group": groups.signature_for(args.task_id),
                "max_concurrency": config.rollout.max_concurrency,
                "run_config_type": type(probe).__name__,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
