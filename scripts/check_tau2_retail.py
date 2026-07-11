from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

from tau3_retail_evolver.config import load_config
from tau3_retail_evolver.envs.split_guard import require_learning_split
from tau3_retail_evolver.envs.task_catalog import RetailTaskCatalog
from tau3_retail_evolver.envs.runtime import Tau2Runtime
from tau3_retail_evolver.envs.tau2_retail import Tau2RetailEnv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check the real Tau2 retail environment.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--inspect", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require_learning_split(args.split)

    config = load_config(args.config)
    runtime = Tau2Runtime.inspect(config.tau2.repo_path)
    catalog = RetailTaskCatalog.from_files(runtime.retail_tasks_path, runtime.retail_split_path)
    if args.task_id not in catalog.task_ids(args.split):
        raise ValueError(f"task {args.task_id!r} is not in the {args.split!r} retail split")

    payload = {
        "status": "ok",
        "tau2_commit": runtime.git_commit,
        "tau2_package_version": runtime.package_version,
        "split": args.split,
        "split_hash": catalog.split_sha256,
        "task_id": args.task_id,
    }
    if args.inspect:
        payload["mode"] = "inspect"
        _print_payload(payload)
        return 0

    try:
        with Tau2RetailEnv(args.task_id, config) as environment:
            reset_result = environment.reset(seed=config.training.seed)
    except RuntimeError as error:
        if "Retail domain does not support solo mode" not in str(error):
            raise
        payload.update(_blocked_reset_summary(str(error), config.tau2))
        payload["mode"] = "reset_close"
        _print_payload(payload)
        return 2

    payload.update(_reset_summary(reset_result.observation, reset_result.info, config.tau2))
    payload["mode"] = "reset_close"
    _print_payload(payload)
    return 0


def _reset_summary(observation: Any, info: Mapping[str, Any], tau2_config: Any) -> dict[str, Any]:
    tools = info.get("tools")
    if not isinstance(tools, Sequence):
        raise RuntimeError("Tau2 reset did not return a sequence of tools")
    policy = info.get("policy")
    if not isinstance(policy, str):
        raise RuntimeError("Tau2 reset did not return a policy string")
    if not isinstance(observation, str):
        raise RuntimeError("Tau2 reset did not return a string observation")
    return {
        "tool_count": len(tools),
        "policy_sha256": hashlib.sha256(policy.encode("utf-8")).hexdigest(),
        "initial_observation_length": len(observation),
        "user_simulator_config": {
            "solo_mode": tau2_config.solo_mode,
            "user_llm": tau2_config.user_llm,
            "user_llm_args": tau2_config.user_llm_args,
        },
    }


def _blocked_reset_summary(reason: str, tau2_config: Any) -> dict[str, Any]:
    return {
        "status": "blocked",
        "block_reason": reason,
        "tool_count": None,
        "policy_sha256": None,
        "initial_observation_length": None,
        "user_simulator_config": {
            "solo_mode": tau2_config.solo_mode,
            "user_llm": tau2_config.user_llm,
            "user_llm_args": tau2_config.user_llm_args,
        },
    }


def _print_payload(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
