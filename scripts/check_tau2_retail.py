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


_REDACTED = "[REDACTED]"
_USER_SIMULATOR_FIELDS = ("solo_mode", "user_llm", "user_llm_args")
_CREDENTIAL_KEY_MARKERS = (
    "apikey",
    "token",
    "authorization",
    "secret",
    "password",
    "credential",
    "privatekey",
    "accesskey",
)


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
    Tau2Runtime.require_pinned_commit(runtime)
    catalog = RetailTaskCatalog.from_files(runtime.retail_tasks_path, runtime.retail_split_path)
    catalog.require_official_compatibility()
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

    gym_factory = Tau2Runtime.load_verified_gym_factory(runtime.repo_path)
    environment = Tau2RetailEnv(args.task_id, config, gym_factory=gym_factory)
    user_simulator_config = environment.user_simulator_config
    try:
        with environment:
            reset_result = environment.reset(seed=config.training.seed)
    except RuntimeError as error:
        if "Retail domain does not support solo mode" not in str(error):
            raise
        payload.update(_blocked_reset_summary(str(error), user_simulator_config))
        payload["mode"] = "reset_close"
        _print_payload(payload)
        return 2

    payload.update(
        _reset_summary(
            reset_result.observation,
            reset_result.info,
            user_simulator_config,
        )
    )
    payload["mode"] = "reset_close"
    _print_payload(payload)
    return 2 if payload["status"] == "blocked" else 0


def _reset_summary(
    observation: Any,
    info: Mapping[str, Any],
    user_simulator_config: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(observation, str):
        raise RuntimeError("Tau2 reset did not return a string observation")
    if not observation.strip():
        return _blocked_reset_summary(
            "Tau2 reset returned an empty initial observation; "
            "the user simulator may have failed to start",
            user_simulator_config,
        )
    tools = info.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        raise RuntimeError("Tau2 reset did not return a sequence of tools")
    policy = info.get("policy")
    if not isinstance(policy, str):
        raise RuntimeError("Tau2 reset did not return a policy string")
    return {
        "tool_count": len(tools),
        "policy_sha256": hashlib.sha256(policy.encode("utf-8")).hexdigest(),
        "initial_observation_length": len(observation),
        "user_simulator_config": _sanitize_user_simulator_config(user_simulator_config),
    }


def _blocked_reset_summary(
    reason: str, user_simulator_config: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "block_reason": reason,
        "tool_count": None,
        "policy_sha256": None,
        "initial_observation_length": None,
        "user_simulator_config": _sanitize_user_simulator_config(user_simulator_config),
    }


def _sanitize_user_simulator_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: _sanitize_value(config[field])
        for field in _USER_SIMULATOR_FIELDS
        if field in config
    }


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _REDACTED if _is_credential_key(key) else _sanitize_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_value(item) for item in value]
    return value


def _is_credential_key(key: Any) -> bool:
    normalized = "".join(character for character in str(key).casefold() if character.isalnum())
    return any(marker in normalized for marker in _CREDENTIAL_KEY_MARKERS)


def _print_payload(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
