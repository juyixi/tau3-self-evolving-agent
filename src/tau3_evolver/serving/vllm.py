from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse

from tau3_evolver.config import ProjectConfig, load_config


def build_vllm_command(
    config: ProjectConfig,
    *,
    model_path: Path,
    executable: str = "vllm",
) -> tuple[str, ...]:
    endpoint = urlparse(config.model.serving_base_url)
    if endpoint.scheme != "http" or endpoint.hostname is None:
        raise ValueError("model.serving_base_url must be an HTTP endpoint")
    if endpoint.path.rstrip("/") != "/v1" or endpoint.query or endpoint.fragment:
        raise ValueError("model.serving_base_url must end at the /v1 API root")
    port = endpoint.port or 80
    return (
        executable,
        "serve",
        str(model_path.expanduser().resolve()),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--host",
        endpoint.hostname,
        "--port",
        str(port),
        "--disable-uvicorn-access-log",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(config.model.max_context_tokens),
        "--served-model-name",
        config.model.served_model_name,
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "qwen3",
        "--gpu-memory-utilization",
        "0.72",
        "--language-model-only",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve the project Qwen policy with the canonical vLLM contract."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(arguments.config)
    command = build_vllm_command(config, model_path=arguments.model_path)
    os.execvp(command[0], command)
    return 2


__all__ = ["build_vllm_command", "main"]
