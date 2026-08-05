from pathlib import Path

import pytest
from pydantic import ValidationError

from tau3_evolver.config import load_config
from tau3_evolver.serving.vllm import build_vllm_command


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_vllm_launcher_uses_canonical_131072_context(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    model_path = tmp_path / "original-qwen"

    command = build_vllm_command(
        config,
        model_path=model_path,
        executable="vllm-test",
    )

    context_index = command.index("--max-model-len")
    assert command[context_index + 1] == "131072"
    assert command[:3] == (
        "vllm-test",
        "serve",
        str(model_path.resolve()),
    )
    assert command[command.index("--served-model-name") + 1] == "Qwen/Qwen3.5-9B"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8000"


def test_model_context_cannot_be_overridden_to_a_smaller_window() -> None:
    with pytest.raises(ValidationError, match="max_context_tokens"):
        load_config(
            PROJECT_ROOT / "configs" / "default.yaml",
            overrides=("model.max_context_tokens=65536",),
        )
