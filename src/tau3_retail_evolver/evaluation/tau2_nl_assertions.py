"""Bind project NL assertion configuration to Tau2 evaluator defaults."""

import importlib
import json
import os
import re
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from functools import wraps
from types import ModuleType
from typing import Any

from tau3_retail_evolver.config import NLAssertionsConfig


_TAU2_EVALUATOR_MODULE = "tau2.evaluator.evaluator_nl_assertions"
_REQUIRED_DEFAULTS = (
    "DEFAULT_LLM_NL_ASSERTIONS",
    "DEFAULT_LLM_NL_ASSERTIONS_ARGS",
)
_NL_ASSERTION_JSON_ATTEMPTS = 3
_JSON_GUARD_MARKER = "_tau3_nl_assertion_json_guard"


def bind_tau2_nl_assertions(
    config: NLAssertionsConfig,
    *,
    environ: Mapping[str, str] | None = None,
    module_loader: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, Any]:
    """Bind NL assertion configuration to Tau2 and return public provenance."""
    environ = os.environ if environ is None else environ
    credential = environ.get(config.api_key_env)
    if not isinstance(credential, str) or not credential.strip():
        raise EnvironmentError(
            f"Missing or blank required environment variable: {config.api_key_env}"
        )

    evaluator_module = module_loader(_TAU2_EVALUATOR_MODULE)
    for member in _REQUIRED_DEFAULTS:
        if not hasattr(evaluator_module, member):
            raise RuntimeError(
                f"Tau2 evaluator module {_TAU2_EVALUATOR_MODULE} is missing {member}"
            )

    evaluator_module.DEFAULT_LLM_NL_ASSERTIONS = config.model
    evaluator_module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = deepcopy(config.model_args)
    _install_json_response_guard(evaluator_module)
    return {
        "model": config.model,
        "model_args": deepcopy(config.model_args),
        "api_key_env": config.api_key_env,
    }


def _install_json_response_guard(evaluator_module: ModuleType) -> None:
    generate = getattr(evaluator_module, "generate", None)
    if not callable(generate) or getattr(generate, _JSON_GUARD_MARKER, False):
        return

    @wraps(generate)
    def guarded_generate(*args: Any, **kwargs: Any) -> Any:
        last_error: ValueError | TypeError | None = None
        for _ in range(_NL_ASSERTION_JSON_ATTEMPTS):
            response = generate(*args, **kwargs)
            content = getattr(response, "content", None)
            try:
                _load_json_object(content)
                return response
            except (TypeError, ValueError) as error:
                last_error = error

            repaired = _repair_json_content(content)
            if repaired is None or repaired == content:
                continue
            try:
                _load_json_object(repaired)
            except (TypeError, ValueError) as error:
                last_error = error
                continue
            return _replace_response_content(response, repaired)

        assert last_error is not None
        raise last_error

    setattr(guarded_generate, _JSON_GUARD_MARKER, True)
    evaluator_module.generate = guarded_generate


def _load_json_object(content: Any) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("NL assertion evaluator returned blank non-text content")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("NL assertion evaluator response must be a JSON object")
    return payload


def _repair_json_content(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    object_start = candidate.find("{")
    object_end = candidate.rfind("}")
    if object_start >= 0 and object_end > object_start:
        candidate = candidate[object_start : object_end + 1]

    candidate = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", candidate)
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", candidate)


def _replace_response_content(response: Any, content: str) -> Any:
    model_copy = getattr(response, "model_copy", None)
    if callable(model_copy):
        return model_copy(update={"content": content})
    cloned = copy(response)
    setattr(cloned, "content", content)
    return cloned
