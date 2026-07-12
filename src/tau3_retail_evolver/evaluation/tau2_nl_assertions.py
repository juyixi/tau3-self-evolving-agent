"""Bind project NL assertion configuration to Tau2 evaluator defaults."""

import importlib
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from types import ModuleType
from typing import Any

from tau3_retail_evolver.config import NLAssertionsConfig


_TAU2_EVALUATOR_MODULE = "tau2.evaluator.evaluator_nl_assertions"
_REQUIRED_DEFAULTS = (
    "DEFAULT_LLM_NL_ASSERTIONS",
    "DEFAULT_LLM_NL_ASSERTIONS_ARGS",
)


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
    return {
        "model": config.model,
        "model_args": deepcopy(config.model_args),
        "api_key_env": config.api_key_env,
    }
