import json
from types import ModuleType, SimpleNamespace

import pytest

from tau3_evolver.config import NLAssertionsConfig
from tau3_evolver.benchmarks.tau2.assertions import bind_tau2_nl_assertions


_TAU2_EVALUATOR_MODULE = "tau2.evaluator.evaluator_nl_assertions"


def _config() -> NLAssertionsConfig:
    return NLAssertionsConfig(
        model="openrouter/openai/gpt-4.1",
        model_args={"temperature": 0.0, "nested": {"retries": 2}},
        api_key_env="TEST_NL_ASSERTIONS_API_KEY",
    )


def _evaluator_module() -> ModuleType:
    module = ModuleType(_TAU2_EVALUATOR_MODULE)
    module.DEFAULT_LLM_NL_ASSERTIONS = "original-model"
    module.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 1.0}
    return module


def test_binds_configured_defaults_and_returns_redacted_provenance() -> None:
    config = _config()
    module = _evaluator_module()
    loaded_names: list[str] = []

    def load_module(name: str) -> ModuleType:
        loaded_names.append(name)
        return module

    provenance = bind_tau2_nl_assertions(
        config,
        environ={config.api_key_env: "literal-test-secret"},
        module_loader=load_module,
    )

    assert loaded_names == [_TAU2_EVALUATOR_MODULE]
    assert module.DEFAULT_LLM_NL_ASSERTIONS == config.model
    assert module.DEFAULT_LLM_NL_ASSERTIONS_ARGS == config.model_args
    assert provenance == {
        "model": config.model,
        "model_args": config.model_args,
        "api_key_env": config.api_key_env,
    }
    assert "literal-test-secret" not in repr(provenance)


def test_defaults_to_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    module = _evaluator_module()
    monkeypatch.setenv(config.api_key_env, "literal-test-secret")

    bind_tau2_nl_assertions(config, module_loader=lambda _: module)

    assert module.DEFAULT_LLM_NL_ASSERTIONS == config.model
    assert module.DEFAULT_LLM_NL_ASSERTIONS_ARGS == config.model_args


def test_model_args_are_deep_copy_isolated_from_the_config() -> None:
    config = _config()
    module = _evaluator_module()

    provenance = bind_tau2_nl_assertions(
        config,
        environ={config.api_key_env: "literal-test-secret"},
        module_loader=lambda _: module,
    )
    module.DEFAULT_LLM_NL_ASSERTIONS_ARGS["nested"]["retries"] = 9
    provenance["model_args"]["nested"]["retries"] = 7

    assert config.model_args == {"temperature": 0.0, "nested": {"retries": 2}}


def test_repairs_invalid_json_escape_without_another_provider_call() -> None:
    config = _config()
    module = _evaluator_module()
    calls = 0
    raw_output = (
        r'{"results":[{"expectedOutcome":"refund","reasoning":"path \q value",'
        r'"metExpectation":true}]}'
    )

    def generate(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(content=raw_output)

    module.generate = generate
    bind_tau2_nl_assertions(
        config,
        environ={config.api_key_env: "literal-test-secret"},
        module_loader=lambda _: module,
    )

    response = module.generate()

    assert calls == 1
    assert json.loads(response.content)["results"][0]["reasoning"] == r"path \q value"


def test_retries_unrepairable_json_twice_then_raises() -> None:
    config = _config()
    module = _evaluator_module()
    calls = 0

    def generate(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(content="not-json")

    module.generate = generate
    bind_tau2_nl_assertions(
        config,
        environ={config.api_key_env: "literal-test-secret"},
        module_loader=lambda _: module,
    )

    with pytest.raises(json.JSONDecodeError):
        module.generate()

    assert calls == 3


def test_json_response_guard_is_idempotent() -> None:
    config = _config()
    module = _evaluator_module()
    calls = 0

    def generate(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(content='{"results":[]}')

    module.generate = generate
    for _ in range(2):
        bind_tau2_nl_assertions(
            config,
            environ={config.api_key_env: "literal-test-secret"},
            module_loader=lambda _: module,
        )

    assert json.loads(module.generate().content) == {"results": []}
    assert calls == 1


@pytest.mark.parametrize("credential", [None, "", "   "])
def test_rejects_missing_or_blank_credentials_before_loading_tau2(
    credential: str | None,
) -> None:
    config = _config()
    loaded = False

    def load_module(_: str) -> ModuleType:
        nonlocal loaded
        loaded = True
        return _evaluator_module()

    environ = {} if credential is None else {config.api_key_env: credential}

    with pytest.raises(EnvironmentError) as error:
        bind_tau2_nl_assertions(config, environ=environ, module_loader=load_module)

    assert config.api_key_env in str(error.value)
    assert "literal-test-secret" not in str(error.value)
    assert not loaded


@pytest.mark.parametrize(
    "missing_member",
    ["DEFAULT_LLM_NL_ASSERTIONS", "DEFAULT_LLM_NL_ASSERTIONS_ARGS"],
)
def test_rejects_tau2_modules_missing_required_contract_members(
    missing_member: str,
) -> None:
    config = _config()
    module = _evaluator_module()
    delattr(module, missing_member)

    with pytest.raises(RuntimeError) as error:
        bind_tau2_nl_assertions(
            config,
            environ={config.api_key_env: "literal-test-secret"},
            module_loader=lambda _: module,
        )

    assert _TAU2_EVALUATOR_MODULE in str(error.value)
    assert missing_member in str(error.value)
    assert "literal-test-secret" not in str(error.value)
    if missing_member == "DEFAULT_LLM_NL_ASSERTIONS_ARGS":
        assert module.DEFAULT_LLM_NL_ASSERTIONS == "original-model"
