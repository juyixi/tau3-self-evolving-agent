# OpenRouter NL Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Tau2 retail NL assertions use configurable OpenRouter GPT-4.1 by default without modifying the pinned Tau2 checkout or leaking credentials.

**Architecture:** Add a strict project configuration model for the evaluator, then bind those values into the already-imported Tau2 evaluator module after the verified Tau2 runtime is loaded. Record only public evaluator provenance in manifest schema version 2 and fail before artifact creation when the configured credential is absent.

**Tech Stack:** Python 3.12-3.13, Pydantic v2, PyYAML, pytest, Tau2, LiteLLM

## Global Constraints

- Default model is exactly `openrouter/openai/gpt-4.1`.
- Default credential variable is exactly `OPENROUTER_API_KEY`; its value must never enter configuration, logs, events, commands, or manifests.
- Default NL assertion temperature is exactly `0.0`.
- Do not modify files under `external/tau2-bench`.
- Preserve Qwen serving, DeepSeek user simulator, no-adapter, and no-memory baseline behavior.
- Every production behavior is implemented test-first and the complete pytest suite must pass.

---

### Task 1: Strict Evaluation Configuration

**Files:**
- Modify: `src/tau3_retail_evolver/config.py`
- Modify: `configs/default.yaml`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `NLAssertionsConfig(model: str, model_args: dict[str, Any], api_key_env: str)`
- Produces: `EvaluationConfig(nl_assertions: NLAssertionsConfig)`
- Extends: `ProjectConfig.evaluation: EvaluationConfig`

- [ ] **Step 1: Write failing default and validation tests**

Add assertions that the default config resolves to `openrouter/openai/gpt-4.1`, `{"temperature": 0.0}`, and `OPENROUTER_API_KEY`. Add parameterized temporary-YAML tests proving blank `model`, blank `api_key_env`, invalid environment variable names, and credential-shaped keys inside `model_args` are rejected.

- [ ] **Step 2: Run configuration tests and verify RED**

Run: `python -m pytest tests/unit/test_config.py -q`

Expected: FAIL because `ProjectConfig` has no `evaluation` field.

- [ ] **Step 3: Implement strict Pydantic models and defaults**

Add `NLAssertionsConfig` and `EvaluationConfig`. Use `Field(min_length=1)` for the model, an environment-variable-name pattern for `api_key_env`, and a field validator that recursively rejects credential keys using the same public-key policy as artifact sanitization. Add the `evaluation` block to `configs/default.yaml`.

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run: `python -m pytest tests/unit/test_config.py -q`

Expected: all configuration tests PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add configs/default.yaml src/tau3_retail_evolver/config.py tests/unit/test_config.py
git commit -m "feat: configure nl assertion evaluator"
```

### Task 2: Tau2 Evaluator Binding

**Files:**
- Create: `src/tau3_retail_evolver/evaluation/__init__.py`
- Create: `src/tau3_retail_evolver/evaluation/tau2_nl_assertions.py`
- Create: `tests/unit/evaluation/test_tau2_nl_assertions.py`

**Interfaces:**
- Consumes: `NLAssertionsConfig`
- Produces: `bind_tau2_nl_assertions(config, *, environ=None, module_loader=importlib.import_module) -> dict[str, Any]`
- Returns: `{"model": str, "model_args": dict[str, Any], "api_key_env": str}` with no credential value

- [ ] **Step 1: Write failing binding tests**

Cover successful binding against a `ModuleType` test double, missing and whitespace-only credentials, missing Tau2 module attributes, deep-copy isolation for `model_args`, and absence of the secret value from the return value and exception messages.

- [ ] **Step 2: Run binding tests and verify RED**

Run: `python -m pytest tests/unit/evaluation/test_tau2_nl_assertions.py -q`

Expected: collection FAIL because `tau3_retail_evolver.evaluation.tau2_nl_assertions` does not exist.

- [ ] **Step 3: Implement the minimal binding module**

Check `api_key_env` in the supplied mapping before loading Tau2. Load `tau2.evaluator.evaluator_nl_assertions`, require `DEFAULT_LLM_NL_ASSERTIONS` and `DEFAULT_LLM_NL_ASSERTIONS_ARGS`, replace both globals with the configured values, and return only public provenance. Raise stable `RuntimeError` or `EnvironmentError` messages that name the variable or missing contract member without including secret values.

- [ ] **Step 4: Run binding tests and verify GREEN**

Run: `python -m pytest tests/unit/evaluation/test_tau2_nl_assertions.py -q`

Expected: all binding tests PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/tau3_retail_evolver/evaluation tests/unit/evaluation
git commit -m "feat: bind tau2 nl assertion evaluator"
```

### Task 3: Baseline and Manifest Integration

**Files:**
- Modify: `scripts/run_baseline.py`
- Modify: `src/tau3_retail_evolver/runs/manifest.py`
- Modify: `tests/unit/scripts/test_run_baseline.py`
- Modify: manifest unit tests discovered under `tests/unit/runs/`

**Interfaces:**
- Consumes: `bind_tau2_nl_assertions(config.evaluation.nl_assertions) -> dict[str, Any]`
- Extends: `create_manifest(..., evaluation_config: Mapping[str, Any], ...)`
- Produces: manifest `schema_version: 2` and `evaluation_config.nl_assertions`

- [ ] **Step 1: Write failing integration and manifest tests**

Assert that `run_baseline.main` calls the binder after verified Tau2 loading but before probe construction and manifest creation. Assert schema version 2, exact public evaluation provenance, and that a known OpenRouter secret appears nowhere in manifest text or stdout. Update direct `create_manifest` tests to require the new argument and expected schema.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/unit/scripts/test_run_baseline.py tests/unit/runs -q`

Expected: FAIL because the binder is not called and `create_manifest` does not accept `evaluation_config`.

- [ ] **Step 3: Implement baseline binding and manifest schema 2**

In `run_baseline.main`, call the binder immediately after `load_verified_gym_factory`. Wrap its returned value as `{"nl_assertions": binding}` and pass it to `create_manifest`. Extend `create_manifest` with the required mapping, sanitize it, write it into the manifest, and set `schema_version` to `2`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest tests/unit/scripts/test_run_baseline.py tests/unit/runs -q`

Expected: all focused tests PASS.

- [ ] **Step 5: Run the complete regression suite**

Run: `python -m pytest -q`

Expected: all tests PASS with only the repository's documented integration skips.

- [ ] **Step 6: Verify repository scope and secret hygiene**

Run: `git diff --check`

Run: `git status --short`

Expected: only planned files plus the pre-existing unstaged `docs/superpowers/plans/2026-07-09-tau3-retail-opd-evolver.md` change are present; no file under `external/tau2-bench` is modified.

- [ ] **Step 7: Commit Task 3**

```powershell
git add scripts/run_baseline.py src/tau3_retail_evolver/runs/manifest.py tests/unit/scripts/test_run_baseline.py tests/unit/runs
git commit -m "feat: use openrouter for tau2 nl evaluation"
```

### Task 4: Real One-Task Validation Guide

**Files:**
- Modify: `docs/archive/2026-07-stage-1-2-delivery.md`

**Interfaces:**
- Consumes: the existing `scripts.run_baseline` CLI
- Produces: exact PowerShell setup and rerun instructions without embedding credentials

- [ ] **Step 1: Add operator instructions**

Document secure current-session setup with `$env:OPENROUTER_API_KEY = Read-Host "OpenRouter API Key"`, retain the existing Qwen tunnel variables, require a new immutable run ID, and explain that the key belongs on the machine running `scripts.run_baseline`, not the AutoDL vLLM host.

- [ ] **Step 2: Check documentation and repository diff**

Run: `git diff --check`

Expected: PASS with no whitespace errors.

- [ ] **Step 3: Commit Task 4**

```powershell
git add docs/archive/2026-07-stage-1-2-delivery.md
git commit -m "docs: add openrouter baseline validation"
```

### Task 5: Final Verification

**Files:**
- No new files

**Interfaces:**
- Verifies all prior task contracts together.

- [ ] **Step 1: Run all tests from a clean process**

Run: `python -m pytest -q`

Expected: all tests PASS with only documented integration skips.

- [ ] **Step 2: Inspect committed scope**

Run: `git log --oneline -6`

Run: `git status --short`

Expected: the feature commits are present; only the pre-existing unrelated plan-file modification remains unstaged.

- [ ] **Step 3: Report the real-world validation boundary**

State explicitly that automated tests verify configuration, binding, ordering, manifest provenance, and secret hygiene. Do not claim the live OpenRouter evaluation has passed until the user runs one real task with valid credentials and reports the resulting summary.
