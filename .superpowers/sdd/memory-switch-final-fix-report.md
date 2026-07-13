# Memory Switch Final Fix Report

## Status

DONE

## Scope

This final review fix preserves the `FastLoopConfig` positional API, validates
the global memory switch strictly, and makes CLI orchestration fail closed when
the switch and memory dependencies disagree. It also corrects the three Stage
4 report self-references.

## Root Cause

- `memory_enabled` was inserted before the existing two positional limit
  fields, so `FastLoopConfig(7, 4)` bound `7` to a boolean field and `4` to
  `retrieve_top_k`.
- `FastLoopConfig` relied on annotations alone and accepted non-boolean values,
  including integer values despite `bool` being an `int` subclass.
- `_run_requested_tasks` used repository presence for snapshots and maintenance
  while the episode runner used `fast_loop_config.memory_enabled`. Direct calls
  with inconsistent arguments could therefore create an environment or access
  a snapshot before failing in the episode runner.

## TDD Evidence

### RED

Command:

```text
python -m pytest -q tests/unit/fast_loop/test_runner.py::test_fast_loop_config_preserves_positional_limit_arguments tests/unit/fast_loop/test_runner.py::test_fast_loop_config_rejects_non_boolean_memory_enabled tests/unit/scripts/test_run_fast_loop.py::test_run_requested_tasks_rejects_mismatched_memory_dependencies_before_side_effects tests/unit/scripts/test_run_fast_loop.py::test_run_requested_tasks_disabled_uses_real_episode_without_memory_access --basetemp=.pytest-tmp/memory-switch-final-fix-red-final
```

Result: `9 failed, 1 passed in 0.55s`.

- `FastLoopConfig(7, 4)` produced `retrieve_top_k=4` instead of preserving the
  original positional mapping.
- String, integer, and `None` values for `memory_enabled` were accepted.
- All four orchestration dependency mismatches did not raise the required
  fail-closed error before entering the task loop; existing behavior either
  reached the episode runner or performed a snapshot/environment side effect.
- The real disabled cross-layer episode test passed on the pre-fix baseline,
  establishing its memory-free behavior as a regression guard rather than a
  test of the newly added validation.

### GREEN

Command:

```text
python -m pytest -q tests/unit/fast_loop/test_runner.py::test_fast_loop_config_preserves_positional_limit_arguments tests/unit/fast_loop/test_runner.py::test_fast_loop_config_rejects_non_boolean_memory_enabled tests/unit/scripts/test_run_fast_loop.py::test_run_requested_tasks_rejects_mismatched_memory_dependencies_before_side_effects tests/unit/scripts/test_run_fast_loop.py::test_run_requested_tasks_disabled_uses_real_episode_without_memory_access --basetemp=.pytest-tmp/memory-switch-final-fix-green
```

Result: `10 passed in 0.26s`.

## Changes

- Restored `FastLoopConfig` positional order to `retrieve_top_k`,
  `max_episode_steps`, then `memory_enabled`.
- Added exact-`bool` validation with `type(value) is bool`, retaining the
  existing positive limit validation.
- Tightened `_run_requested_tasks` annotations to `MemoryRepository | None`,
  `Retriever | None`, and `FastLoopConfig`.
- Added a single entry validation before the task loop. Enabled memory requires
  both dependencies; disabled memory requires neither. Every mismatch raises
  `ValueError` before snapshot, environment factory, or maintenance calls.
- Routed input, per-task, maintenance, and output snapshot decisions through
  `fast_loop_config.memory_enabled`.
- Added an integration-shaped disabled test that invokes the real
  `run_fast_loop_episode` through `_run_requested_tasks` with a fake
  environment and policy. It confirms no memory prompt payload, memory
  lifecycle events, selected IDs, written IDs, or maintenance calls occur.
- Corrected the three report self-reference paths.

## Files

- `src/tau3_retail_evolver/fast_loop/runner.py`
- `scripts/run_fast_loop.py`
- `tests/unit/fast_loop/test_runner.py`
- `tests/unit/scripts/test_run_fast_loop.py`
- `.superpowers/sdd/memory-switch-task-1-report.md`
- `.superpowers/sdd/memory-switch-task-2-report.md`
- `.superpowers/sdd/memory-switch-task-3-report.md`
- `.superpowers/sdd/memory-switch-final-fix-report.md`

## Verification

```text
python -m pytest -q tests/unit/fast_loop/test_runner.py tests/unit/scripts/test_run_fast_loop.py --basetemp=.pytest-tmp/memory-switch-final-fix
61 passed in 0.93s

python -m pytest -q --basetemp=.pytest-tmp/memory-switch-final-fix-full
401 passed, 3 skipped in 6.65s

python -m compileall -q src scripts tests
exit 0

git diff --check
exit 0
```

The `git diff --check` invocation printed only line-ending advisories and no
whitespace errors.

## Concerns

- The pre-existing untracked `.pytest-tmp-review/` directory was left
  untouched and is excluded from staging.

## Final review wave 2

### Root Cause

- `run_fast_loop_episode` accepted any non-`None` retriever. A fabricated
  object therefore reached `environment.reset()` before failing at
  `retriever.retrieve()`.
- `_run_requested_tasks` duplicated a weaker presence-only dependency check.
  It dereferenced `fast_loop_config.memory_enabled` without verifying the
  config type, then could snapshot a read-only or fabricated repository or
  construct an environment before the episode runner rejected the input.

### TDD Evidence

#### RED

The default pytest temporary directory was inaccessible in this environment,
so the first invocation failed during fixture setup with `PermissionError`.
The required workspace-local base temp command then ran the new tests:

```text
python -m pytest -q tests/unit/fast_loop/test_runner.py tests/unit/scripts/test_run_fast_loop.py -k "memory_dependency_contract_fails_before_reset or validates_runtime_contract_before_side_effects" --basetemp=.pytest-tmp/memory-switch-final-review-wave2
```

Result: `5 failed, 4 passed, 57 deselected in 0.44s`.

- The runner called `reset()` before rejecting a fabricated retriever.
- Orchestration raised `AttributeError` for a fabricated config, called
  `snapshot()` on a read-only repository, called `snapshot()` on a fabricated
  repository, and constructed an environment before rejecting a fabricated
  retriever.

#### GREEN

```text
python -m pytest -q tests/unit/fast_loop/test_runner.py tests/unit/scripts/test_run_fast_loop.py --basetemp=.pytest-tmp/memory-switch-final-review-wave2
```

Result: `66 passed in 1.01s`.

### Changes

- Added reusable `validate_fast_loop_dependencies` in the runner module. It
  requires an actual `FastLoopConfig`; enabled memory requires a mutable
  `MemoryRepository` and a `Retriever`; disabled memory requires both
  dependencies to be `None`.
- Called that validator at the start of both `run_fast_loop_episode` and
  `_run_requested_tasks`, before reset, snapshot, environment construction, or
  maintenance.
- Added runner coverage for a fabricated retriever and orchestration coverage
  for fabricated config, read-only repository, fabricated repository, and
  fabricated retriever. Each case attaches a snapshot spy when available and
  asserts zero snapshot, environment-factory, and maintenance calls.
- Corrected both plan references to the canonical `FastLoopConfig` field
  order: `retrieve_top_k`, `max_episode_steps`, `memory_enabled`.

### Files

- `src/tau3_retail_evolver/fast_loop/runner.py`
- `scripts/run_fast_loop.py`
- `tests/unit/fast_loop/test_runner.py`
- `tests/unit/scripts/test_run_fast_loop.py`
- `docs/superpowers/plans/2026-07-13-memory-ablation-switch.md`
- `.superpowers/sdd/memory-switch-final-fix-report.md`

### Verification

```text
python -m pytest -q --basetemp=.pytest-tmp/memory-switch-final-review-wave2-full
406 passed, 3 skipped in 6.40s

python -m compileall -q src scripts tests
exit 0

git diff --check
exit 0 (line-ending advisories only)
```

The pre-existing `.pytest-tmp-review/` directory remains untouched and is not
staged.
