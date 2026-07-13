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
