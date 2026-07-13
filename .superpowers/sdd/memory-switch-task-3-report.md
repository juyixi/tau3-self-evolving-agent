# Task 3: CLI Orchestration, Manifest, And Summary Provenance

## Status

Complete. The fast-loop CLI now honors `config.memory.enabled` while retaining
the existing enabled-memory behavior and recording the switch in run artifacts.

## RED

Before modifying production code, the CLI test configured `memory.enabled=False`
and replaced `open_training_memory`, `build_embedding_provider`, and
`run_due_maintenance` with call-failing test doubles.

The first required command could not enter the tests because this environment
denied access to the default shared pytest temporary directory. Re-running the
same collection with an isolated writable base temp produced the valid RED:

```text
python -m pytest tests/unit/scripts/test_run_fast_loop.py -q --basetemp <isolated-temp>/pytest-task3
```

Result: `2 failed, 12 passed in 0.54s`.

- The enabled-path assertion failed because `rollout_options.memory_enabled`
  was not present.
- The disabled-path regression failed at the existing unconditional
  `open_training_memory(...)` call, with `AssertionError: memory dependency
  must not be called when disabled`.

This proved the required factory guard was absent before the implementation.

## GREEN

The minimal CLI change builds the repository, embedding provider, retriever,
and input snapshot only when memory is enabled. Disabled runs pass `None`
repository/retriever dependencies, use null snapshot IDs, and skip task-level,
maintenance, and output snapshots. The runner receives
`FastLoopConfig(memory_enabled=config.memory.enabled)`.

Focused verification:

```text
python -m pytest tests/unit/scripts/test_run_fast_loop.py -q --basetemp <isolated-temp>/pytest-task3
```

Result: `14 passed in 0.48s`.

Brief regression command:

```text
python -m pytest tests/unit/scripts/test_run_fast_loop.py tests/unit/runs/test_manifest.py tests/integration/test_fast_loop_tau2_retail.py -q --basetemp <isolated-temp>/pytest-task3-full
```

Result: `40 passed, 1 skipped in 0.54s`.

## Files

- `scripts/run_fast_loop.py`
- `tests/unit/scripts/test_run_fast_loop.py`
- `tests/integration/test_fast_loop_tau2_retail.py`
- `.superpowers/sdd/task-3-report.md`

## Self-Check

- Disabled CLI runs never call `open_training_memory`,
  `build_embedding_provider`, or `run_due_maintenance`; the regression test
  makes each dependency fail immediately if it is invoked.
- Disabled artifacts contain `rollout_options.memory_enabled=false`, a null
  manifest snapshot ID, `summary.memory_enabled=false`, two null summary
  snapshot IDs, and an empty maintenance-round list.
- The disabled regression asserts that no project `history` directory exists
  and that episode contexts use a null memory snapshot.
- Enabled unit and real-smoke assertions now require
  `rollout_options.memory_enabled=true` and `summary.memory_enabled=true`.
- `git diff --check` completed with exit code 0. Git printed only LF-to-CRLF
  advisories for the modified files.

## Commit

Commit message: `feat: record no-memory fast-loop ablations`

## Concerns

- The real TAU2 smoke test was skipped because its explicit endpoint and
  credential opt-in variables were not set; its enabled-memory provenance
  assertions were updated and collected successfully.
- The default pytest temporary directory is inaccessible in this environment,
  so verification used an isolated writable `--basetemp` outside the worktree.
- The pre-existing untracked `.pytest-tmp-review/` directory was not modified
  or staged.
