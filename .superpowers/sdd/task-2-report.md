# Task 2 Report: Memory Ablation Switch In Fast-Loop Runner

## Status

Complete. Memory-disabled episodes bypass retrieval, selection, and write
lifecycles while preserving the existing action, failure, and close behavior.

## Files

- `src/tau3_retail_evolver/fast_loop/runner.py`
- `tests/unit/fast_loop/test_runner.py`
- `.superpowers/sdd/task-2-report.md`

## RED

Initial brief command:

```text
python -m pytest tests/unit/fast_loop/test_runner.py -q
```

Output: `1 failed, 2 passed, 34 errors in 4.87s`. The new disabled-memory test
failed because `FastLoopConfig` did not accept `memory_enabled`; the 34 setup
errors were the known permission denial while pytest scanned the default user
temp directory.

Focused RED command with a workspace-local temp directory:

```text
python -m pytest tests/unit/fast_loop/test_runner.py::test_disabled_memory_bypasses_the_memory_lifecycle tests/unit/fast_loop/test_runner.py::test_memory_dependency_contract_fails_before_reset -q --basetemp=.pytest-tmp/task-2-red
```

Output: `5 failed in 0.32s`. Each failure was the expected missing
`FastLoopConfig(memory_enabled=...)` interface.

## GREEN

Focused GREEN command:

```text
python -m pytest tests/unit/fast_loop/test_runner.py::test_disabled_memory_bypasses_the_memory_lifecycle tests/unit/fast_loop/test_runner.py::test_memory_dependency_contract_fails_before_reset -q --basetemp=.pytest-tmp/task-2-green
```

Output: `5 passed in 0.23s`.

Brief regression command, using a workspace-local temp directory:

```text
python -m pytest tests/unit/fast_loop/test_runner.py tests/unit/models/test_policy.py -q --basetemp=.pytest-tmp/task-2-regression
```

Output: `90 passed in 0.78s`.

## Commit

`feat: bypass memory lifecycle in fast loop`

## Self-Check

- `FastLoopConfig` now defaults `memory_enabled` to `True`.
- Memory dependencies are checked before `environment.reset`; enabled episodes
  require a mutable repository and retriever, while disabled episodes reject
  either dependency being supplied.
- Disabled episodes emit exactly one `MemoryDisabled(reason="config")` event,
  send the policy only action prompts, omit `memories` from the action payload,
  and return empty selected and written IDs.
- Disabled episodes do not emit retrieval, selection, proposal, commit, or
  write-failure memory events.
- Retrieval/selection and write/persistence remain within the existing outer
  `try`/`except`/close flow, so action and environment failures retain their
  previous failure evidence and cleanup semantics.
- `python -m compileall -q src/tau3_retail_evolver/fast_loop/runner.py tests/unit/fast_loop/test_runner.py` completed with exit code 0.
- `git diff --check` completed with exit code 0; Git printed only existing
  LF-to-CRLF advisories.

## Concerns

- The default pytest temporary directory is inaccessible in this environment,
  so focused and regression verification use `--basetemp` inside the worktree.
- An unrelated untracked `.pytest-tmp-review/` directory was already present
  and was not modified or staged.
