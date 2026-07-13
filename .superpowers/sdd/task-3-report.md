# Task 3 Report: Periodic Memory Maintenance

## Status

Complete on branch `codex/stage-4-fast-loop`.

Implementation commit: `8c97d26f308ad73470793f49dac30910a8e0bb47`

The report is committed separately in the commit containing this file.

## TDD Evidence

### RED

Before any production edit, the owned Task 3 test file was created and run with:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py -q --basetemp=.pytest-tmp/maintenance-red
```

Result: collection failed with `ModuleNotFoundError` for
`tau3_retail_evolver.fast_loop.maintenance`, the expected missing feature.

### GREEN

Command:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py -q --basetemp=.pytest-tmp/task3
```

The first implementation run produced `59 passed, 1 failed`; the sole failure
was a test expectation that said `memory not found` while the reviewed Stage 3
API correctly raises `unknown memory`. After correcting only that assertion,
the required GREEN result was `60 passed in 0.96s`.

## Regression And Static Verification

Command:

```text
python -m pytest tests/unit/fast_loop tests/unit/memory -q --basetemp=.pytest-tmp/task3-regression
```

Result: `188 passed in 5.18s`.

Additional checks:

```text
python -m py_compile src/tau3_retail_evolver/fast_loop/maintenance.py tests/unit/fast_loop/test_maintenance.py
git diff --check
git diff --cached --check
```

All completed with exit code 0. The worktree Python environment does not have
`ruff` installed, so the optional Ruff invocation could not run; it is not a
configured or brief-required check.

## Contention And Persistence

- A deterministic two-thread test holds the first policy call with events,
  starts a competing call against a separately reopened repository, and then
  releases the first call without sleeps. Exactly one result has
  `executed=True`, the shared policy is called once, and only the successful
  three-event sequence is emitted.
- Round 1 remains completed after reopening `MemoryRepository`; repeated and
  resumed task-30 calls do not invoke policy. Task 60 then records round 2.
- A task-61 call with no prior round-2 state executes current round 2 once.
- State bytes are canonical JSON:
  `{"completed_rounds":[1,2],"schema_version":1}\n`.
- A simulated atomic state-write failure leaves no completion record after a
  real soft delete commits. Replaying the same command succeeds through Stage
  3 idempotence without incrementing the retired item's version again.

## Files Changed

- `src/tau3_retail_evolver/fast_loop/maintenance.py`
  - Adds frozen scheduler state/result interfaces, bounded public diagnostics,
    process-locked due-round execution, one-shot repair, real batch operations,
    atomic state persistence, and sanitized maintenance events.
- `tests/unit/fast_loop/test_maintenance.py`
  - Adds 60 tests covering scheduling, persistence, diagnostics, commands,
    guards, failures, event evidence, replay, and thread contention.
- `.superpowers/sdd/task-3-report.md`
  - Records Task 3 implementation and verification evidence.

Task 1 decisions/prompts, Task 2 runner/events, Stage 3 memory code, and
`fast_loop.__init__` were reused unchanged.

## State Schema

`maintenance_state.json` is scheduler metadata at the repository root, not a
memory tier. It contains exactly:

```json
{"completed_rounds":[1,2],"schema_version":1}
```

Completed rounds must be positive, sorted, and unique. Loading rejects invalid
JSON, extra/missing structure, unknown schema, zero/negative rounds,
duplicates, noncanonical ordering, and rounds later than the current schedule.
The file is written with `write_bytes_atomic` only after `MemoryOperations`
successfully applies the validated command batch.

## Event Schema

All events use the existing `RunContext` envelope and provenance task key
`maintenance-round-<round>`. No task or run ID enters the policy prompt.

Successful order:

```text
MaintenanceStarted
MaintenanceProposed
MaintenanceCommitted
```

- `MaintenanceStarted` records round, completed train-task count, period, and
  the four bounded per-tier item counts.
- `MaintenanceProposed` records canonical validated commands and
  `repair_used`; raw output, repaired output, parse text, attribution, and
  hidden memory fields are never included.
- `MaintenanceCommitted` records looked-up, created, and updated IDs plus the
  canonical completed-round list.
- Failures after start emit `MaintenanceFailed` with round and generic
  `{type, message}` evidence. If failure logging also fails, that secondary
  error is attached as a note and the original exception is re-raised.
- Round zero and already-completed rounds emit no events.

## Self-Review

- Confirmed schedule integers, train/learn context, and concrete mutable
  repository are checked before policy calls or state mutation.
- Confirmed the maintenance process lock covers state reload, due decision,
  diagnostics/policy, command application, and completion persistence.
- Confirmed diagnostics include every tier in enum order, active items only,
  deterministic ID order, the configured bound, and exactly the six approved
  public item fields.
- Confirmed decision parsing performs at most one policy repair and validates
  mixed lookup/write batches and exact write-command round before proposal.
- Confirmed cross-tier merge, soft delete, lookup, merge replay, and batch
  atomicity remain delegated to the reviewed Stage 3 operations.
- Confirmed invalid repair, operation failure, and state-write failure do not
  mark a round complete and emit sanitized evidence after start.
- Confirmed only brief-owned implementation, test, and report files were
  changed. The unrelated untracked Stage 4 plan remains untouched and unstaged.

## Commits

- `8c97d26f308ad73470793f49dac30910a8e0bb47` - implementation and tests.
- Report commit - the commit containing this report.

## Concerns

- Ruff was unavailable in the active Python environment; required compilation,
  diff checks, focused tests, and regression tests all passed.
- Contention is exercised with two threads and separate repository instances,
  using the same existing process-lock implementation used across processes.
  No external policy, real Qwen endpoint, or multi-process CI worker was run.

## Fix Review Findings

### RED Evidence

The two attribution regressions were added before the production edit and run
with:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py::test_nested_camelcase_attribution_triggers_clean_repair tests/unit/fast_loop/test_maintenance.py::test_attribution_separator_variant_after_repair_fails_without_mutation -q --basetemp=.pytest-tmp/task3-review-red-attribution
```

Both tests failed for the reviewed behavior:

- Nested `attributionScore` was accepted and persisted without a repair call.
- A repaired nested `attribution.score` was accepted instead of raising, so
  the maintenance round and merge were committed.

The replacement contention test was also run before the production edit:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py::test_two_threads_execute_same_round_only_once -q --basetemp=.pytest-tmp/task3-review-lock
```

It passed while wrapping, rather than replacing, the actual scheduler lock.
The wrapper signals immediately before each real lock `__enter__`; the main
thread now waits for the second attempt signal before releasing the first
policy call. A delegating `apply_batch` wrapper counts real applications.

### Minimal Fix

- Maintenance semantic validation now recursively visits merge metadata
  mappings and sequences, normalizes each key with Unicode NFKC, case folding,
  and removal of non-alphanumeric separators, and rejects any normalized key
  containing `attributionscore`.
- Rejection happens inside the existing decision validator, so initial invalid
  metadata receives exactly one repair opportunity. Invalid repaired metadata
  fails before `MaintenanceProposed`, memory operations, or state completion.
- Attribution is rejected rather than removed. Valid repaired metadata is the
  only metadata emitted and persisted.
- The contention test tracks two real scheduler-lock entry attempts and waits
  deterministically without sleeps. It asserts two attempts, one policy call,
  one real operation application, one three-event sequence, and one completed
  round.

### GREEN And Regression Evidence

Targeted review tests:

```text
3 passed in 0.29s
```

Task 3 suite:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py -q --basetemp=.pytest-tmp/task3-review
62 passed in 1.27s
```

Fast-loop and memory regression:

```text
python -m pytest tests/unit/fast_loop tests/unit/memory -q --basetemp=.pytest-tmp/task3-review-regression
190 passed in 5.70s
```

Changed-module compilation and `git diff --check` completed with exit code 0.
The review fix is committed in the commit containing this section.

### Contention Review Follow-Up

#### Why The Previous Synchronization Did Not Prove Contention

The previous tracking context manager set its second-attempt event before it
delegated to the real `ReentrantFileLock.__enter__`. Waiting for that event only
proved that the second worker had reached the wrapper. The first worker could
release the real RLock after the event was set but before the second worker
actually tried to acquire it, so the green test did not prove overlapping lock
contention. This is recorded as the RED-equivalent diagnosis permitted by the
review request; no production behavior or production code changed.

The active CPython runtime reports the real `_thread.RLock.acquire` signature
as `(blocking=True, timeout=-1)` and accepts `blocking=False`.

#### Revised Deterministic Synchronization

- The test obtains the actual scheduler `ReentrantFileLock` from the production
  `reentrant_process_lock` factory for the real repository root and namespace.
- It saves that object's real `_thread_lock` RLock and replaces only the field
  with a proxy. Production `ReentrantFileLock.__enter__`/`__exit__`, depth
  tracking, OS/file locking, and the real underlying RLock remain active.
- On the second calling thread, proxy `acquire()` first delegates
  `actual_rlock.acquire(blocking=False)`. If that succeeds, the contention
  event remains unset and the test times out. Only an explicit `False`, proving
  that the first thread currently owns the real RLock, sets
  `second_contended`; the proxy then delegates the real blocking `acquire()`.
- Proxy `release()` delegates directly to the saved real RLock. No no-lock fake
  or substituted scheduler context manager is used.
- The main thread waits for `second_contended` before releasing the blocked
  first policy call. Assertions require two lock-acquire calls, one policy
  generation, one delegated real `apply_batch`, one three-event sequence, and
  one completed state round.

#### Verification

Targeted contention test:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py::test_two_threads_execute_same_round_only_once -q --basetemp=.pytest-tmp/task3-contention-rlock
1 passed in 0.25s
```

Task 3 suite:

```text
python -m pytest tests/unit/fast_loop/test_maintenance.py -q --basetemp=.pytest-tmp/task3-contention-review
62 passed in 1.03s
```

Fast-loop and memory regression:

```text
python -m pytest tests/unit/fast_loop tests/unit/memory -q --basetemp=.pytest-tmp/task3-contention-review-regression
190 passed in 5.22s
```

Only the Task 3 test and this report changed in this follow-up.
