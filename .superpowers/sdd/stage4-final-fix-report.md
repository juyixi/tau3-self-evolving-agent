# Stage 4 Final Fix Report

## Scope

Completed every finding in `stage4-final-fix-brief.md` without reverting unrelated work. The worktree initially contained an uncommitted partial RED diff in `tests/unit/fast_loop/test_runner.py`; that test work was preserved, completed, and verified before production edits.

## Implemented Findings

1. Unified write metadata validation now applies Unicode NFKC, casefold, and alphanumeric comparison, rejects any nested attribution-score variant or credential-bearing key, performs exactly one policy repair, and retains a defensive proposal-time filter/assertion.
2. Reset policy, tools, observation, and task instruction now pass through the public prompt projection immediately after reset and before events, retrieval, embeddings, or policy calls. Credential values are redacted and forbidden nested fields fail with only a generic `EpisodeFailed` event before environment close.
3. `MemorySelected` now stores canonical selected IDs/evidence, repair and parse booleans, sampling/latency data, and SHA-256 output hashes. Raw output, repaired output, and validation text are not persisted.
4. The CLI assigns every requested task to the `retail` task group and uses an exact pre-episode Memory snapshot in a replaced episode context. Maintenance checks receive a replaced post-episode snapshot context; the manifest remains the immutable run-input snapshot.
5. Maintenance diagnostics deterministically truncate each content field to `MAX_DIAGNOSTIC_CONTENT_CHARS` without mutating Memory. `MaintenanceStarted` records the complete sanitized diagnostics actually supplied to the maintenance prompt.
6. `OpenAICompatibleHttpClient` has a finite positive `request_timeout_s` option with a 120-second default. Only the default stdlib transport binds it to `urlopen`; injected three-argument transports are unchanged. The opt-in real subprocess has a 1800-second timeout and the Chinese validation guide documents both bounds.
7. Partial write progress separately records newly committed IDs and safe replay IDs, preserving order and duplicate/multiset evidence. Successful write events retain their prior `written_memory_ids` behavior.
8. The real five-task smoke now checks selection is a subset of candidates, terminal step completion/reward consistency, nonempty official terminal mappings, per-episode snapshot consistency, and the existing manifest/source provenance.

## TDD Evidence

### RED

- Runner/public/selection/write progress: `python -m pytest -q tests/unit/fast_loop/test_runner.py --basetemp=.pytest-tmp/stage4-final-red-runner` -> 9 failed, 18 passed. Failures were the missing safe selection fields, late public validation/redaction, absent write metadata repair, and replay IDs counted as commits.
- Maintenance: `python -m pytest -q tests/unit/fast_loop/test_maintenance.py --basetemp=.pytest-tmp/stage4-final-red-maintenance` -> 2 failed, 61 passed. Failures were unbounded content and missing `MaintenanceStarted.diagnostics`.
- CLI snapshots/task group: focused two-task orchestration test -> 1 failed because both episodes reused the manifest snapshot and task groups were empty.
- HTTP timeout: focused timeout tests -> 5 failed because no timeout option/default forwarding existed.
- Strict timeout types: focused invalid-timeout test -> 2 failed, 4 passed because `True` was accepted and string input raised `TypeError` instead of the generic validation error.

### GREEN

- Batch A: runner 27 passed; CLI 12 passed; public prompt tests 16 passed.
- Batch B: maintenance plus prompt tests 79 passed.
- Batch C: policy tests initially 51 passed, then 53 passed after strict timeout-type coverage; CLI 12 passed; integration module 2 passed and 1 opt-in real smoke skipped.

## Final Verification

- `python -m pytest -q --basetemp=.pytest-tmp/stage4-final-fix` -> 374 passed, 3 skipped in 7.89s.
- `python -m compileall -q src scripts tests` -> exit 0.
- `git diff --check` -> exit 0. Git emitted only the repository's existing LF-to-CRLF checkout warnings.

## Residual Concern

The credential-gated real five-task Tau2/Qwen smoke remained skipped because the required external services and credentials were not enabled in this run. Its collection, timeout, and local assertion helpers passed; the full external execution still requires `RUN_FAST_LOOP_TAU2_INTEGRATION=1` and the documented environment variables.

## Final Re-review Addendum

### Additional Findings Fixed

1. Runner metadata validation now preserves camelCase through Unicode NFKC normalization. Attribution matching uses a separately casefolded/alphanumeric compact key, while credential matching receives the original NFKC string. The proposal-time defensive filter follows the same rule. Initial and repaired write decisions containing `dbPassword` or `refreshToken` are rejected before proposals or persistence. Maintenance merge metadata recursively applies the same credential rule; an `apiToken` decision receives one clean repair and the rejected value reaches neither events nor Memory.
2. `RunContext.default_task_group` was added after the existing default fields with a backward-compatible `"baseline"` default. Fast-loop CLI contexts set it to `"retail"`, so synthetic `maintenance-round-*` events are also classified as retail while baseline behavior is unchanged.
3. `--completed-train-tasks-before` is now an argparse-required option with no silent zero default. All CLI, unit, integration, and guide invocations provide it explicitly; omission fails during argument parsing.
4. If replay lookup fails after a duplicate-add `ValueError`, the lookup exception now receives the exact already-committed and already-replayed progress before it is re-raised. `MemoryWriteFailed` therefore remains retry-safe even when replay verification itself fails.

### Additional TDD Evidence

- RED runner: `dbPassword` and `refreshToken` produced four credential-validation failures. After correcting the replay fixture to avoid retrieval-time lookup, its focused RED reached `MemoryWriteFailed` with `committed_memory_ids=[]` instead of the expected prior commit.
- RED maintenance: 1 failed, 63 passed because nested `apiToken` metadata bypassed semantic repair.
- RED CLI/task group: 3 failed, 10 passed because the cumulative-count option was optional and `RunContext` had no retail fallback for synthetic maintenance events.
- GREEN runner: 32 passed.
- GREEN maintenance: 64 passed.
- GREEN CLI plus baseline compatibility: 20 passed.
- GREEN integration collection: 2 passed, 1 credential-gated real smoke skipped.

### Updated Final Verification

- `python -m pytest -q --basetemp=.pytest-tmp/stage4-final-review-fix` -> 381 passed, 3 skipped in 6.24s.
- `python -m compileall -q src scripts tests` -> exit 0.
- `git diff --check` -> exit 0, with only LF-to-CRLF checkout warnings.
