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
