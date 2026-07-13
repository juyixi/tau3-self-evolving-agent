# Stage 4 Task 1 Report

## Status

Completed and committed on branch `codex/stage-4-fast-loop`.

## TDD Evidence

### RED

Command:

```powershell
python -m pytest tests/unit/fast_loop/test_decisions.py tests/unit/fast_loop/test_prompts.py -q --basetemp=.pytest-tmp/task1
```

Result: failed during collection as expected because both required modules did not yet
exist:

- `ModuleNotFoundError: No module named 'tau3_retail_evolver.fast_loop.decisions'`
- `ModuleNotFoundError: No module named 'tau3_retail_evolver.fast_loop.prompts'`

### GREEN

Command:

```powershell
python -m pytest tests/unit/fast_loop/test_decisions.py tests/unit/fast_loop/test_prompts.py -q --basetemp=.pytest-tmp/task1
```

Result: `12 passed in 0.20s`.

Relevant existing-memory and fast-loop regression command:

```powershell
python -m pytest tests/unit/memory tests/unit/fast_loop -q --basetemp=.pytest-tmp/task1-regression
```

Result: `92 passed in 4.03s`.

Additional verification:

```powershell
python -m compileall -q src/tau3_retail_evolver/fast_loop/decisions.py src/tau3_retail_evolver/fast_loop/prompts.py
git diff --check
```

Result: both completed successfully with no output.

## Files Changed

- `src/tau3_retail_evolver/fast_loop/decisions.py`
- `src/tau3_retail_evolver/fast_loop/prompts.py`
- `tests/unit/fast_loop/test_decisions.py`
- `tests/unit/fast_loop/test_prompts.py`
- `.superpowers/sdd/task-1-report.md`

## Implementation Summary

- Added strict, frozen, extra-forbidding decision models for selection, action, memory
  writes, maintenance commands, and parser results.
- Added single-object JSON parsing with optional semantic validation, at most one repair,
  and explicit failure results that never produce a maintenance command from invalid
  output.
- Added JSON-safe lifecycle prompts that project Memory candidates to the allowed public
  fields and reject hidden evaluation, attribution, embedding, metadata, hindsight, and
  task-ID fields.
- Added bounded maintenance diagnostics with schemas for the existing typed lookup,
  merge, and delete commands.

## Self-Review

- Confirmed every new Pydantic model is frozen, strict, and forbids extra fields.
- Confirmed selection IDs are trimmed, nonblank, unique, and can be semantically checked
  against the supplied candidate ID set through `validate_candidates`.
- Confirmed the parser rejects non-object JSON, schema failures, semantic failures, and a
  failed repaired response without falling back to a command.
- Confirmed selection and action prompts include only the Memory allowlist: ID, tier,
  content, version, rank, and similarity.
- Confirmed no unrelated tracked files were modified. The pre-existing untracked plan
  file `docs/superpowers/plans/2026-07-13-stage-4-fast-loop.md` was left untouched.

## Commit

Implementation commit: `1599112` (`Add fast loop lifecycle decisions and prompts`).

## Concerns

None. Git emitted existing line-ending conversion warnings for the newly added Python
files; they do not affect test or compilation results.

## Fix Review Findings

### Findings Addressed

- Forbidden prompt keys are normalized with Unicode NFKC, case folding, and removal of
  separators before comparison. Nested camelCase, mixed-case, hyphenated, dotted, and
  tuple-contained variants are rejected. History messages use an exact `role` and
  `content` whitelist.
- Maintenance command payloads pass through strict fast-loop input models before they
  are exposed as the existing Stage 3 `LookupCommand`, `MergeCommand`, and
  `DeleteCommand` types. String, float, and boolean values cannot be coerced into
  `updated_round`.
- Parsing `SelectionDecision` now requires `candidate_ids`; missing sets and unknown IDs
  return explicit failures, while valid subsets parse successfully.
- Maintenance diagnostics now require exactly the `trajectory`, `tip`, `skill`, and
  `tool` tiers. Each tier has an `items` list capped at 100 entries, and each item permits
  only `id`, `content`, `version`, `usage_count`, `success_count`, and `last_used` with
  strict bounded types.

### Review RED

Command:

```powershell
python -m pytest tests/unit/fast_loop/test_decisions.py tests/unit/fast_loop/test_prompts.py -q --basetemp=.pytest-tmp/task1-review-red
```

Result: `13 failed, 12 passed in 0.28s`. The failures reproduced missing mandatory
candidate sets, unknown candidate handling, maintenance scalar coercion, forbidden-key
variants, history leakage, and unbounded diagnostics.

Additional direct-construction boundary command:

```powershell
python -m pytest tests/unit/fast_loop/test_prompts.py -q --basetemp=.pytest-tmp/task1-review-red-tuple
```

Result: `1 failed, 15 passed in 0.24s`. The failing test demonstrated that a JSON-safe
nested tuple could bypass recursive hidden-field scanning.

### Review GREEN

Task 1 command:

```powershell
python -m pytest tests/unit/fast_loop/test_decisions.py tests/unit/fast_loop/test_prompts.py -q --basetemp=.pytest-tmp/task1-review-final-2
```

Result: `28 passed in 0.24s`.

Relevant regression command:

```powershell
python -m pytest tests/unit/memory tests/unit/fast_loop -q --basetemp=.pytest-tmp/task1-review-regression-2
```

Result: `108 passed in 4.07s`.

Additional verification:

```powershell
python -m compileall -q src/tau3_retail_evolver/fast_loop/decisions.py src/tau3_retail_evolver/fast_loop/prompts.py
git diff --check
```

Result: both completed successfully. `git diff --check` reported only the existing Git
line-ending conversion warnings.

### Fix Commit

Implementation fix commit: `edaa2aa` (`Fix Task 1 lifecycle review findings`).

### Fix Concerns

`pytest-cov` is not installed in the current environment, so no percentage-based
coverage report was generated. All new review cases have focused regression tests, and
the complete related fast-loop and memory unit suites pass.
