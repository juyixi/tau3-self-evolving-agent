# Task 2 Report: Episode Fast-Loop Runner And Attribution Evidence

## Status

Complete on branch `codex/stage-4-fast-loop`.

Implementation commit: `aa9baab488825d6736569f90af3cd517e6aeb3d3`

## TDD Evidence

### Initial RED

Command:

```text
python -m pytest tests/unit/fast_loop/test_runner.py -q --basetemp=.pytest-tmp/runner-red
```

Result: collection failed as expected because `RunMode` did not yet exist in
`fast_loop.events`. No production files had been edited before this RED run.

### Boundary RED/GREEN Cycles

- Added an official Tau2 truncation test. RED failed because
  `EpisodeFinished` did not distinguish official truncation from project
  max-step truncation. The runner now emits both `truncated` and
  `project_truncated`, and `EpisodeResult.truncated` covers either source.
- Added an attribution-storage assertion. RED showed model-proposed
  `attribution_score` surviving inside `MemoryWriteProposed.raw_output`. The
  event now records only canonical filtered proposals plus sampling/repair
  audit fields, so no attribution score is persisted.

### GREEN

Command:

```text
python -m pytest tests/unit/fast_loop/test_runner.py tests/unit/fast_loop/test_baseline_runner.py -q --basetemp=.pytest-tmp/task2
```

Result: `23 passed in 0.47s`.

## Regression And Static Verification

Command:

```text
python -m pytest tests/unit/fast_loop tests/unit/memory -q --basetemp=.pytest-tmp/task2-regression
```

Result: `124 passed in 4.33s`.

Additional checks:

```text
python -m compileall -q src/tau3_retail_evolver/fast_loop/events.py src/tau3_retail_evolver/fast_loop/runner.py tests/unit/fast_loop/test_runner.py
git diff --check
```

Both completed with exit code 0. Git emitted only the existing Windows
LF-to-CRLF advisory.

## Files Changed

- `src/tau3_retail_evolver/fast_loop/events.py`
  - Added `RunMode`.
  - Generalized adapter and memory snapshot revisions.
  - Added backward-compatible baseline mode default and mode provenance.
- `src/tau3_retail_evolver/fast_loop/runner.py`
  - Added lifecycle policy/config/result interfaces and the episode runner.
  - Added retrieval, selection, on-policy action, terminal/truncation, write,
    replay, failure evidence, and exactly-once cleanup behavior.
- `tests/unit/fast_loop/test_runner.py`
  - Added real repository/retriever happy-path coverage and all required
    guard, repair, failure, truncation, replay, and cleanup scenarios.

Task 1 `decisions.py` and `prompts.py` were reused unchanged.

## Event Schema Summary

Every event includes the canonical `RunContext` envelope: schema version,
event type, run/iteration/split/mode, task provenance/group, model/adapter/
memory revisions, and seed.

Successful order:

```text
EpisodeStarted
MemoryCandidatesRetrieved
MemorySelected
(DecisionMade, EnvironmentStepped)+
EpisodeFinished
MemoryWriteProposed
MemoryWriteCommitted
```

- Candidate and selected evidence includes memory ID, version, tier, rank,
  and similarity. Selection records raw/repaired/error provenance.
- Decision and environment events contain public observation/action/result
  data only; reset and step metadata use strict allowlists.
- Episode completion records final `StepResult.reward`, JSON-safe official
  terminal mappings, and official versus project truncation.
- Write proposals contain canonical, attribution-free metadata with reserved
  run/iteration/reward/selection provenance. Commit evidence is emitted only
  after every write is added or verified as a stable-ID replay.
- Partial writes emit `MemoryWriteFailed` with committed IDs and a sanitized
  error. Other lifecycle failures emit sanitized `EpisodeFailed` after prior
  evidence.

## Self-Review

- Confirmed pre-reset guards execute before environment reset, policy calls,
  retrieval, or repository writes for test/non-learning/read-only inputs.
- Confirmed `task_id` is used only for event/source provenance and is absent
  from every lifecycle prompt.
- Confirmed selection is parsed against candidate IDs and action/write output
  receives at most one repair attempt.
- Confirmed action strings passed to Tau2 exactly match parsed policy output.
- Confirmed terminal reward is not accumulated and official terminal JSON
  uses baseline-equivalent safety checks.
- Confirmed reserved metadata wins over model metadata and attribution keys
  are recursively removed from persisted and event proposal metadata.
- Confirmed replay requires matching stable ID, tier, canonical content, and
  exact source task; unsafe duplicates still fail.
- Confirmed cleanup executes exactly once and cleanup failure is attached to
  the primary exception as a note.
- Confirmed only brief-owned implementation/test/report files were touched;
  an unrelated untracked plan file was neither modified nor staged.

## Concerns

- The OpenAI HTTP adapter and CLI remain intentionally out of scope.
- No external policy or real Tau2 integration test was run; deterministic
  unit tests use the required scripted lifecycle policy and fake Tau2
  environment while exercising the real repository and retriever.
