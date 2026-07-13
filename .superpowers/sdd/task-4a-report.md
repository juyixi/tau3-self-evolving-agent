# Task 4A Report: OpenAI-Compatible Qwen Fast-Loop Policy

## Status

Implemented `OpenAICompatibleFastLoopPolicy` as a `FastLoopPolicy` adapter without changing the reviewed fast-loop modules or existing baseline policy behavior.

## TDD Evidence

### RED

- Required RED command: `python -m pytest tests/unit/models/test_policy.py -q --basetemp=.pytest-tmp/qwen-fast-policy-red`
- Result: collection failed with `ImportError: cannot import name 'OpenAICompatibleFastLoopPolicy'`, confirming the new interface was absent before production edits.
- Follow-up malformed structured-action RED: the focused test failed because a non-sequence `tool_calls` value raised `ValueError: Qwen completion has no text content` instead of returning parser-invalid output for runner repair.

### GREEN

- Required target command: `python -m pytest tests/unit/models/test_policy.py -q --basetemp=.pytest-tmp/task4a`
- Result: `30 passed in 0.27s`.
- Regression command: `python -m pytest tests/unit/models tests/unit/fast_loop -q --basetemp=.pytest-tmp/task4a-regression`
- Result: `160 passed in 1.54s`.
- Compile: `python -m py_compile src/tau3_retail_evolver/models/openai_compatible.py tests/unit/models/test_policy.py` passed.
- `git diff --check` passed; Git emitted only the existing Windows LF-to-CRLF conversion warnings.

## Client-Call Examples

Selection sends two messages, no tools, and canonical public JSON:

```json
{"messages":[{"role":"system","content":"Return exactly one strict JSON object matching SelectionDecision..."},{"role":"user","content":"{\"history\":[...],\"memories\":[...],\"observation\":...,\"policy\":...,\"task_instruction\":...,\"tools\":[...]}"}],"tools":[],"temperature":0.7,"top_p":0.9}
```

Action sends the same canonical public prompt payload and passes the exact official Tau2 tools separately. A Qwen structured call is codec-validated and returned as canonical decision JSON:

```json
{"action":"{\"arguments\":{\"order_id\":\"123\"},\"name\":\"find_order\"}"}
```

Repair sends one new request containing exactly the public `LifecyclePrompt`, `invalid_output`, and `validation_error`. Action repair retains the official tools; non-action repair uses no tools.

## Files Changed

- `src/tau3_retail_evolver/models/openai_compatible.py`
- `tests/unit/models/test_policy.py`
- `.superpowers/sdd/task-4a-report.md`

No reviewed runner, decisions, prompts, maintenance, action codec, HTTP client contract, baseline prompt behavior, or package export was modified.

## Self-Review

- Sampling values must be finite; every response records canonical temperature/top-p values and nonnegative measured latency.
- Generation serializes only the supplied public prompt data. No task/run IDs, evaluator criteria, attribution, hidden memory metadata, persistence, or logging were added.
- Selection, write, and maintenance pass no tools and preserve assistant text for strict runner parsing; structured tool calls become parser-invalid raw assistant JSON.
- Action prefers one Qwen structured call, validates through `Tau2ActionCodec`, and canonicalizes `ActionDecision`; valid text actions use the same codec.
- Malformed action text or structured calls return invalid raw output for the runner's one repair attempt. Client failures remain policy failures with sanitized exception text and no chained cause.
- Existing baseline tests were retained unchanged and pass in the models regression suite.

## Commits

- `999281c Add OpenAI-compatible fast-loop policy`
- This report is committed separately in the commit containing this file.

## Concerns

- A pre-existing untracked `docs/superpowers/plans/2026-07-13-stage-4-fast-loop.md` belongs to other work and was intentionally neither modified nor committed.
- No live Qwen endpoint was called; behavior is verified against exact OpenAI-compatible client calls and representative Qwen response shapes.
