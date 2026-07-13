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

## Review Remediation

### Findings Addressed

- Action extraction, Qwen tool parsing, and `Tau2ActionCodec` decoding now share one failure boundary. Any failure returns canonical `{"invalid_action_output":"action decoding failed"}` with no legal top-level `action`, forcing runner repair.
- The bypass case where raw output was itself `{"action":"{\"name\":\"unknown\",...}"}` is covered and cannot be accepted as `ActionDecision`.
- Missing/null/empty `tool_calls` with null content, malformed mapping/string calls, and non-JSON-serializable structured messages all become parser-invalid action responses instead of adapter exceptions.
- Selection, write, and maintenance use a fast-loop-only structured-call guard. Any present `tool_calls` value other than `None` or an empty list rejects mixed valid content, including list, mapping, string, and scalar shapes.
- The shared baseline `_raw_output`, baseline policy, reviewed runner, decisions, prompts, maintenance, and codec were not changed.

### Review TDD Evidence

- RED: `python -m pytest tests/unit/models/test_policy.py -q --basetemp=.pytest-tmp/task4a-review-red`
- RED result: `18 failed, 28 passed`; failures reproduced the accepted action shell, premature action extraction exceptions, accepted non-action mixed content, and environment execution of the invalid `unknown` action.
- GREEN: `python -m pytest tests/unit/models/test_policy.py -q --basetemp=.pytest-tmp/task4a-review-green`
- GREEN result: `46 passed in 0.30s`.
- Regression: `python -m pytest tests/unit/models tests/unit/fast_loop -q --basetemp=.pytest-tmp/task4a-review-regression`
- Regression result: `176 passed in 1.46s`.
- Compile and `git diff --check` passed.

### Runner Integration Evidence

The integration unit test runs `run_fast_loop_episode` with a real mutable `MemoryRepository` and `Retriever`. The action generate response is codec-invalid but has a syntactically valid `ActionDecision` shell; repair returns one valid structured Qwen call. Assertions prove:

- The environment executes only canonical `find_order` from repair.
- Exactly two client requests carry action tools: generate plus one repair.
- The complete episode uses four client calls: selection, action generate, action repair, and write.
- Audited events contain neither the invalid `unknown` raw action nor the internal invalid-action wrapper.

### Review Commit

- `e75d397 Force repair for invalid fast-loop actions`
