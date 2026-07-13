# Stage 4 OPD-Evolver Fast Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 tau2 retail train 任务上实现当前 Qwen 策略驱动的 retrieve/select/act/write/maintain 快循环，并产出 Stage 5 Outcome-calibrated attribution 所需的完整、无特权泄漏证据。

**Architecture:** 四类生命周期输出先经 Pydantic 严格校验，再由 runner 顺序编排 Retriever、tau2 环境和 MemoryRepository。训练 Memory 仅允许 `split=train` 且 `RunMode.LEARN` 的上下文写入；候选、选择、轨迹、终局结果、写入提案和提交结果进入 rollout JSONL，归因分数留给 Stage 5 单独生成。

**Tech Stack:** Python 3.12、Pydantic 2、tau2 retail adapter、JSON/JSONL、pytest。

## Global Constraints

- 基础模型固定为 `Qwen/Qwen3.5-9B`；Stage 4 的 action 必须由当前策略 on-policy 生成。
- Memory 固定使用 `history/agents/retail/memory/` 下四个 JSON 权威文件，不按 `run_id` 隔离。
- 默认 `retrieve_top_k=50`、`teacher_memory_cap=20`、`maintenance_period=30`、`max_episode_steps=40`。
- 只有 `split=train` 且 `RunMode.LEARN` 可以修改训练 Memory；test、baseline 和 evaluation 路径必须只读。
- Prompt 不得包含 test task ID、evaluation criteria、attribution score 或 privileged hindsight。
- Stage 4 只记录 `C_t`、`S_t`、轨迹、终局 reward 和写入 provenance；Outcome-calibrated attribution 在 Stage 5 写入 `runs/<run_id>/attribution/scores.jsonl`。

---

### Task 1: Typed Lifecycle Decisions And Public Prompts

**Files:**
- Create: `src/tau3_retail_evolver/fast_loop/decisions.py`
- Create: `src/tau3_retail_evolver/fast_loop/prompts.py`
- Test: `tests/unit/fast_loop/test_decisions.py`
- Test: `tests/unit/fast_loop/test_prompts.py`

**Interfaces:**
- Produces: `SelectionDecision(memory_ids: tuple[str, ...])`
- Produces: `ActionDecision(action: str)`
- Produces: `MemoryWrite(tier, content, retrieval_text, metadata)` and `WriteDecision(memories)`
- Produces: `MaintenanceDecision(commands)` using the existing typed lookup/merge/delete commands.
- Produces: `parse_decision(raw, decision_type, validator, repair=None)` with at most one repair attempt and an explicit `DecisionParseResult` failure.
- Produces: `build_selection_prompt`, `build_action_prompt`, `build_write_prompt`, and `build_maintenance_prompt` returning JSON-safe public prompt payloads.

- [ ] **Step 1: Write failing decision tests**

Cover valid JSON, duplicate selection IDs, unknown IDs, blank action, invalid tier, unsafe maintenance operation, one successful repair, and exhausted repair. Assert invalid output never silently becomes a valid command.

- [ ] **Step 2: Run decision tests and verify RED**

Run: `python -m pytest tests/unit/fast_loop/test_decisions.py -q --basetemp=.pytest-tmp/decisions-red`

Expected: collection fails because `fast_loop.decisions` does not exist.

- [ ] **Step 3: Implement the minimal typed models and parser**

Use `ConfigDict(extra="forbid", frozen=True)`, discriminated existing maintenance commands, canonical unique IDs, and `TypeAdapter` validation. Return `{decision, raw_output, repaired_output, error}` provenance rather than hiding parse failures.

- [ ] **Step 4: Write failing prompt tests**

Assert the selection/action prompts contain task instruction, official policy/tools, current observation/history and only allowed Memory fields. Assert forbidden keys and sentinel values for evaluator criteria, hidden task ID, attribution and hindsight are absent from the serialized prompt.

- [ ] **Step 5: Implement public prompt builders and run GREEN**

Run: `python -m pytest tests/unit/fast_loop/test_decisions.py tests/unit/fast_loop/test_prompts.py -q --basetemp=.pytest-tmp/task1`

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/tau3_retail_evolver/fast_loop/decisions.py src/tau3_retail_evolver/fast_loop/prompts.py tests/unit/fast_loop/test_decisions.py tests/unit/fast_loop/test_prompts.py
git commit -m "feat: add typed fast-loop lifecycle decisions"
```

### Task 2: Episode Fast-Loop Runner And Attribution Evidence

**Files:**
- Create: `src/tau3_retail_evolver/fast_loop/runner.py`
- Modify: `src/tau3_retail_evolver/fast_loop/events.py`
- Modify: `src/tau3_retail_evolver/fast_loop/__init__.py`
- Test: `tests/unit/fast_loop/test_runner.py`

**Interfaces:**
- Produces: `RunMode` with `LEARN`, `EVALUATE`, and `BASELINE`.
- Produces: `FastLoopPolicy.select/act/write` protocol so a Qwen-backed adapter and deterministic tests share the same lifecycle contract.
- Produces: `FastLoopConfig(retrieve_top_k, max_episode_steps)`.
- Produces: `run_fast_loop_episode(task_id, env, policy, repository, retriever, config, context) -> EpisodeResult`.
- Event order: `EpisodeStarted`, `MemoryCandidatesRetrieved`, `MemorySelected`, repeated `DecisionMade`/`EnvironmentStepped`, `EpisodeFinished`, `MemoryWriteProposed`, then `MemoryWriteCommitted` or `MemoryWriteFailed`.

- [ ] **Step 1: Write failing happy-path runner test**

Use a real temporary `MemoryRepository`, deterministic embedding provider, fake tau2 environment and scripted lifecycle policy. Assert selected IDs are a candidate subset, every action comes from the policy, terminal reward/evaluation are preserved, writes include `source_task_ids=(task_id,)`, and the event stream contains `C_t`, `S_t`, action trajectory and commit IDs.

- [ ] **Step 2: Run the happy-path test and verify RED**

Run: `python -m pytest tests/unit/fast_loop/test_runner.py -q --basetemp=.pytest-tmp/runner-red`

Expected: collection fails because `fast_loop.runner` does not exist.

- [ ] **Step 3: Implement minimal retrieve/select/act/write orchestration**

Build the retrieval query only from public task instruction, policy, tool names and observation. Apply validated writes through `MemoryRepository.add`; emit proposal before persistence and commit only after all requested writes succeed idempotently.

- [ ] **Step 4: Add failure and capability tests**

Cover unknown selected ID, invalid action decision, policy exception/timeout, environment exception, max-step truncation, repository write failure, `split=test`, and non-LEARN run modes. Assert environment cleanup always runs and no forbidden mode changes the training repository.

- [ ] **Step 5: Implement explicit failure events and mutation guard**

Reject mutating contexts before reset. Preserve the primary failure, attach cleanup failures as notes, and emit `EpisodeFailed` or `MemoryWriteFailed` with sanitized diagnostics and complete prior evidence.

- [ ] **Step 6: Run Task 2 GREEN and regressions**

Run: `python -m pytest tests/unit/fast_loop/test_runner.py tests/unit/fast_loop/test_baseline_runner.py -q --basetemp=.pytest-tmp/task2`

Expected: all runner and baseline tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/tau3_retail_evolver/fast_loop tests/unit/fast_loop/test_runner.py
git commit -m "feat: add opd evolver fast-loop runner"
```

### Task 3: Periodic Memory Maintenance

**Files:**
- Create: `src/tau3_retail_evolver/fast_loop/maintenance.py`
- Test: `tests/unit/fast_loop/test_maintenance.py`

**Interfaces:**
- Produces: `MaintenanceState(completed_rounds)` persisted at the agent Memory root.
- Produces: `bounded_diagnostics(repository, per_tier_limit) -> RepositoryDiagnostics`.
- Produces: `run_due_maintenance(completed_train_tasks, period, policy, operations, context) -> MaintenanceResult`.
- Maintenance input contains bounded active Memory metadata/content only; commands remain typed `lookup`, same-tier `merge`, and soft `delete`.

- [ ] **Step 1: Write failing scheduler and command tests**

Assert no run at tasks 1-29, one run at 30, no duplicate after resume at 30, one new run at 60, bounded diagnostics, same-tier atomic merge/delete, and full proposal/commit/failure events.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/unit/fast_loop/test_maintenance.py -q --basetemp=.pytest-tmp/maintenance-red`

Expected: collection fails because `fast_loop.maintenance` does not exist.

- [ ] **Step 3: Implement durable idempotent scheduling**

Derive `maintenance_round=completed_train_tasks // period`; skip zero/already-completed rounds, validate policy output, apply commands via `MemoryOperations.apply_batch`, then atomically persist completed rounds only after a successful commit.

- [ ] **Step 4: Run Task 3 GREEN**

Run: `python -m pytest tests/unit/fast_loop/test_maintenance.py -q --basetemp=.pytest-tmp/task3`

Expected: all maintenance tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/tau3_retail_evolver/fast_loop/maintenance.py tests/unit/fast_loop/test_maintenance.py
git commit -m "feat: add periodic memory maintenance"
```

### Task 4: Stage Gate And Real Smoke Preparation

**Files:**
- Modify: `README.md`
- Create: `tests/integration/test_fast_loop_tau2_retail.py`

**Interfaces:**
- Produces an opt-in five-task real tau2 retail smoke guarded by environment variables.
- Documents the Qwen endpoint, user simulator, train-only mutation, output JSONL and persistent Memory paths.

- [ ] **Step 1: Add skipped-by-default real smoke test**

The test selects five fixed train task IDs, uses the existing tau2 adapter and configured Qwen endpoint, writes rollout JSONL under an isolated run directory, and requires an explicit temporary Memory root during automated testing.

- [ ] **Step 2: Add a no-cost 30-task maintenance integration test**

Use fake environment/policy and a temporary Memory root; assert exactly one maintenance round and no external API calls.

- [ ] **Step 3: Run complete verification**

Run: `python -m pytest -q --basetemp=.pytest-tmp/stage4-full`

Expected: all local tests pass; external integration tests are skipped unless explicitly enabled.

Run: `python -m compileall -q src scripts tests`

Expected: exit code 0.

- [ ] **Step 4: Commit**

```bash
git add README.md tests/integration/test_fast_loop_tau2_retail.py
git commit -m "test: add stage 4 fast-loop smoke coverage"
```

---

## Self-Review

- Spec coverage: Tasks 1-4 cover retrieve/select/act/write/maintain, Q=30, public prompt isolation, on-policy action generation, evidence logging, train-only mutation, resume idempotency and real tau2 smoke preparation.
- Attribution boundary: No Stage 4 type or Memory item stores Outcome-calibrated attribution; Stage 4 emits only the evidence Stage 5 needs.
- Storage boundary: Memory remains JSON under the shared agent namespace; logs remain JSONL under run artifacts.
- Type consistency: `SelectionDecision`, `WriteDecision`, `MaintenanceDecision`, `RunMode`, `EpisodeResult` and maintenance signatures are introduced before their consumers.
- Placeholder scan: No deferred implementation markers remain; every task has exact files, behaviors, commands and expected outcomes.
