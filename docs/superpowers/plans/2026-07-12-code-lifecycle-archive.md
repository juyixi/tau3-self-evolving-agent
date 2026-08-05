# Stage 1/2 Code Lifecycle Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reclassify Stage 1/2 validation assets so product entrypoints, operational diagnostics, test-only helpers, core runtime code, and historical documentation have durable and enforceable boundaries.

**Architecture:** Keep the Tau2 smoke capability as an explicitly non-core `tools.preflight` module, move scripted test policy behavior into `tests/support`, remove the behavior-free environment factory, and make lifecycle auditing a stage gate in project documentation. Preserve all environment and rollout behavior while changing ownership and import paths.

**Tech Stack:** Python 3.12-3.13, pytest, Git, Markdown.

## Global Constraints

- `src/` contains only code used by rollout, memory, OPD training, or evaluation at runtime.
- `scripts/` contains only long-lived product workflow entrypoints.
- `tools/` contains operational diagnostics and must never be imported by the core runtime.
- Test helpers live under `tests/support/` and are not exported by the production package.
- Existing unrelated working-tree changes must not be staged or overwritten.
- The old `python -m scripts.check_tau2_retail` path is intentionally removed without a compatibility shim.
- `.superpowers/` remains ignored local scratch and is not copied into the delivery archive.

---

### Task 1: Relocate the Tau2 preflight command

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/preflight/__init__.py`
- Move: `scripts/check_tau2_retail.py` to `tools/preflight/check_tau2_retail.py`
- Move: `tests/unit/scripts/test_check_tau2_retail.py` to `tests/unit/tools/preflight/test_check_tau2_retail.py`
- Modify: `tests/integration/test_real_tau2_retail.py`
- Modify: `docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md`

**Interfaces:**
- Produces: `python -m tools.preflight.check_tau2_retail --split train --task-id <id> [--inspect]`
- Preserves: JSON payload fields, exit codes, credential redaction, pinned checkout checks, split checks, and reset/close behavior.

- [x] **Step 1: Change focused tests to the target module path**

Update the unit test import:

```python
from tools.preflight import check_tau2_retail
```

Update the integration subprocess command:

```python
[sys.executable, "-m", "tools.preflight.check_tau2_retail", *args]
```

- [x] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.cache\tau2-venv\Scripts\python.exe -m pytest -q `
  tests/unit/tools/preflight/test_check_tau2_retail.py `
  tests/integration/test_real_tau2_retail.py `
  --basetemp .cache/pytest-preflight-red
```

Expected: collection fails because `tools.preflight.check_tau2_retail` does not exist.

- [x] **Step 3: Move the implementation and add package markers**

Move the existing implementation without behavior changes. Add empty `tools/__init__.py` and `tools/preflight/__init__.py` so module execution is stable.

- [x] **Step 4: Update active documentation**

Replace the Stage 1 smoke command and file path with:

```text
tools/preflight/check_tau2_retail.py
python -m tools.preflight.check_tau2_retail --split train --task-id 0
```

- [x] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command with `--basetemp .cache/pytest-preflight-green`.

Expected: preflight unit tests pass and both integration tests skip unless `RUN_TAU2_INTEGRATION=1`.

- [x] **Step 6: Verify the old module path is gone**

Run:

```powershell
rg -n "scripts\.check_tau2_retail|scripts/check_tau2_retail" scripts tools tests docs
```

Expected: no active command or import remains; historical mention is allowed only inside the archive design/spec explaining the migration.

---

### Task 2: Remove test-only and behavior-free code from `src/`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/support/__init__.py`
- Create: `tests/support/policy.py`
- Modify: `tests/unit/models/test_policy.py`
- Modify: `tests/unit/scripts/test_run_baseline.py`
- Modify: `tests/unit/fast_loop/test_baseline_runner.py`
- Modify: `tests/unit/envs/test_tau2_retail_adapter.py`
- Modify: `src/tau3_evolver/models/policy.py`
- Modify: `src/tau3_evolver/models/__init__.py`
- Delete: `src/tau3_evolver/envs/factory.py`

**Interfaces:**
- Production retains: `DecisionRequest`, `DecisionResponse`, and abstract `Policy`.
- Tests gain: `tests.support.policy.ScriptedPolicy` with the same constructor, `requests` property, and `generate` behavior.
- Environment callers construct `Tau2RetailEnv` directly.

- [x] **Step 1: Point tests at the target support module and direct adapter**

Replace test imports with:

```python
from tests.support.policy import ScriptedPolicy
```

Replace the factory-only adapter test with direct construction:

```python
environment = Tau2RetailEnv("task-17", config, gym_factory=FakeGymEnv)
assert isinstance(environment, Tau2RetailEnv)
```

Then remove the duplicate assertion if direct construction is already covered, so the behavior-free wrapper test is deleted rather than renamed.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.cache\tau2-venv\Scripts\python.exe -m pytest -q `
  tests/unit/models/test_policy.py `
  tests/unit/scripts/test_run_baseline.py `
  tests/unit/fast_loop/test_baseline_runner.py `
  tests/unit/envs/test_tau2_retail_adapter.py `
  --basetemp .cache/pytest-support-red
```

Expected: collection fails because `tests.support.policy` does not exist.

- [x] **Step 3: Add the test-only scripted policy**

Create `tests/support/policy.py` with:

```python
from __future__ import annotations

from collections.abc import Sequence

from tau3_evolver.models.policy import DecisionRequest, DecisionResponse, Policy


class ScriptedPolicy(Policy):
    def __init__(self, responses: Sequence[DecisionResponse]) -> None:
        self._responses = iter(responses)
        self._requests: list[DecisionRequest] = []

    @property
    def requests(self) -> tuple[DecisionRequest, ...]:
        return tuple(self._requests)

    def generate(self, request: DecisionRequest) -> DecisionResponse:
        try:
            response = next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted policy has no remaining responses") from error
        self._requests.append(request)
        return response
```

- [x] **Step 4: Remove test-only production exports and the factory**

Delete `ScriptedPolicy` from `models/policy.py`, remove it from `models/__init__.py`, and delete `envs/factory.py`.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command with `--basetemp .cache/pytest-support-green`.

Expected: all focused tests pass, with one fewer test after removal of the factory wrapper assertion.

- [x] **Step 6: Verify production ownership boundaries**

Run:

```powershell
rg -n "ScriptedPolicy" src scripts tools
rg -n "create_tau2_retail_env|envs\.factory" src scripts tools tests
```

Expected: no `ScriptedPolicy` under production paths and no factory references anywhere.

---

### Task 3: Establish the permanent lifecycle rule and milestone archive

**Files:**
- Create: `docs/development/code-lifecycle.md`
- Create: `docs/archive/2026-07-stage-1-2-delivery.md`
- Modify: `docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md`
- Modify: `docs/superpowers/plans/2026-07-10-tau3-retail-opd-evolver-staged.md`

**Interfaces:**
- Produces: one normative lifecycle document inherited by all future stage gates.
- Produces: one concise historical Stage 1/2 delivery summary without raw scratch artifacts.

- [x] **Step 1: Write the normative lifecycle document**

Include these required sections:

```markdown
# 代码生命周期与归档规范
## 目录职责
## 阶段结束审计
## 保留、迁移与删除判定
## 测试与验收门禁
## 禁止事项
```

State that `src/`, `scripts/`, `tools/`, `tests/`, and `docs/archive/` have the exact ownership rules defined in the approved design.

- [x] **Step 2: Write the concise milestone archive**

Record the Stage 1 and Stage 2.1/2.2 commit ranges, durable deliverables, moved/deleted files, and the final test gate. Do not copy `.superpowers` review diffs, briefs, reports, or progress logs.

- [x] **Step 3: Make the rule normative in current project docs**

Add a short “代码生命周期与归档” section to the main design and a global constraint plus stage-gate checklist item to the staged plan, both linking to:

```text
docs/development/code-lifecycle.md
```

- [x] **Step 4: Verify documentation consistency**

Run:

```powershell
rg -n "code-lifecycle|代码生命周期|tools\.preflight\.check_tau2_retail" docs
rg -n "scripts\.check_tau2_retail" docs
```

Expected: normative docs reference the lifecycle policy and current preflight command; the old command appears only in migration-history context.

- [x] **Step 5: Run the complete default suite**

Run:

```powershell
.cache\tau2-venv\Scripts\python.exe -m pytest -q `
  --basetemp .cache/pytest-lifecycle-final
```

Expected: `93 passed, 2 skipped`.

- [x] **Step 6: Review and commit the implementation**

Run:

```powershell
git diff --check
git status --short
```

Stage only lifecycle cleanup files; do not stage unrelated pre-existing documentation line-ending changes. Commit with:

```text
refactor: archive stage validation utilities
```
