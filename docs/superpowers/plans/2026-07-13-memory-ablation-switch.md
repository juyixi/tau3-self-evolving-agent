# Fast Loop Memory 消融开关实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 canonical retail fast loop 增加默认开启、严格可审计的 `memory.enabled` 总开关，使无 Memory 消融不加载、不读取、不写入任何 Memory 能力。

**Architecture:** `MemoryConfig.enabled` 是唯一配置源；`FastLoopConfig.memory_enabled` 将值传入 episode runner。Runner 在关闭时只执行 action 生命周期并记录 `MemoryDisabled`，CLI 不构建 repository、embedding、retriever 或 maintenance，同时通过 manifest/summary 的显式字段证明消融状态。

**Tech Stack:** Python 3.12+、Pydantic v2、PyYAML、pytest、现有 JSONL event 与 atomic JSON artifact API。

## Global Constraints

- `memory.enabled` 默认值必须为 `true`，保持现有 Stage 4 行为兼容。
- 配置只接受严格 YAML boolean；字符串和整数不得隐式转换。
- `false` 必须关闭 retrieve、select、prompt memory context、write、embedding 和 maintenance。
- 无 Memory action prompt 不得包含 `memories` 键或 Memory 占位内容。
- 无 Memory run 不得创建或修改 `history/`。
- test/base split 隔离、凭证清理和现有错误传播语义不得改变。
- 只修改本功能涉及的代码和文档，不改动依赖版本。

---

### Task 1: 严格配置契约与无 Memory Action Prompt

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/tau3_retail_evolver/config.py`
- Modify: `src/tau3_retail_evolver/fast_loop/prompts.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/fast_loop/test_prompts.py`

**Interfaces:**
- Produces: `MemoryConfig.enabled: StrictBool = True`
- Produces: `build_action_prompt(..., include_memory_context: bool = True) -> LifecyclePrompt`

- [ ] **Step 1: 编写配置失败测试**

在 `tests/unit/test_config.py` 增加断言：默认配置解析后 `config.memory.enabled is True`；临时 YAML 中 `enabled: false` 解析为 `False`；`enabled: "false"` 和 `enabled: 0` 都抛出 `ValidationError`。

```python
assert load_config(default_config).memory.enabled is True
with pytest.raises(ValidationError):
    load_config(config_with_memory_enabled('"false"'))
```

- [ ] **Step 2: 运行配置测试并确认失败**

Run: `python -m pytest tests/unit/test_config.py -q`

Expected: FAIL，因为 `MemoryConfig` 尚无 `enabled` 字段。

- [ ] **Step 3: 实现严格配置字段**

在 YAML 的 `memory` 节增加 `enabled: true`，并在配置模型中使用严格布尔：

```python
from pydantic import StrictBool

class MemoryConfig(_ConfigModel):
    enabled: StrictBool = True
```

- [ ] **Step 4: 编写 prompt 失败测试**

在 `tests/unit/fast_loop/test_prompts.py` 断言：

```python
prompt = build_action_prompt(
    task_instruction="task",
    policy="policy",
    tools=[],
    observation="obs",
    include_memory_context=False,
)
assert "memories" not in prompt.payload
```

同时保留默认调用仍产生 `memories` 字段的回归断言。

- [ ] **Step 5: 实现 action prompt 字段省略**

让 `build_action_prompt` 先构造 public context，仅在 `include_memory_context=True` 时添加规范化的 `memories`：

```python
payload = project_public_context(...)
if include_memory_context:
    payload["memories"] = [_public_memory(memory) for memory in memories]
return LifecyclePrompt(kind="action", payload=payload, command_schemas=())
```

- [ ] **Step 6: 运行聚焦测试**

Run: `python -m pytest tests/unit/test_config.py tests/unit/fast_loop/test_prompts.py -q`

Expected: PASS。

- [ ] **Step 7: 提交 Task 1**

```bash
git add configs/default.yaml src/tau3_retail_evolver/config.py src/tau3_retail_evolver/fast_loop/prompts.py tests/unit/test_config.py tests/unit/fast_loop/test_prompts.py
git commit -m "feat: add strict memory feature switch"
```

### Task 2: Episode Runner 的完整 Memory Bypass

**Files:**
- Modify: `src/tau3_retail_evolver/fast_loop/runner.py`
- Modify: `tests/unit/fast_loop/test_runner.py`

**Interfaces:**
- Consumes: `FastLoopConfig(memory_enabled: bool, retrieve_top_k: int, max_episode_steps: int)`
- Produces: `run_fast_loop_episode(..., repository: MemoryRepository | None, retriever: Retriever | None, ...) -> EpisodeResult`
- Event: `MemoryDisabled(reason="config")`

- [ ] **Step 1: 编写 runner 失败测试**

构造 `memory_enabled=False` 的 fake episode，传入 `repository=None`、`retriever=None`，并断言：

```python
assert policy.prompt_kinds == ["action"]
assert "memories" not in policy.prompts[0].payload
assert result.selected_memory_ids == ()
assert result.written_memory_ids == ()
assert [event["event_type"] for event in events].count("MemoryDisabled") == 1
assert not forbidden_memory_events.intersection(event_types)
```

另加参数契约测试：enabled 时缺 repository/retriever、disabled 时仍传任一对象，都在 `environment.reset` 前抛出 `ValueError`。

- [ ] **Step 2: 运行 runner 聚焦测试并确认失败**

Run: `python -m pytest tests/unit/fast_loop/test_runner.py -q`

Expected: FAIL，因为 runner 仍无条件要求并访问 Memory。

- [ ] **Step 3: 扩展 `FastLoopConfig` 和依赖校验**

```python
@dataclass(frozen=True, slots=True)
class FastLoopConfig:
    memory_enabled: bool = True
    retrieve_top_k: int = 50
    max_episode_steps: int = 40

def _require_memory_dependencies(*, enabled, repository, retriever):
    if enabled:
        # require mutable MemoryRepository and retriever
    elif repository is not None or retriever is not None:
        raise ValueError("disabled memory requires no repository or retriever")
```

- [ ] **Step 4: 实现 bypass 生命周期**

在 reset 和 public context 建立后分支：

```python
if config.memory_enabled:
    candidates = retriever.retrieve(...)
    selected = ...
else:
    candidates = []
    selected = []
    selected_ids = ()
    _emit(context, task_id, "MemoryDisabled", reason="config")
```

Action prompt 传入 `include_memory_context=config.memory_enabled`。Episode terminal event 后仅在 enabled 分支生成 write decision 和持久化；disabled 分支直接构造空 selected/written IDs 的 `EpisodeResult`。

- [ ] **Step 5: 保持失败与 close 语义**

确认 bypass 分支仍位于现有 `try/except/finally` 控制流内，action/environment/close 异常继续追加 `FastLoopFailed` 并传播；不得复制另一套 episode loop。

- [ ] **Step 6: 运行 runner 与 policy 回归测试**

Run: `python -m pytest tests/unit/fast_loop/test_runner.py tests/unit/models/test_policy.py -q`

Expected: PASS。

- [ ] **Step 7: 提交 Task 2**

```bash
git add src/tau3_retail_evolver/fast_loop/runner.py tests/unit/fast_loop/test_runner.py
git commit -m "feat: bypass memory lifecycle in fast loop"
```

### Task 3: CLI 编排、Manifest 与 Summary Provenance

**Files:**
- Modify: `scripts/run_fast_loop.py`
- Modify: `tests/unit/scripts/test_run_fast_loop.py`
- Modify: `tests/integration/test_fast_loop_tau2_retail.py`

**Interfaces:**
- `rollout_options.memory_enabled: bool`
- `fast_loop_summary.json.memory_enabled: bool`
- Disabled snapshot fields: `None`

- [ ] **Step 1: 编写 CLI 失败测试**

在脚本测试中加载 `memory.enabled=false` 配置，并将以下 factory 替换为“调用即失败”的 spy：`open_training_memory`、`build_embedding_provider`、`run_due_maintenance`。运行 `main` 后断言：

```python
assert manifest["rollout_options"]["memory_enabled"] is False
assert manifest["memory_snapshot_id"] is None
assert summary["memory_enabled"] is False
assert summary["input_memory_snapshot_id"] is None
assert summary["output_memory_snapshot_id"] is None
assert summary["maintenance_rounds_executed"] == []
assert not (project_root / "history").exists()
```

- [ ] **Step 2: 运行 CLI 测试并确认失败**

Run: `python -m pytest tests/unit/scripts/test_run_fast_loop.py -q`

Expected: FAIL，因为 CLI 仍无条件创建 Memory 依赖和 snapshot。

- [ ] **Step 3: 实现可选 Memory 资源编排**

根据 `config.memory.enabled` 设置：

```python
if config.memory.enabled:
    repository = open_training_memory(...)
    retriever = Retriever(build_embedding_provider(...))
    input_snapshot_id = repository.snapshot().memory_snapshot_id
else:
    repository = None
    retriever = None
    input_snapshot_id = None
```

将 `memory_enabled` 写入 `rollout_options` 和 summary，并把 `FastLoopConfig.memory_enabled` 传入 runner。

- [ ] **Step 4: 让 task orchestration 跳过 snapshot 与 maintenance**

`_run_requested_tasks` 接受可选 repository/retriever。关闭时 episode context snapshot 为 `None`，episode 完成后不调用 repository snapshot 或 `run_due_maintenance`；开启时保持现有逻辑逐行等价。

- [ ] **Step 5: 更新真实 smoke 的 provenance 断言**

现有真实五任务测试增加：

```python
assert manifest["rollout_options"]["memory_enabled"] is True
assert summary["memory_enabled"] is True
```

不新增需要外部凭证的无 Memory 集成测试；无 Memory 外部 smoke 使用同一 CLI 和专用 YAML 即可。

- [ ] **Step 6: 运行 CLI、manifest 和集成测试收集测试**

Run: `python -m pytest tests/unit/scripts/test_run_fast_loop.py tests/unit/runs/test_manifest.py tests/integration/test_fast_loop_tau2_retail.py -q`

Expected: PASS，真实测试在未设置 opt-in 环境变量时保持 skip。

- [ ] **Step 7: 提交 Task 3**

```bash
git add scripts/run_fast_loop.py tests/unit/scripts/test_run_fast_loop.py tests/integration/test_fast_loop_tau2_retail.py
git commit -m "feat: record no-memory fast-loop ablations"
```

### Task 4: 文档、审计与完整回归

**Files:**
- Modify: `docs/stage-4-fast-loop-validation.md`
- Modify: `docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md`
- Test: full repository suite

**Interfaces:**
- Documents: `memory.enabled=true|false` 的运行和产物契约

- [ ] **Step 1: 更新长期设计与阶段计划**

在主设计的快循环章节明确总开关和无 Memory fake run 验收；不得改变 Stage 5 attribution 公式或 Stage 8 test 隔离协议。独立的本功能设计与实施计划承担变更追踪，不修改存在用户未提交工作的分阶段总计划。

- [ ] **Step 2: 增加消融运行指南**

在验证文档给出专用 YAML 示例：

```yaml
memory:
  enabled: false
```

说明仍使用 `python -m scripts.run_fast_loop`，并检查 manifest/summary、`MemoryDisabled`、无 `history/` 和无 Memory lifecycle events。

- [ ] **Step 3: 运行完整测试**

Run: `python -m pytest -q --basetemp=.pytest-tmp/memory-switch-final`

Expected: 全部默认测试 PASS；仅外部集成测试 SKIP。

- [ ] **Step 4: 运行静态与差异检查**

Run: `python -m compileall -q src scripts tests`

Run: `git diff --check`

Expected: 两条命令 exit code 0。

- [ ] **Step 5: 执行代码生命周期审计**

确认新增逻辑只位于现有配置、fast-loop、CLI 和测试文件；无一次性脚本进入 `scripts/`，无运行数据进入 Git，`history/` 和 `runs/` 仍被忽略。

- [ ] **Step 6: 提交 Task 4**

```bash
git add docs/stage-4-fast-loop-validation.md docs/superpowers/specs/2026-07-09-tau3-retail-opd-evolver-design.md
git commit -m "docs: document memory ablation workflow"
```

- [ ] **Step 7: 最终 review gate**

检查 `git status --short` 只包含预期内容或为空；审查 enabled 与 disabled 两条路径的 provenance、凭证安全、异常路径和 Stage 5 数据隔离，再决定合并。
