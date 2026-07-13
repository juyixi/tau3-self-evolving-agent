# Task 4B 交付报告

## 状态

完成真实 `python -m scripts.run_fast_loop`、manifest learning provenance、五任务 opt-in tau2 smoke、30-task 无外部 API 调度 smoke 和中文验证指南。实现提交为 `82c7dd1`（`Add real fast-loop train CLI`）；本报告由后续独立提交保存。

## TDD 证据

### RED

执行：

```text
python -m pytest tests/unit/scripts/test_run_fast_loop.py tests/unit/runs/test_manifest.py tests/integration/test_fast_loop_tau2_retail.py -q --basetemp=.pytest-tmp/stage4-cli-red
```

结果：collection error，`scripts.run_fast_loop` 不存在，符合新 CLI 尚未实现的预期失败。

随后独立执行 manifest 测试：

```text
python -m pytest tests/unit/runs/test_manifest.py -q --basetemp=.pytest-tmp/stage4-manifest-red
```

结果：5 failed、15 passed。新增 provenance 用例因 `create_manifest()` 不接受 `adapter_revision` 失败，空 provenance 用例同样失败；原 baseline 用例保持通过。

### GREEN

执行：

```text
python -m pytest tests/unit/scripts/test_run_fast_loop.py tests/unit/runs/test_manifest.py tests/integration/test_fast_loop_tau2_retail.py -q --basetemp=.pytest-tmp/task4b
```

最终结果：`32 passed, 1 skipped in 0.39s`。跳过项是显式 opt-in 的真实 tau2 五任务集成测试。

### 全量验证

```text
python -m pytest -q --basetemp=.pytest-tmp/task4b-regression
python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

结果：全量 `353 passed, 3 skipped in 5.89s`；compileall 返回 0；两个 diff check 均返回 0。Git 只报告工作区 LF/CRLF 转换提示，没有 whitespace error。

## Artifact Schema

- `runs/<run_id>/manifest.json`：保留 schema version 2 和 baseline 精确默认输出；learning run 写入非空 `adapter_revision`（可选）与 immutable 输入 `memory_snapshot_id`，`parent_checkpoint` 支持 `str | None`。
- `runs/<run_id>/rollouts/events.jsonl`：使用真实 `JsonlWriter`；RunContext 固定 `mode=learn`、`split=train` 并携带 model、adapter、输入 snapshot provenance。
- `runs/<run_id>/fast_loop_summary.json`：canonical、原子写入；字段为 `run_id`、`episode_count`、`total_terminal_reward`、`successful_task_ids`、`maintenance_rounds_executed`、累计任务 before/after、输入/输出 snapshot ID。
- `history/agents/<agent_id>/memory/*_memory.json`：复用已审核四层 MemoryRepository；写入保留 `source_task_ids`。
- `history/agents/<agent_id>/memory/snapshots/<snapshot_id>/`：运行前后 immutable snapshots。
- `history/agents/<agent_id>/memory/maintenance_state.json`：复用已审核 maintenance scheduler 的 completed rounds 状态。

所有 manifest、event、summary 测试均检查凭证不泄漏；summary 不包含 raw model output 或 attribution。

## 30-Task Smoke

`test_thirty_successful_tasks_execute_exactly_maintenance_round_one` 使用真实临时 MemoryRepository、RunContext、JsonlWriter 和真实 `run_due_maintenance`。30 个 episode 均为确定性成功 fake，不访问 tau2、模型或 embedding API；maintenance policy 只返回合法空命令。

验证结果：前 29 个累计值不执行维护；第 30 个累计值触发且只触发 round 1；policy 调用 1 次；`maintenance_state.json.completed_rounds == [1]`；唯一 `MaintenanceStarted.completed_train_tasks == 30`。

## 五任务 Opt-In Smoke

`tests/integration/test_fast_loop_tau2_retail.py` 仅在以下条件全部满足时运行：

- `RUN_FAST_LOOP_TAU2_INTEGRATION=1`
- `QWEN_BASE_URL`
- `QWEN_MODEL_REVISION`
- `OPENROUTER_API_KEY`
- `configs/default.yaml` 对应 simulator 所需的其他 provider 凭证

测试从 pinned 官方 catalog 选择前五个 train ID `0, 1, 2, 3, 4`，将 run output 和 Memory 重定向到 pytest temp，调用真实 CLI，并检查五个 terminal episodes、candidate/selection/action/write 顺序、官方 terminal reward mappings、proposal/Memory source task provenance，以及 no-op write 时 snapshot 相等、有 canonical write 时 snapshot 改变。

本机未设置 opt-in 变量且 worktree 未配置完整 tau2 checkout，因此没有产生真实模型成本；该测试按契约跳过。

## 文件变更

- 新增 `scripts/run_fast_loop.py`
- 新增 `tests/unit/scripts/test_run_fast_loop.py`
- 新增 `tests/integration/test_fast_loop_tau2_retail.py`
- 新增 `docs/stage-4-fast-loop-validation.md`
- 修改 `src/tau3_retail_evolver/runs/manifest.py`
- 修改 `tests/unit/runs/test_manifest.py`
- 新增 `.superpowers/sdd/task-4b-report.md`

未修改已审核 fast-loop、memory、environment、model adapter、baseline script 或配置。未跟踪的 `docs/superpowers/plans/2026-07-13-stage-4-fast-loop.md` 属于其他开发者，未读取、修改或暂存。

## 自审

- 非 train split 在 config、tau2、Memory 访问前拒绝。
- run ID、iteration、累计任务数、model/adapter revision 在副作用前校验。
- task IDs 必须唯一且属于 pinned 官方 train split。
- runtime commit、catalog hash、gym factory、NL assertion 和 simulator probe 全部复用 baseline 路径。
- Memory 在 manifest 前 snapshot；manifest 和 RunContext 使用同一输入 snapshot ID。
- 每个任务创建 fresh Tau2RetailEnv；公开 instruction 不含 task ID，也不读取隐藏 task metadata。
- maintenance 仅在成功 episode 后按显式累计值调用。
- 失败保留 manifest/JSONL evidence，继续抛出原异常，不写 success summary。
- 已有 output run ID 在 config 访问前拒绝；manifest 继续拒绝 overwrite。
- baseline manifest 的 schema version、字段和 `None` 默认值由原精确输出测试覆盖。

## Concerns

唯一剩余 concern 是真实五任务 smoke 未在本机执行；它需要完整 pinned tau2 checkout、Qwen endpoint 和 simulator/OpenRouter 凭证，会产生真实模型调用成本。所有无成本单元、调度、全量回归和编译检查均已执行。
